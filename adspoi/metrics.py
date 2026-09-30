"""Per-event ranking metrics and optional paired t/Wilcoxon diagnostics."""
from typing import Dict, List, Sequence

import numpy as np

# Test-report cutoffs are separate from validation cutoffs in eval_ks.
REPORT_KS = (1, 5, 10, 20)
import torch


def poi_ranks(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Rank targets against real POIs only, excluding padding independently of its score."""
    if scores.ndim != 2 or scores.shape[1] < 2 or scores.shape[0] == 0:
        raise ValueError('POI scores must be a nonempty [batch, padding + vocabulary] tensor')
    if targets.ndim != 1 or len(targets) != len(scores):
        raise ValueError('One target POI is required per score row')
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError('Target POI IDs must be integers')
    if ((targets < 1) | (targets >= scores.shape[1])).any():
        raise ValueError('Target must be a real POI in 1..vocabulary_size, never padding')
    return per_sample_ranks(scores[:, 1:], targets.long() - 1)


def per_sample_ranks(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """One-based mid-rank of each target. Reject non-finite scores."""
    if not torch.isfinite(scores).all():
        n_bad = int((~torch.isfinite(scores)).sum())
        raise ValueError(
            f"{n_bad} of {scores.numel()} scores are NaN or Inf. Ranking them would "
            f"report rank 1 for every sample (every comparison with NaN is False), "
            f"i.e. a perfect 100% score from a broken model. Check for a diverged "
            f"loss or an unsupported device dtype.")
    tgt = scores.gather(1, targets.unsqueeze(1))
    greater = (scores > tgt).sum(1)
    tied = (scores == tgt).sum(1) - 1          # exclude the target itself
    return greater + 1 + tied.float() / 2.0


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = (5, 10, 20)) -> Dict[str, np.ndarray]:
    """Per-sample metric vectors. Mean of each vector is the reported number."""
    out: Dict[str, np.ndarray] = {}
    for k in ks:
        out[f"HR@{k}"] = (ranks <= k).astype(np.float64)
        out[f"NDCG@{k}"] = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    out["MRR"] = 1.0 / ranks
    return out


def summarize(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(v.mean()) for k, v in per_sample.items()}


def format_metrics(m: Dict[str, float], pct: bool = True) -> str:
    order = sorted(m.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1]) if "@" in s else 0))
    return " | ".join(f"{k}: {m[k] * 100:.2f}" if pct else f"{k}: {m[k]:.4f}" for k in order)


def aggregate_seeds(seed_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean and sample standard deviation across seed metrics."""
    if not seed_metrics:
        return {}
    keys = seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "values": [float(v) for v in vals],
        }
    return out


def paired_tests(ours: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    """Paired t-test and Wilcoxon signed-rank over per-sample metric values.

    Both arrays must be aligned sample-for-sample (same test set, same order).
    Reports Cohen's d_z for the paired design.
    """
    from scipy import stats

    ours = np.asarray(ours, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if ours.shape != baseline.shape:
        raise ValueError(f"unaligned arrays: {ours.shape} vs {baseline.shape}")

    diff = ours - baseline
    res = {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "t_stat": float("nan"), "t_p": float("nan"),
        "wilcoxon_stat": float("nan"), "wilcoxon_p": float("nan"),
        "cohens_dz": float("nan"),
    }
    if np.allclose(diff, 0):
        res["t_p"] = res["wilcoxon_p"] = 1.0
        return res

    t_stat, t_p = stats.ttest_rel(ours, baseline)
    res["t_stat"], res["t_p"] = float(t_stat), float(t_p)
    sd = diff.std(ddof=1)
    res["cohens_dz"] = float(diff.mean() / sd) if sd > 0 else float("nan")
    try:
        w_stat, w_p = stats.wilcoxon(ours, baseline, zero_method="wilcox")
        res["wilcoxon_stat"], res["wilcoxon_p"] = float(w_stat), float(w_p)
    except ValueError:
        pass
    return res


def significance_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
