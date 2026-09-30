"""Strict readers and sample-level inference for saved experiments.

The bootstrap averages a metric across fixed trained seeds for each check-in,
then resamples check-ins with replacement. Its uncertainty is conditional on
those trained models. It neither resamples users nor estimates training-seed
uncertainty. Repeated events from a user may be dependent. The reported p value
is the two-sided bootstrap tail estimate used by the source experiment, not a
randomization test. A zero estimate means no sampled replicate crossed zero.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Run:
    seed: int
    path: Path
    metadata: dict
    ids: np.ndarray
    ranks: np.ndarray


def common_arguments(parser):
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--tag', default='main')
    parser.add_argument('--datasets', nargs='+', choices=['nyc', 'tky', 'ca'], default=['nyc', 'tky', 'ca'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])


def bootstrap_arguments(parser):
    parser.add_argument('--nboot', type=int, default=20000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)


def source_identity(metadata):
    """Legacy artifacts may all omit source_hash, but may not mix with new ones."""
    if 'source_hash' not in metadata:
        return None
    value = metadata['source_hash']
    if not isinstance(value, str) or not value:
        raise ValueError('source_hash must be a nonempty string when present')
    return value


def load_runs(root, tag, dataset, model, seeds=(42, 43, 44), config_filter=None):
    """Select exactly one full-objective run per requested seed, or fail.

    No validation/test-based selection occurs here. Put a selected configuration
    in a dedicated tag before aggregation. Extra unrequested seeds are ignored.
    config_filter selects exact saved configuration values, e.g. one sweep K.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide a nonempty list of unique seeds')
    if not tag or any(c in tag for c in '/*?[]') or tag in ('.', '..'):
        raise ValueError('tag must be a single literal directory name')
    selected = {s: [] for s in seeds}
    for path in sorted((Path(root) / tag / dataset / model).glob('*/seed*/metrics.json')):
        with path.open() as f:
            meta = json.load(f)
        seed = meta.get('seed')
        if seed not in selected:
            continue
        if config_filter and any(meta.get('config', {}).get(k) != v for k, v in config_filter.items()):
            continue
        if path.parent.name != f'seed{seed}':
            raise ValueError(f'Seed metadata/path mismatch: {path}')
        if meta.get('dataset') != dataset or meta.get('variant') != model:
            raise ValueError(f'Dataset/model metadata mismatch: {path}')
        if meta.get('config', {}).get('train_objective') != 'full':
            raise ValueError(f'Expected full training objective: {path}')
        selected[seed].append((path, meta))
    runs = []
    signature = fingerprint = reference_ids = source = None
    for seed in sorted(seeds):
        matches = selected[seed]
        if len(matches) != 1:
            raise ValueError(f'{dataset}/{model}/seed{seed}: expected one run, found {len(matches)} in tag {tag}')
        path, meta = matches[0]
        cfg_sig = json.dumps(meta['config'], sort_keys=True, allow_nan=False)
        fp = meta.get('data_fingerprint')
        if not isinstance(fp, str) or not fp:
            raise ValueError(f'Missing data fingerprint: {path}')
        if signature is not None and (signature != cfg_sig or fingerprint != fp):
            raise ValueError(f'Mixed configurations or data across seeds: {path}')
        current_source = source_identity(meta)
        if runs and current_source != source:
            raise ValueError(f'Mixed source hashes across seeds: {path}')
        signature, fingerprint, source = cfg_sig, fp, current_source
        ids = np.load(path.parent / 'check_ins_id.npy', allow_pickle=False)
        ranks = np.load(path.parent / 'ranks.npy', allow_pickle=False)
        if ids.ndim != 1 or ranks.ndim != 1 or len(ids) != len(ranks) or len(ids) == 0:
            raise ValueError(f'Invalid rank/ID shapes: {path}')
        if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
            raise ValueError(f'IDs must be unique integers: {path}')
        if not np.isfinite(ranks).all() or (ranks < 1).any():
            raise ValueError(f'Invalid ranks: {path}')
        if meta.get('n_test_samples', len(ids)) != len(ids):
            raise ValueError(f'Sample count mismatch: {path}')
        order = np.argsort(ids)
        ids, ranks = ids[order], ranks[order].astype(np.float64)
        if reference_ids is not None and not np.array_equal(reference_ids, ids):
            raise ValueError(f'ID sets differ across seeds: {path}')
        reference_ids = ids
        runs.append(Run(seed, path.parent, meta, ids, ranks))
    return runs


def align_samples(a, b):
    if [r.seed for r in a] != [r.seed for r in b]:
        raise ValueError('Model seed sets differ')
    for x, y in zip(a, b):
        if not np.array_equal(x.ids, y.ids):
            raise ValueError(f'Model ID sets differ at seed {x.seed}')
        if source_identity(x.metadata) != source_identity(y.metadata):
            raise ValueError('Model source hashes differ, or mix present and absent hashes')
        if x.metadata['data_fingerprint'] != y.metadata['data_fingerprint']:
            raise ValueError('Model data fingerprints differ')


def align_models(a, b):
    align_samples(a, b)
    for x, y in zip(a, b):
        # Model-specific architecture settings may differ, but experimental
        # protocol and training settings must match for this controlled test.
        keys = ('train_objective', 'protocol_rev', 'max_history_len', 'repeat_history_len',
                'repeat_gate_hidden', 'learning_rate', 'batch_size', 'num_epochs',
                'warmup_epochs', 'early_stopping_patience', 'weight_decay', 'dropout',
                'label_smoothing', 'explore_weight', 'bpr_weight', 'bpr_margin',
                'gradient_clip', 'eval_ks', 'optimizer', 'schedule_profile')
        for key in keys:
            if x.metadata['config'].get(key) != y.metadata['config'].get(key):
                raise ValueError(f'Model training/protocol mismatch: {key}')


def metric(ranks, name):
    ranks = np.asarray(ranks, dtype=float)
    if name == 'MRR':
        return 1.0 / ranks
    family, k = name.split('@')
    hit = ranks <= int(k)
    if family == 'HR':
        return hit.astype(float)
    if family == 'NDCG':
        return hit / np.log2(ranks + 1.0)
    raise ValueError(f'Unknown metric: {name}')


def mean_metric(runs, name):
    return np.mean([metric(r.ranks, name) for r in runs], axis=0)


def paired_bootstrap(a, b, nboot=20000, seed=0, max_cells=262144):
    """Percentile paired bootstrap with bounded temporary sampling arrays.

    max_cells bounds each index matrix and each gathered difference matrix,
    independently of the sample count. The RNG draw layout is implementation
    dependent. It preserves the resampling distribution, not old bitwise draws.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Bootstrap requires nonempty aligned 1D samples')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite bootstrap input')
    if nboot < 1 or max_cells < 1:
        raise ValueError('nboot and max_cells must be positive')
    d, rng = a - b, np.random.default_rng(seed)
    boot = np.empty(nboot)
    rows = min(32, max_cells)
    for start in range(0, nboot, rows):
        count = min(rows, nboot - start)
        sums = np.zeros(count)
        width = max(1, max_cells // count)
        for col in range(0, len(d), width):
            idx = rng.integers(0, len(d), size=(count, min(width, len(d) - col)))
            sums += d[idx].sum(axis=1)
        boot[start:start + count] = sums / len(d)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return dict(n=len(d), a=float(a.mean()) * 100, b=float(b.mean()) * 100,
                delta=float(d.mean()) * 100, lo=float(lo) * 100, hi=float(hi) * 100,
                p=p, nboot=nboot, bootstrap_seed=seed, p_resolution=2 / nboot)


def provenance(runs):
    return [{'seed': r.seed, 'run_dir': str(r.path),
             'config_hash': r.metadata.get('config_hash'),
             'source_hash': source_identity(r.metadata),
             'data_fingerprint': r.metadata['data_fingerprint']} for r in runs]


LIMITATIONS = ('Sample bootstrap after averaging metrics across fixed trained seeds. '
              'This does not estimate seed-level uncertainty or account for user-level dependence. '
              'The p value is a bootstrap tail estimate. Zero is a finite-resampling estimate, '
              'not proof of zero probability. No causal attribution to an individual component follows.')


def emit(payload):
    print(json.dumps(payload, indent=2, allow_nan=False))


def cli(main):
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
