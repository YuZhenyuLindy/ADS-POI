"""Build training and rolling evaluation samples from supplied official splits."""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DataUnavailable(RuntimeError):
    """Kept for API compatibility. Real data errors now raise OfficialDataMissing."""


def derive_tau_d(train_data: Dict) -> float:
    """Median displacement in kilometres between consecutive training events, floored at 0.1."""
    R = 6371.0
    d = []
    for traj in train_data.values():
        if len(traj) < 2:
            continue
        lat = np.radians(np.array([c["latitude"] for c in traj]))
        lon = np.radians(np.array([c["longitude"] for c in traj]))
        dlat, dlon = np.diff(lat), np.diff(lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
        d.append(R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    if not d:
        raise ValueError("cannot derive tau_d: no multi-check-in trajectories")
    med = float(np.median(np.concatenate(d)))
    return max(med, 0.1)  # a degenerate median would switch the decay off


def _pack_repeat_history(history: List[Dict], max_len: int) -> List[int]:
    """Return the most recent POI IDs within the repeat-feature window."""
    return [c["poi_idx"] for c in history[-max_len:]]


def _pack(history: List[Dict], max_len: int, default_lat: float, default_lon: float,
          t_ref: float = 0.0):
    """Truncate to the last max_len, then LEFT-pad to a fixed width.

    Timestamps come out as HOURS SINCE t_ref, not raw epoch seconds. See
    official.load_official for why: differences are all the model uses, and the
    smaller magnitude is what makes float32 safe (and MPS possible)."""
    h = history[-max_len:]
    n = len(h)
    pad = max_len - n
    poi = [0] * pad + [c["poi_idx"] for c in h]
    ts = [(h[0]["ts_utc"] - t_ref) / 3600.0] * pad + [(c["ts_utc"] - t_ref) / 3600.0 for c in h]
    hour = [0.0] * pad + [c["hour"] for c in h]
    dow = [0] * pad + [c["dow"] for c in h]
    loc = [[default_lat, default_lon]] * pad + [[c["latitude"], c["longitude"]] for c in h]
    return poi, ts, hour, dow, loc, n


def _tensors(poi, ts, hour, dow, loc, n, target, t_ref=0.0, rep_hist=None, rep_len=0):
    d = {
        "poi_ids": torch.tensor(poi, dtype=torch.long),
        "ts_hours": torch.tensor(ts, dtype=torch.float32),   # hours since t_ref
        "hour": torch.tensor(hour, dtype=torch.float32),
        "dow": torch.tensor(dow, dtype=torch.long),
        "locations": torch.tensor(loc, dtype=torch.float32),
        "seq_len": torch.tensor(n, dtype=torch.long),
        "target_poi": torch.tensor(target["poi_idx"], dtype=torch.long),
        "target_hour": torch.tensor(target["hour"], dtype=torch.float32),
        "target_dow": torch.tensor(target["dow"], dtype=torch.long),
        "target_location": torch.tensor([target["latitude"], target["longitude"]], dtype=torch.float32),
    }
    if rep_hist is not None:
        pad = rep_len - len(rep_hist)
        d["repeat_hist"] = torch.tensor([0] * pad + rep_hist, dtype=torch.long)
    return d


class POITrainDataset(Dataset):
    def __init__(self, train_data, num_pois, poi_popularity, config,
                 default_lat=0.0, default_lon=0.0, t_ref=0.0):
        self.num_pois = num_pois
        self.t_ref = t_ref
        self.sample_negatives = config.get("train_objective", "sampled") != "full"
        self.num_negatives = config["num_negatives"]
        self.max_history_len = config["max_history_len"]
        self.repeat_history_len = config.get("repeat_history_len", 512)
        self.default_lat, self.default_lon = default_lat, default_lon
        self.valid_idx = np.where(poi_popularity > 0)[0]
        p = poi_popularity[self.valid_idx]
        self.valid_probs = p / p.sum()
        self._cdf = np.cumsum(self.valid_probs)
        self._cdf[-1] = 1.0

        self.samples = []
        for uid, traj in train_data.items():
            seen = set()
            for i in range(1, len(traj)):
                seen.add(traj[i - 1]["poi_idx"])
                self.samples.append({"history": traj[:i], "target": traj[i],
                                     "is_explore": traj[i]["poi_idx"] not in seen})

    def __len__(self):
        return len(self.samples)

    def _sample_popularity(self, size: int) -> np.ndarray:
        """Popularity-weighted draw via the prebuilt CDF. Same distribution as
        np.random.choice(p=self.valid_probs), without its per-call O(|V|) setup."""
        return self.valid_idx[np.searchsorted(self._cdf, np.random.random(size))]

    def _sample_negatives(self, target: int) -> List[int]:
        """Draw popularity and uniform negatives, excluding the target."""
        n_pop = self.num_negatives // 2
        neg = set()
        for _ in range(8):
            if len(neg) >= n_pop:
                break
            neg.update(int(c) for c in self._sample_popularity(n_pop * 2) if c != target)
            if len(neg) > n_pop:
                neg = set(list(neg)[:n_pop])
        for _ in range(8):
            if len(neg) >= self.num_negatives:
                break
            for c in np.random.randint(1, self.num_pois, size=(self.num_negatives - len(neg)) * 2):
                if c != target:
                    neg.add(int(c))
                if len(neg) >= self.num_negatives:
                    break
        out = list(neg)[: self.num_negatives]
        while len(out) < self.num_negatives:
            r = int(np.random.randint(1, self.num_pois))
            if r != target:
                out.append(r)
        return out

    def __getitem__(self, idx):
        s = self.samples[idx]
        packed = _pack(s["history"], self.max_history_len, self.default_lat,
                       self.default_lon, self.t_ref)
        d = _tensors(*packed, s["target"], self.t_ref,
                     _pack_repeat_history(s["history"], self.repeat_history_len),
                     self.repeat_history_len)
        if self.sample_negatives:
            d["neg_ids"] = torch.tensor(self._sample_negatives(s["target"]["poi_idx"]), dtype=torch.long)
        d["is_explore"] = torch.tensor(float(s["is_explore"]), dtype=torch.float32)
        return d


class POIEvalDataset(Dataset):
    def __init__(self, history_base: Dict, eval_data: Dict, config,
                 default_lat=0.0, default_lon=0.0, t_ref=0.0):
        self.max_history_len = config["max_history_len"]
        self.repeat_history_len = config.get("repeat_history_len", 512)
        self.t_ref = t_ref
        self.default_lat, self.default_lon = default_lat, default_lon
        self.samples = []
        for uid, traj in eval_data.items():
            if uid not in history_base or not history_base[uid]:
                raise ValueError(f"evaluation user {uid} has no prior history; refusing to drop samples")
            base = history_base[uid]
            for label, events in (("history", base), ("evaluation", traj)):
                times = np.array([e["ts_utc"] for e in events], dtype=np.float64)
                if not np.isfinite(times).all() or (np.diff(times) < 0).any():
                    raise ValueError(f"user {uid}: {label} events must have finite, nondecreasing timestamps")
            if traj and base[-1]["ts_utc"] > traj[0]["ts_utc"]:
                raise ValueError(f"user {uid}: evaluation boundary includes future history")
            # Tied timestamps retain supplied event order. Sessions do not reset history.
            for i in range(len(traj)):
                self.samples.append({"history": base + traj[:i], "target": traj[i]})

    def __len__(self):
        return len(self.samples)

    @property
    def check_ins_ids(self) -> np.ndarray:
        """Return target check-in IDs in sample order.

        Samples are grouped by user, so their order can differ from CSV rows.
        Join external results by these IDs rather than row position.
        """
        return np.array([s["target"]["check_ins_id"] for s in self.samples], dtype=np.int64)

    def __getitem__(self, idx):
        s = self.samples[idx]
        packed = _pack(s["history"], self.max_history_len, self.default_lat,
                       self.default_lon, self.t_ref)
        return _tensors(*packed, s["target"], self.t_ref,
                        _pack_repeat_history(s["history"], self.repeat_history_len),
                        self.repeat_history_len)


def collate_fn(batch):
    keys = ["poi_ids", "ts_hours", "hour", "dow", "locations", "target_poi",
            "target_hour", "target_dow", "target_location"]
    if "repeat_hist" in batch[0]:
        keys = keys + ["repeat_hist"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["seq_lengths"] = torch.stack([b["seq_len"] for b in batch])
    for k in ("neg_ids", "is_explore"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def _worker_init(worker_id: int) -> None:
    """Initialize a separate NumPy random stream for each worker."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def create_dataloaders(data: Dict, config: Dict, num_workers: int = 0):
    t_ref = data.get("t_ref", 0.0)
    train_ds = POITrainDataset(data["train_data"], data["num_pois"], data["poi_popularity"],
                               config, data["default_lat"], data["default_lon"], t_ref)
    val_ds = POIEvalDataset(data["train_data"], data["val_data"], config,
                            data["default_lat"], data["default_lon"], t_ref)
    train_plus_val = {uid: traj + data["val_data"].get(uid, [])
                      for uid, traj in data["train_data"].items()}
    test_ds = POIEvalDataset(train_plus_val, data["test_data"], config,
                             data["default_lat"], data["default_lon"], t_ref)

    ebs = config.get("eval_batch_size", config["batch_size"])
    mk = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh, collate_fn=collate_fn,
                                       num_workers=num_workers,
                                       worker_init_fn=_worker_init if num_workers else None,
                                       persistent_workers=num_workers > 0,
                                       pin_memory=torch.cuda.is_available())
    print(f"[data] samples: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    return (mk(train_ds, config["batch_size"], True), mk(val_ds, ebs, False), mk(test_ds, ebs, False))
