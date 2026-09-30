"""Load prepared STHGCN/LLM4POI split files without further filtering or splitting.

See DATA.md for the input schema and ID conventions. Timestamps are parsed
from UTCTimeOffset. Index zero is reserved for padding."""
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .timeparse import TIME_SPEC, assert_human_rhythm, parse_times

DEFAULT_OFFICIAL = Path(__file__).resolve().parent.parent / "data_official"

FILES = {
    "train": "train_sample.csv",
    "val": "validate_sample_with_traj.csv",
    "test": "test_sample_with_traj.csv",
}

SOURCE = {
    "nyc": {"note": "Foursquare New York City", "platform": "Foursquare",
            "provenance": "GETNext pre-split NYC_{train,val,test}.csv"},
    "tky": {"note": "Foursquare Tokyo", "platform": "Foursquare",
            "provenance": "STHGCN filter(9,9) + global chronological 80/10/10"},
    "ca": {"note": "Gowalla California", "platform": "Gowalla",
           "provenance": "STHGCN filter(9,9) applied TWICE + global chronological 80/10/10"},
}

ID_CONVENTION = {"nyc": "zero_indexed", "tky": "one_indexed", "ca": "one_indexed"}


class OfficialDataMissing(RuntimeError):
    pass


def _read(ds: str, root: Path) -> Dict[str, pd.DataFrame]:
    d = root / ds
    out = {}
    for split, fn in FILES.items():
        p = d / fn
        if not p.exists():
            raise OfficialDataMissing(
                f"{p} not found. Prepare the official split files as described "
                f"in DATA.md. No substitute preprocessing is applied.")
        out[split] = pd.read_csv(p, low_memory=False)
    return out


REQUIRED_COLUMNS = ("UserId", "PoiId", "UTCTimeOffset", "Latitude", "Longitude",
                    "pseudo_session_trajectory_id", "check_ins_id")
ID_COLUMNS = ("UserId", "PoiId", "pseudo_session_trajectory_id", "check_ins_id")


def _validate_raw(raw: Dict[str, pd.DataFrame], ds: str) -> None:
    """Reject invalid prepared splits without filtering, renumbering or re-splitting.

    Equal timestamps are permitted. Their order is the original per-user CSV
    row order, with train preceding validation and validation preceding test.
    """
    seen_event_ids = set()
    parsed = {}
    for name in FILES:
        frame = raw[name]
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{ds}/{name}: missing required columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{ds}/{name}: empty prepared split")
        for col in (*ID_COLUMNS, "Latitude", "Longitude"):
            try:
                numeric = pd.to_numeric(frame[col], errors="raise")
                values = numeric.to_numpy(dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{ds}/{name}: {col} must be numeric") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{ds}/{name}: {col} contains missing or non-finite values")
            if col in ID_COLUMNS:
                if (values != np.floor(values)).any() or (values < 0).any() or (values >= 2**63).any():
                    raise ValueError(f"{ds}/{name}: {col} must contain nonnegative int64 IDs")
            elif col == "Latitude" and (np.abs(values) > 90).any():
                raise ValueError(f"{ds}/{name}: Latitude outside [-90,90]")
            elif col == "Longitude" and (np.abs(values) > 180).any():
                raise ValueError(f"{ds}/{name}: Longitude outside [-180,180]")
        ids = frame["check_ins_id"].astype("int64")
        if ids.duplicated().any() or seen_event_ids.intersection(ids.tolist()):
            raise ValueError(f"{ds}/{name}: duplicate check_ins_id within or across splits; "
                             "provide unique upstream event IDs, do not silently renumber or deduplicate")
        seen_event_ids.update(ids.tolist())
        times = parse_times(frame, ds)
        if not np.isfinite(times["ts_utc"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{ds}/{name}: non-finite parsed timestamps")
        for uid, group in times.groupby("UserId", sort=False):
            if (np.diff(group["ts_utc"].to_numpy()) < 0).any():
                raise ValueError(f"{ds}/{name}: user {uid} events are out of chronological order")
        parsed[name] = times

    train_pois = set(raw["train"]["PoiId"].astype("int64"))
    train_users = set(raw["train"]["UserId"].astype("int64"))
    first = 0 if ID_CONVENTION[ds] == "zero_indexed" else 1
    if min(train_pois) != first or max(train_pois) - first + 1 != len(train_pois):
        raise ValueError(f"{ds}/train: POI vocabulary must be contiguous from {first}; "
                         "holes would introduce candidates absent from training")
    for name in ("val", "test"):
        if not set(raw[name]["PoiId"].astype("int64")).issubset(train_pois):
            raise ValueError(f"{ds}/{name}: unseen POI not in the actual training vocabulary")
        if not set(raw[name]["UserId"].astype("int64")).issubset(train_users):
            raise ValueError(f"{ds}/{name}: unknown user without training history")

    latest = {}
    for name in FILES:
        for uid, group in parsed[name].groupby("UserId", sort=False):
            first_time, last_time = group["ts_utc"].iloc[0], group["ts_utc"].iloc[-1]
            if uid in latest and first_time < latest[uid]:
                raise ValueError(f"{ds}/{name}: user {uid} split boundary includes future history")
            latest[uid] = last_time


def poi_shift(ds: str) -> int:
    """Offset mapping the official PoiId onto our vocabulary, where 0 = padding.

    zero_indexed (nyc): official ids 0..N-1 -> 1..N, shift +1.
    one_indexed  (tky, ca): official ids are already 1..N with 0 as their own
                            padding bucket -> no shift.
    """
    return 1 if ID_CONVENTION[ds] == "zero_indexed" else 0


def real_id_range(ds: str, train: pd.DataFrame, col: str) -> Tuple[int, int]:
    """(n_distinct_real, our_max_index) for a label-encoded column."""
    if ID_CONVENTION[ds] == "zero_indexed":
        n = int(train[col].max()) + 1        # 0..max are all real. padding is max+1
    else:
        n = int(train[col].max())            # 1..max are real. 0 is padding
    return n, n


def _trajectories(df: pd.DataFrame, ds: str) -> Dict[int, List[Dict]]:
    df = parse_times(df, ds)
    df = df.assign(_source_order=np.arange(len(df))).sort_values(
        ["UserId", "ts_utc", "_source_order"], kind="mergesort")
    has_cat = "PoiCategoryName" in df.columns
    cols = ["UserId", "PoiId", "ts_utc", "local_hour", "local_dow",
            "Latitude", "Longitude", "pseudo_session_trajectory_id", "check_ins_id"]
    if has_cat:
        cols.append("PoiCategoryName")

    shift = poi_shift(ds)
    trajs: Dict[int, List[Dict]] = {}
    for row in df[cols].itertuples(index=False, name=None):
        trajs.setdefault(int(row[0]), []).append({
            "poi_idx": int(row[1]) + shift,      # 0 reserved for padding
            "ts_utc": float(row[2]),
            "hour": float(row[3]),
            "dow": int(row[4]),
            "latitude": float(row[5]),
            "longitude": float(row[6]),
            "traj_id": int(row[7]),
            "check_ins_id": int(row[8]),
            "category": row[9] if has_cat else "Unknown",
        })
    return trajs


def fingerprint(splits: Dict[str, Dict], num_pois: int) -> str:
    h = hashlib.sha256()
    h.update(f"OFFICIAL_V1|{num_pois}".encode())
    for name in ("train", "val", "test"):
        h.update(f"|{name}|".encode())
        for uid in sorted(splits[name]):
            h.update(f"{uid}:".encode())
            h.update(np.array([c["poi_idx"] for c in splits[name][uid]], dtype=np.int64).tobytes())
            h.update(np.array([c["ts_utc"] for c in splits[name][uid]], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def load_official(ds: str, root: Path = None, verify_reference: bool = True) -> Dict:
    ds = ds.lower()
    root = Path(root or DEFAULT_OFFICIAL)
    info = SOURCE[ds]
    print(f"\n{'=' * 68}\n[official] {ds.upper()} ({info['note']}, {info['platform']})\n"
          f"[official] provenance: {info['provenance']}\n{'=' * 68}")

    raw = _read(ds, root)
    _validate_raw(raw, ds)
    file_hashes = {}
    for split, filename in FILES.items():
        digest = hashlib.sha256()
        with (root / ds / filename).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        file_hashes[split] = digest.hexdigest()
    complete_fingerprint = hashlib.sha256(
        '|'.join(f'{key}:{file_hashes[key]}' for key in sorted(file_hashes)).encode()
    ).hexdigest()
    for k, df in raw.items():
        print(f"[official] {k:5s}: {len(df):>7,} check-ins  "
              f"({(root / ds / FILES[k]).stat().st_size / 1e6:.1f} MB)")

    shift = poi_shift(ds)
    n_train_pois, max_idx = real_id_range(ds, raw["train"], "PoiId")
    n_users = int(raw["train"]["UserId"].nunique())
    num_pois = max_idx + 1                           # index 0 is our padding

    splits = {k: _trajectories(df, ds) for k, df in raw.items()}
    used = {c["poi_idx"] for s in splits.values() for t in s.values() for c in t}
    assert 0 not in used, f"{ds}: padding index 0 leaked into the data"
    dead = set(range(1, num_pois)) - used
    if dead:
        print(f"[official] note: {len(dead)} vocabulary slots never appear in any split")

    all_df = parse_times(pd.concat(raw.values()), ds)
    st = assert_human_rhythm(all_df["local_hour"].values, ds)

    # File membership, not an optional user-supplied column, defines training data.
    train_df = parse_times(raw["train"], ds)
    poi_locations = np.zeros((num_pois, 2), dtype=np.float64)
    poi_categories: Dict[int, str] = {}
    agg = {"Latitude": "mean", "Longitude": "mean"}
    if "PoiCategoryName" in train_df.columns:
        agg["PoiCategoryName"] = lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"
    for pid, row in train_df.groupby("PoiId").agg(agg).iterrows():
        i = int(pid) + shift
        if not 1 <= i < num_pois:
            continue                                 # the official padding bucket
        poi_locations[i] = [row["Latitude"], row["Longitude"]]
        poi_categories[i] = row.get("PoiCategoryName", "Unknown")

    counts = np.zeros(num_pois, dtype=np.float64)
    for traj in splits["train"].values():
        for c in traj:
            counts[c["poi_idx"]] += 1
    counts[0] = 0.0
    popularity = counts / counts.sum()

    n_check = sum(len(df) for df in raw.values())
    n_traj = int(pd.concat(raw.values())["pseudo_session_trajectory_id"].nunique())
    stats = {
        "dataset": ds.upper(), "platform": info["platform"], "note": info["note"],
        "provenance": info["provenance"], "id_convention": ID_CONVENTION[ds],
        "users": n_users, "pois": n_train_pois, "checkins": n_check,
        "trajectories": n_traj,
        "sparsity_pct": 100 * (1 - n_check / (n_users * n_train_pois)),
        "avg_traj_len": n_check / n_traj,
        "train_checkins": len(raw["train"]), "val_checkins": len(raw["val"]),
        "test_checkins": len(raw["test"]),
        "train_users": len(splits["train"]), "test_users": len(splits["test"]),
        "tz": TIME_SPEC[ds]["tz"], "time_column_means": TIME_SPEC[ds]["column_means"],
        "rhythm_trough_hour": st["argmin_hour"], "rhythm_day_night_ratio": st["day_night_ratio"],
        "num_pois_incl_pad": num_pois,
        "sequence_fingerprint": fingerprint(splits, num_pois),
        "data_fingerprint": complete_fingerprint,
        "source_files_sha256": file_hashes,
    }
    print(f"[official] users={n_users:,} POIs={n_train_pois:,} check-ins={n_check:,} "
          f"trajectories={n_traj:,}")
    print(f"[official] local time OK (trough {st['argmin_hour']:02d}:00, "
          f"day/night {st['day_night_ratio']:.1f}) via {TIME_SPEC[ds]['tz']}, "
          f"column read as {TIME_SPEC[ds]['column_means']}")
    print(f"[official] fingerprint {stats['data_fingerprint']}")

    t_ref = min(c["ts_utc"] for t in splits["train"].values() for c in t)

    result = {
        "t_ref": float(t_ref),
        "train_data": splits["train"], "val_data": splits["val"], "test_data": splits["test"],
        "poi_locations": poi_locations, "poi_categories": poi_categories,
        "poi_popularity": popularity, "num_pois": num_pois, "stats": stats,
        "default_lat": float(train_df["Latitude"].mean()),
        "default_lon": float(train_df["Longitude"].mean()),
    }
    stats["reference_protocol_verified"] = False
    if verify_reference:
        from .protocol import validate_reference
        metadata = validate_reference(result)
        stats["reference_protocol_verified"] = True
        stats["reference_protocol"] = metadata
    return result
