# Data preparation

The code consumes preprocessed chronological splits for Foursquare NYC/TKY and Gowalla CA. Obtain the datasets and preprocessing instructions from the upstream [LLM4POI repository](https://github.com/neolifer/LLM4POI) and its [dataset distribution](https://huggingface.co/datasets/w11wo/LLM4POI). Follow the upstream data terms. This repository does not redistribute check-ins or silently download data.

Use the split files from the original experiment when available. A fresh upstream preprocessing run must be checked against the reference sizes and fingerprints before comparing results. The exact upstream preprocessing commit used by the historical runs is not bundled, so row counts alone do not establish byte-for-byte identity.

## Required layout

```text
data_official/
  nyc/
    train_sample.csv
    validate_sample_with_traj.csv
    test_sample_with_traj.csv
  tky/
    train_sample.csv
    validate_sample_with_traj.csv
    test_sample_with_traj.csv
  ca/
    train_sample.csv
    validate_sample_with_traj.csv
    test_sample_with_traj.csv
```

If an upstream checkout is at `LLM4POI/` and its preprocessing has completed, copy the files with:

```bash
python scripts/prepare_data.py --source LLM4POI/datasets --out data_official
python scripts/check_data.py --official-dir data_official
```

The source may instead be an existing directory containing `nyc/`, `tky/` and `ca/` split files. `prepare_data.py` only copies files and writes checksums. It refuses to overwrite different existing files. `LLM4POI/datasets` in the command denotes a local upstream checkout, not a bundled directory.

## Schema and conventions

Required columns are `UserId`, `PoiId`, `UTCTimeOffset`, `Latitude`, `Longitude`, `pseudo_session_trajectory_id` and `check_ins_id`. `PoiCategoryName` is optional. Check-in IDs must uniquely identify evaluation events.

NYC uses zero-based real POI IDs, which the loader shifts by one. TKY/CA use one-based real POI IDs. Index zero in model tensors always denotes padding.

`UTCTimeOffset` is interpreted as local wall time for NYC/TKY and UTC for CA. Local hour and weekday are derived with the dataset time zone. `UTCTimeOffsetEpoch` is not used. Training data determine the vocabulary, popularity, location metadata and median displacement scale.

Reference sizes from the available chronological experiment records:

| Dataset | Training events | Validation events | Test events | Real POIs |
|---|---:|---:|---:|---:|
| NYC | 83,228 | 9,458 | 9,074 | 4,980 |
| TKY | 326,258 | 38,653 | 38,237 | 7,832 |
| CA | 190,627 | 18,299 | 12,791 | 9,689 |

These are reference checks, not a command to trim another dataset to those sizes. The model is evaluated on each eligible event using prior history. Do not substitute a per-user leave-one-out split for these experiments.

The loader preserves stable order for tied timestamps. It records complete input-file SHA-256 hashes and a separate historical sequence fingerprint. Code hashes capture loader changes. The complete file fingerprint also detects changed coordinates or metadata that a POI/time-only fingerprint would miss.

The default loader verifies the reference sizes above and the recorded sequence fingerprints: NYC `c2e725a8fcb7e66a`, TKY `0052dc5000529136`, CA `f71323bfb1675312`. These fingerprints come from the supplied full-objective experiment outputs. They constrain the event sequences and candidate sizes, not every coordinate or metadata value in the original files. Input-file hashes from the historical runs are not available in this release.

All held-out users and POIs must be actual training members, real POI IDs must be contiguous, and event IDs must be globally unique across the three splits. Future history, malformed timestamps and missing histories cause an error. The loader neither drops invalid rows nor silently repairs a split. See [EVALUATION.md](EVALUATION.md) for event selection, visible history, candidate ranking and aggregation conventions.
