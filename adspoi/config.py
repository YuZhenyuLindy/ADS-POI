"""Shared chronological-split experiment configuration.

The default historical profile retains the source experiment AdamW and schedule.
The optional paper profile uses Adam and a three-epoch linear warmup.
The headline model is repeat_gate with full-vocabulary training. Preprocessing and splits are supplied as data_official CSV files.
"""
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

# tau_d="auto" resolves to median training displacement before hashing/saving.
# num_negatives is used only for explicitly selected sampled training.
BASE_CONFIG = {'total_state_dim': 128,
 'num_substates': 4,
 'substate_dim': 32,
 'poi_embed_dim': 128,
 'slot_embed_dim': 16,
 'spatial_dim': 32,
 'dist_embed_dim': 16,
 'num_dist_buckets': 8,
 'num_time_slots': 4,
 'agg_hidden_dim': 64,
 'temperature': 1.0,
 'lambda_init': 0.1,
 'mu_init': 0.1,
 'tau_t': 24.0,
 'tau_d': 'auto',
 'train_objective': 'full',
 'num_negatives': 499,
 'batch_size': 512,
 'eval_batch_size': 1024,
 'num_epochs': 50,
 'learning_rate': 0.002,
 'optimizer': 'adamw',
 'schedule_profile': 'historical',
 'weight_decay': 0.0001,
 'dropout': 0.1,
 'gradient_clip': 5.0,
 'early_stopping_patience': 10,
 'warmup_epochs': 3,
 'label_smoothing': 0.02,
 'explore_weight': 1.5,
 'bpr_weight': 0.5,
 'bpr_margin': 1.0,
 'max_history_len': 50,
 'eval_ks': [5, 10],
 'protocol_rev': 2,
 'repeat_gate_hidden': 32,
 'repeat_history_len': 512}

DATASETS = ("nyc", "tky", "ca")
VARIANTS = ("full", "full_routed", "no_decomposition", "no_aggregation",
            "no_heterogeneous", "single_state", "single_state_32", "single_state_128",
            "repeat_gate", "repeat_fixed")


def resolve_config(dataset: str, overrides=None, profile="historical") -> Dict[str, Any]:
    """Resolve shared settings, rejecting silent width truncation in K sweeps."""
    ds = dataset.lower()
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    if profile not in ("paper", "historical"):
        raise ValueError("profile must be paper or historical")
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["optimizer"] = "adam" if profile == "paper" else "adamw"
    cfg["schedule_profile"] = profile
    if overrides:
        unknown = set(overrides) - set(BASE_CONFIG)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    expected_optimizer = "adam" if profile == "paper" else "adamw"
    if cfg["optimizer"] != expected_optimizer or cfg["schedule_profile"] != profile:
        raise ValueError("optimizer and schedule_profile must agree with the selected profile")
    if not isinstance(cfg["warmup_epochs"], int) or cfg["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    cfg["dataset"] = ds
    K, width = cfg["num_substates"], cfg["total_state_dim"]
    if not isinstance(K, int) or not isinstance(width, int) or K < 1 or width < 1:
        raise ValueError("K and total state width must be positive integers")
    if width % K:
        raise ValueError("total state width must be divisible by K")
    cfg["substate_dim"] = width // K
    for key in ("num_epochs", "batch_size", "eval_batch_size", "poi_embed_dim",
                "early_stopping_patience", "max_history_len", "repeat_history_len"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["learning_rate"] <= 0 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("learning_rate must be positive and dropout in [0,1)")
    if cfg["train_objective"] not in ("full", "sampled"):
        raise ValueError("train_objective must be full or sampled")
    if cfg["max_history_len"] < 2:
        raise ValueError("max_history_len must be at least 2 for spatial context")
    return cfg


def source_hash() -> str:
    """Hash executable experiment sources using portable relative filenames."""
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "adspoi").glob("*.py")) + [root / "run_experiment.py"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def config_hash(cfg, variant=None, model="adspoi", code_hash=None,
                data_fingerprint=None) -> str:
    """Identify resolved settings, model/variant, executable code and data.

    All settings are included conservatively, including repeat features for
    baselines. Changing an unused setting can create a new run, never reuse an
    incompatible one. No private paths or machine identifiers enter the hash.
    """
    payload = {"config": cfg, "model": model, "variant": variant,
               "source_hash": code_hash or source_hash(),
               "data_fingerprint": data_fingerprint}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


PROTOCOL_KEYS = ("max_history_len", "eval_ks", "train_objective", "protocol_rev",
                 "optimizer", "schedule_profile")


def assert_uniform(configs, keys=None, strict_model=True):
    if len(configs) < 2:
        return
    check = list(keys) if keys else list(PROTOCOL_KEYS)
    if strict_model and keys is None:
        check += [k for k in BASE_CONFIG if k != "tau_d"]
    for key in set(check):
        values = {json.dumps(c.get(key), sort_keys=True) for c in configs.values()}
        if len(values) > 1:
            raise ValueError(f"configurations differ on {key}")
