#!/usr/bin/env python3
"""Train one experiment and save auditable results.

Default: ADS-POI repeat_gate, full-vocabulary loss, shared learning rate 0.002.
Completed runs are reused only after loading the input data and verifying the
code/config/data identity and required artifacts. Checkpoints restore inference
models. they do not contain optimizer state for resuming interrupted training.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from adspoi.config import config_hash, resolve_config, source_hash
from adspoi.protocol import evaluation_protocol
from adspoi.model import VARIANT_NAMES as VARIANTS
from adspoi.data import create_dataloaders
from adspoi.official import OfficialDataMissing, load_official
from adspoi.engine import evaluate, measure_inference, state_analysis, train_model
from adspoi.model import build_model
from adspoi.metrics import REPORT_KS, format_metrics
from adspoi.utils import (append_jsonl, count_params, env_info, pick_device,
                          set_seed, setup_logger, write_json, read_json)


def run_dir(root: Path, tag: str, dataset: str, variant: str, cfg_hash: str, seed: int) -> Path:
    return root / tag / dataset / variant / cfg_hash / f"seed{seed}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["nyc", "tky", "ca"])
    p.add_argument("--model", default="adspoi",
                   choices=["adspoi", "gru4rec", "sasrec", "gru4rec_st", "sasrec_st",
                            "gru4rec_repeat", "sasrec_repeat",
                            "gru4rec_st_repeat", "sasrec_st_repeat"],
                   help="adspoi = the paper's model; others are sequence-only "
                        "baselines reimplemented under this package's training protocol")
    p.add_argument("--variant", default="repeat_gate", choices=list(VARIANTS),
                   help="ablation variant; ignored unless --model adspoi")
    p.add_argument("--profile", choices=["paper", "historical"], default="historical",
                   help="historical: source AdamW/schedule (default); paper: Adam and linear warmup")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="main", help="experiment group, e.g. main / ablation / sweep_K")

    p.add_argument("--K", type=int, default=None, dest="num_substates")
    p.add_argument("--d", type=int, default=None, dest="total_state_dim")
    p.add_argument("--epochs", type=int, default=None, dest="num_epochs")
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max-history-len", type=int, default=None)
    p.add_argument("--train-objective", default=None, choices=["sampled", "full"],
                   help="full-vocabulary training (default), or explicit sampled-loss comparison")

    p.add_argument("--out", default="runs", type=Path)
    p.add_argument("--official-dir", default=None, type=Path,
                   help="dir holding the official STHGCN/LLM4POI splits (default: ./data_official)")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force", action="store_true", help="rerun even if metrics.json exists")
    p.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                   help="save a restorable best-validation model (default: enabled)")
    p.add_argument("--no-state-analysis", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    overrides = {k: getattr(args, k) for k in
                 ("num_substates", "total_state_dim", "num_epochs", "learning_rate",
                  "batch_size", "dropout", "max_history_len", "train_objective")}
    # Vary stored state width independently of the shared POI embedding table.
    cfg = resolve_config(args.dataset, overrides, profile=args.profile)
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    cfg["num_workers"] = args.num_workers
    effective_variant = args.variant if args.model == "adspoi" else "full"
    run_name = effective_variant if args.model == "adspoi" else args.model
    if not args.tag or Path(args.tag).name != args.tag or args.tag in (".", ".."):
        raise ValueError("tag must be a single directory name")
    try:
        data = load_official(args.dataset, args.official_dir)
    except OfficialDataMissing as e:
        print(str(e), file=sys.stderr)
        return 2
    fingerprint = data["stats"]["data_fingerprint"]
    tau_d_rule = cfg["tau_d"]
    if tau_d_rule == "auto":
        from adspoi.data import derive_tau_d
        cfg["tau_d"] = round(derive_tau_d(data["train_data"]), 4)
    code_hash = source_hash()
    cfg_hash = config_hash(cfg, effective_variant, args.model, code_hash, fingerprint)
    out_dir = run_dir(args.out, args.tag, args.dataset, run_name, cfg_hash, args.seed)
    required = ["config.json", "history.json", "ranks.npy", "targets.npy",
                "check_ins_id.npy", "topk.npy"]
    if args.save_checkpoint:
        required.append("checkpoint.pt")
    if not args.no_state_analysis:
        required.append("state_analysis.json")
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        previous = read_json(metrics_path)
        matches = (previous.get("config_hash") == cfg_hash
                   and previous.get("source_hash") == code_hash
                   and previous.get("data_fingerprint") == fingerprint
                   and previous.get("model") == args.model
                   and previous.get("variant") == run_name
                   and previous.get("seed") == args.seed)
        if matches and all((out_dir / name).is_file() for name in required):
            print(f"[skip] verified completed {args.dataset}/{run_name}/seed{args.seed}")
            return 0
        raise RuntimeError("Existing result identity or artifacts are incomplete; inspect the run or use --force")
    out_dir.mkdir(parents=True, exist_ok=True)
    # metrics.json is the completion marker. a failed forced run must not retain it.
    if metrics_path.exists():
        metrics_path.unlink()
    logger = setup_logger(out_dir / "log.txt")
    device = pick_device(args.device)
    logger.info(f"dataset={args.dataset} model={args.model} variant={effective_variant} seed={args.seed}")
    logger.info(f"device={device} fingerprint={fingerprint} effective_tau_d={cfg['tau_d']}")
    provenance = env_info()
    set_seed(args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(data, cfg, args.num_workers)
    model = build_model(data["num_pois"], cfg, effective_variant, args.model)
    n_params = count_params(model)
    label = args.model if args.model != "adspoi" else f"adspoi/{args.variant}"
    logger.info(f"model   : {label} K={model.K} d_s={model.d_s} params={n_params:,}")

    actual_shape = {"K": int(model.K), "d_s": int(model.d_s),
                    "stored_state_width": int(model.K * model.d_s)}
    identity = {"dataset": args.dataset, "config": cfg, "config_hash": cfg_hash, "source_hash": code_hash,
                "evaluation_protocol": evaluation_protocol(cfg),
                "data_fingerprint": fingerprint, "model": args.model,
                "variant": run_name, "model_variant": effective_variant,
                "seed": args.seed, "num_pois": data["num_pois"],
                "actual_state": actual_shape, "tau_d_rule": tau_d_rule}
    write_json(out_dir / "config.json", {**identity, "env": provenance})

    t0 = time.time()
    model, history = train_model(model, train_loader, val_loader, cfg, device, logger)
    train_seconds = time.time() - t0

    # Report all test cutoffs while retaining the configured validation cutoffs.
    test_metrics, per_sample, extras = evaluate(model, test_loader, device, REPORT_KS,
                                                collect_alpha=False, collect_topk=True)
    logger.info(f"TEST    : {format_metrics(test_metrics)}")

    # Per-sample ranks are what make a paired test possible later.
    np.save(out_dir / "ranks.npy", extras["ranks"])
    np.save(out_dir / "targets.npy", extras["targets"])
    # Check-in IDs align results from different implementations.
    np.save(out_dir / "check_ins_id.npy", extras["check_ins_id"])
    # Save recommended POIs for coverage and diversity analyses.
    np.save(out_dir / "topk.npy", extras["topk"])

    timing = measure_inference(model, test_loader, device, cfg["eval_batch_size"])
    logger.info(f"latency : {timing['per_query_latency_ms']:.3f} ms/query, "
                f"{timing['throughput_queries_per_s']:.1f} q/s (batch={timing['batch_size']})")

    if not args.no_state_analysis:
        sa = state_analysis(model, test_loader, device, data["poi_categories"])
        write_json(out_dir / "state_analysis.json", sa)

    if args.save_checkpoint:
        checkpoint = {**identity, "state_dict": model.state_dict(),
                      "best_epoch": history["best_epoch"], "format_version": 1}
        temporary = out_dir / "checkpoint.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(out_dir / "checkpoint.pt")

    write_json(out_dir / "history.json", history)
    write_json(out_dir / "metrics.json", {
        **identity,
        "dataset": args.dataset, "variant": run_name, "model": args.model,
        "seed": args.seed, "config_hash": cfg_hash, "config": cfg,
        "data_fingerprint": data["stats"]["data_fingerprint"],
        "test": test_metrics,
        "best_val_hr10": history["best_val_hr10"],
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["epochs"]),
        "stopped_early": history["stopped_early"],
        "train_seconds": train_seconds,
        "num_params": n_params,
        "timing": timing,
        "dataset_stats": data["stats"],
        "n_test_samples": int(len(extras["ranks"])),
        "env": provenance,
    })

    append_jsonl(args.out / "results.jsonl", {
        "tag": args.tag, "dataset": args.dataset, "variant": run_name,
        "model": args.model, "seed": args.seed, "config_hash": cfg_hash,
        "K": model.K, "d_s": model.d_s, "d": model.K * model.d_s,
        "test": test_metrics, "run_dir": out_dir.relative_to(args.out).as_posix(),
        "source_hash": code_hash, "data_fingerprint": fingerprint,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    logger.info(f"done in {train_seconds / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
