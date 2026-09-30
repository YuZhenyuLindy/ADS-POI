#!/usr/bin/env python3
"""Serial, fail-fast experiment suites. Use --dry-run to inspect commands.

main: repeat_gate and the equal-input SASRec repeat baseline.
ablation: ungated full, no_decomposition (128-wide), single_state (32-wide),
          no_aggregation, no_heterogeneous.
sweep: repeat_gate at K=1,2,4,8 with total stored state width fixed at 128.

All jobs use full-vocabulary training, a fixed learning rate, and seeds 42/43/44
unless overridden explicitly. This runner does not select the best test result.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def commands(args):
    root = Path(__file__).resolve().parent.parent
    if args.suite == "main":
        variants = [("adspoi", "repeat_gate", None),
                    ("sasrec_st_repeat", "full", None)]
    elif args.suite == "ablation":
        variants = [("adspoi", v, None) for v in
                    ("full", "no_decomposition", "single_state", "no_aggregation", "no_heterogeneous")]
    else:
        variants = [("adspoi", "repeat_gate", k) for k in (1, 2, 4, 8)]
    for dataset in args.datasets:
        for model, variant, k in variants:
            for seed in args.seeds:
                cmd = [sys.executable, str(root / "run_experiment.py"),
                       "--dataset", dataset, "--model", model, "--variant", variant,
                       "--seed", str(seed), "--tag", args.suite, "--train-objective", "full",
                       "--profile", args.profile,
                       "--lr", str(args.lr), "--official-dir", str(args.official_dir),
                       "--out", str(args.out), "--device", args.device,
                       "--num-workers", str(args.num_workers), "--save-checkpoint"]
                if k is not None:
                    cmd += ["--K", str(k), "--d", "128"]
                yield cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=("main", "ablation", "sweep"))
    parser.add_argument("--datasets", nargs="+", default=["nyc", "tky", "ca"], choices=("nyc", "tky", "ca"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--profile", choices=("paper", "historical"), default="historical")
    parser.add_argument("--official-dir", type=Path, default=Path("data_official"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("lr must be positive and num-workers nonnegative")
    for command in commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
