# Release validation

Local validation used Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1 and pytest 8.4.2 on an ARM64 CPU. The exact installed packages are listed in `requirements-tested.txt`. GPU benchmark training was not run for this release.

## Completed checks

- All 58 tests passed with `python -m pytest -q`.
- Forward and backward passes passed for the headline model, five ungated variants, SASRec+ST+Repeat and GRU4Rec+ST+Repeat.
- Changing the target coordinates did not change model scores.
- Synthetic one-epoch training, aligned ranks, checkpoint restoration and standalone checkpoint evaluation passed.
- Cache tests verified code/data identity checks and rejection of incomplete artifacts.
- Statistical tests covered missing seeds, duplicate runs, configuration/source mismatches, event alignment, fractional ranks and bootstrap behavior.
- All 10 command-line entry points accepted `--help`.
- Dry runs produced 18 main jobs, 45 ablation jobs and 36 repeat-gated state-count sweep jobs.
- Protocol tests reject malformed splits, unseen candidates, repeated event IDs, future histories, alternate reference splits and invalid targets. They check same-user rolling histories across sessions and the distinction between 50-event, 512-event and complete-history windows.
- Optimizer tests instantiate Adam and AdamW and verify both the paper warmup sequence and historical schedule. Explicit single-state variants are checked against their original numerical implementations at default widths.

Tests use synthetic fixtures. Benchmark metrics, p values and runtime measurements have not been verified on the full datasets.

The included GitHub Actions workflow runs CPU tests when used in a repository. It has not been executed on GitHub as part of this local preparation.
