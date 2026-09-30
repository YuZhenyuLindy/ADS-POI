# ADS-POI

Implementation of ADS-POI for next POI recommendation.

ADS-POI represents a user's history with parallel recurrent states, applies state-specific temporal and spatial decay, and combines the states using the query context. The headline configuration also uses history-based repeat features. The repository includes training, full-vocabulary evaluation, matched-input baselines, ablations, state-count sweeps and paired statistical analysis.

## Installation

Python 3.9 or newer is required. A CUDA GPU is recommended for the complete experiments. CPU execution is supported for tests and small examples.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For a CUDA installation, install the appropriate PyTorch build for your environment before the remaining dependencies. Automatic device selection uses CUDA when available and CPU otherwise.

The locally tested package versions and scope are recorded in [VALIDATION.md](VALIDATION.md) and `requirements-tested.txt`.

## Data

Prepare the existing NYC, TKY and CA chronological split files in `data_official/` following [DATA.md](DATA.md). No dataset, checkpoint or historical run output is bundled. The loader does not re-filter the data or substitute a different split.

```bash
python scripts/check_data.py --official-dir data_official
```

The model receives the query time and previously observed locations. The target POI coordinates are not model inputs. Validation and test evaluation use a rolling history consisting only of previously observed events.

## Train and evaluate

Run one headline experiment:

```bash
python run_experiment.py --dataset nyc --seed 42 --device cuda
```

Defaults are `--profile historical --model adspoi --variant repeat_gate --train-objective full`, with K=4, state width 32, total stored state width 128 and learning rate 0.002. The default reproduces the source experiment optimizer (AdamW) and scheduler ordering. All datasets use the same default hyperparameters. Best checkpoint selection uses validation HR@10. Testing ranks every real POI in the vocabulary.

`--profile paper` is an optional alternative using Adam and a conventional three-epoch linear warmup. It follows the manuscript wording but changes the original experiment. Use a separate output root, for example `python scripts/run_suite.py --suite main --profile paper --out runs_paper --dry-run`. Do not mix the two profiles or attribute historical scores to the alternative profile.

The complete main comparison uses ADS-POI and SASRec+ST+Repeat, each on three datasets and seeds 42/43/44. Preview commands before starting the 18 training jobs:

```bash
python scripts/run_suite.py --suite main --dry-run
python scripts/run_suite.py --suite main --device cuda
```

The baseline uses the same ST input features, repeat-feature module and training objective. Its Transformer encoder differs from the ADS-POI recurrent encoder. This comparison is between complete architectures, not a single-component ablation.

Other shared-protocol baselines are available via `--model`, including `gru4rec`, `sasrec`, `gru4rec_st`, `sasrec_st` and `gru4rec_st_repeat`. These are local adaptations under this repository's training recipe.

## Ablations and state-count sweep

```bash
python scripts/run_suite.py --suite ablation --dry-run
python scripts/run_suite.py --suite ablation --device cuda
python scripts/run_suite.py --suite sweep --dry-run
python scripts/run_suite.py --suite sweep --device cuda
```

The ablation suite runs 45 jobs. The repeat-gated K sweep runs 36 jobs with K=1/2/4/8 and fixed total state width 128. These suites do not select configurations using test performance.

The variant names below describe their **executable configurations**:

| Variant | States | Width per state | Repeat mechanism |
|---|---:|---:|---|
| `repeat_gate` | 4 | 32 | Context-dependent gate |
| `full` | 4 | 32 | None |
| `no_decomposition` | 1 | 128 | None |
| `single_state` | 1 | 32 | None |
| `no_aggregation` | 4 | 32 | None, uniform aggregation |
| `no_heterogeneous` | 4 | 32 | None, shared decay parameters |

Each run records its actual K and state width. The default ablation suite retains the original experiment identifiers. Optional explicit variants `single_state_32` and `single_state_128` fix their stated widths. A display label in the manuscript does not override a run's recorded dimensions. Historical numerical rows are not reassigned by this release. Here `full` names the ungated model variant, while `--train-objective full` names the training objective. They are separate settings. Additional exploratory variants `full_routed` and `repeat_fixed` are implemented but excluded from the default suites.

## Summaries and paired tests

After all main runs finish:

```bash
python dump_clean_table.py --runs runs --tag main --models repeat_gate sasrec_st_repeat
python paired_sig_test.py --runs runs --tag main
python stratified_sig_test.py --runs runs --tag main --official-dir data_official --window win
python dump_state_analysis.py --runs runs --tag main --models repeat_gate
```

For ablations and sweeps:

```bash
python dump_clean_table.py --runs runs --tag ablation --models full no_decomposition single_state no_aggregation no_heterogeneous
python dump_sweep.py --runs runs --tag sweep
```

The exact evaluation contract and its relation to the manuscript are specified in [EVALUATION.md](EVALUATION.md).

Reports are JSON. Redirect stdout to a file to save a report. Collectors require exactly one configuration per requested seed, matching data fingerprints and identical check-in ID sets. Missing runs, duplicate configurations and incompatible protocols cause an error instead of producing a partial table.

Paired bootstrap uses 20,000 resamples by default. Metrics are first averaged across the fixed trained seeds for each test event, then paired events are resampled. This quantifies sample uncertainty conditional on those models. It does not estimate training-seed uncertainty or account for within-user event dependence. `--window win` defines revisits using the saved 512-event repeat window. `--window full` uses all visible history. The report records the choice.

## Saved artifacts

Each run is stored under `runs/<tag>/<dataset>/<model-or-variant>/<hash>/seed<seed>/` and includes:

- `config.json` and `metrics.json`, with effective settings, actual state dimensions, code and data fingerprints and anonymous software metadata.
- `history.json`, with training loss, actual training learning rate and validation metrics.
- `ranks.npy`, `targets.npy`, `check_ins_id.npy` and `topk.npy` for aligned analysis.
- `checkpoint.pt`, containing the best-validation model and reconstruction settings.
- `state_analysis.json`, unless `--no-state-analysis` is selected.

Completed experiments are reused only when code, configuration, data and required artifacts match. Interrupted experiments restart training. Checkpoints support model restoration but do not contain optimizer state for mid-training resume. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact implementation conventions and release scope.

Use `python evaluate_checkpoint.py --help` for standalone evaluation of a saved checkpoint. Evaluation checks the checkpoint's code and data fingerprints before restoring the model.

## Repository layout

```text
adspoi/                 Models, datasets, training, metrics and result validation
scripts/                Data preparation and experiment suites
tests/                  Synthetic model, protocol and statistical checks
run_experiment.py       Single-run entry point
paired_sig_test.py       Overall paired bootstrap
stratified_sig_test.py   Revisit/explore paired bootstrap
dump_clean_table.py     Seed-level metric summaries
dump_sweep.py           State-count sweep summaries
dump_state_analysis.py  Per-seed routing diagnostics
```

Tests use synthetic data. Benchmark scores require completed runs on the prepared datasets.
