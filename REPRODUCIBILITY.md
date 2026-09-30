# Implementation conventions

## Release scope

The package consolidates the full-vocabulary experiment implementation and analysis scripts. It contains no historical metrics, pretrained weights or raw ranks. The benchmark tables must be generated from completed runs. Synthetic tests validate code paths rather than benchmark accuracy.

The release retains the core model equations and tensor operations and adds explicit dimension names for single-state ablations. The default historical profile retains the source training recipe. The optional paper profile changes optimizer and warmup to follow the manuscript wording. Packaging changes include explicit defaults, anonymous metadata, strict result selection, bounded-memory bootstrap, complete data hashes and exclusion of padding from exported top-k lists for small vocabularies. Run identities differ from the notebook identities, preventing accidental reuse of incompatible cached results.

## Training

Default settings are defined in `adspoi/config.py`. The full objective combines label-smoothed cross-entropy over real POIs, a margin term over the ten highest-scoring non-target POIs and explore-event reweighting. Padding receives no smoothing mass. Evaluation uses the complete POI vocabulary.

The optional **paper** profile uses Adam with weight decay 0.0001. The first three epoch rates are base/3, 2×base/3 and base, followed by cosine decay. The manuscript specifies the warmup duration but not the subsequent decay formula, which is made explicit here.

The default **historical** profile uses AdamW and retains the source experiment's scheduler order: epoch 1 uses the base learning rate, then the scheduler updates after each epoch. With a three-epoch warmup, epoch 2 uses approximately one-third of the base rate. `history.json` distinguishes the rate used for training from the next epoch's rate. Selecting the alternative optimizer and schedule changes new experiments and does not establish a new explanation for historical benchmark scores.

Sequence history is capped at 50 events, repeat features at 512. Training explore weights use whether the target appeared anywhere in its preceding training history. Query time is assumed available. Query coordinates are derived from observed history. Location data for the target are carried by the dataset for bookkeeping but are never passed to the model's query encoder.

## Model and ablations

The `repeat_gate` variant combines the K-state recurrent model with count, recency and visited features. Repeat weights are unconstrained learned values conditioned on the decision state and context. The `full` variant omits this mechanism.

The default ablation suite retains the source identifiers: `no_decomposition` uses the configured total width (128 by default), and `single_state` uses the configured sub-state width (32 by default). Optional `single_state_32` and `single_state_128` provide unambiguous fixed dimensions. These are implementation definitions, not a reconstruction of the manuscript's numerical row provenance. Original run records are needed to resolve conflicting display labels.

State-width changes can also alter the output projection. Holding total stored state width fixed does not imply identical parameter count. The SASRec+ST+Repeat comparison holds inputs and the repeat module fixed but changes the sequence encoder.

`full_routed` is an exploratory multiplicative-routing variant, not the default additive-attention model. It is excluded from the main and ablation suites.

## Metrics and analysis

Ranking uses one-based mid-ranks for score ties. HR and NDCG are computed per test event, then averaged. Non-finite scores cause an error. Mean and sample standard deviation across seeds are reported separately. Reports never infer significance from a fixed difference threshold.

Bootstrap aligns events by check-in ID and averages each event's metric over the selected seeds before resampling paired events. It uses bounded temporary allocations. A p estimate of zero means no resample crossed zero at the chosen resolution. It does not mean the true probability is zero.

All requested seeds must be present exactly once, with the same effective settings, code identity and data fingerprint. No collector selects the best test result. Latent state indices are not assumed aligned across seeds, so routing vectors remain separate by seed.

## Timing

The training runner records the timing scope implemented in `measure_inference`. Consult the saved fields and the function before comparing measurements. Backbone-only forward timing is different from full candidate scoring and from end-to-end data-loading latency. Parameter counts are total trainable parameters, including the POI embedding table. They must not be relabeled as backbone-only counts without an explicit exclusion rule.

## Anonymous artifacts

The repository excludes notebooks, execution identities, personal paths and git history. Runtime environment metadata contains software versions and hardware type, without usernames, hostnames, git remotes or command-line paths. Locally generated reports may include run-directory paths supplied by the caller. Use relative output paths if sharing those reports anonymously.
