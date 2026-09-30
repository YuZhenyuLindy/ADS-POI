# Evaluation contract

This document distinguishes the manuscript's explicit requirements from implementation details supplied by the experiment code. The default runner uses this contract for both ADS-POI and its reimplemented baselines.

## Dataset and test instances

Section 5.1.1 specifies official STHGCN/LLM4POI preprocessing, per-user trajectories and chronological train/validation/test splits. It does not state that the split itself must be performed separately for each user. This release reads the supplied split files without re-splitting them. It does not replace them with per-user leave-one-out or session-last evaluation.

The experiment code evaluates every test-file event once. History is grouped by user across session boundaries. Validation uses training history and earlier validation events. Test uses training history, validation events and earlier test events. The current target and later events are excluded. Model parameters are not updated during validation or testing.

The loader requires all held-out users and POIs to occur in training. It rejects missing histories rather than silently dropping test events. It checks unique event IDs, finite input values, contiguous real POI IDs and chronological ordering within each user and across split boundaries. Invalid inputs cause an error, not filtering or re-splitting.

Raw timestamps can tie. Such events retain their input-file order within each split, with training preceding validation and validation preceding test. This is an explicit implementation convention for tied events, whereas the mathematical trajectory definition uses strictly increasing times. Events are not silently deduplicated.

The standard loader also verifies the event counts, candidate vocabulary size and sequence fingerprint recorded in the available full-objective experiment logs. Those reference checks prevent substituting LOO, shortened evaluation or a different event sequence. They do not replace verification of the actual CSV files. Complete file hashes are recorded for every new run.

## Inputs and candidate ranking

Section 3 gives the query time as an input. The model therefore receives the target event's hour and weekday, together with the observed history. Query spatial context comes from the last two observed locations. Ground-truth target coordinates are never passed to the model.

Every real POI in the training vocabulary is scored for every evaluation event. There is no negative sampling, visited-item exclusion, geographic candidate restriction or target-dependent candidate construction. Padding is explicitly removed from ranking and top-k output, irrespective of its score.

The recurrent/Transformer input uses the latest 50 history events. Repeat features use the latest 512. These window lengths come from the implementation and are not specified in the PDF. Prior events outside these windows remain part of the conceptual complete history used for the training explore-weight label.

## Metrics

Section 5.1.3 specifies HR@5, HR@10, NDCG@5, NDCG@10 and MRR under full ranking. This implementation uses one relevant target per event:

```text
HR@k   = 1 if target rank <= k, otherwise 0
NDCG@k = 1 / log2(target rank + 1) if rank <= k, otherwise 0
MRR    = 1 / target rank, without truncation
```

Ranks are one-based. Tied real POIs receive the average of their occupied ranks. Metrics are averaged over events within a seed, then over the specified training seeds. They are not averaged equally over users or unweighted batches. Additional saved cutoffs do not change the five reported metrics. Non-finite scores, invalid target IDs and empty evaluation sets are rejected.

The PDF does not specify tie handling or event-versus-user weighting. These are explicit implementation conventions, not additional claims about text in the manuscript.

## Seeds, checkpoint selection and statistics

The main recipe uses seeds 42, 43 and 44. Each run selects the best checkpoint by validation HR@10. No test metric controls early stopping, model selection or learning-rate selection.

For the Section 5.3 comparison, predictions must have identical event ID sets across models and seeds. Duplicate configurations, missing requested seeds, different data or different optimizer profiles are rejected. The paired bootstrap first averages each event's metric over the fixed trained seeds, then resamples paired events 20,000 times. It returns the mean difference, percentile interval and a two-sided tail estimate.

The PDF specifies a per-sample paired bootstrap but not its resampling count or seed-aggregation convention. These follow the supplied statistical implementation. This is not a user-cluster bootstrap and does not estimate variability over newly trained seeds.

## Revisit and explore strata

Section 5.4 defines revisit targets using visible history. The supplied statistical code supports two explicit interpretations:

- `--window win`, the default, checks the 512-event repeat-feature window.
- `--window full` checks all earlier observed events for that user.

Both exclude the current and future targets. The chosen definition, sample counts and fractions are recorded in the report. The PDF does not specify the window length. Without the original labels and raw ranks, the published stratum percentages and p values cannot be independently certified. The code does not alter the window to fit those percentages.
