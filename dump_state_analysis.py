#!/usr/bin/env python3
"""Read state diagnostics without assuming state identities align across seeds."""
import argparse
import json
import numpy as np
from adspoi.results import common_arguments, load_runs, align_samples, provenance, emit, cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--models', nargs='+', default=['repeat_gate'])
    args = parser.parse_args()
    output = {}
    for ds in args.datasets:
        output[ds] = {}
        reference = None
        for model in args.models:
            runs = load_runs(args.runs, args.tag, ds, model, args.seeds)
            if reference is not None:
                align_samples(reference, runs)
            reference = runs
            records = []
            for run in runs:
                with (run.path / 'state_analysis.json').open() as f:
                    record = json.load(f)
                for key in ['alpha_entropy_mean', 'context_spread_max_range']:
                    if not np.isfinite(record[key]):
                        raise ValueError(f'Invalid {key}: {run.path}')
                records.append(dict(seed=run.seed, analysis=record))
            summaries = {}
            for key in ['alpha_entropy_mean', 'context_spread_max_range']:
                vals = [r['analysis'][key] for r in records]
                summaries[key] = dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
            output[ds][model] = dict(summary=summaries, per_seed=records, runs=provenance(runs))
    emit(dict(note='Latent state indices are not aligned across independently trained seeds. '
                   'State vectors are reported per seed, never averaged by state index.', results=output))


if __name__ == '__main__':
    cli(main)
