#!/usr/bin/env python3
"""Summarize a single configuration across explicitly selected trained seeds."""
import argparse
import numpy as np
from adspoi.results import common_arguments, load_runs, align_samples, metric, provenance, emit, cli



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument('--models', nargs='+', default=['repeat_gate', 'sasrec_st_repeat'])
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
            row = {}
            for name in ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR']:
                values = [float(metric(r.ranks, name).mean()) * 100 for r in runs]
                row[name] = dict(mean=float(np.mean(values)),
                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    n=len(values), values=values)
            output[ds][model] = dict(metrics=row, runs=provenance(runs),
                actual_state=runs[0].metadata.get('actual_state'))
    emit(dict(units='percent', sd='sample standard deviation across trained seeds', results=output))


if __name__ == '__main__':
    cli(main)
