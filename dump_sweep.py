#!/usr/bin/env python3
"""Summarize fixed-total-width K sweeps from complete saved seed groups."""
import argparse
import numpy as np
from adspoi.results import common_arguments, load_runs, align_models, metric, provenance, emit, cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.set_defaults(tag='sweep')
    parser.add_argument('--variant', default='repeat_gate')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--d', type=int, default=128, help='Fixed total state width')
    args = parser.parse_args()
    if args.d < 1 or len(set(args.ks)) != len(args.ks) or any(k < 1 or args.d % k for k in args.ks):
        raise ValueError('K values must be unique positive divisors of positive total width d')
    output = {}
    for ds in args.datasets:
        output[ds] = {}
        reference = None
        for k in sorted(args.ks):
            runs = load_runs(args.runs, args.tag, ds, args.variant, args.seeds,
                             config_filter={'num_substates': k, 'total_state_dim': args.d})
            cfg = runs[0].metadata['config']
            if cfg.get('substate_dim') != args.d // k:
                raise ValueError(f'{ds}/K={k}: saved substate width differs from d/K')
            if reference is not None:
                align_models(reference, runs)
                for key in set(cfg) | set(reference[0].metadata['config']):
                    if key not in ('num_substates', 'substate_dim') and cfg.get(key) != reference[0].metadata['config'].get(key):
                        raise ValueError(f'{ds}: sweep changed a setting other than K/width: {key}')
            reference = runs
            metrics = {}
            for name in ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR']:
                vals = [float(metric(r.ranks, name).mean()) * 100 for r in runs]
                metrics[name] = dict(mean=float(np.mean(vals)),
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                    n=len(vals), values=vals)
            output[ds][str(k)] = dict(substate_dim=args.d // k, metrics=metrics, runs=provenance(runs))
    emit(dict(total_state_dim=args.d, variant=args.variant, units='percent',
              sd='sample standard deviation across trained seeds', results=output))


if __name__ == '__main__':
    cli(main)
