#!/usr/bin/env python3
"""Compare full-objective ADS-POI and its equal-information baseline."""
import argparse
import numpy as np
from adspoi.results import (common_arguments, bootstrap_arguments, load_runs,
    align_models, mean_metric, metric, paired_bootstrap, provenance, emit, cli, LIMITATIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    bootstrap_arguments(parser)
    args = parser.parse_args()
    output = {}
    for ds in args.datasets:
        a = load_runs(args.runs, args.tag, ds, 'repeat_gate', args.seeds)
        b = load_runs(args.runs, args.tag, ds, 'sasrec_st_repeat', args.seeds)
        align_models(a, b)
        result = paired_bootstrap(mean_metric(a, 'HR@10'), mean_metric(b, 'HR@10'),
                                  args.nboot, args.bootstrap_seed)
        result['discordant_counts'] = [dict(seed=x.seed,
            a_only=int(np.sum((metric(x.ranks, 'HR@10') == 1) & (metric(y.ranks, 'HR@10') == 0))),
            b_only=int(np.sum((metric(x.ranks, 'HR@10') == 0) & (metric(y.ranks, 'HR@10') == 1))))
            for x, y in zip(a, b)]
        result['runs'] = dict(a=provenance(a), b=provenance(b))
        output[ds] = result
    emit(dict(metric='HR@10', units='percentage points', limitations=LIMITATIONS, results=output))


if __name__ == '__main__':
    cli(main)
