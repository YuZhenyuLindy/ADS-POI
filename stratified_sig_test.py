#!/usr/bin/env python3
"""Paired sample bootstrap within explicitly defined revisit/explore strata."""
import argparse
from pathlib import Path
import numpy as np
from adspoi.results import (common_arguments, bootstrap_arguments, load_runs,
    align_models, mean_metric, paired_bootstrap, provenance, emit, cli, LIMITATIONS)


def build_labels(dataset, official_dir, cfg, expected_fingerprint):
    from adspoi.official import load_official
    from adspoi.data import POIEvalDataset
    # Keep loader status output off stdout so the result remains valid JSON.
    import contextlib
    import sys
    with contextlib.redirect_stdout(sys.stderr):
        data = load_official(dataset, root=official_dir)
    if data['stats']['data_fingerprint'] != expected_fingerprint:
        raise ValueError('Official data fingerprint differs from saved runs')
    train_plus_val = {uid: traj + data['val_data'].get(uid, [])
                      for uid, traj in data['train_data'].items()}
    test = POIEvalDataset(train_plus_val, data['test_data'], cfg,
        data['default_lat'], data['default_lon'], data.get('t_ref', 0.0))
    window = int(cfg['repeat_history_len'])
    if window < 1:
        raise ValueError('repeat_history_len must be positive')
    labels = {}
    for sample in test.samples:
        target = sample['target']
        cid = int(target['check_ins_id'])
        if cid in labels:
            raise ValueError(f'Duplicate official check-in ID: {cid}')
        history = sample['history']
        labels[cid] = dict(full=target['poi_idx'] in {c['poi_idx'] for c in history},
                          win=target['poi_idx'] in {c['poi_idx'] for c in history[-window:]})
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    bootstrap_arguments(parser)
    parser.add_argument('--official-dir', type=Path, default=Path('data_official'))
    parser.add_argument('--window', choices=['full', 'win'], default='win',
                        help='win uses the saved repeat_history_len, full uses all visible history')
    args = parser.parse_args()
    output = {}
    for ds in args.datasets:
        a = load_runs(args.runs, args.tag, ds, 'repeat_gate', args.seeds)
        b = load_runs(args.runs, args.tag, ds, 'sasrec_st_repeat', args.seeds)
        align_models(a, b)
        labels = build_labels(ds, args.official_dir, a[0].metadata['config'],
                              a[0].metadata['data_fingerprint'])
        ids = a[0].ids
        if set(labels) != set(ids.tolist()):
            raise ValueError(f'{ds}: label IDs must exactly equal run IDs')
        rev = np.array([labels[int(i)][args.window] for i in ids], dtype=bool)
        results = {}
        for name in ['HR@10', 'NDCG@5', 'MRR']:
            ma, mb = mean_metric(a, name), mean_metric(b, name)
            results[name] = {}
            for stratum, mask in [('all', np.ones(len(ids), dtype=bool)), ('revisit', rev), ('explore', ~rev)]:
                results[name][stratum] = (paired_bootstrap(ma[mask], mb[mask], args.nboot,
                    args.bootstrap_seed) if mask.any() else dict(n=0, status='empty stratum'))
        output[ds] = dict(metrics=results, revisit_count=int(rev.sum()), sample_count=len(ids),
            repeat_history_len=a[0].metadata['config']['repeat_history_len'],
            runs=dict(a=provenance(a), b=provenance(b)))
    emit(dict(window=args.window, units='percentage points', limitations=LIMITATIONS, results=output))


if __name__ == '__main__':
    cli(main)
