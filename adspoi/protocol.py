"""Reference chronological evaluation settings from the supplied experiment records.

The PDF specifies chronological splits and full ranking. Event counts and
sequence fingerprints below come from the corresponding saved experiment logs.
They are not new measurements of datasets bundled in this repository.
"""
REFERENCE = {
    'nyc': dict(train=83228, val=9458, test=9074, pois=4980,
                sequence_fingerprint='c2e725a8fcb7e66a'),
    'tky': dict(train=326258, val=38653, test=38237, pois=7832,
                sequence_fingerprint='0052dc5000529136'),
    'ca': dict(train=190627, val=18299, test=12791, pois=9689,
               sequence_fingerprint='f71323bfb1675312'),
}


def validate_reference(data):
    """Reject alternate splits, shortened test sets or changed event sequences."""
    stats = data['stats']
    ds = stats['dataset'].lower()
    ref = REFERENCE[ds]
    actual = {name: sum(len(v) for v in data[f'{name}_data'].values())
              for name in ('train', 'val', 'test')}
    actual['pois'] = data['num_pois'] - 1
    actual['sequence_fingerprint'] = stats['sequence_fingerprint']
    differences = {key: dict(expected=value, actual=actual[key])
                   for key, value in ref.items() if actual[key] != value}
    if differences:
        raise ValueError(f'{ds}: input differs from the recorded chronological protocol: '
                         f'{differences}. Do not re-split or trim data to match counts. See DATA.md.')
    return dict(name='official_chronological_all_events_v1', reference=ref,
                evidence='saved full-objective experiment logs',
                excludes='No leave-one-out, session-last or sampled-candidate substitution')


def evaluation_protocol(cfg):
    """Machine-readable protocol saved with each experiment."""
    return dict(name='official_chronological_all_events_v1',
        split='supplied chronological CSV splits, not recomputed',
        test_instances='every test CSV event, one prediction per event',
        history='same-user train + validation + earlier test events',
        validation_history='same-user train + earlier validation events',
        sessions='history does not reset at session boundaries',
        equal_timestamps='stable CSV order within split, train then validation then test',
        query_time='given', query_location='last observed location',
        candidates='all real training-vocabulary POIs, including visited POIs',
        padding_candidate=False, negative_sampling_at_evaluation=False,
        metric_aggregation='mean over test events, then mean over trained seeds',
        ties='mid-rank', mrr='untruncated',
        sequence_window=cfg['max_history_len'], repeat_window=cfg['repeat_history_len'],
        training_explore_history='complete training prefix',
        stratified_default_history='repeat window',
        bootstrap='paired event resampling after averaging per-event metrics over fixed seeds')
