"""Fail closed on malformed prepared splits. preserve valid rolling evaluation."""
import copy

import numpy as np
import pandas as pd
import pytest

from adspoi import official
from adspoi.data import POIEvalDataset, create_dataloaders
from test_model_pipeline import tiny_problem


def csv_splits():
    def row(user, poi, event, hour, session=1):
        return dict(UserId=user, PoiId=poi, check_ins_id=event,
                    UTCTimeOffset=f'2024-01-01 {hour:02d}:00:00',
                    Latitude=40.0, Longitude=-73.0,
                    pseudo_session_trajectory_id=session)
    return {
        'train': pd.DataFrame([row(0, 0, 1, 8), row(0, 1, 2, 9),
                               row(1, 1, 3, 8), row(1, 0, 4, 9)]),
        'val': pd.DataFrame([row(0, 1, 5, 10), row(1, 0, 6, 10)]),
        'test': pd.DataFrame([row(0, 0, 7, 11), row(0, 1, 8, 11),
                              row(1, 1, 9, 11), row(1, 0, 10, 12)]),
    }


def test_csv_loader_valid_equal_times_and_full_fingerprint(tmp_path, monkeypatch):
    raw = csv_splits()
    for frame in raw.values():
        frame['_split'] = 'train'  # Optional CSV labels must not override file membership.
    raw['test']['Latitude'] = 70.0
    folder = tmp_path / 'nyc'
    folder.mkdir()
    for split, frame in raw.items():
        frame.to_csv(folder / official.FILES[split], index=False)
    # A tiny fixture has no meaningful day/night population distribution.
    monkeypatch.setattr(official, 'assert_human_rhythm', lambda *a:
                        {'argmin_hour': 4, 'day_night_ratio': 10.})
    data = official.load_official('nyc', tmp_path, verify_reference=False)
    assert data['stats']['reference_protocol_verified'] is False
    assert data['num_pois'] == 3 and data['stats']['pois'] == 2
    assert data['stats']['users'] == 2
    assert data['default_lat'] == 40.0
    assert (data['poi_locations'][1:, 0] == 40.0).all()
    assert [e['check_ins_id'] for e in data['test_data'][0]] == [7, 8]
    assert len(data['stats']['data_fingerprint']) == 64
    assert len(data['stats']['source_files_sha256']) == 3
    with pytest.raises(ValueError, match='recorded chronological protocol'):
        official.load_official('nyc', tmp_path)
    from adspoi import protocol
    def accept_reference(result):
        assert result['stats']['reference_protocol_verified'] is False
        return {'name': 'synthetic-reference'}
    monkeypatch.setattr(protocol, 'validate_reference', accept_reference)
    checked = official.load_official('nyc', tmp_path)
    assert checked['stats']['reference_protocol_verified'] is True
    assert checked['stats']['reference_protocol'] == {'name': 'synthetic-reference'}


@pytest.mark.parametrize('case,match', [
    ('missing', 'missing required'), ('nan', 'non-finite'),
    ('fractional_id', 'int64'), ('coordinate', 'Latitude outside'),
    ('hole', 'contiguous'), ('unknown_poi', 'unseen POI'),
    ('unknown_user', 'unknown user'), ('duplicate_across', 'duplicate check_ins_id'),
    ('duplicate_within', 'duplicate check_ins_id'), ('future_train', 'future history'),
    ('future_val', 'future history'), ('unsorted', 'chronological order'),
])
def test_reject_malformed_splits(case, match):
    raw = csv_splits()
    if case == 'missing':
        raw['test'] = raw['test'].drop(columns='Latitude')
    elif case == 'nan':
        raw['train'].loc[0, 'Longitude'] = np.nan
    elif case == 'fractional_id':
        raw['train']['PoiId'] = raw['train']['PoiId'].astype(float)
        raw['train'].loc[0, 'PoiId'] = .5
    elif case == 'coordinate':
        raw['train'].loc[0, 'Latitude'] = 91
    elif case == 'hole':
        raw['train'].loc[raw['train']['PoiId'] == 1, 'PoiId'] = 2
        # The evaluation POI 1 is in [0,max] but absent from the true vocabulary.
    elif case == 'unknown_poi':
        raw['test'].loc[0, 'PoiId'] = 2
    elif case == 'unknown_user':
        raw['test'].loc[0, 'UserId'] = 99
    elif case == 'duplicate_across':
        raw['test'].loc[0, 'check_ins_id'] = 1
    elif case == 'duplicate_within':
        raw['test'].loc[1, 'check_ins_id'] = 7
    elif case == 'future_train':
        raw['val'].loc[0, 'UTCTimeOffset'] = '2024-01-01 07:00:00'
    elif case == 'future_val':
        raw['val'].loc[0, 'UTCTimeOffset'] = '2024-01-01 12:00:00'
    elif case == 'unsorted':
        raw['train'].loc[0, 'UTCTimeOffset'] = '2024-01-01 10:00:00'
    with pytest.raises(ValueError, match=match):
        official._validate_raw(raw, 'nyc')


def test_equal_boundary_times_preserve_split_and_file_order():
    raw = csv_splits()
    raw['val'].loc[0, 'UTCTimeOffset'] = '2024-01-01 09:00:00'
    raw['test'].loc[0:1, 'UTCTimeOffset'] = '2024-01-01 09:00:00'
    official._validate_raw(raw, 'nyc')
    assert [e['check_ins_id'] for e in official._trajectories(raw['test'], 'nyc')[0]] == [7, 8]


def test_rolling_history_is_user_local_and_crosses_sessions():
    cfg, data = tiny_problem()
    for split in ('train_data', 'val_data', 'test_data'):
        for uid, events in data[split].items():
            for event in events:
                event['poi_idx'] = uid  # disjoint user vocabularies
                event['traj_id'] = event['check_ins_id']  # every event is a new session
    _, _, loader = create_dataloaders(data, cfg)
    assert len(loader.dataset) == 4
    for sample in loader.dataset.samples:
        uid = sample['target']['check_ins_id'] // 100
        assert {e['poi_idx'] for e in sample['history']} == {uid}
        assert all(e['ts_utc'] <= sample['target']['ts_utc'] for e in sample['history'])
        assert sample['target']['check_ins_id'] not in {e['check_ins_id'] for e in sample['history']}
        expected = 7 if sample['target']['check_ins_id'] % 100 == 7 else 8
        assert len(sample['history']) == expected  # includes validation and earlier test events


@pytest.mark.parametrize('case', ['unknown_user', 'empty_history', 'future', 'unsorted_base', 'unsorted_eval'])
def test_eval_constructor_refuses_silent_repairs(case):
    cfg, data = tiny_problem()
    base, target = copy.deepcopy(data['train_data']), copy.deepcopy(data['test_data'])
    if case == 'unknown_user':
        del base[1]
    elif case == 'empty_history':
        base[1] = []
    elif case == 'future':
        base[1][-1]['ts_utc'] = target[1][0]['ts_utc'] + 1
    elif case == 'unsorted_base':
        base[1] = list(reversed(base[1]))
    elif case == 'unsorted_eval':
        target[1] = list(reversed(target[1]))
    with pytest.raises(ValueError):
        POIEvalDataset(base, target, cfg)


def test_full_history_revisit_label_and_truncated_model_windows():
    from adspoi.config import resolve_config
    from adspoi.data import POITrainDataset
    cfg = resolve_config('nyc')
    events = [dict(poi_idx=1 if i in (0, 514) else 2, ts_utc=float(i * 3600),
                   hour=float(i % 24), dow=0, latitude=40., longitude=-73.,
                   check_ins_id=i) for i in range(515)]
    dataset = POITrainDataset({1: events}, 3, np.array([0., .5, .5]), cfg)
    sample = dataset[len(dataset) - 1]
    assert sample['target_poi'].item() == 1
    assert sample['is_explore'].item() == 0  # full prior history contains POI 1
    assert sample['poi_ids'].shape == (50,)
    assert sample['repeat_hist'].shape == (512,)
    assert not (sample['poi_ids'] == 1).any()
    assert not (sample['repeat_hist'] == 1).any()
