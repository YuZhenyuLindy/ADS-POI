"""Adversarial checks of candidate ranking and explicit evaluation conventions."""
import copy
import numpy as np
import pytest
import torch
from adspoi.config import resolve_config
from adspoi.metrics import poi_ranks, metrics_from_ranks
from adspoi.model import build_model
from adspoi.data import POITrainDataset, POIEvalDataset
from adspoi.protocol import REFERENCE, validate_reference, evaluation_protocol


def test_full_ranking_excludes_padding_without_filtering_revisits():
    # Padding has the highest score, but the real target is ranked first.
    scores = torch.tensor([[1000., 4., 3., 2.], [1000., 0., 0., 0.]])
    ranks = poi_ranks(scores, torch.tensor([1, 2]))
    torch.testing.assert_close(ranks, torch.tensor([1., 2.]))
    values = metrics_from_ranks(np.array([1., 6., 11.]), ks=[5, 10])
    np.testing.assert_array_equal(values['HR@5'], [1., 0., 0.])
    np.testing.assert_array_equal(values['HR@10'], [1., 1., 0.])
    assert values['MRR'][2] == pytest.approx(1 / 11)
    assert values['NDCG@10'][1] == pytest.approx(1 / np.log2(7))


@pytest.mark.parametrize('target', [0, -1, 4])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError, match='real POI'):
        poi_ranks(torch.zeros(1, 4), torch.tensor([target]))


def test_nonfinite_real_candidates_rejected():
    with pytest.raises(ValueError):
        poi_ranks(torch.tensor([[0., float('nan'), 1.]]), torch.tensor([2]))


def test_explicit_single_state_dimensions_and_legacy_equivalence():
    cfg = resolve_config('nyc')
    for explicit, legacy, width in [('single_state_32', 'single_state', 32),
                                    ('single_state_128', 'no_decomposition', 128)]:
        torch.manual_seed(3)
        old = build_model(7, cfg, legacy)
        torch.manual_seed(3)
        new = build_model(7, cfg, explicit)
        assert new.K == 1 and new.d_s == width
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key])
        changed = resolve_config('nyc', {'num_substates': 2, 'total_state_dim': 64})
        assert build_model(7, changed, explicit).d_s == width


def test_long_history_windows_and_training_explore_definition():
    cfg = resolve_config('nyc', {'tau_d': 1.0})
    def event(i, poi):
        return dict(poi_idx=poi, ts_utc=float(i * 3600), hour=float(i % 24),
                    dow=0, latitude=40., longitude=-73., check_ins_id=i)
    trajectory = [event(0, 1)] + [event(i, 2) for i in range(1, 514)] + [event(514, 1)]
    train = POITrainDataset({1: trajectory}, 3, np.array([0., .1, .9]), cfg)
    sample = train[len(train) - 1]
    assert sample['is_explore'].item() == 0
    assert sample['poi_ids'].shape == (50,)
    assert sample['repeat_hist'].shape == (512,)
    assert (sample['poi_ids'] == 2).all()
    assert (sample['repeat_hist'] == 2).all()
    assert sample['target_poi'].item() == 1
    test = POIEvalDataset({1: trajectory[:-1]}, {1: trajectory[-1:]}, cfg)
    assert len(test) == 1
    assert test.samples[0]['history'][-1]['check_ins_id'] == 513


def test_reference_gate_rejects_alternate_test_set():
    data = dict(stats=dict(dataset='NYC', sequence_fingerprint='wrong'),
        train_data={1: [1]}, val_data={1: [2]}, test_data={1: [3]}, num_pois=7)
    with pytest.raises(ValueError, match='recorded chronological protocol'):
        validate_reference(data)
    cfg = resolve_config('nyc')
    protocol = evaluation_protocol(cfg)
    assert protocol['sequence_window'] == 50
    assert protocol['repeat_window'] == 512
    assert protocol['padding_candidate'] is False
    assert REFERENCE['tky']['test'] == 38237
