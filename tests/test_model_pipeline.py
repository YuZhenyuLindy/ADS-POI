"""Synthetic checks of causality, full-vocabulary loss and model interfaces."""
import numpy as np
import pytest
import torch

from adspoi.config import resolve_config
from adspoi.data import create_dataloaders
from adspoi.engine import _forward, train_model, evaluate
from adspoi.metrics import per_sample_ranks
from adspoi.model import build_model, ADSPOILoss


def tiny_problem():
    cfg = resolve_config('nyc', dict(total_state_dim=16, num_substates=4,
        poi_embed_dim=16, tau_d=1.0, max_history_len=4, repeat_history_len=8,
        batch_size=4, eval_batch_size=4, num_epochs=1, dropout=0.0))
    def event(i, uid):
        return dict(poi_idx=i % 6 + 1, ts_utc=float(i * 3600), hour=float(i % 24),
            dow=0, latitude=40 + i * .001, longitude=-73 + i * .001,
            check_ins_id=uid * 100 + i, category='synthetic')
    data = dict(train_data={u: [event(i, u) for i in range(6)] for u in (1, 2)},
        val_data={u: [event(6, u)] for u in (1, 2)},
        test_data={u: [event(7, u), event(8, u)] for u in (1, 2)},
        num_pois=7, poi_popularity=np.array([0] + [1 / 6] * 6),
        default_lat=40., default_lon=-73., t_ref=0.)
    return cfg, data


@pytest.mark.parametrize('model,variant', [
    ('adspoi', 'repeat_gate'), ('adspoi', 'full'),
    ('adspoi', 'no_decomposition'), ('adspoi', 'single_state'),
    ('adspoi', 'no_aggregation'), ('adspoi', 'no_heterogeneous'),
    ('sasrec_st_repeat', 'full'), ('gru4rec_st_repeat', 'full')])
def test_full_objective_backward_and_target_location_invariance(model, variant):
    torch.manual_seed(4)
    cfg, data = tiny_problem()
    tr, _, _ = create_dataloaders(data, cfg)
    batch = next(iter(tr))
    assert 'neg_ids' not in batch
    net = build_model(data['num_pois'], cfg, variant, model).eval()
    h, _, rep = _forward(net, batch)
    scores = net.compute_all_scores(h, rep)
    changed = dict(batch, target_location=torch.full_like(batch['target_location'], 999.))
    h2, _, rep2 = _forward(net, changed)
    torch.testing.assert_close(scores, net.compute_all_scores(h2, rep2))
    assert scores.shape == (4, 7)
    assert (scores[:, 0] < -1e6).all()
    loss = ADSPOILoss().forward_full(scores, batch['target_poi'], batch['is_explore'])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_one_epoch_training_and_rank_persistence_contract():
    torch.manual_seed(12)
    cfg, data = tiny_problem()
    tr, val, test = create_dataloaders(data, cfg)
    net = build_model(data['num_pois'], cfg, 'repeat_gate')
    net, history = train_model(net, tr, val, cfg, torch.device('cpu'))
    metrics, _, extras = evaluate(net, test, torch.device('cpu'), collect_topk=True)
    assert history['best_epoch'] == 1
    assert 0 <= metrics['HR@10'] <= 1
    assert len(extras['ranks']) == len(extras['check_ins_id']) == 4
    assert len(set(extras['check_ins_id'])) == 4
    assert (extras['topk'] > 0).all()


def test_rolling_history_excludes_current_and_future_targets():
    cfg, data = tiny_problem()
    _, _, loader = create_dataloaders(data, cfg)
    for sample in loader.dataset.samples:
        assert all(e['ts_utc'] < sample['target']['ts_utc'] for e in sample['history'])


def test_midrank_and_nonfinite_rejection():
    scores = torch.tensor([[-1e9, 2., 2., 0.]])
    assert per_sample_ranks(scores, torch.tensor([1])).item() == 1.5
    with pytest.raises(ValueError):
        per_sample_ranks(torch.tensor([[0., float('nan')]]), torch.tensor([1]))


def test_single_state_widths_are_explicit():
    cfg = resolve_config('nyc')
    assert build_model(7, cfg, 'no_decomposition').d_s == 128
    assert build_model(7, cfg, 'single_state').d_s == 32
