"""Runner artifacts, checkpoint restoration and verified reuse on synthetic data."""
import copy
import json
import numpy as np
import pytest
import torch
import run_experiment as runner
import evaluate_checkpoint as checkpoint_cli
import adspoi.config as config_module
from adspoi.data import create_dataloaders
from adspoi.engine import evaluate
from adspoi.model import build_model
from test_model_pipeline import tiny_problem


@pytest.mark.parametrize('model,state', [
    ('adspoi', {'K': 4, 'd_s': 4, 'stored_state_width': 16}),
    ('sasrec_st_repeat', {'K': 1, 'd_s': 16, 'stored_state_width': 16}),
])
def test_checkpoint_and_verified_skip(tmp_path, monkeypatch, model, state):
    cfg, data = tiny_problem()
    cfg['tau_d'] = 'auto'
    data['stats'] = {'data_fingerprint': 'synthetic-v1'}
    data['poi_categories'] = {i: 'synthetic' for i in range(1, 7)}
    loads = []
    def load_data(dataset, directory):
        loads.append(dataset)
        return copy.deepcopy(data)
    monkeypatch.setattr(runner, 'load_official', load_data)
    monkeypatch.setattr(runner, 'resolve_config', lambda dataset, overrides, profile="paper": copy.deepcopy(cfg))
    args = ['--dataset', 'nyc', '--model', model, '--device', 'cpu', '--out', str(tmp_path)]
    assert runner.main(args) == 0
    path, = tmp_path.glob('main/nyc/*/*/seed42/metrics.json')
    result = json.loads(path.read_text())
    directory = path.parent
    saved_config = json.loads((directory / 'config.json').read_text())
    assert result['actual_state'] == saved_config['actual_state'] == state
    assert result['config']['train_objective'] == 'full'
    assert result['tau_d_rule'] == 'auto'
    assert isinstance(result['config']['tau_d'], float) and result['config']['tau_d'] > 0
    assert result['source_hash'] == runner.source_hash()
    assert result['data_fingerprint'] == 'synthetic-v1'
    assert result['config_hash'] == runner.config_hash(result['config'],
        result['model_variant'], model, result['source_hash'], result['data_fingerprint'])
    assert not {'hostname', 'git_commit', 'argv', 'slurm_job_id'} & result['env'].keys()
    assert (directory / 'state_analysis.json').is_file()
    assert result['epochs_run'] == 1 and result['n_test_samples'] == 4
    checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
    restored = build_model(checkpoint['num_pois'], checkpoint['config'],
                           checkpoint['model_variant'], checkpoint['model'])
    restored.load_state_dict(checkpoint['state_dict'])
    _, _, loader = create_dataloaders(data, checkpoint['config'])
    _, _, extras = evaluate(restored, loader, torch.device('cpu'))
    np.testing.assert_array_equal(extras['ranks'], np.load(directory / 'ranks.npy'))
    assert checkpoint['actual_state'] == state
    monkeypatch.setattr(checkpoint_cli, 'load_official', lambda ds, directory: copy.deepcopy(data))
    assert checkpoint_cli.main(['--checkpoint', str(directory / 'checkpoint.pt'), '--device', 'cpu']) == 0
    def training_must_not_run(*args, **kwargs):
        pytest.fail('completed experiment unexpectedly retrained')
    monkeypatch.setattr(runner, 'train_model', training_must_not_run)
    before = path.read_bytes()
    assert runner.main(args) == 0
    assert loads == ['nyc', 'nyc']
    assert path.read_bytes() == before
    assert len((tmp_path / 'results.jsonl').read_text().splitlines()) == 1
    (directory / 'checkpoint.pt').unlink()
    with pytest.raises(RuntimeError, match='identity or artifacts'):
        runner.main(args)
    data['stats']['data_fingerprint'] = 'synthetic-v2'
    class ReachedNewTraining(Exception):
        pass
    def observe_new_training(*args, **kwargs):
        raise ReachedNewTraining
    monkeypatch.setattr(runner, 'train_model', observe_new_training)
    with pytest.raises(ReachedNewTraining):
        runner.main(args)
    configs = [json.loads(p.read_text()) for p in tmp_path.glob('main/nyc/*/*/seed42/config.json')]
    assert len(configs) == 2
    assert len({c['config_hash'] for c in configs}) == 2
    assert {c['data_fingerprint'] for c in configs} == {'synthetic-v1', 'synthetic-v2'}


def test_source_identity_ignores_tests_and_bytecode(tmp_path, monkeypatch):
    package = tmp_path / 'adspoi'
    package.mkdir()
    source = package / 'config.py'
    source.write_text('CONFIG = 1\n')
    (tmp_path / 'run_experiment.py').write_text('RUNNER = 1\n')
    monkeypatch.setattr(config_module, '__file__', str(source))
    original = config_module.source_hash()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_new.py').write_text('assert True\n')
    (package / '__pycache__').mkdir()
    (package / '__pycache__' / 'config.pyc').write_bytes(b'bytecode')
    assert config_module.source_hash() == original
    source.write_text('CONFIG = 2\n')
    assert config_module.source_hash() != original
