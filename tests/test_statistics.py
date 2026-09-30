"""Synthetic artifact checks. No training or real result artifacts required."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adspoi.results import load_runs, align_models, metric, mean_metric, paired_bootstrap


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_artifact(self, model, seed, cfg='one', ids=(3, 1, 2), ranks=(2., 20., 1.), lr=0.002):
        path = self.root / 'test' / 'nyc' / model / cfg / f'seed{seed}'
        path.mkdir(parents=True, exist_ok=True)
        meta = dict(seed=seed, dataset='nyc', variant=model, data_fingerprint='data1',
                    config=dict(train_objective='full', learning_rate=lr), n_test_samples=len(ids))
        (path / 'metrics.json').write_text(json.dumps(meta))
        np.save(path / 'check_ins_id.npy', np.array(ids))
        np.save(path / 'ranks.npy', np.array(ranks))
        return path

    def test_missing_and_duplicate_fail(self):
        self.run_artifact('repeat_gate', 42)
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate')
        self.run_artifact('repeat_gate', 42, cfg='two')
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42])

    def test_configuration_and_id_differences_fail(self):
        self.run_artifact('repeat_gate', 42)
        self.run_artifact('repeat_gate', 43, lr=0.001)
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42, 43])
        self.run_artifact('repeat_gate', 43, ids=(1, 2, 4))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42, 43])

    def test_seed_and_id_alignment(self):
        for seed in (44, 42, 43):
            self.run_artifact('repeat_gate', seed)
            self.run_artifact('sasrec_st_repeat', seed, ids=(1, 2, 3), ranks=(20, 1, 2))
        a = load_runs(self.root, 'test', 'nyc', 'repeat_gate')
        b = load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat')
        align_models(a, b)
        self.assertEqual([r.seed for r in a], [42, 43, 44])
        np.testing.assert_array_equal(mean_metric(a, 'HR@10'), mean_metric(b, 'HR@10'))
        self.run_artifact('sasrec_st_repeat', 43, ids=(1, 2, 4))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat')

    def test_models_must_share_ids_and_protocol(self):
        self.run_artifact('repeat_gate', 42)
        self.run_artifact('sasrec_st_repeat', 42, ids=(1, 2, 4))
        a = load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42])
        b = load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat', [42])
        with self.assertRaises(ValueError):
            align_models(a, b)
        self.run_artifact('sasrec_st_repeat', 42, lr=0.001)
        b = load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat', [42])
        with self.assertRaises(ValueError):
            align_models(a, b)

    def test_source_hashes_must_match_across_seeds_and_models(self):
        a42 = self.run_artifact('repeat_gate', 42)
        a43 = self.run_artifact('repeat_gate', 43)
        b42 = self.run_artifact('sasrec_st_repeat', 42)
        def put(path, value):
            fn = path / 'metrics.json'
            meta = json.loads(fn.read_text())
            meta['source_hash'] = value
            fn.write_text(json.dumps(meta))
        put(a42, 'source-a')
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42, 43])
        put(a43, 'source-b')
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42, 43])
        put(a43, 'source-a')
        load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42, 43])
        a = load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42])
        b = load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat', [42])
        with self.assertRaises(ValueError):
            align_models(a, b)
        put(b42, 'source-b')
        b = load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat', [42])
        with self.assertRaises(ValueError):
            align_models(a, b)
        put(b42, 'source-a')
        align_models(a, load_runs(self.root, 'test', 'nyc', 'sasrec_st_repeat', [42]))

    def test_config_filter_selects_exact_sweep(self):
        for k in (1, 4):
            for seed in (42, 43, 44):
                path = self.run_artifact('repeat_gate', seed, cfg=f'k{k}')
                fn = path / 'metrics.json'
                meta = json.loads(fn.read_text())
                meta['config'].update(num_substates=k, total_state_dim=128, substate_dim=128//k)
                fn.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate')
        runs = load_runs(self.root, 'test', 'nyc', 'repeat_gate',
                         config_filter={'num_substates': 4, 'total_state_dim': 128})
        self.assertEqual(len(runs), 3)
        self.assertTrue(all(r.metadata['config']['num_substates'] == 4 for r in runs))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', config_filter={'num_substates': 8})

    def test_duplicate_ids_and_nonfinite_ranks_fail(self):
        self.run_artifact('repeat_gate', 42, ids=(1, 1, 2))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42])
        self.run_artifact('repeat_gate', 42, ranks=(1, np.nan, 3))
        with self.assertRaises(ValueError):
            load_runs(self.root, 'test', 'nyc', 'repeat_gate', [42])

    def test_bootstrap_null_and_constant_effect(self):
        x = np.array([0, 1, 1., 0])
        result = paired_bootstrap(x, x, nboot=67, max_cells=3)
        self.assertEqual((result['delta'], result['lo'], result['hi'], result['p']), (0, 0, 0, 1))
        result = paired_bootstrap(x + 1, x, nboot=67, max_cells=3)
        self.assertEqual((result['delta'], result['lo'], result['hi'], result['p']), (100, 100, 100, 0))
        self.assertEqual(result, paired_bootstrap(x + 1, x, nboot=67, max_cells=3))

    def test_fractional_ranks_preserved(self):
        np.testing.assert_allclose(metric(np.array([1.5, 5.5]), 'MRR'), [2/3, 1/5.5])
        np.testing.assert_allclose(metric(np.array([1.5, 5.5]), 'NDCG@5'), [1/np.log2(2.5), 0])


if __name__ == '__main__':
    unittest.main()
