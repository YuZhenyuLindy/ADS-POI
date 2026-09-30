"""Optimizer and learning-rate protocol checks without dataset training."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from adspoi.config import resolve_config, config_hash

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from adspoi.engine import WarmupCosineScheduler, build_optimizer
else:
    # Exercise the actual dependency-free scheduler class even in environments
    # without Torch. No training/model code is executed by this fallback.
    path = Path(__file__).resolve().parents[1] / 'adspoi' / 'engine.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WarmupCosineScheduler')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    WarmupCosineScheduler = namespace['WarmupCosineScheduler']


class ConfigurationTests(unittest.TestCase):
    def test_profile_defaults_and_identity(self):
        paper = resolve_config('nyc', profile='paper')
        historical = resolve_config('nyc')
        self.assertEqual((paper['optimizer'], paper['schedule_profile']), ('adam', 'paper'))
        self.assertEqual((historical['optimizer'], historical['schedule_profile']), ('adamw', 'historical'))
        self.assertNotEqual(config_hash(paper, code_hash='code'), config_hash(historical, code_hash='code'))
        with self.assertRaises(ValueError):
            resolve_config('nyc', {'optimizer': 'adamw'}, profile='paper')
        with self.assertRaises(ValueError):
            resolve_config('nyc', profile='unknown')


class TrainingProtocolTests(unittest.TestCase):
    def schedule(self, profile, warmup=3, total=8):
        cfg = resolve_config('nyc', profile=profile)
        optimizer = (build_optimizer(torch.nn.Linear(1, 1), cfg) if torch is not None
                     else SimpleNamespace(param_groups=[{'lr': cfg['learning_rate']}]))
        scheduler = WarmupCosineScheduler(optimizer, warmup, total, profile=profile)
        rates = []
        for _ in range(total):
            rates.append(scheduler.get_lr())
            scheduler.step()
        return rates, optimizer

    def test_paper_adam_and_warmup_then_cosine(self):
        rates, optimizer = self.schedule('paper')
        for observed, expected in zip(rates[:3], [0.002 / 3, 0.002 * 2 / 3, 0.002]):
            self.assertAlmostEqual(observed, expected, places=14)
        self.assertTrue(all(a > b for a, b in zip(rates[2:], rates[3:])))
        self.assertAlmostEqual(rates[-1], 1e-6, places=14)

    def test_historical_exact_original_sequence(self):
        rates, optimizer = self.schedule('historical')
        expected = [0.002]
        for epoch in range(1, 8):
            factor = epoch / 3 if epoch <= 3 else 0.5 * (1 + math.cos(math.pi * (epoch - 3) / 5))
            expected.append(1e-6 + (0.002 - 1e-6) * factor)
        self.assertEqual(rates, expected)

    @unittest.skipIf(torch is None, 'PyTorch is required to instantiate optimizers')
    def test_optimizer_classes(self):
        _, paper = self.schedule('paper')
        _, historical = self.schedule('historical')
        self.assertIs(type(paper), torch.optim.Adam)
        self.assertIs(type(historical), torch.optim.AdamW)

    def test_no_warmup_and_short_runs(self):
        rates, _ = self.schedule('paper', warmup=0, total=3)
        self.assertAlmostEqual(rates[0], 0.002)
        self.assertAlmostEqual(rates[-1], 1e-6)
        rates, _ = self.schedule('paper', warmup=3, total=1)
        self.assertAlmostEqual(rates[0], 0.002 / 3)


if __name__ == '__main__':
    unittest.main()
