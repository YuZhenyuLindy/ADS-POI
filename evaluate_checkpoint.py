#!/usr/bin/env python3
"""Evaluate a saved release checkpoint against its exact input data."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import torch
from adspoi.config import source_hash
from adspoi.data import create_dataloaders
from adspoi.engine import evaluate
from adspoi.metrics import REPORT_KS
from adspoi.model import build_model
from adspoi.official import load_official
from adspoi.utils import pick_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--official-dir', type=Path, default=Path('data_official'))
    p.add_argument('--device', default='auto')
    args = p.parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    if ckpt['source_hash'] != source_hash():
        raise ValueError('Checkpoint source hash differs from this release')
    with contextlib.redirect_stdout(sys.stderr):
        data = load_official(ckpt['dataset'], args.official_dir)
        if data['stats']['data_fingerprint'] != ckpt['data_fingerprint']:
            raise ValueError('Checkpoint data fingerprint differs from the supplied data')
        if data['num_pois'] != ckpt['num_pois']:
            raise ValueError('Checkpoint vocabulary differs from the supplied data')
        _, _, loader = create_dataloaders(data, ckpt['config'])
    model = build_model(ckpt['num_pois'], ckpt['config'], ckpt['model_variant'], ckpt['model'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    device = pick_device(args.device)
    model.to(device)
    metrics, _, _ = evaluate(model, loader, device, REPORT_KS)
    print(json.dumps(dict(dataset=ckpt['dataset'], model=ckpt['model'],
        seed=ckpt['seed'], test=metrics), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
