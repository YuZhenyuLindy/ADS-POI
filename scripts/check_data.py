#!/usr/bin/env python3
"""Validate supplied datasets through the same loader used for training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adspoi.official import load_official


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-dir', type=Path, default=Path('data_official'))
    parser.add_argument('--datasets', nargs='+', default=['nyc', 'tky', 'ca'], choices=['nyc', 'tky', 'ca'])
    args = parser.parse_args()
    for ds in args.datasets:
        data = load_official(ds, args.official_dir)
        print(json.dumps(data['stats'], indent=2))


if __name__ == '__main__':
    main()
