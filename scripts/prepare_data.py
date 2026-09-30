#!/usr/bin/env python3
"""Copy existing official split files into the repository input layout.

This command does not download, filter or re-split data. The source may contain
<dataset>/<file> or the upstream layout <dataset>/preprocessed/<file>.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES = ('train_sample.csv', 'validate_sample_with_traj.csv', 'test_sample_with_traj.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('data_official'))
    p.add_argument('--datasets', nargs='+', default=['nyc', 'tky', 'ca'], choices=['nyc', 'tky', 'ca'])
    args = p.parse_args()
    copies = []
    for ds in args.datasets:
        for filename in FILES:
            choices = [args.source / ds / filename, args.source / ds / 'preprocessed' / filename]
            found = [path for path in choices if path.is_file()]
            if len(found) != 1:
                p.error(f'Expected exactly one source for {ds}/{filename}, found {len(found)}')
            src, dst = found[0], args.out / ds / filename
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                p.error(f'Refusing to replace a different existing file: {dst}')
            copies.append((src, dst, digest))
    manifest = {}
    for src, dst, digest in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst)
        manifest[dst.relative_to(args.out).as_posix()] = digest
    (args.out / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
