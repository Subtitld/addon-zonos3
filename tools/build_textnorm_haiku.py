#!/usr/bin/env python3
"""Build zyphra-textnorm on Haiku and lay it out in build/haiku-lib/ for tools/build_haiku.py.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python3.14 tools/build_textnorm_haiku.py [--jobs N]

PyPI has no zyphra-textnorm wheel for Haiku, and HaikuPorts has neither pip
nor maturin, so this builds the pinned sdist's Python extension with cargo
(the rust_bin package) and writes what maturin's wheel holds:
zyphra_textnorm/ with the module (named for this Python's extension suffix),
maturin's two-line __init__.py, and a dist-info with the version and the
licences.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
SRC = BUILD / 'textnorm-src'
OUT = BUILD / 'haiku-lib'
DOWNLOADS = BUILD / 'downloads'

VERSION = '0.3.1'
SDIST = ('https://files.pythonhosted.org/packages/90/02/0d90815e5e5fe008eac8cdd5c4b7df3d4f121ffa55cb8d7e055275055bb0/'
         'zyphra_textnorm-0.3.1.tar.gz', 'f3d14af5e2d048dcd7ff805fd08fbf914733e97e53ce015d9779a8e7ddb5d000')

# What maturin writes for a module with no Python source of its own.
INIT = '''from .zyphra_textnorm import *

__doc__ = zyphra_textnorm.__doc__
if hasattr(zyphra_textnorm, "__all__"):
    __all__ = zyphra_textnorm.__all__
'''


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, expected: str) -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / url.rsplit('/', 1)[1]
    if not (target.is_file() and sha256(target) == expected):
        print('+ fetch', url, flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, target.open('wb') as out:
            shutil.copyfileobj(response, out)
    if sha256(target) != expected:
        sys.exit(f'FATAL: {target.name} does not match its pinned sha256')
    return target


def main() -> int:
    if not sys.platform.startswith('haiku'):
        sys.exit('FATAL: run this on Haiku')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()

    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir(parents=True)
    with tarfile.open(fetch(*SDIST)) as tf:
        tf.extractall(SRC, filter='data')
    source = SRC / f'zyphra_textnorm-{VERSION}'

    env = dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(BUILD / 'textnorm-target'))
    run(['cargo', 'build', '--release', '--locked', '--jobs', str(args.jobs),
         '--manifest-path', source / 'python' / 'Cargo.toml', '--features', 'pyo3/extension-module'],
        cwd=source, env=env)
    built = BUILD / 'textnorm-target' / 'release' / 'libzyphra_textnorm.so'
    if not built.is_file():
        sys.exit(f'FATAL: cargo made no {built}')

    package = OUT / 'zyphra_textnorm'
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    module = package / f'zyphra_textnorm{sysconfig.get_config_var("EXT_SUFFIX")}'
    shutil.copyfile(built, module)
    run(['strip', '--strip-unneeded', module])
    (package / '__init__.py').write_text(INIT, encoding='utf-8')

    info = OUT / f'zyphra_textnorm-{VERSION}.dist-info'
    if info.exists():
        shutil.rmtree(info)
    (info / 'licenses').mkdir(parents=True)
    (info / 'METADATA').write_text(f'Metadata-Version: 2.4\nName: zyphra-textnorm\nVersion: {VERSION}\n',
                                   encoding='utf-8')
    for name in ('LICENSE', 'NOTICE'):
        shutil.copyfile(source / name, info / 'licenses' / name)
    shutil.copytree(source / 'licenses', info / 'licenses' / 'licenses')

    sys.path.insert(0, str(OUT))
    import zyphra_textnorm
    print(zyphra_textnorm.Normalizer().normalize('$5.32', 'en') if hasattr(zyphra_textnorm, 'Normalizer') else '')
    print(f'{module.name}: {module.stat().st_size / 2**20:.1f} MiB; loads')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
