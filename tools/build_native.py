"""Build libzonos3_c (native/) and put it, with the upstream data it needs, into build/assets/.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The add-on calls native/zonos3_c.h through ctypes. This script checks out
Zyphra's ZONOS3-Mini at a pinned commit (with its ggml submodule only: no
textnorm, no Rust), builds native/ against it, and collects:

- build/assets/<library>: libzonos3_c.so, libzonos3_c.dylib or zonos3_c.dll,
  with ggml linked in. x86-64 builds need AVX2.
- build/assets/zonos3/laci_heads/*.json: the alignment heads of each language,
  which LACI reads (assets/laci_heads/ upstream; they are not on Hugging Face).
- build/assets/zonos3/languages.json: upstream's language registry.
- build/assets/native.json: where it all came from, and each file's SHA-256.
- build/licences/zonos3/: the licences of what the library compiles in.

Linux builds run in the AlmaLinux 8 container (tools/build_native_linux.sh),
so the library loads on glibc 2.28. Windows needs Visual Studio's C++ tools,
macOS the Xcode command-line tools; both need CMake and git. Haiku builds run
on Haiku, against the system's libroot and libstdc++ (on every install).

    python tools/build_native.py [--jobs N]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
ASSETS = BUILD / 'assets'
LICENCES = BUILD / 'licences' / 'zonos3'
SRC = BUILD / 'zonos3-src'
NATIVE_BUILD = BUILD / 'native'

REPO = 'https://github.com/Zyphra/ZONOS3-Mini'
COMMIT = 'a50e14117d99181ba7f28047b451ba03ba1a09a9'  # main, 2026-11-17

LIBRARY = {'linux': 'libzonos3_c.so', 'windows': 'zonos3_c.dll', 'macos': 'libzonos3_c.dylib',
           'haiku': 'libzonos3_c.so'}

# Haiku: libroot is its C library; libstdc++ and libgcc_s come with every
# install (gcc_syslibs).
HAIKU_SYSTEM = {'libroot.so', 'libstdc++.so.6', 'libgcc_s.so.1', 'libnetwork.so', 'libbsd.so'}

# Libraries every Linux desktop has; the build may link nothing else dynamically.
LINUX_SYSTEM = {
    'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
    'ld-linux-x86-64.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
}
# The newest symbol version each AlmaLinux 8 (glibc 2.28, GCC 8) library provides.
LINUX_CEILINGS = {'GLIBC': (2, 28), 'GLIBCXX': (3, 4, 25), 'CXXABI': (1, 3, 11), 'GCC': (8, 0, 0)}


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def output(cmd, cwd=None):
    env = dict(os.environ, LC_ALL='C')
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


# Files as committed: Zyphra's ggml patches don't apply to a tree with Windows line endings. `-c` reaches the
# submodule's git too.
GIT = ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf']


def checkout():
    """The pinned commit and its ggml submodule, as shallow as git allows."""
    if not (SRC / '.git').exists():
        SRC.mkdir(parents=True, exist_ok=True)
        run([*GIT, 'init', '-q'], cwd=SRC)
        run([*GIT, 'remote', 'add', 'origin', REPO + '.git'], cwd=SRC)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SRC, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        run([*GIT, 'fetch', '-q', '--depth', '1', 'origin', COMMIT], cwd=SRC)
        # --force: CMake applies Zyphra's ggml patches inside the submodule, which a re-checkout must undo.
        run([*GIT, 'checkout', '-q', '--force', COMMIT], cwd=SRC)
    run([*GIT, 'submodule', 'update', '--init', '--force', '--depth', '1', 'servers/ggml/ggml'], cwd=SRC)


def build(platform, jobs):
    configure = ['cmake', '-S', ROOT / 'native', '-B', NATIVE_BUILD, f'-DZONOS3_DIR={SRC}',
                 f'-DZS_UPSTREAM_COMMIT={COMMIT}']
    if platform == 'windows':
        configure += ['-A', 'x64']
    else:
        configure += ['-DCMAKE_BUILD_TYPE=Release']
    if platform == 'macos':
        # Apple Silicon only, as the catalog ships; macOS 13.3 like the other add-ons.
        configure += ['-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3']
    run(configure)
    run(['cmake', '--build', NATIVE_BUILD, '--config', 'Release', '--parallel', str(jobs), '--target', 'zonos3_c'])
    name = LIBRARY[platform]
    for candidate in (NATIVE_BUILD / name, NATIVE_BUILD / 'Release' / name):
        if candidate.is_file():
            return candidate
    sys.exit(f'FATAL: the build made no {name} in {NATIVE_BUILD}')


# ---- Linux: refuse anything AlmaLinux 8 couldn't load ---------------------

def needed(lib):
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', output(['readelf', '-d', lib]))


def version_needs(lib):
    needs, current = {}, None
    for line in output(['readelf', '-V', '--wide', lib]).splitlines():
        m = re.search(r'File: (\S+)', line)
        if m:
            current = needs.setdefault(m.group(1), [])
            continue
        m = re.search(r'Name: (\S+)', line)
        if m and current is not None:
            current.append(m.group(1))
    return needs


def check_linux(lib):
    problems = [f'{lib.name} needs {dep}, which is not a system library'
                for dep in needed(lib) if dep not in LINUX_SYSTEM]
    for dep, versions in version_needs(lib).items():
        for version in versions:
            m = re.fullmatch(r'([A-Z]+)_([0-9.]+)', version)
            if not m or m.group(1) not in LINUX_CEILINGS:
                continue
            if tuple(int(x) for x in m.group(2).split('.')) > LINUX_CEILINGS[m.group(1)]:
                problems.append(f'{lib.name} needs {version} from {dep}')
    exported = [line.split()[-1] for line in output(['nm', '-D', '--defined-only', lib]).splitlines()
                if line.split()[-1:] and not line.split()[-1].startswith('zs_') and ' T ' in line]
    if exported:
        problems.append(f'{lib.name} exports more than zs_*: {exported[:5]}')
    if problems:
        sys.exit('not loadable on glibc 2.28 / GCC 8:\n  ' + '\n  '.join(sorted(set(problems))))
    print('the library loads on glibc 2.28 with the GCC 8 runtime, and exports only zs_*')


def check_haiku(lib):
    problems = [f'{lib.name} needs {dep}, which is not a system library'
                for dep in needed(lib) if dep not in HAIKU_SYSTEM]
    exported = [line.split()[-1] for line in output(['nm', '-D', '--defined-only', lib]).splitlines()
                if line.split()[-1:] and not line.split()[-1].startswith('zs_') and ' T ' in line]
    if exported:
        problems.append(f'{lib.name} exports more than zs_*: {exported[:5]}')
    if problems:
        sys.exit('not loadable on Haiku:\n  ' + '\n  '.join(sorted(set(problems))))
    print('the library needs only Haiku system libraries, and exports only zs_*')


# ---- licences ---------------------------------------------------------------

def leading_comment(path):
    """The licence block at the top of a vendored header."""
    text = path.read_text(encoding='utf-8', errors='replace')
    m = re.match(r'\s*/\*(.*?)\*/', text, re.S)
    if not m:
        sys.exit(f'FATAL: no licence comment at the top of {path}')
    return m.group(1).strip('\n') + '\n'


def collect_licences():
    fresh(LICENCES)
    shutil.copyfile(SRC / 'LICENSE', LICENCES / 'LICENSE')
    shutil.copyfile(SRC / 'NOTICE', LICENCES / 'NOTICE')
    third = LICENCES / 'third_party'
    (third / 'ggml').mkdir(parents=True)
    shutil.copyfile(SRC / 'servers/ggml/ggml/LICENSE', third / 'ggml' / 'LICENSE')
    (third / 'lmdb').mkdir()
    for name in ('LICENSE', 'COPYRIGHT'):
        shutil.copyfile(SRC / 'servers/ggml/vendor/lmdb' / name, third / 'lmdb' / name)
    (third / 'pocketfft').mkdir()
    (third / 'pocketfft' / 'LICENSE').write_text(
        leading_comment(SRC / 'servers/ggml/vendor/pocketfft_hdronly.hpp'), encoding='utf-8')
    (third / 'nlohmann-json').mkdir()
    header = (SRC / 'servers/ggml/vendor/json.hpp').read_text(encoding='utf-8', errors='replace')
    spdx = re.findall(r'^// SPDX-[^\n]*$', header[:4000], re.M)
    if 'SPDX-License-Identifier: MIT' not in '\n'.join(spdx):
        sys.exit('FATAL: vendor/json.hpp is no longer marked MIT; check its licence')
    (third / 'nlohmann-json' / 'LICENSE').write_text(
        '\n'.join(spdx) + '\n\n' + MIT_TEXT.replace('<COPYRIGHT>', 'Copyright (c) 2013-2025 Niels Lohmann'),
        encoding='utf-8')


MIT_TEXT = '''MIT License

<COPYRIGHT>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def main():
    host = 'haiku' if sys.platform.startswith('haiku') else {'linux': 'linux', 'win32': 'windows',
                                                               'darwin': 'macos'}[sys.platform]
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()

    checkout()
    library = build(host, args.jobs)
    fresh(ASSETS)
    target = ASSETS / LIBRARY[host]
    shutil.copyfile(library, target)
    if host == 'linux':
        check_linux(target)
    elif host == 'haiku':
        check_haiku(target)
    data = ASSETS / 'zonos3'
    shutil.copytree(SRC / 'assets' / 'laci_heads', data / 'laci_heads')
    shutil.copyfile(SRC / 'model/src/zonos3/registries/languages.json', data / 'languages.json')
    collect_licences()

    files = {p.relative_to(ASSETS).as_posix(): sha256(p) for p in sorted(ASSETS.rglob('*')) if p.is_file()}
    record = {'runtime': 'ZONOS3-Mini (servers/ggml) through native/zonos3_c', 'commit': COMMIT,
              'platform': host, 'entry': LIBRARY[host], 'files': files}
    (ASSETS / 'native.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f'{target.name}: {target.stat().st_size / 2**20:.1f} MiB; {len(files)} files in {ASSETS}')


if __name__ == '__main__':
    main()
