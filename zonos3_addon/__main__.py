"""ZONOS3-Mini add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline text-to-speech with Zyphra's ZONOS3-Mini, in 43 languages, at
44.1 kHz, in 27 stock voices or the speaker's own voice cloned from a clip.
Serves `tts.synthesize`. Zyphra's ggml runtime runs on the CPU in-process
(engine.py, through native/zonos3_c). The model downloads the first time a
line needs it (download.py), the speaker encoder the first time a voice is
cloned. After that nothing leaves the machine.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:zonos3]` and surfaces.

Options (the host sets them from Configure as ZONOS3_<KEY>)
-----------------------------------------------------------
threads: CPU threads, 0 for automatic; voice_ref_audio: a clip the speaker's
own voice falls back on when the host sends none.

Limits
------
One line at a time. A cancel stops the line within a frame (1/15 s of speech),
or stops the download (which resumes on the next request).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from zonos3_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from zonos3_addon import audio, download, tls, voices

ADDON_VERSION = __version__

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_UNSUPPORTED_VOICE = 'unsupported_voice'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'
ERR_OOM = 'out_of_memory'

# No space left on the device, or over the user's disk quota.
_DISK_FULL_ERRNOS = {errno.ENOSPC, getattr(errno, 'EDQUOT', 122)}
# Beside a package module: frozen, this script itself runs from _internal/.
_PINS_FILE = Path(download.__file__).resolve().parent / 'pins.json'

log = logging.getLogger('zonos3_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`, so nothing else (the native
    runtime prints to stdout and stderr) can write a stray line into the
    protocol. Falls back to the Python-level swap if the fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _env_number(name: str, default: float, low: float, high: float) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        log.warning('%s=%r is not a number; using %s', name, raw, default)
        value = default
    return max(low, min(high, value))


def _options() -> dict:
    return {
        'threads': int(_env_number('ZONOS3_THREADS', 0, 0, 256)),
        'voice_ref_audio': _env('ZONOS3_VOICE_REF_AUDIO'),
    }


def _resolve_language(raw) -> voices.Language:
    language = voices.language_for(raw)
    if language is not None:
        return language
    if not voices.split_tag(raw)[0]:
        raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                          "Set the project's language first: ZONOS3-Mini speaks 43 languages.")
    spoken = ', '.join(sorted({lang.name for lang in voices.LANGUAGE_BY_CODE.values()}))
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'ZONOS3-Mini does not speak {raw!r}. It speaks {spoken}; '
                      f'pick another voice engine for other languages.')


def _rate(params: dict) -> float:
    try:
        return float(params.get('rate') or 0)
    except (TypeError, ValueError):
        return 0.0


def _seed(params: dict) -> int:
    try:
        return int(params.get('seed', 1)) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# ffmpeg, for references that are not 16-bit PCM WAV at 16 or 44.1 kHz
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _read_reference(path: str) -> tuple:
    """(samples, rate) at 16 or 44.1 kHz: `path` itself if it is such a WAV, else via ffmpeg."""
    try:
        samples, rate = audio.read_wav(path)
        if rate in (16000, 44100):
            return samples, rate
    except audio.UnreadableAudio:
        pass
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise _AddonError(ERR_BAD_PARAMS,
                          f'The reference {Path(path).name} is not 16-bit WAV at 16 or 44.1 kHz, and no ffmpeg '
                          f'was found to convert it. Use such a WAV file.')
    scratch = Path(tempfile.mkdtemp(prefix='zonos3-', dir=_env('SUBTITLD_TEMP_DIR') or None))
    try:
        out = scratch / 'reference.wav'
        cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', path,
               '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', '-t', '60', str(out)]
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=_child_env(),
                              startupinfo=_no_window_startupinfo())
        if done.returncode != 0 or not out.is_file():
            tail = ' / '.join(done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:])
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {tail}')
        try:
            return audio.read_wav(out)
        except audio.UnreadableAudio as exc:
            raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {exc}') from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The model: downloaded and loaded on first use, shared by every request
# ---------------------------------------------------------------------------
_pins: dict | None = None
_loaded = None              # engine.Zonos3
_loaded_key = None          # (threads, with the speaker encoder)
_clones: dict = {}          # (path, size, mtime) of a clip -> embedding
_model_lock = threading.Lock()


def pins(group: str) -> list[dict]:
    """Every pinned file of a group, each with the repository and revision it is downloaded from."""
    global _pins
    if _pins is None:
        _pins = json.loads(_PINS_FILE.read_text(encoding='utf-8'))['groups']
    entry = _pins[group]
    return [{**pin, 'repo': entry['repo'], 'revision': entry['revision']} for pin in entry['files']]


def _file(name: str) -> Path:
    for group in ('core', 'clone'):
        for pin in pins(group):
            if pin['path'].endswith('/' + name):
                return download.local_path(pin)
    raise KeyError(name)


def _megabytes(count: int) -> str:
    return f'{count / 1e6:,.0f} MB'


_GROUP_NAMES = {'core': 'the ZONOS3-Mini model', 'clone': 'the voice-cloning encoder'}


def _ensure(state: '_Request', groups: list[str]) -> None:
    """Download whatever of `groups` is missing; progress runs 0 to 0.3."""
    todo = [pin for group in groups for pin in download.missing(pins(group))]
    if not todo:
        return
    total = sum(pin['size'] for pin in todo)
    names = [_GROUP_NAMES[g] for g in groups if download.missing(pins(g))]
    message = f'Downloading {" and ".join(names)} ({_megabytes(total)})...'
    _emit_progress(state.req_id, 0.0, message)
    last = [0.0]

    def progress(done: int, total_bytes: int) -> None:
        now = time.monotonic()
        if now - last[0] >= 0.25 or done >= total_bytes:
            last[0] = now
            _emit_progress(state.req_id, 0.3 * done / max(1, total_bytes), message)

    started = time.monotonic()
    download.ensure(todo, progress, state.cancel.is_set)
    log.info('downloaded %s in %.1fs', _megabytes(total), time.monotonic() - started)


def _load(state: '_Request', options: dict, clone: bool):
    """Download (once) and load. The caller holds _model_lock."""
    global _loaded, _loaded_key
    from zonos3_addon import engine

    groups = ['core', 'clone'] if clone else ['core']
    _ensure(state, groups)
    threads = options['threads'] or engine.default_threads()
    # The speaker encoder stays loaded once a clone asked for it.
    with_encoder = clone or (_loaded is not None and _loaded.can_clone)
    key = (threads, with_encoder)
    if _loaded is None or _loaded_key != key:
        _emit_progress(state.req_id, 0.3, 'Loading ZONOS3-Mini...')
        started = time.monotonic()
        if _loaded is not None:
            _loaded.close()
        _loaded = None
        _loaded = engine.Zonos3(_file('backbone-ptq.gguf'), _file('zac-keep2.gguf'),
                                _file('spk-encoder.gguf') if with_encoder else None, threads)
        _loaded_key = key
        log.info('loaded the model with %d threads%s in %.1fs', threads,
                 ' and the speaker encoder' if with_encoder else '', time.monotonic() - started)
    return _loaded


def _clone_embedding(model, reference: str) -> bytes:
    stat = os.stat(reference)
    key = (os.path.abspath(reference), stat.st_size, stat.st_mtime_ns)
    if key in _clones:
        return _clones[key]
    samples, rate = _read_reference(reference)
    if len(samples) < rate * 1.0:
        raise _AddonError(ERR_BAD_PARAMS, 'The reference clip is shorter than a second; '
                                          '5 to 20 seconds of one person speaking works best.')
    embedding = model.embed(samples, rate)
    while len(_clones) >= 16:
        _clones.pop(next(iter(_clones)))
    _clones[key] = embedding
    return embedding


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        from zonos3_addon.engine import Cancel
        self.req_id = req_id
        self.cancel = Cancel()   # the native runtime polls it every frame
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _handle_synthesize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    text = ' '.join(str(params.get('text') or '').split())
    output_path = params.get('output_path')
    if not text:
        _emit_error(req_id, ERR_BAD_PARAMS, 'text is empty')
        return
    if not isinstance(output_path, str) or not output_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_path is required')
        return
    try:
        language = _resolve_language(params.get('language'))
    except _silent_type() as exc:
        log.warning('synthesize %s: %s', req_id, exc)
        outcome = ('error', ERR_INTERNAL, str(exc), True)
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    voice_id = str(params.get('voice') or '') or voices.CLONE
    voice = voices.by_id(voice_id)
    if voice is None:
        _emit_error(req_id, ERR_UNSUPPORTED_VOICE, f'unknown voice {voice_id!r}')
        return
    options = _options()
    seed = _seed(params)

    reference = ''
    if voice.id == voices.CLONE:
        reference = str(params.get('voice_ref_audio') or '') or options['voice_ref_audio']
        if not reference:
            _emit_error(req_id, ERR_BAD_PARAMS,
                        "The speaker's own voice needs a clip of it, and none came with the request. "
                        'Pick a reference clip in Configure, or a stock voice.')
            return
        if not os.path.isfile(reference):
            _emit_error(req_id, ERR_BAD_PARAMS, f'The reference clip is missing: {reference}')
            return

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        from zonos3_addon import engine
        if state.cancel.is_set():
            raise engine.Cancelled()
        spoken = engine.normalize(text, language)
        if not spoken:
            raise _AddonError(ERR_BAD_PARAMS, 'Nothing in the line could be spoken.')
        rate = engine.speaking_rate(_rate(params))
        with _model_lock:
            model = _load(state, options, voice.id == voices.CLONE)
            if state.cancel.is_set():
                raise engine.Cancelled()
            embedding = _clone_embedding(model, reference) if voice.id == voices.CLONE else voice.embedding
            _emit_progress(req_id, 0.35, 'Speaking...')
            log.info('synthesize %s: voice=%s language=%s seed=%d rate=%.1f chars=%d', req_id, voice.id,
                     language.code, seed, rate, len(spoken))
            started = time.monotonic()
            samples, record = model.synthesize(spoken, language, embedding, seed=seed, rate=rate,
                                               cancel=state.cancel)
            took = time.monotonic() - started
        if state.cancel.is_set():
            raise engine.Cancelled()
        if not len(samples):
            raise _AddonError(ERR_BAD_PARAMS, 'Nothing in the line could be spoken.')
        audio.write_wav(Path(output_path), samples, model.sample_rate)
        duration = len(samples) / model.sample_rate
        log.info('synthesize %s: %.2fs of audio in %.2fs (%s)', req_id, duration, took,
                 ', '.join(f'{k}={record[k]}' for k in ('end', 'attempts', 'takes', 'tail_cut', 'seed', 'laci')
                           if k in record))
        outcome = ('result', {
            'path': output_path,
            'duration_sec': round(duration, 3),
            'sample_rate': model.sample_rate,
            'channels': 1,
            'voice': voice.id,
        })
    except (download.Cancelled, _cancelled_type()):
        log.info('synthesize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except download.DownloadError as exc:
        log.warning('synthesize %s: download failed: %s', req_id, exc)
        outcome = ('error', ERR_NETWORK, f'Could not download the model: {exc}', True)
    except _AddonError as exc:
        log.warning('synthesize %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory. ZONOS3-Mini needs about 2 GB free.', False)
    except OSError as exc:
        log.exception('synthesize %s failed', req_id)
        if getattr(exc, 'errno', None) in _DISK_FULL_ERRNOS:
            outcome = ('error', ERR_DISK_FULL, f'Out of disk space: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('synthesize %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _silent_type():
    from zonos3_addon import engine
    return engine.SilentTakes


def _cancelled_type():
    from zonos3_addon import engine
    return engine.Cancelled


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'zonos3-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'tts.synthesize',
            'languages': list(voices.LANGUAGES),
            'voices': voices.manifest_voices(),
            'voice_clone': True,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'tts.synthesize':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('tts.synthesize without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_synthesize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or the model.

    The native runtime must load and report its interface, the text
    normalization must work, the voices and pins must be there, and the TLS
    and hashing modules the downloads use must be present.
    """
    import ssl

    import zyphra_textnorm

    from zonos3_addon import engine

    english = voices.LANGUAGE_BY_CODE['en_us']
    normalized = engine.normalize('It costs $5.32.', english)
    heads = engine.laci_heads(voices.LANGUAGE_BY_CODE['de'])
    ok = ('five dollars' in normalized and heads.is_file() and heads.name.endswith('_de.json')
          and len(voices.stock()) == 27)
    report = {
        'ok': ok,
        'version': ADDON_VERSION,
        'runtime': engine.runtime_version(),
        'textnorm': getattr(zyphra_textnorm, '__version__', ''),
        'normalized': normalized,
        'voices': len(voices.all_voices()),
        'pins': {group: len(pins(group)) for group in ('core', 'clone')},
        'openssl': ssl.OPENSSL_VERSION,
        'sha256': hashlib.sha256(b'').hexdigest()[:8],
        'ca_bundle': os.environ.get('SSL_CERT_FILE', ''),
    }
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if ok else 1


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'zonos3-addon {ADDON_VERSION}')
        return 0
    # Lines in other scripts reach the log; a Windows stderr is cp1252 and would raise mid-synthesis.
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(
        level=(_env('ZONOS3_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('zonos3-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        _stop_workers()
    return rc


if __name__ == '__main__':
    sys.exit(main())
