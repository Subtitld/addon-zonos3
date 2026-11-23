"""ZONOS3-Mini through libzonos3_c, loaded in-process with ctypes.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

native/zonos3_c.h is a small C interface over Zyphra's ggml runtime
(servers/ggml of ZONOS3-Mini): load the backbone, the waveform decoder and the
speaker encoder once, then embed voices and speak lines. tools/build_native.py
puts the library into build/assets/; the frozen add-on carries it in
_internal/assets/. Text is normalized here first, with Zyphra's
zyphra-textnorm ("$5.32" -> "five dollars thirty two cents").

Every line runs under LACI (Zyphra's alignment check, ported from their
server): the alignment heads follow which part of the text is being read,
skips and stalls are taken again, and the line ends once the text has been
read. The tail cut then ends it at the first silence after that, as
upstream's own short-text fix cannot run on this checkpoint (it has no pause
tokens); see native/zonos3_c.cpp.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import platform
import sys
import threading
from array import array
from ctypes import POINTER, Structure, byref, c_char_p, c_double, c_float, c_int, c_int64, c_uint32, c_void_p
from pathlib import Path

from zonos3_addon.voices import Language

log = logging.getLogger('zonos3_addon.engine')

SAMPLE_RATE = 44100
SPEAKER_DIM = 192
INTERFACE = '1'
_PACKAGE = Path(__file__).resolve().parent

LIBRARY = {'linux': 'libzonos3_c.so', 'win32': 'zonos3_c.dll', 'darwin': 'libzonos3_c.dylib'}

# The speaking-rate condition is in UTF-8 bytes per second; 14.5 is about the middle of the model's scale
# (bucket 15 of 30), the pace of ordinary speech in the Latin-script languages.
_NATURAL_RATE = 14.5
# A take is redone with the next seed when it holds less voice than this (the model, rarely, says nothing), or when
# LACI ran out of attempts at it ("capped": the last one still skipped or stopped short). A cloned voice in a language
# its clip isn't in does both often: an English clip in Gujarati gave 10 silent and 2 half-read takes in 16 seeds.
_MIN_VOICED_SEC = 0.12
_TAKES = 12
# A take LACI gave up on still holds speech, often all of it, and costs a full reading plus LACI's own retries (an
# English clip in Greek: every take capped, at about 10 s each). After this many of them, the longest is kept.
_CAPPED_TAKES = 3



class EngineError(Exception):
    """The runtime failed, or could not be loaded."""


class Cancelled(Exception):
    """The request was cancelled."""

class SilentTakes(EngineError):
    """Every take of a line came out (nearly) silent."""


def assets_dir() -> Path:
    """Where the native library is: $ZONOS3_ASSETS, the frozen bundle's, or build/assets/."""
    override = (os.environ.get('ZONOS3_ASSETS') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'assets'
    return _PACKAGE.parent / 'build' / 'assets'


def library_path() -> Path:
    return assets_dir() / LIBRARY.get(sys.platform, LIBRARY['linux'])


def laci_heads(language: Language) -> Path:
    """The alignment heads of a language; English's are the default file."""
    folder = assets_dir() / 'zonos3' / 'laci_heads'
    own = folder / f'zonos3_mini_heads_{language.speech}.json'
    return own if own.is_file() else folder / 'zonos3_mini_heads.json'


def default_threads() -> int:
    """Every logical processor, at most 8: more is rarely faster for one line at a time."""
    return max(1, min(8, os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# The CPU: x86-64 builds need AVX2
# ---------------------------------------------------------------------------
def _is_x86_64() -> bool:
    return platform.machine().lower() in ('x86_64', 'amd64', 'x64')


def cpu_lacks_avx2() -> bool:
    """True when this is an x86-64 processor known not to have AVX2.

    ggml is built for x86-64-v3 (AVX2, FMA, F16C), and on an older processor
    it would die on an illegal instruction, taking the add-on with it. When
    the check cannot tell, it lets the load go ahead.
    """
    if not _is_x86_64():
        return False
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as handle:
                for line in handle:
                    if line.startswith('flags'):
                        return 'avx2' not in line.split(':', 1)[1].split()
        except OSError:
            return False
        return False
    if sys.platform == 'win32':
        # PF_AVX2_INSTRUCTIONS_AVAILABLE. A Windows too old to know the
        # feature answers False for it, so a False only counts on Windows 11.
        try:
            if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
                return False
            return sys.getwindowsversion().build >= 22000
        except (AttributeError, OSError):
            return False
    return False


# ---------------------------------------------------------------------------
# The C interface (native/zonos3_c.h)
# ---------------------------------------------------------------------------
class Request(Structure):
    _fields_ = [('text', c_char_p), ('language', c_char_p), ('embedding', POINTER(c_float)),
                ('cfg_scale', c_float), ('speaking_rate', c_double), ('seed', c_uint32), ('max_frames', c_int),
                ('laci_heads', c_char_p), ('short_text', c_int), ('tail_cut', c_int),
                ('cancel', POINTER(c_int))]


_lib = None
_lib_lock = threading.Lock()


def load_library():
    """The library, loaded once. Raises EngineError when it is missing or cannot load."""
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = library_path()
        if not path.is_file():
            raise EngineError(f'the speech runtime is missing: {path}')
        if cpu_lacks_avx2():
            raise EngineError('this processor has no AVX2, which ZONOS3-Mini needs (Intel since 2013, AMD since 2015)')
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise EngineError(f'could not load {path.name}: {exc}') from exc
        lib.zs_version.restype = c_char_p
        lib.zs_version.argtypes = []
        lib.zs_load.restype = c_void_p
        lib.zs_load.argtypes = [c_char_p, c_char_p, c_char_p, c_int, c_char_p, c_int]
        lib.zs_unload.restype = None
        lib.zs_unload.argtypes = [c_void_p]
        lib.zs_sample_rate.restype = c_int
        lib.zs_sample_rate.argtypes = [c_void_p]
        lib.zs_speaker_dim.restype = c_int
        lib.zs_speaker_dim.argtypes = [c_void_p]
        lib.zs_embed.restype = c_int
        lib.zs_embed.argtypes = [c_void_p, POINTER(c_float), c_int64, c_int, POINTER(c_float)]
        lib.zs_synthesize.restype = c_int
        lib.zs_synthesize.argtypes = [c_void_p, POINTER(Request), POINTER(POINTER(c_float)), POINTER(c_int64),
                                      c_char_p, c_int, c_char_p, c_int]
        lib.zs_free.restype = None
        lib.zs_free.argtypes = [POINTER(c_float)]
        interface = lib.zs_version().decode('ascii', 'replace').split()[0]
        if interface != INTERFACE:
            raise EngineError(f'{path.name} has interface {interface}, the add-on needs {INTERFACE}')
        _lib = lib
        return lib


def runtime_version() -> str:
    return load_library().zs_version().decode('ascii', 'replace')


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
_normalizer = None


def normalize(text: str, language: Language) -> str:
    """The text as it is spoken, in the language's grammar: numbers, dates, units, money, symbols."""
    global _normalizer
    import zyphra_textnorm
    if _normalizer is None:
        _normalizer = zyphra_textnorm.Normalizer()
    return _normalizer.normalize(text, language.nemo).strip()


def speaking_rate(rate_percent: float) -> float:
    """The host's rate, -100..100 %, as the model's speaking-rate condition; -1 leaves the voice's own pace."""
    if not rate_percent:
        return -1.0
    return _NATURAL_RATE * max(0.5, min(2.0, 1.0 + rate_percent / 100.0))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Cancel:
    """A cancel the runtime polls every frame."""

    def __init__(self):
        self._flag = c_int(0)

    def set(self) -> None:
        self._flag.value = 1

    def is_set(self) -> bool:
        return bool(self._flag.value)

    @property
    def pointer(self):
        return ctypes.pointer(self._flag)


def voiced_seconds(samples: array, rate: int = SAMPLE_RATE, threshold: float = 0.02) -> float:
    """How much of `samples` is louder than `threshold` (RMS over 20 ms windows)."""
    window = rate // 50
    voiced = 0
    for start in range(0, len(samples) - window + 1, window):
        chunk = samples[start:start + window]
        if math.sqrt(sum(x * x for x in chunk) / window) >= threshold:
            voiced += 1
    return voiced * window / rate


class Zonos3:
    """One loaded model. Lines are spoken one at a time (the caller serializes them)."""

    def __init__(self, backbone: Path, decoder: Path, speaker_encoder: Path | None, threads: int):
        lib = load_library()
        err = ctypes.create_string_buffer(1024)
        handle = lib.zs_load(str(backbone).encode(), str(decoder).encode(),
                             str(speaker_encoder).encode() if speaker_encoder else None, threads, err, len(err))
        if not handle:
            raise EngineError(err.value.decode('utf-8', 'replace') or 'could not load the model')
        self._lib = lib
        self._handle = handle
        self.sample_rate = lib.zs_sample_rate(handle)
        self.can_clone = speaker_encoder is not None
        if lib.zs_speaker_dim(handle) != SPEAKER_DIM:
            raise EngineError(f'the model has {lib.zs_speaker_dim(handle)}-value speaker embeddings, not {SPEAKER_DIM}')

    def close(self) -> None:
        if self._handle:
            self._lib.zs_unload(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — interpreter shutdown
            pass

    def embed(self, samples: array, rate: int) -> bytes:
        """The speaker embedding of a clip: mono float32 at 44100 or 16000 Hz (the first 60 s count)."""
        if not self.can_clone:
            raise EngineError('loaded without the speaker encoder')
        if rate not in (16000, 44100):
            raise EngineError(f'the speaker encoder takes 16 or 44.1 kHz, not {rate} Hz')
        clip = (c_float * len(samples)).from_buffer(samples)
        out = (c_float * SPEAKER_DIM)()
        if self._lib.zs_embed(self._handle, clip, len(samples), rate, out):
            raise EngineError('could not embed the reference clip')
        return bytes(out)

    def _once(self, text: str, language: Language, embedding: bytes | None, seed: int, rate: float,
              laci: bool, cancel: Cancel) -> tuple[array, dict]:
        emb = (c_float * SPEAKER_DIM).from_buffer_copy(embedding) if embedding else None
        heads = str(laci_heads(language)).encode() if laci else None
        request = Request(text.encode('utf-8'), language.code.encode(), emb, 1.0, rate, seed & 0xFFFFFFFF, 0,
                          heads, 1 if laci else 0, 1 if laci else 0, cancel.pointer)
        out, count = POINTER(c_float)(), c_int64()
        err = ctypes.create_string_buffer(1024)
        info = ctypes.create_string_buffer(16384)
        rc = self._lib.zs_synthesize(self._handle, byref(request), byref(out), byref(count), err, len(err),
                                     info, len(info))
        if cancel.is_set():
            if out:
                self._lib.zs_free(out)
            raise Cancelled()
        if rc:
            raise EngineError(err.value.decode('utf-8', 'replace') or 'synthesis failed')
        samples = array('f')
        if count.value:
            samples.frombytes(ctypes.string_at(out, count.value * 4))
            self._lib.zs_free(out)
        record = {}
        for line in info.value.decode('utf-8', 'replace').splitlines():
            try:
                record.update(json.loads(line))
            except ValueError:
                pass
        return samples, record

    def synthesize(self, text: str, language: Language, embedding: bytes | None, seed: int = 0,
                   rate: float = -1.0, cancel: Cancel | None = None) -> tuple[array, dict]:
        """(samples at sample_rate, what happened). `text` is normalized already.

        A take that is silent, or that LACI gave up on, is taken again with
        the next seed, up to _TAKES (and _CAPPED_TAKES of those LACI gave up
        on); then the longest take with voice is kept, unless all were silent
        (SilentTakes). LACI's probe reads at most 2048
        text rows; a longer line is spoken without it.
        """
        cancel = cancel or Cancel()
        laci = True
        best: tuple[array, dict] | None = None
        capped = 0
        for attempt in range(_TAKES):
            try:
                samples, record = self._once(text, language, embedding, seed + attempt, rate, laci, cancel)
            except EngineError as exc:
                if laci and 'too long for the alignment probe' in str(exc):
                    log.info('the line is too long for LACI; speaking it without')
                    laci = False
                    samples, record = self._once(text, language, embedding, seed + attempt, rate, laci, cancel)
                else:
                    raise
            record['seed'] = seed + attempt
            record['laci'] = laci
            record['takes'] = attempt + 1
            silent = voiced_seconds(samples, self.sample_rate) < _MIN_VOICED_SEC
            if not silent and not record.get('capped'):
                return samples, record
            log.info('take %d %s', attempt + 1, 'holds almost no voice' if silent else 'is one LACI gave up on')
            if not silent:
                capped += 1
                if best is None or len(samples) > len(best[0]):
                    best = (samples, record)
                if capped >= _CAPPED_TAKES:
                    break
        if best is None:
            raise SilentTakes(f'{_TAKES} takes of the line came out silent. With a cloned voice, a clip in the '
                              'line\'s language helps; or pick a stock voice.')
        return best
