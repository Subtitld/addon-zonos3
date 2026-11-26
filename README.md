# ZONOS3-Mini add-on for Subtitld

Offline text-to-speech for [Subtitld](https://subtitld.org) with Zyphra's
[ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini), on Zyphra's own
ggml runtime, with no PyTorch:

- **43 languages**: Arabic, Bengali, Bulgarian, Chinese (Mandarin), Czech,
  Danish, Dutch, English (US and British), Finnish, French, Georgian, German,
  Greek, Gujarati, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese,
  Kannada, Korean, Malay, Malayalam, Marathi, Norwegian, Odia, Polish,
  Portuguese, Punjabi, Romanian, Russian, Slovak, Spanish, Swedish, Tagalog,
  Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu and Vietnamese.
- **27 stock voices**, Zyphra's default voices, recorded in English, French,
  German, Hindi, Italian, Korean, Mandarin, Portuguese, Spanish and Telugu.
  Every voice speaks every language, but with its own accent, which can be
  strong: an English voice's Russian is hard to follow. For the other
  languages, clone the speaker's voice.
- **The speaker's own voice**, cloned from a clip of the video: 5 to 20
  seconds of one person speaking is enough.
- 44.1 kHz mono.

The model downloads the first time a line is spoken (1,115 MB); the speaker
encoder the first time a voice is cloned (45 MB).

## Speed and memory

Measured on an Intel i7-13700T laptop CPU, with other work running:

| | |
|---|---|
| Speed | about twice as fast as real time (a 2 s line in about 1 s) |
| Loading the model | about 4 s, once |
| Memory | 1.6 GB, 1.85 GB after cloning |
| Cloning a voice | about 5 s for an 11 s clip, once per clip |

x86-64 processors need AVX2 (Intel since 2013, AMD since 2015). The model
runs on the CPU on every platform.

## Settings

| Setting | Default | |
|---|---|---|
| CPU threads | 0 | 0 uses the logical processors, at most 8. |
| Fallback reference clip | none | Cloned when Subtitld has no clip of the speaker to send. |

Subtitld's speed setting (−100 to +100 %) moves the model's speaking-rate
condition; at 0 each voice keeps its own pace.

## How it works

- **Text.** Zyphra's [zyphra-textnorm](https://github.com/Zyphra/zyphra-textnorm)
  writes each line out as it is spoken, in its language's grammar
  ("$5.32" → "five dollars thirty two cents", "3. Oktober" → "dritten
  Oktober"), and drops markup and emoji names into words.
- **Speech.** `native/` builds `libzonos3_c`, a small C interface over
  ZONOS3-Mini's ggml runtime (`servers/ggml` of
  [Zyphra/ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini), at the commit
  `tools/build_native.py` pins), which the add-on loads in-process with
  `ctypes`.
- **Lines that end where the text ends.** Left alone, the model often keeps
  talking after a short line ("Hello there." → "Hello there. I was a stud.").
  Every line runs under LACI, Zyphra's alignment check, ported from their
  server: alignment heads follow which part of the text is being read, a
  skipped or stalled passage is taken again, and the line is pushed to stop
  once the text has been read. The add-on then cuts at the first silence
  after that point, with Zyphra's codes-only silence detector. That cut is
  this add-on's own rule, not Zyphra's: their fix for short lines adds a
  filler sentence after a pause token, and this checkpoint has no pause
  tokens. On short subtitle lines, transcribed back with Whisper, 61 of 64
  takes were clean with both, against 37 of 64 without.
- **Takes again.** A take with almost no voice in it, or one LACI gave up on
  (it ran out of its own retries and the reading still skipped or stopped
  short), is taken again with the next seed: up to 12 takes, of which at
  most 3 that LACI gave up on; then the longest is kept. A cloned voice in a
  language its clip isn't in needs this most: with an English clip, Gujarati
  came out silent in 10 seeds of 16 and half-read in 2 more under 26.11.1,
  which tried 3 seeds. When every take is silent, the line fails with a
  message instead of writing silence. Such clones keep a strong accent
  either way, and in some languages (Greek, in our tests) lose the words;
  a reference clip in the line's language, or a stock voice, does better.
- **Downloads** come from Zyphra's Hugging Face repository, pinned to one
  revision and checked by SHA-256 (`zonos3_addon/pins.json`), into
  Subtitld's model folder (`$SUBTITLD_MODELS_DIR/zonos3/`). An interrupted
  download resumes on the next request.

## Licences

| Part | Licence |
|---|---|
| This add-on | Apache-2.0 |
| ZONOS3-Mini weights and default voices | Apache-2.0, by Zyphra |
| ZONOS3-Mini runtime | MIT, with ggml and nlohmann/json (MIT), LMDB (OpenLDAP) and pocketfft (BSD-3-Clause) |
| zyphra-textnorm | MIT, Apache-2.0 (NeMo-text-processing grammars) and Unicode-3.0 (CLDR data) |

Each release archive carries every notice in `THIRD_PARTY_LICENSES/`.

## Building

    tools/build_linux.sh                   # Linux: runtime, bundle and archive, in AlmaLinux 8 (Docker)

    python tools/build_native.py           # Windows, macOS: the runtime into build/assets/
    pip install -e '.[build,dev]'
    pyinstaller pyinstaller.spec --noconfirm

`tools/pin_models.py` and `tools/make_voices.py` rewrite `pins.json` and
`voices.json` for another revision of the model; `tools/crate_licences.py`
rewrites the Rust crate notices for another version of zyphra-textnorm.
Tests: `pytest` (the native runtime and model-backed tests need
`build/assets/` and `ZONOS3_TEST_MODELS_DIR`; `tests/test_installed.py` runs
a frozen binary).

## Haiku

libzonos3_c and zyphra-textnorm (a Rust extension with no Haiku wheel, built with cargo from the rust_bin package) are built on Haiku, so the Haiku archive is built on a Haiku machine and added to a release by hand. Haiku has no PyInstaller: the add-on runs on HaikuPorts' Python 3.14.

```bash
python3.14 tools/build_native.py
python3.14 tools/build_textnorm_haiku.py
python3.14 tools/build_haiku.py
```
