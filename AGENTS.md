# Repository Instructions

## Project Context

This repository is for building a realtime speech-to-text package. The package should take raw microphone audio as input and produce completed transcript text as output.

Current target hardware:

- Development machine with GPU acceleration.
- Final deployment target: NVIDIA Jetson Orin Nano running JetPack 6.2.2.
- Microphone: ReSpeaker Lite.
- STT model direction: `faster-whisper` with the Whisper `small` model.

## Working Area

New work should happen in `audio_in/` unless the user explicitly asks otherwise.

Do not use the older scripts outside `audio_in/` as design references. They were exploratory scratch work and should not guide the new architecture unless the user explicitly asks to inspect or port a specific idea.

For now, ignore LLM integration. The near-term behavior is: listen to the microphone, detect the end of a spoken utterance, then print the final transcript in the CLI.

## Engineering Guidelines

- Prefer a modular pipeline over one large script:
  - audio device discovery and capture
  - resampling/channel handling
  - VAD/speech detection
  - utterance buffering
  - partial transcription
  - end-of-utterance detection
  - final transcript event output
  - metrics/logging
- Keep realtime latency visible. Track capture latency, partial transcript interval, finalization delay, and transcription time.
- Always preserve a GPU path. For `faster-whisper`, use `device="cuda"` when available.
- Keep Jetson portability in mind:
  - avoid unnecessary heavyweight dependencies
  - keep audio I/O backend assumptions explicit
  - document CUDA, cuDNN, CTranslate2, and Python version constraints
  - make model/device/compute type configurable
- Avoid hard-coded microphone device IDs in final code. Provide device listing and config-based selection.
- Prefer 16 kHz mono PCM internally.
- Keep the core package independent from the CLI. The CLI should be a thin runner around package APIs.
- Separate internal partial hypotheses from final utterances.
- Do not send low-confidence silence, noise, or Whisper hallucination filler as final user text.

## Default Runtime Choices

Initial defaults for development:

- sample rate: `16000`
- channels: `1`
- chunk duration: `0.2` to `0.5` seconds
- model: `small`
- device: `cuda`
- compute type on desktop GPU: `float16` or `int8_float16`
- compute type on Jetson: benchmark `int8`, `int8_float16`, and `float16`
- partial transcript interval: around `0.8` to `2.0` seconds after speech starts
- final silence threshold: around `0.8` to `1.5` seconds, tuned with EOU logic

## Testing Expectations

When changing the pipeline, include at least one way to test without speaking into the mic:

- recorded WAV replay
- synthetic audio fixture
- saved debug chunks

For live changes, verify:

- microphone device can be selected
- CUDA is actually used
- partial transcripts update while speaking
- final transcript fires once per completed utterance
- silence/noise does not produce user text
- interruption and long utterances do not crash the process

## Style

- Keep code readable and practical. This is a realtime system, so clear control flow is more valuable than clever abstractions.
- Add short comments only where timing, buffering, or device behavior is non-obvious.
- Prefer configuration files or CLI flags over editing constants in code.

## Crucial Design Discoveries & Optimizations

- **VAD & Audio Gating Calibration**: 
  - Never run dynamic noise floor calibration on active utterances using `calibration_seconds=0.0`. Doing so evaluates only the very first frame of the active utterance, which is highly unstable (often capturing breath, lip smacks, or quiet speech start). This inflates the threshold and causes the gating logic to reject valid speech (leaving the UI stuck at `Speaking...` and dropping finalized sentences).
  - Always use the robust session-level calibrated VAD threshold for both partial and final audio-gating (`calibration_seconds=-1.0`, `min_threshold=vad.threshold`).
- **ReSpeaker Lite Channel Selection**:
  - The ReSpeaker Lite is a 2-channel array. Channel 0 contains processed hardware-cleaned audio (with onboard Acoustic Echo Cancellation, Noise Suppression, Automatic Gain Control, and Beamforming). Channel 1 contains Raw, unprocessed mic capsule audio.
  - Keep `channels=1` in the PortAudio configuration. This automatically grabs Channel 0 (the hardware-processed stream), ensuring Whisper receives the cleanest possible audio without extra CPU downmixing.
- **Single-Line Overwrite (--clean mode)**:
  - When designing clean, real-time overwrite displays using carriage returns (`\r\033[K`), always check the terminal column width using `shutil.get_terminal_size()`. 
  - Truncate long strings on the left side to keep them shorter than the terminal width. This prevents automatic terminal wrapping, which orphans previous lines and corrupts the display.
