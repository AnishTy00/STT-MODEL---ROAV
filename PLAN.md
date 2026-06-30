# Audio-In Package Plan

## Goal

Build a single Python package named `audio_in` whose job is simple and strict: raw microphone audio in, final transcript text out.

For now, do not build the LLM integration. The first usable CLI should listen to the ReSpeaker Lite, detect when the user has finished speaking, and print the transcript immediately after end-of-utterance is detected.

The implementation must stay operating-system agnostic. Use cross-platform audio capture abstractions only, and do not depend on Windows-specific device IDs, host APIs, paths, or naming behavior. The deployment target is Linux on NVIDIA Jetson Orin Nano with JetPack 6.2.2.

## Target Architecture

```text
ReSpeaker Lite mic
  -> audio capture
  -> normalize to 16 kHz mono PCM
  -> speech detector / VAD
  -> utterance buffer
  -> faster-whisper small on GPU
  -> end detection
  -> print final transcript in CLI
```

## Current Principle

Do not copy architecture from the older exploratory scripts outside this folder. Build this package from first principles and only add complexity when a test shows it is needed.

## Package Shape

- `audio_in/__init__.py`: public package marker/version.
- `audio_in/devices.py`: list/select input devices, especially ReSpeaker Lite.
- `audio_in/capture.py`: raw microphone stream API.
- `audio_in/mic_test.py`: CLI for validating mic selection and input levels.
- `audio_in/audio_gate.py`: raw-sample energy gate to avoid transcribing noise.
- `audio_in/transcriber.py`: GPU `faster-whisper` wrapper.
- `audio_in/transcribe_mic.py`: fixed-duration mic recording and transcription test.
- `audio_in/vad.py`: later speech detection.
- `audio_in/pipeline.py`: later raw-audio-to-transcript orchestration.
- `audio_in/live_cli.py`: later final CLI that prints completed transcripts.

## Step 1: Test The Microphone

Goal: prove the ReSpeaker Lite is visible and that live samples are arriving.

Implement:

- device listing
- selection by index or name substring
- automatic preference for names containing `respeaker`
- short live level meter using raw 16 kHz mono `float32` frames
- RMS and peak display
- clear errors when `sounddevice` or the mic device is unavailable

Success looks like:

- `python -m audio_in.mic_test --list` shows the ReSpeaker Lite.
- `python -m audio_in.mic_test` shows changing RMS/peak levels while speaking.
- No hard-coded device ID is required.

## Step 2: Capture API

After the mic test works, harden the reusable capture layer:

- callback-based input stream
- internal 16 kHz mono `float32`
- frame size around 20-50 ms
- queue or generator API
- clean shutdown on Ctrl+C

## Step 3: Speech Detection

Start with a simple baseline, then upgrade:

1. RMS/noise-floor calibration.
2. WebRTC VAD or Silero VAD if latency/noise handling needs improvement.
3. Optional ReSpeaker multi-channel processing later if needed.

Initial rules:

- Calibrate ambient noise for 1-2 seconds.
- Start utterance after consecutive speech frames.
- End candidate after silence reaches about 0.8-1.5 seconds.
- Keep max utterance length, around 20-30 seconds.
- Drop very short utterances unless the text is clearly meaningful.

## Step 4: GPU Transcription

Wrap `faster-whisper` in a reusable transcriber:

- Load `WhisperModel("small", device="cuda", compute_type=...)`.
- Fail loudly if CUDA is expected but unavailable.
- Log model load time and memory usage.
- Use `beam_size=1`, `best_of=1`, `temperature=0.0`, `condition_on_previous_text=False` for low-latency conversational turns.
- Start with `word_timestamps=False`; add timestamps only if needed for UX or debugging.

Benchmark compute types:

- Desktop GPU: `float16`, `int8_float16`, `int8`.
- Jetson Orin Nano: `int8`, `int8_float16`, `float16`.

Measure:

- realtime factor
- time to first partial
- finalization delay after user stops
- GPU memory
- CPU/RAM usage

First test command:

```text
python -m audio_in.transcribe_mic --device respeaker --seconds 10
```

## Step 5: End Detection And CLI Output

First working behavior:

- listen continuously
- buffer one utterance
- transcribe after silence/end is detected
- print only the final transcript

Later, add partial hypotheses if they help the user experience.

## Step 6: Better End-of-Utterance Detection

Use layered EOU logic:

- silence duration
- transcript punctuation
- question/command patterns
- continuation phrases like "because", "and then", "I want to"
- maximum silence timeout

Start simple: silence after speech means final. Add transcript-aware EOU only after the basic loop works.

## Step 7: Hallucination And Noise Filtering

Prevent common Whisper filler from reaching the LLM:

- Drop empty transcripts.
- Drop very short low-energy utterances.
- Drop known hallucinations like "thank you", "thanks for watching", "music", "applause" when audio evidence is weak.
- Track speech duration and average energy with each transcript.

## Jetson Porting Plan

For Jetson Orin Nano with JetPack 6.2.2:

- Confirm Python version and CUDA stack from JetPack.
- Install a compatible PyTorch/CTranslate2/faster-whisper path.
- Verify `nvidia-smi` may not be available on Jetson; use Jetson-specific tools like `tegrastats`.
- Benchmark CPU vs GPU fallback but keep GPU as the intended path.
- Test ReSpeaker Lite enumeration under ALSA/PulseAudio/PipeWire as available.
- Keep dependency versions pinned after the first successful Jetson run.

Jetson acceptance checks:

- model loads with CUDA
- live mic capture works with ReSpeaker Lite
- partial transcript latency is acceptable
- final transcript emits after natural pauses
- memory usage stays stable over a 10-20 minute session

## Milestones

1. List and select ReSpeaker Lite reliably.
2. Live capture stream prints RMS/VAD state without Whisper.
3. GPU Whisper wrapper transcribes a WAV file.
4. Live utterance segmentation produces clean audio chunks.
5. Final transcripts work from live speech.
6. EOU logic improves conversational turn-taking.
7. Desktop benchmark report is captured.
8. Jetson install notes and benchmark report are captured.

## First Implementation Step

Start with `devices.py` and `mic_test.py`.
