"""Record from the microphone and transcribe with faster-whisper."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

try:
    from .capture import RawMicStream
    from .devices import find_input_device, format_device_table, list_input_devices
    from .transcriber import WhisperTranscriber
except ImportError:  # Allows direct execution: python audio_in/transcribe_mic.py
    from capture import RawMicStream
    from devices import find_input_device, format_device_table, list_input_devices
    from transcriber import WhisperTranscriber


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_MS = 50
DEFAULT_SECONDS = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record microphone audio and transcribe it with faster-whisper."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available input devices and exit.",
    )
    parser.add_argument(
        "--device",
        help="Input device index or name substring. Defaults to ReSpeaker-like device if found.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"How long to record before transcription. Default: {DEFAULT_SECONDS}.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Requested input sample rate. Default: {DEFAULT_SAMPLE_RATE}.",
    )
    parser.add_argument(
        "--block-ms",
        type=int,
        default=DEFAULT_BLOCK_MS,
        help=f"Input callback block size in milliseconds. Default: {DEFAULT_BLOCK_MS}.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size/name. Default: small.",
    )
    parser.add_argument(
        "--device-type",
        default="cuda",
        help="faster-whisper device. Default: cuda.",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="faster-whisper compute type. Default: float16.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Transcription language. Default: en.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(format_device_table(list_input_devices()))
        return 0

    if args.seconds <= 0:
        raise ValueError("Transcription test needs a positive --seconds value.")

    input_device = find_input_device(args.device)
    print("Selected input device:")
    print(format_device_table([input_device]))
    print()

    audio = record_audio(
        device_selector=str(input_device.index),
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        seconds=args.seconds,
    )

    print("Loading faster-whisper model...")
    transcriber = WhisperTranscriber(
        model_size=args.model,
        device=args.device_type,
        compute_type=args.compute_type,
        language=args.language,
    )
    transcriber.load()

    print("Transcribing...")
    transcript = transcriber.transcribe(audio, sample_rate=args.sample_rate)

    print()
    print(f"Language: {transcript.language}")
    print(f"Audio duration: {transcript.duration_seconds:0.2f}s")
    print(f"Transcription time: {transcript.transcribe_seconds:0.2f}s")
    print(f"Realtime factor: {transcript.transcribe_seconds / max(transcript.duration_seconds, 0.001):0.2f}x")
    print()
    print(f"Transcript: {transcript.text}")
    return 0


def record_audio(
    device_selector: str,
    sample_rate: int,
    block_ms: int,
    seconds: float,
) -> np.ndarray:
    print(
        f"Recording raw {sample_rate} Hz mono float32 audio for {seconds:0.1f}s. "
        "Speak after recording starts."
    )
    print()

    start = time.monotonic()
    chunks: list[np.ndarray] = []

    with RawMicStream(
        device=device_selector,
        sample_rate=sample_rate,
        block_ms=block_ms,
    ) as stream:
        try:
            while time.monotonic() - start < seconds:
                frame = stream.read(timeout=0.5)
                if frame is None:
                    _print_meter(time.monotonic() - start, 0.0, 0.0, stream.frames_seen)
                    continue

                chunks.append(frame.samples.copy())
                rms = _rms(frame.samples)
                peak = float(np.max(np.abs(frame.samples))) if len(frame.samples) else 0.0
                _print_meter(time.monotonic() - start, rms, peak, frame.frames_seen)
        except KeyboardInterrupt:
            print("\nRecording stopped.")

    print("\nRecording complete.")
    if not chunks:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32, copy=False)


def _rms(samples: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(samples * samples)))) if len(samples) else 0.0


def _print_meter(elapsed: float, rms: float, peak: float, frames: int) -> None:
    bar_width = 40
    level = min(1.0, peak * 8.0)
    filled = int(level * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\r{elapsed:6.1f}s  rms={rms:0.5f}  peak={peak:0.5f}  "
        f"frames={frames:<9} [{bar}]",
        end="",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
