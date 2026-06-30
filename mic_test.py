"""CLI utility for validating live microphone input."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

try:
    from .capture import RawMicStream
    from .devices import find_input_device, format_device_table, list_input_devices
except ImportError:  # Allows direct execution: python audio_in/mic_test.py
    from capture import RawMicStream
    from devices import find_input_device, format_device_table, list_input_devices


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_MS = 50
DEFAULT_SECONDS = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and test audio input devices for the audio_in package."
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
        help=f"How long to record before playback. Default: {DEFAULT_SECONDS}.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(format_device_table(list_input_devices()))
        return 0

    device = find_input_device(args.device)
    print("Selected input device:")
    print(format_device_table([device]))
    print()

    run_level_test(
        device_index=device.index,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        seconds=args.seconds,
    )
    return 0


def run_level_test(
    device_index: int,
    sample_rate: int,
    block_ms: int,
    seconds: float,
) -> None:
    if seconds <= 0:
        raise ValueError("Playback test needs a positive --seconds value.")

    print(
        f"Testing raw {sample_rate} Hz mono float32 input in {block_ms} ms blocks. "
        "Speak into the mic; captured audio will play back afterward."
    )
    print("Press Ctrl+C to stop before playback.\n")

    start = time.monotonic()
    chunks: list[np.ndarray] = []

    with RawMicStream(
        device=str(device_index),
        sample_rate=sample_rate,
        block_ms=block_ms,
    ) as stream:
        try:
            while time.monotonic() - start < seconds:
                frame = stream.read(timeout=0.5)
                if frame is None:
                    _print_meter(
                        elapsed=time.monotonic() - start,
                        rms=0.0,
                        peak=0.0,
                        frames=stream.frames_seen,
                    )
                    continue

                samples = frame.samples
                chunks.append(samples.copy())
                rms = float(math.sqrt(float(np.mean(samples * samples)))) if len(samples) else 0.0
                peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
                _print_meter(
                    elapsed=time.monotonic() - start,
                    rms=rms,
                    peak=peak,
                    frames=frame.frames_seen,
                )
        except KeyboardInterrupt:
            print("\nMic test stopped before playback.")
            return

    print("\nRecording complete.")

    if not chunks:
        print("No audio frames were captured.")
        return

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    _playback(audio, sample_rate)


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


def _playback(audio: np.ndarray, sample_rate: int) -> None:
    try:
        from .devices import require_sounddevice
    except ImportError:
        from devices import require_sounddevice

    sounddevice = require_sounddevice()
    duration = len(audio) / sample_rate if sample_rate else 0.0
    print(f"Playing back {duration:0.1f} seconds of captured audio...")
    sounddevice.play(audio, samplerate=sample_rate)
    sounddevice.wait()
    print("Playback complete.")


if __name__ == "__main__":
    raise SystemExit(main())
