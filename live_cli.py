"""Continuous CLI runner for realtime speech transcription."""

from __future__ import annotations

import argparse
from datetime import datetime

try:
    from .devices import format_device_table, list_input_devices
    from .pipeline import PipelineConfig, RealtimeSpeechPipeline
except ImportError:  # Allows direct execution: python audio_in/live_cli.py
    from devices import format_device_table, list_input_devices
    from pipeline import PipelineConfig, RealtimeSpeechPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously listen and print transcripts after speech ends."
    )
    parser.add_argument("--list", action="store_true", help="List input devices and exit.")
    parser.add_argument("--device", help="Input device index or name substring.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--block-ms", type=int, default=50)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device-type", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--calibration-seconds", type=float, default=1.0)
    parser.add_argument("--end-silence-seconds", type=float, default=0.9)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--min-threshold", type=float, default=0.015)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print calibration, speech start/end, rejected audio, and timing diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(format_device_table(list_input_devices()))
        return 0

    config = PipelineConfig(
        device=args.device,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        calibration_seconds=args.calibration_seconds,
        vad_threshold_multiplier=args.threshold_multiplier,
        vad_min_threshold=args.min_threshold,
        end_silence_seconds=args.end_silence_seconds,
        model=args.model,
        whisper_device=args.device_type,
        compute_type=args.compute_type,
        language=args.language,
    )

    if args.verbose:
        print("Loading faster-whisper model...")
    pipeline = RealtimeSpeechPipeline(config)
    pipeline.transcriber.load()
    if args.verbose:
        print("Listening. Press Ctrl+C to stop.\n")

    try:
        for event in pipeline.listen():
            _print_event(event, verbose=args.verbose)
    except KeyboardInterrupt:
        if args.verbose:
            print("\nStopped.")
    return 0


def _print_event(event: dict, verbose: bool = False) -> None:
    event_type = event.get("type")

    if event_type == "final":
        result = event["result"]
        print(f"[{_timestamp()}] {result.text}", flush=True)

        if verbose:
            transcript = result.transcript
            print(
                f"  audio={transcript.duration_seconds:0.2f}s, "
                f"transcribe={transcript.transcribe_seconds:0.2f}s, "
                f"rtf={transcript.transcribe_seconds / max(transcript.duration_seconds, 0.001):0.2f}x"
            )
            if transcript.no_speech_probability is not None:
                print(f"  no_speech_probability={transcript.no_speech_probability:0.2f}")
            print()
        return

    if not verbose:
        return

    if event_type == "status":
        print(event["message"])
    elif event_type == "calibrated":
        print(
            "Calibrated: "
            f"noise_rms={event['noise_rms']:0.5f}, "
            f"threshold={event['threshold']:0.5f}\n"
        )
    elif event_type == "speech_start":
        print("Speech started...")
    elif event_type == "speech_end":
        print(f"Speech ended. Utterance duration: {event['duration']:0.2f}s")
    elif event_type == "rejected":
        print("Rejected noise/silence.\n")


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
