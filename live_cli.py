"""Continuous CLI runner for realtime speech transcription."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil

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
    parser.add_argument("--end-silence-seconds", type=float, default=3.0)
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=300.0,
        help="Maximum continuous utterance duration before forcing finalization.",
    )
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--min-threshold", type=float, default=0.015)
    parser.add_argument(
        "--disable-partials",
        action="store_true",
        help="Disable real-time partial transcripts.",
    )
    parser.add_argument(
        "--disable-transcript-eou",
        action="store_true",
        help="Disable transcript-aware End-of-Utterance detection.",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=1.0,
        help="Interval in seconds between partial transcriptions.",
    )
    parser.add_argument(
        "--eou-short-silence",
        type=float,
        default=0.5,
        help="Short silence duration in seconds for transcript-aware EOU.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print calibration, speech start/end, rejected audio, and timing diagnostics.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Enable clean mode with single-line real-time states and clean final transcripts.",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Enable local Ollama LLM integration to process final transcripts.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:1.5b",
        help="Name of the Ollama model to use.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host address.",
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
        max_utterance_seconds=args.max_utterance_seconds,
        model=args.model,
        whisper_device=args.device_type,
        compute_type=args.compute_type,
        language=args.language,
        enable_partial_transcripts=not args.disable_partials,
        enable_transcript_eou=not args.disable_transcript_eou,
        partial_interval_seconds=args.partial_interval,
        eou_short_silence_seconds=args.eou_short_silence,
    )

    if args.clean:
        print("[Loading model...] ⏳", end="", flush=True)
    elif args.verbose:
        print("Loading faster-whisper model...")

    pipeline = RealtimeSpeechPipeline(config)
    pipeline.transcriber.load()

    if args.verbose and not args.clean:
        print("Listening. Press Ctrl+C to stop.\n")

    try:
        for event in pipeline.listen():
            if args.clean:
                if event.get("type") == "final" and args.ollama:
                    result = event["result"]
                    print(f"\r\033[K[{_timestamp()}] {result.text}", flush=True)
                    print("\r\033[K🤖 Thinking... ⏳", end="", flush=True)
                    print("\r\033[K🤖 AI > ", end="", flush=True)
                    try:
                        from urllib.request import Request, urlopen
                        import urllib.error
                        
                        data = json.dumps({
                            "model": args.ollama_model,
                            "prompt": result.text,
                            "stream": True
                        }).encode("utf-8")
                        
                        req = Request(
                            f"{args.ollama_host}/api/generate",
                            data=data,
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                        
                        with urlopen(req) as response:
                            for line in response:
                                if line:
                                    chunk = json.loads(line.decode("utf-8"))
                                    print(chunk.get("response", ""), end="", flush=True)
                        print("\n", flush=True)
                    except urllib.error.URLError as e:
                        print(f"\n[Ollama Connection Error: {e.reason}. Make sure Ollama is running!]\n", flush=True)
                    except Exception as e:
                        print(f"\n[Ollama Error: {e}]\n", flush=True)
                    print("\r\033[K🎙️ Ready & Listening...", end="", flush=True)
                else:
                    _print_event_clean(event)
            else:
                if event.get("type") == "final" and args.ollama:
                    result = event["result"]
                    _print_event(event, verbose=args.verbose)
                    print("🤖 AI > ", end="", flush=True)
                    try:
                        from urllib.request import Request, urlopen
                        import urllib.error
                        
                        data = json.dumps({
                            "model": args.ollama_model,
                            "prompt": result.text,
                            "stream": True
                        }).encode("utf-8")
                        
                        req = Request(
                            f"{args.ollama_host}/api/generate",
                            data=data,
                            headers={"Content-Type": "application/json"},
                            method="POST"
                        )
                        
                        with urlopen(req) as response:
                            for line in response:
                                if line:
                                    chunk = json.loads(line.decode("utf-8"))
                                    print(chunk.get("response", ""), end="", flush=True)
                        print("\n", flush=True)
                    except urllib.error.URLError as e:
                        print(f"\n[Ollama Connection Error: {e.reason}]\n", flush=True)
                    except Exception as e:
                        print(f"\n[Ollama Error: {e}]\n", flush=True)
                else:
                    _print_event(event, verbose=args.verbose)
    except KeyboardInterrupt:
        if args.clean:
            print("\r\033[K", end="", flush=True)
        elif args.verbose:
            print("\nStopped.")
    return 0


def _format_clean_line(text: str, prefix: str) -> str:
    """Format a line to fit perfectly within the terminal width to prevent auto-wrapping."""
    try:
        columns = shutil.get_terminal_size().columns
    except Exception:
        columns = 80
    
    # Keep columns at least 40 to avoid extreme edge cases
    columns = max(40, columns)
    
    # Total available width for the text, leaving 5 characters for cursor safety padding
    avail = columns - len(prefix) - 5
    if len(text) <= avail:
        return f"{prefix}{text}"
    
    # Truncate on the left side and prepend with ellipsis
    return f"{prefix}... {text[-(avail - 4):]}"


def _print_event(event: dict, verbose: bool = False) -> None:
    event_type = event.get("type")

    if event_type == "partial":
        text = event["text"]
        line = _format_clean_line(text, "  (partial) ")
        print(f"\r\033[K{line}", end="", flush=True)
        return

    if event_type == "final":
        result = event["result"]
        print(f"\r\033[K[{_timestamp()}] {result.text}", flush=True)

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
        print(f"\r\033[K{event['message']}")
    elif event_type == "calibrated":
        print(
            f"\r\033[KCalibrated: "
            f"noise_rms={event['noise_rms']:0.5f}, "
            f"threshold={event['threshold']:0.5f}\n"
        )
    elif event_type == "speech_start":
        print("\r\033[KSpeech started...")
    elif event_type == "speech_end":
        print(f"\r\033[KSpeech ended. Utterance duration: {event['duration']:0.2f}s")
    elif event_type == "rejected":
        print("\r\033[KRejected noise/silence.\n")


_last_clean_text = ""


def _print_event_clean(event: dict) -> None:
    global _last_clean_text
    event_type = event.get("type")

    if event_type == "status":
        print("\r\033[K[Calibrating noise floor...] ⏳", end="", flush=True)
    elif event_type == "calibrated":
        print("\r\033[K🎙️ Ready & Listening...", end="", flush=True)
    elif event_type == "speech_start":
        _last_clean_text = ""
        print("\r\033[K🎙️ Speaking...", end="", flush=True)
    elif event_type == "partial":
        text = event["text"]
        _last_clean_text = text
        if text:
            line = _format_clean_line(text, "🎙️ ")
            print(f"\r\033[K{line}", end="", flush=True)
        else:
            print("\r\033[K🎙️ Speaking...", end="", flush=True)
    elif event_type == "speech_end":
        line = _format_clean_line(_last_clean_text, "⏳ ")
        print(f"\r\033[K{line}", end="", flush=True)
    elif event_type == "rejected":
        print("\r\033[K🎙️ Ready & Listening...", end="", flush=True)
    elif event_type == "final":
        result = event["result"]
        print(f"\r\033[K[{_timestamp()}] {result.text}", flush=True)
        print("\r\033[K🎙️ Ready & Listening...", end="", flush=True)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
