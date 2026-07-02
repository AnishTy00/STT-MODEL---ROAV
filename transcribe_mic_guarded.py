"""Experimental microphone transcription with stricter hallucination guards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import numpy as np

try:
    from .audio_gate import GateResult, gate_audio
    from .capture import RawMicStream
    from .devices import find_input_device, format_device_table, list_input_devices
    from .vad import EnergyVad, UtteranceSegmenter, frame_rms
except ImportError:  # Allows direct execution: python audio_in/transcribe_mic_guarded.py
    from audio_gate import GateResult, gate_audio
    from capture import RawMicStream
    from devices import find_input_device, format_device_table, list_input_devices
    from vad import EnergyVad, UtteranceSegmenter, frame_rms


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_MS = 50
DEFAULT_SECONDS = 10.0

HALLUCINATION_PHRASES = (
    "thank you",
    "thanks for watching",
    "thanks for listening",
    "subscribe",
    "subtitles by",
    "subtitle by",
    "music",
    "[music]",
    "(music)",
    "you",
)


@dataclass(frozen=True)
class GuardedTranscript:
    text: str
    accepted: bool
    reject_reason: str | None
    language: str
    audio_seconds: float
    transcribe_seconds: float
    no_speech_probability: float | None
    average_log_probability: float | None
    compression_ratio: float | None
    segment_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously listen to the microphone in real-time, segment speech utterances, "
            "gate weak inputs, and transcribe using guarded faster-whisper decode settings "
            "to reject silence hallucinations."
        )
    )
    parser.add_argument("--list", action="store_true", help="List input devices and exit.")
    parser.add_argument(
        "--device",
        help="Input device index or name substring. Defaults to ReSpeaker-like device if found.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Execution duration limit in seconds. If <= 0, runs indefinitely. Default: 0.0 (continuous).",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--block-ms", type=int, default=DEFAULT_BLOCK_MS)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device-type", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="en")

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Higher is usually more accurate and slower. Default: 5.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="Candidate count used by faster-whisper. Default: 5.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--condition-on-previous-text", action="store_true")

    parser.add_argument("--gate-threshold-multiplier", type=float, default=4.0)
    parser.add_argument("--gate-min-threshold", type=float, default=0.02)
    parser.add_argument("--gate-min-speech-seconds", type=float, default=0.45)
    parser.add_argument("--gate-min-speech-ratio", type=float, default=0.06)
    parser.add_argument("--gate-padding-seconds", type=float, default=0.25)

    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=1.0,
        help="Calibration duration for noise floor estimation. Default: 1.0.",
    )
    parser.add_argument(
        "--end-silence-seconds",
        type=float,
        default=3.0,
        help="Silence duration in seconds before finalizing an utterance. Default: 3.0.",
    )
    parser.add_argument(
        "--pre-speech-seconds",
        type=float,
        default=0.4,
        help="Audio duration in seconds to keep before speech starts. Default: 0.4.",
    )
    parser.add_argument(
        "--speech-start-seconds",
        type=float,
        default=0.15,
        help="Speech duration in seconds to trigger utterance start. Default: 0.15.",
    )
    parser.add_argument(
        "--min-utterance-seconds",
        type=float,
        default=0.4,
        help="Discard segmented utterances shorter than this. Default: 0.4.",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=300.0,
        help="Force finalize utterance if it exceeds this duration. Default: 300.0.",
    )

    parser.add_argument(
        "--reject-no-speech-probability",
        type=float,
        default=0.65,
        help="Reject text when Whisper says the audio was probably non-speech.",
    )
    parser.add_argument(
        "--reject-average-log-probability",
        type=float,
        default=-1.0,
        help="Reject low-confidence text below this average log probability.",
    )
    parser.add_argument(
        "--reject-compression-ratio",
        type=float,
        default=2.4,
        help="Reject repetitive text above this compression ratio.",
    )
    parser.add_argument(
        "--min-output-chars",
        type=int,
        default=3,
        help="Reject very short final text. Default: 3.",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Print rejected transcript text for debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(format_device_table(list_input_devices()))
        return 0

    input_device = find_input_device(args.device)
    print("Selected input device:")
    print(format_device_table([input_device]))
    print()

    print("Loading faster-whisper model...")
    model = load_model(args.model, args.device_type, args.compute_type)
    print("Model loaded successfully.")
    print()

    vad = EnergyVad(
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        calibration_seconds=args.calibration_seconds,
        threshold_multiplier=args.gate_threshold_multiplier,
        min_threshold=args.gate_min_threshold,
    )
    segmenter = UtteranceSegmenter(
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        pre_speech_seconds=args.pre_speech_seconds,
        speech_start_seconds=args.speech_start_seconds,
        end_silence_seconds=args.end_silence_seconds,
        min_utterance_seconds=args.min_utterance_seconds,
        max_utterance_seconds=args.max_utterance_seconds,
    )

    print(f"Opening microphone stream at {args.sample_rate} Hz...")
    with RawMicStream(
        device=str(input_device.index),
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
    ) as stream:
        print(f"Calibrating noise floor for {args.calibration_seconds} seconds. Please remain silent...")

        deadline = time.monotonic() + args.calibration_seconds
        calibration_rms_values = []
        while time.monotonic() < deadline:
            frame = stream.read(timeout=0.5)
            if frame is not None:
                calibration_rms_values.append(frame_rms(frame.samples))

        vad.calibrate(calibration_rms_values)
        print(f"Calibration done: noise_rms={vad.noise_rms:.5f}, threshold={vad.threshold:.5f}")
        print("Listening... Speak into the microphone. Press Ctrl+C to stop.")
        print()

        start_time = time.monotonic()
        try:
            while True:
                if args.seconds > 0.0 and (time.monotonic() - start_time) >= args.seconds:
                    print(f"\nReached execution limit of {args.seconds} seconds.")
                    break

                frame = stream.read(timeout=0.5)
                if frame is None:
                    continue

                decision = vad.decide(frame.samples)
                was_active = segmenter.active
                utterance = segmenter.push(frame.samples, decision.is_speech)

                if not was_active and segmenter.active:
                    print("\n[Speech started...]")

                if utterance is not None:
                    print(f"\n[Speech ended. Utterance duration: {len(utterance)/args.sample_rate:.2f}s. Processing...]")

                    # Gate the utterance
                    gate = gate_audio(
                        utterance,
                        sample_rate=args.sample_rate,
                        frame_ms=args.block_ms,
                        calibration_seconds=0.0,  # Already calibrated noise floor
                        threshold_multiplier=args.gate_threshold_multiplier,
                        min_threshold=args.gate_min_threshold,
                        min_speech_seconds=args.gate_min_speech_seconds,
                        min_speech_ratio=args.gate_min_speech_ratio,
                        padding_seconds=args.gate_padding_seconds,
                    )

                    if args.show_rejected or gate.should_transcribe:
                        print_gate_result(gate, args.sample_rate)

                    if not gate.should_transcribe:
                        print("Transcript rejected before Whisper (audio gate).")
                        print()
                        continue

                    # Transcribe with guarded decoding parameters
                    transcript = transcribe_guarded(model, gate.samples, args)
                    print_transcript(transcript, show_rejected=args.show_rejected)
                    print()
        except KeyboardInterrupt:
            print("\nReal-time transcription stopped by user.")

    return 0


def load_model(model_size: str, device: str, compute_type: str):
    if device == "cuda" and not _cuda_available():
        raise RuntimeError("CUDA was requested for faster-whisper, but no CUDA GPU is available.")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("The 'faster-whisper' package is required for transcription.") from exc

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_guarded(model, samples: np.ndarray, args: argparse.Namespace) -> GuardedTranscript:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    audio_seconds = len(mono) / args.sample_rate if args.sample_rate else 0.0

    start = time.monotonic()
    segments, info = model.transcribe(
        mono,
        language=args.language,
        task="transcribe",
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        condition_on_previous_text=args.condition_on_previous_text,
        compression_ratio_threshold=args.reject_compression_ratio,
        log_prob_threshold=args.reject_average_log_probability,
        no_speech_threshold=args.reject_no_speech_probability,
        word_timestamps=False,
        vad_filter=False,
    )
    segment_list = list(segments)
    transcribe_seconds = time.monotonic() - start

    text = " ".join(segment.text.strip() for segment in segment_list).strip()
    no_speech_probability = _max_attr(segment_list, "no_speech_prob")
    average_log_probability = _mean_attr(segment_list, "avg_logprob")
    compression_ratio = _max_attr(segment_list, "compression_ratio")
    reject_reason = rejection_reason(
        text=text,
        no_speech_probability=no_speech_probability,
        average_log_probability=average_log_probability,
        compression_ratio=compression_ratio,
        args=args,
    )

    return GuardedTranscript(
        text=text,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        language=info.language,
        audio_seconds=audio_seconds,
        transcribe_seconds=transcribe_seconds,
        no_speech_probability=no_speech_probability,
        average_log_probability=average_log_probability,
        compression_ratio=compression_ratio,
        segment_count=len(segment_list),
    )


def rejection_reason(
    text: str,
    no_speech_probability: float | None,
    average_log_probability: float | None,
    compression_ratio: float | None,
    args: argparse.Namespace,
) -> str | None:
    normalized = " ".join(text.lower().strip(" .,!?:;[]()").split())

    if len(text.strip()) < args.min_output_chars:
        return "too little transcript text"

    if normalized in HALLUCINATION_PHRASES:
        return "known silence/noise hallucination phrase"

    if no_speech_probability is not None:
        if no_speech_probability >= args.reject_no_speech_probability:
            return "high no_speech_probability"

    if average_log_probability is not None:
        if average_log_probability <= args.reject_average_log_probability:
            return "low average log probability"

    if compression_ratio is not None:
        if compression_ratio >= args.reject_compression_ratio:
            return "high compression ratio"

    return None


def record_audio(
    device_selector: str,
    sample_rate: int,
    block_ms: int,
    seconds: float,
) -> np.ndarray:
    print(
        f"Recording raw {sample_rate} Hz mono float32 audio for {seconds:0.1f}s. "
        "Try silence, noise, and normal speech as separate test runs."
    )
    print()

    start = time.monotonic()
    chunks: list[np.ndarray] = []

    with RawMicStream(device=device_selector, sample_rate=sample_rate, block_ms=block_ms) as stream:
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


def print_gate_result(gate: GateResult, sample_rate: int) -> None:
    duration = len(gate.samples) / sample_rate if sample_rate else 0.0
    print()
    print("Gate:")
    print(f"  decision={gate.should_transcribe}")
    print(f"  reason={gate.reason}")
    print(f"  noise_rms={gate.noise_rms:0.5f}")
    print(f"  threshold={gate.threshold:0.5f}")
    print(f"  speech_seconds={gate.speech_seconds:0.2f}")
    print(f"  speech_ratio={gate.speech_ratio:0.2f}")
    print(f"  trimmed_audio_seconds={duration:0.2f}")
    print()


def print_transcript(transcript: GuardedTranscript, show_rejected: bool) -> None:
    print()
    print(f"Language: {transcript.language}")
    print(f"Segments: {transcript.segment_count}")
    print(f"Audio duration: {transcript.audio_seconds:0.2f}s")
    print(f"Transcription time: {transcript.transcribe_seconds:0.2f}s")
    print(
        "Realtime factor: "
        f"{transcript.transcribe_seconds / max(transcript.audio_seconds, 0.001):0.2f}x"
    )
    print(f"no_speech_probability: {_format_optional(transcript.no_speech_probability)}")
    print(f"avg_logprob: {_format_optional(transcript.average_log_probability)}")
    print(f"compression_ratio: {_format_optional(transcript.compression_ratio)}")
    print()

    if transcript.accepted:
        print(f"Accepted transcript: {transcript.text}")
        return

    print(f"Rejected transcript: {transcript.reject_reason}")
    if show_rejected:
        print(f"Rejected text: {transcript.text}")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    return bool(torch.cuda.is_available())


def _mean_attr(segments: list[object], attr: str) -> float | None:
    values = [
        float(value)
        for segment in segments
        if (value := getattr(segment, attr, None)) is not None
    ]
    return sum(values) / len(values) if values else None


def _max_attr(segments: list[object], attr: str) -> float | None:
    values = [
        float(value)
        for segment in segments
        if (value := getattr(segment, attr, None)) is not None
    ]
    return max(values) if values else None


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:0.2f}"


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
