"""Energy-based audio gating before transcription."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GateResult:
    samples: np.ndarray
    should_transcribe: bool
    reason: str
    noise_rms: float
    threshold: float
    speech_seconds: float
    speech_ratio: float


def gate_audio(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 50,
    calibration_seconds: float = 1.0,
    threshold_multiplier: float = 3.0,
    min_threshold: float = 0.015,
    min_speech_seconds: float = 0.35,
    min_speech_ratio: float = 0.04,
    padding_seconds: float = 0.25,
) -> GateResult:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(mono) == 0:
        return _result(mono, False, "no audio captured", 0.0, min_threshold, 0.0, 0.0)

    frame_size = max(1, int(sample_rate * (frame_ms / 1000.0)))
    frame_rms = _frame_rms(mono, frame_size)
    if len(frame_rms) == 0:
        return _result(mono, False, "audio too short", 0.0, min_threshold, 0.0, 0.0)

    calibration_frames = max(1, int(calibration_seconds * 1000 / frame_ms))
    noise_frames = frame_rms[: min(calibration_frames, len(frame_rms))]
    noise_rms = float(np.percentile(noise_frames, 80))
    threshold = max(min_threshold, noise_rms * threshold_multiplier)

    speech_mask = frame_rms >= threshold
    speech_frames = int(np.count_nonzero(speech_mask))
    speech_seconds = speech_frames * frame_ms / 1000.0
    speech_ratio = speech_frames / len(frame_rms)

    if speech_seconds < min_speech_seconds:
        return _result(
            mono,
            False,
            "not enough speech-like audio",
            noise_rms,
            threshold,
            speech_seconds,
            speech_ratio,
        )

    if speech_ratio < min_speech_ratio:
        return _result(
            mono,
            False,
            "speech-like audio ratio too low",
            noise_rms,
            threshold,
            speech_seconds,
            speech_ratio,
        )

    speech_indices = np.flatnonzero(speech_mask)
    first_frame = int(speech_indices[0])
    last_frame = int(speech_indices[-1])
    padding_frames = max(1, int(padding_seconds * 1000 / frame_ms))

    start_frame = max(0, first_frame - padding_frames)
    end_frame = min(len(frame_rms), last_frame + padding_frames + 1)
    start_sample = start_frame * frame_size
    end_sample = min(len(mono), end_frame * frame_size)

    return _result(
        mono[start_sample:end_sample],
        True,
        "speech-like audio detected",
        noise_rms,
        threshold,
        speech_seconds,
        speech_ratio,
    )


def _frame_rms(samples: np.ndarray, frame_size: int) -> np.ndarray:
    frame_count = int(math.ceil(len(samples) / frame_size))
    values = np.empty(frame_count, dtype=np.float32)

    for index in range(frame_count):
        start = index * frame_size
        end = min(len(samples), start + frame_size)
        frame = samples[start:end]
        values[index] = math.sqrt(float(np.mean(frame * frame))) if len(frame) else 0.0

    return values


def _result(
    samples: np.ndarray,
    should_transcribe: bool,
    reason: str,
    noise_rms: float,
    threshold: float,
    speech_seconds: float,
    speech_ratio: float,
) -> GateResult:
    return GateResult(
        samples=samples,
        should_transcribe=should_transcribe,
        reason=reason,
        noise_rms=noise_rms,
        threshold=threshold,
        speech_seconds=speech_seconds,
        speech_ratio=speech_ratio,
    )
