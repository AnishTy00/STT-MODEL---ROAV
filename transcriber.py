"""GPU-backed Whisper transcription."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration_seconds: float
    transcribe_seconds: float
    no_speech_probability: float | None


class WhisperTranscriber:
    """Small wrapper around faster-whisper for raw mono float32 audio."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        if self.device == "cuda" and not _cuda_available():
            raise RuntimeError(
                "CUDA was requested for faster-whisper, but no CUDA GPU is available."
            )

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'faster-whisper' package is required for transcription."
            ) from exc

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> Transcript:
        self.load()

        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        duration_seconds = len(mono) / sample_rate if sample_rate else 0.0

        start = time.monotonic()
        segments, info = self._model.transcribe(
            mono,
            language=self.language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.6,
            word_timestamps=False,
            vad_filter=False,
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        transcribe_seconds = time.monotonic() - start
        no_speech_probability = _max_no_speech_probability(segment_list)

        return Transcript(
            text=text,
            language=info.language,
            duration_seconds=duration_seconds,
            transcribe_seconds=transcribe_seconds,
            no_speech_probability=no_speech_probability,
        )


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    return bool(torch.cuda.is_available())


def _max_no_speech_probability(segments) -> float | None:
    values = [
        float(segment.no_speech_prob)
        for segment in segments
        if getattr(segment, "no_speech_prob", None) is not None
    ]
    return max(values) if values else None
