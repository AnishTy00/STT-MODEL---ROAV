"""Frame-level speech detection for realtime utterance segmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VadDecision:
    rms: float
    is_speech: bool
    threshold: float


class EnergyVad:
    """Simple noise-floor VAD for the first realtime pipeline."""

    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        calibration_seconds: float = 1.0,
        threshold_multiplier: float = 3.0,
        min_threshold: float = 0.015,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.calibration_seconds = calibration_seconds
        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold
        self.noise_rms = 0.0
        self.threshold = min_threshold

    def calibrate(self, rms_values: list[float]) -> None:
        if not rms_values:
            self.noise_rms = 0.0
            self.threshold = self.min_threshold
            return

        self.noise_rms = float(np.percentile(np.asarray(rms_values), 80))
        self.threshold = max(self.min_threshold, self.noise_rms * self.threshold_multiplier)

    def decide(self, samples: np.ndarray) -> VadDecision:
        rms = frame_rms(samples)
        return VadDecision(
            rms=rms,
            is_speech=rms >= self.threshold,
            threshold=self.threshold,
        )


class UtteranceSegmenter:
    """Turns frame-level speech decisions into complete utterance buffers."""

    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        pre_speech_seconds: float = 0.4,
        speech_start_seconds: float = 0.15,
        end_silence_seconds: float = 0.9,
        min_utterance_seconds: float = 0.4,
        max_utterance_seconds: float = 20.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.pre_speech_frames = max(1, _seconds_to_frames(pre_speech_seconds, block_ms))
        self.speech_start_frames = max(1, _seconds_to_frames(speech_start_seconds, block_ms))
        self.end_silence_frames = max(1, _seconds_to_frames(end_silence_seconds, block_ms))
        self.min_utterance_frames = max(1, _seconds_to_frames(min_utterance_seconds, block_ms))
        self.max_utterance_frames = max(1, _seconds_to_frames(max_utterance_seconds, block_ms))
        self._pre_speech: deque[np.ndarray] = deque(maxlen=self.pre_speech_frames)
        self._utterance: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def push(self, samples: np.ndarray, is_speech: bool) -> np.ndarray | None:
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)

        if not self._active:
            self._pre_speech.append(chunk.copy())
            self._speech_run = self._speech_run + 1 if is_speech else 0

            if self._speech_run >= self.speech_start_frames:
                self._active = True
                self._utterance = [frame.copy() for frame in self._pre_speech]
                self._silence_run = 0
            return None

        self._utterance.append(chunk.copy())

        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        utterance_too_long = len(self._utterance) >= self.max_utterance_frames
        silence_finished = self._silence_run >= self.end_silence_frames

        if utterance_too_long or silence_finished:
            return self._finish()

        return None

    def _finish(self) -> np.ndarray | None:
        utterance = self._utterance
        self._pre_speech.clear()
        self._utterance = []
        self._speech_run = 0
        self._silence_run = 0
        self._active = False

        if len(utterance) < self.min_utterance_frames:
            return None

        return np.concatenate(utterance).astype(np.float32, copy=False)


def frame_rms(samples: np.ndarray) -> float:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    return math.sqrt(float(np.mean(mono * mono))) if len(mono) else 0.0


def _seconds_to_frames(seconds: float, block_ms: int) -> int:
    return int(math.ceil(seconds * 1000 / block_ms))
