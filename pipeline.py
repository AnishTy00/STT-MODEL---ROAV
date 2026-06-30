"""Realtime raw-audio-to-transcript pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    from .audio_gate import GateResult, gate_audio
    from .capture import RawMicStream
    from .transcriber import Transcript, WhisperTranscriber
    from .vad import EnergyVad, UtteranceSegmenter, frame_rms
except ImportError:  # Allows direct execution through scripts importing pipeline.py
    from audio_gate import GateResult, gate_audio
    from capture import RawMicStream
    from transcriber import Transcript, WhisperTranscriber
    from vad import EnergyVad, UtteranceSegmenter, frame_rms


@dataclass(frozen=True)
class PipelineConfig:
    device: str | None = None
    sample_rate: int = 16000
    block_ms: int = 50
    calibration_seconds: float = 1.0
    vad_threshold_multiplier: float = 3.0
    vad_min_threshold: float = 0.015
    pre_speech_seconds: float = 0.4
    speech_start_seconds: float = 0.15
    end_silence_seconds: float = 0.9
    min_utterance_seconds: float = 0.4
    max_utterance_seconds: float = 20.0
    model: str = "small"
    whisper_device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"


@dataclass(frozen=True)
class FinalTranscript:
    text: str
    transcript: Transcript
    gate: GateResult
    utterance_seconds: float


class RealtimeSpeechPipeline:
    """Continuously listens, segments utterances, and transcribes final audio."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.transcriber = WhisperTranscriber(
            model_size=config.model,
            device=config.whisper_device,
            compute_type=config.compute_type,
            language=config.language,
        )

    def listen(self):
        config = self.config
        vad = EnergyVad(
            sample_rate=config.sample_rate,
            block_ms=config.block_ms,
            calibration_seconds=config.calibration_seconds,
            threshold_multiplier=config.vad_threshold_multiplier,
            min_threshold=config.vad_min_threshold,
        )
        segmenter = UtteranceSegmenter(
            sample_rate=config.sample_rate,
            block_ms=config.block_ms,
            pre_speech_seconds=config.pre_speech_seconds,
            speech_start_seconds=config.speech_start_seconds,
            end_silence_seconds=config.end_silence_seconds,
            min_utterance_seconds=config.min_utterance_seconds,
            max_utterance_seconds=config.max_utterance_seconds,
        )

        with RawMicStream(
            device=config.device,
            sample_rate=config.sample_rate,
            block_ms=config.block_ms,
        ) as stream:
            yield {
                "type": "status",
                "message": f"Selected input device: {stream.device.name}",
            }

            rms_values = self._collect_calibration_rms(stream)
            vad.calibrate(rms_values)
            yield {
                "type": "calibrated",
                "noise_rms": vad.noise_rms,
                "threshold": vad.threshold,
            }

            while True:
                frame = stream.read(timeout=0.5)
                if frame is None:
                    continue

                decision = vad.decide(frame.samples)
                was_active = segmenter.active
                utterance = segmenter.push(frame.samples, decision.is_speech)
                if not was_active and segmenter.active:
                    yield {"type": "speech_start", "rms": decision.rms}

                if utterance is None:
                    continue

                yield {
                    "type": "speech_end",
                    "duration": len(utterance) / config.sample_rate,
                }

                result = self._transcribe_utterance(utterance)
                if result is None:
                    yield {"type": "rejected"}
                    continue

                yield {
                    "type": "final",
                    "result": result,
                }

    def _collect_calibration_rms(self, stream: RawMicStream) -> list[float]:
        deadline = time.monotonic() + self.config.calibration_seconds
        values: list[float] = []
        while time.monotonic() < deadline:
            frame = stream.read(timeout=0.5)
            if frame is not None:
                values.append(frame_rms(frame.samples))
        return values

    def _transcribe_utterance(self, utterance: np.ndarray) -> FinalTranscript | None:
        gate = gate_audio(
            utterance,
            sample_rate=self.config.sample_rate,
            frame_ms=self.config.block_ms,
            calibration_seconds=0.0,
        )
        if not gate.should_transcribe:
            return None

        transcript = self.transcriber.transcribe(gate.samples, self.config.sample_rate)
        text = transcript.text.strip()
        if not text:
            return None

        return FinalTranscript(
            text=text,
            transcript=transcript,
            gate=gate,
            utterance_seconds=len(utterance) / self.config.sample_rate,
        )
