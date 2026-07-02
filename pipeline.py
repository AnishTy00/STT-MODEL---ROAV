"""Realtime raw-audio-to-transcript pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    from .audio_gate import GateResult, gate_audio
    from .capture import RawMicStream
    from .eou import evaluate_eou, is_hallucination
    from .transcriber import Transcript, WhisperTranscriber
    from .vad import EnergyVad, UtteranceSegmenter, frame_rms
except ImportError:  # Allows direct execution through scripts importing pipeline.py
    from audio_gate import GateResult, gate_audio
    from capture import RawMicStream
    from eou import evaluate_eou, is_hallucination
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
    end_silence_seconds: float = 3.0
    min_utterance_seconds: float = 0.4
    max_utterance_seconds: float = 300.0
    model: str = "small"
    whisper_device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    enable_partial_transcripts: bool = True
    enable_transcript_eou: bool = True
    partial_interval_seconds: float = 1.0
    eou_short_silence_seconds: float = 0.5


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

            last_partial_transcribe_seconds = 0.0
            checked_eou_for_current_pause = False
            last_partial_text = ""

            while True:
                frame = stream.read(timeout=0.5)
                if frame is None:
                    continue

                decision = vad.decide(frame.samples)
                was_active = segmenter.active
                utterance = segmenter.push(frame.samples, decision.is_speech)
                if not was_active and segmenter.active:
                    yield {"type": "speech_start", "rms": decision.rms}
                    last_partial_transcribe_seconds = 0.0
                    checked_eou_for_current_pause = False
                    last_partial_text = ""

                if decision.is_speech:
                    checked_eou_for_current_pause = False

                if segmenter.active and utterance is None:
                    current_dur = segmenter.current_duration_seconds
                    silence_sec = segmenter.silence_seconds

                    should_partial = (
                        config.enable_partial_transcripts
                        and (current_dur - last_partial_transcribe_seconds >= config.partial_interval_seconds)
                    )

                    should_eou_check = (
                        config.enable_transcript_eou
                        and (silence_sec >= config.eou_short_silence_seconds)
                        and not checked_eou_for_current_pause
                    )

                    if should_partial or should_eou_check:
                        partial_audio = segmenter.get_current_audio()
                        if len(partial_audio) > 0:
                            # Apply audio gate to trim trailing silence and validate speech energy
                            gate = gate_audio(
                                partial_audio,
                                sample_rate=config.sample_rate,
                                frame_ms=config.block_ms,
                                calibration_seconds=-1.0,
                                min_threshold=vad.threshold,
                            )
                            
                            text = ""
                            if gate.should_transcribe:
                                transcript = self.transcriber.transcribe(gate.samples, config.sample_rate)
                                cand_text = transcript.text.strip()
                                # Filter out known hallucination phrases
                                if not is_hallucination(cand_text):
                                    text = cand_text

                            last_partial_transcribe_seconds = current_dur
                            if silence_sec >= config.eou_short_silence_seconds:
                                checked_eou_for_current_pause = True

                            # Yield event only if text changed from last partial (prevent redundant spam/flicker)
                            if text != last_partial_text:
                                yield {
                                    "type": "partial",
                                    "text": text,
                                    "duration": current_dur,
                                    "silence_seconds": silence_sec,
                                }
                                last_partial_text = text

                            if should_eou_check and evaluate_eou(
                                text=text,
                                silence_seconds=silence_sec,
                                eou_short_silence_seconds=config.eou_short_silence_seconds,
                                end_silence_seconds=config.end_silence_seconds,
                                enable_transcript_eou=config.enable_transcript_eou,
                            ):
                                utterance = segmenter.force_finish()

                if utterance is None:
                    continue

                yield {
                    "type": "speech_end",
                    "duration": len(utterance) / config.sample_rate,
                }

                result = self._transcribe_utterance(utterance, vad.threshold)
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

    def _transcribe_utterance(self, utterance: np.ndarray, threshold: float) -> FinalTranscript | None:
        gate = gate_audio(
            utterance,
            sample_rate=self.config.sample_rate,
            frame_ms=self.config.block_ms,
            calibration_seconds=-1.0,
            min_threshold=threshold,
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
