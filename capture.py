"""Operating-system agnostic raw microphone capture."""

from __future__ import annotations

from dataclasses import dataclass
import queue
from types import TracebackType

import numpy as np

try:
    from .devices import InputDevice, find_input_device, require_sounddevice
except ImportError:  # Allows direct execution through scripts importing capture.py
    from devices import InputDevice, find_input_device, require_sounddevice


@dataclass(frozen=True)
class MicFrame:
    """One block of mono microphone audio."""

    samples: np.ndarray
    sample_rate: int
    frames_seen: int


class RawMicStream:
    """Read raw mono float32 frames from the selected microphone.

    This class intentionally stays above operating-system details. The backend is
    PortAudio through sounddevice; the rest of the package sees only numpy frames.
    """

    def __init__(
        self,
        device: InputDevice | str | None = None,
        sample_rate: int = 16000,
        block_ms: int = 50,
        queue_size: int = 100,
    ) -> None:
        self.sounddevice = require_sounddevice()
        self.device = find_input_device(device) if isinstance(device, str) or device is None else device
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.blocksize = max(1, int(sample_rate * (block_ms / 1000.0)))
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stream = None
        self._frames_seen = 0

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def __enter__(self) -> "RawMicStream":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = self.sounddevice.InputStream(
            device=self.device.index,
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None

    def read(self, timeout: float = 0.5) -> MicFrame | None:
        try:
            chunk = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

        samples = chunk.reshape(-1).astype(np.float32, copy=False)
        self._frames_seen += len(samples)
        return MicFrame(
            samples=samples,
            sample_rate=self.sample_rate,
            frames_seen=self._frames_seen,
        )

    def _on_audio(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            # Status is backend-neutral PortAudio information. The consumer can
            # detect missing frames from the stream without platform-specific code.
            pass

        try:
            self._queue.put_nowait(indata.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(indata.copy())
