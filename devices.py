"""Audio input device discovery and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - exercised in real env setup
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc
else:
    SOUNDDEVICE_IMPORT_ERROR = None


RESPEAKER_HINTS = ("respeaker", "re speaker", "seeed", "arrayuac")


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float

    @property
    def is_respeaker_like(self) -> bool:
        normalized = self.name.lower()
        return any(hint in normalized for hint in RESPEAKER_HINTS)


def require_sounddevice():
    if sd is None:
        raise RuntimeError(
            "The 'sounddevice' package is required for microphone access. "
            "Install it in this environment before running audio_in mic tests."
        ) from SOUNDDEVICE_IMPORT_ERROR
    return sd


def list_input_devices() -> list[InputDevice]:
    sounddevice = require_sounddevice()
    devices = []

    for index, info in enumerate(sounddevice.query_devices()):
        max_input_channels = int(info.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue

        devices.append(
            InputDevice(
                index=index,
                name=str(info.get("name", "")),
                max_input_channels=max_input_channels,
                default_samplerate=float(info.get("default_samplerate", 0.0)),
            )
        )

    return devices


def find_input_device(selector: str | None = None) -> InputDevice:
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No audio input devices were found.")

    if selector:
        selected = _match_selector(devices, selector)
        if selected:
            return selected

        raise RuntimeError(
            f"No input device matched {selector!r}. Run with --list to see available devices."
        )

    for device in devices:
        if device.is_respeaker_like:
            return device

    return devices[0]


def format_device_table(devices: Iterable[InputDevice]) -> str:
    rows = ["index  channels  default_hz  name"]
    for device in devices:
        marker = "  *" if device.is_respeaker_like else ""
        rows.append(
            f"{device.index:>5}  {device.max_input_channels:>8}  "
            f"{device.default_samplerate:>10.0f}  {device.name}{marker}"
        )
    return "\n".join(rows)


def _match_selector(devices: list[InputDevice], selector: str) -> InputDevice | None:
    selector = selector.strip()

    if selector.isdigit():
        index = int(selector)
        for device in devices:
            if device.index == index:
                return device

    lowered = selector.lower()
    for device in devices:
        if lowered in device.name.lower():
            return device

    return None
