from dataclasses import dataclass
from typing import List

from audio_backend import get_sounddevice


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int

    def label(self) -> str:
        return f"[{self.index}] {self.name}"


def list_audio_devices() -> List[AudioDevice]:
    sd = get_sounddevice()
    result: List[AudioDevice] = []
    for idx, dev in enumerate(sd.query_devices()):
        result.append(
            AudioDevice(
                index=idx,
                name=str(dev["name"]),
                max_input_channels=int(dev["max_input_channels"]),
                max_output_channels=int(dev["max_output_channels"]),
            )
        )
    return result


def list_input_devices() -> List[AudioDevice]:
    return [d for d in list_audio_devices() if d.max_input_channels > 0]


def list_output_devices() -> List[AudioDevice]:
    return [d for d in list_audio_devices() if d.max_output_channels > 0]


def parse_index_from_label(label: str) -> int:
    # label format: [12] Device name
    left = label.find("[")
    right = label.find("]")
    if left == -1 or right == -1 or right <= left + 1:
        raise ValueError(f"Invalid device label: {label}")
    return int(label[left + 1:right])
