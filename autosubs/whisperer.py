from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import (
    DEFAULT_BEAM_SIZE, DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL_SIZE,
    NO_SPEECH_THRESHOLD
)

# Импортируем faster-whisper здесь, чтобы остальной код поднимался даже без него
try:
    from faster_whisper import WhisperModel
except ImportError as e:
    WhisperModel = None  # type: ignore

def _select_device(user_device: str) -> str:
    d = (user_device or "auto").lower()
    if d in ("metal", "mps"):
        print("Предупреждение: 'metal' не поддерживается. Использую CPU.")
        return "cpu"
    if d == "auto":
        return "cpu"
    return d

def _select_compute_type(user_compute: str) -> str:
    c = (user_compute or "default").lower()
    return "default" if c == "auto" else c

class WhisperTranscriber:
    def __init__(self,
                 model_size: str = DEFAULT_MODEL_SIZE,
                 device: str = DEFAULT_DEVICE,
                 compute_type: str = DEFAULT_COMPUTE_TYPE) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "Требуется пакет faster-whisper. Установите: pip install faster-whisper"
            )
        self.device = _select_device(device)
        self.compute = _select_compute_type(compute_type)
        print(f"Загружаю модель faster-whisper: {model_size} (device={self.device}, compute={self.compute})…")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute)

    def transcribe(self,
                   audio_path: Path,
                   language: str | None = None,
                   translate_to_english: bool = False,
                   vad_filter: bool = True,
                   beam_size: int = DEFAULT_BEAM_SIZE
                   ) -> tuple[List[Tuple[float, float, str]], object]:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task="translate" if translate_to_english else "transcribe",
            vad_filter=vad_filter,
            beam_size=beam_size,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=True,
        )
        print(f"Определён язык: {info.language} (prob={info.language_probability:.2f})")
        segments = [(seg.start, seg.end, seg.text.strip()) for seg in segments_iter]
        return segments, info
