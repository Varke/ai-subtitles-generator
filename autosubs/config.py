from __future__ import annotations

import re
from pathlib import Path

# ---------- Константы и настройки (общие) ----------
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_LINE_MAX = 42                 # макс. символов в строке
DEFAULT_MAX_LINES = 2                 # макс. строк в одном кью
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_DEVICE = "auto"               # cpu|cuda|rocm|auto
DEFAULT_COMPUTE_TYPE = "auto"         # default|int8|int8_float16|float16|int16|auto
DEFAULT_BEAM_SIZE = 5
NO_SPEECH_THRESHOLD = 0.6

# ===== ASS defaults =====
ASS_PLAYRES_X = 1920
ASS_PLAYRES_Y = 1080
ASS_FONT_SIZE = 48
ASS_FADE_IN_MS = 200
ASS_FADE_OUT_MS = 120

# Разводка по файлам/имёнам
RESULT_DIR_NAME = "result"
TEMP_AUDIO_NAME = "audio.wav"
VTT_HEADER = "WEBVTT\n\n"
RU_SUBBED_SUFFIX = "_ru_subbed"
EN_DUBBED_SUFFIX = "_en_dubbed"

# Фоновая музыка (опционально)
BACKGROUND_MUSIC_PATH = "lo_fi_only.WAV"    # напр.: "/path/to/bg.mp3" ; пусто = не использовать
BACKGROUND_MUSIC_DB = -24.0   # громкость фоновой музыки в dB

# Авто-сдвиг EN таймингов под TTS
MIN_TTS_GAP_SEC = 0.12        # минимальный зазор между озвучками (120 мс)

# Регексы для фильтрации текста перед TTS
NUMERIC_ONLY_RE = re.compile(r"^\s*[\d\s\.,:;\/\\\-–—]+\s*$")
TIMECODEY_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?([,\.]\d{1,3})?")
DIGIT_WORD_RE = re.compile(r"\S*\d\S*")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

# Цвета/стили ASS (libass): AA BB GG RR (AA=00 непрозрачно)
PRIMARY_COLOUR = "&H00FFFFFF&"  # белый текст
BACK_COLOUR    = "&H00000000&"  # чёрный фон (opaque)
BORDER_STYLE   = 3              # 3 = opaque box
ALIGNMENT      = 5              # центр снизу (5 — центр экрана)
MARGIN_V       = 50
OUTLINE        = 6              # толщина «поля» бокса
SHADOW         = 0

# Прочее
VOLUME_BOOST_DB = 6  # усиление итоговой TTS-дорожки перед выводом WAV

# Утилита для путей
def ensure_resolved(p: str | Path | None) -> Path | None:
    if not p:
        return None
    return Path(p).expanduser().resolve()
