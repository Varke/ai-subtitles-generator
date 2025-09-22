#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autosubs.py
Автор: github.com/varke

Краткое описание:
Скрипт автоматизирует процесс создания субтитров и дубляжа:
  1) Распознавание / перевод речи (Whisper) → .ru/.en.srt
  2) Вшивание русских субтитров в исходное видео (ориг. аудио)
  3) Синтез англ. озвучки (Piper) → WAV
  4) Мультиплекс видео + TTS-дорожка → финальное видео с EN-сабами

В целях надёжности перед TTS производится жёсткая фильтрация текста:
  - удаляются таймкоды, чисто цифровые строки, URL, маркеры, слова с цифрами и т.п.
  - текст передаётся в Piper через STDIN (без передачи флага -t)
  - для управления скоростью речи length_scale задаётся в конфиге .json,
    а не в CLI (чтобы не попасть в синтезируемый текст).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------- Константы и настройки ----------
# Частота семплирования при извлечении аудио (для Whisper)
DEFAULT_SAMPLE_RATE = 16_000

# Максимальная длина строки для сабов при переносе
DEFAULT_LINE_MAX = 42

# Параметры по умолчанию для Whisper
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_DEVICE = "auto"       # cpu|cuda|rocm|auto
DEFAULT_COMPUTE_TYPE = "auto" # default|int8|int8_float16|float16|int16|auto
DEFAULT_BEAM_SIZE = 5
NO_SPEECH_THRESHOLD = 0.6

# Именование выходных артефактов и временных файлов
RESULT_DIR_NAME = "result"
TEMP_AUDIO_NAME = "audio.wav"
VTT_HEADER = "WEBVTT\n\n"
RU_SUBBED_SUFFIX = "_ru_subbed"
EN_DUBBED_SUFFIX = "_en_dubbed"

# Регексы для фильтрации текста перед TTS
_NUMERIC_ONLY_RE = re.compile(r"^\s*[\d\s\.,:;\/\\\-–—]+\s*$")
_TIMECODEY_RE = re.compile(r"\d{1,2}:\d{2}(:\d{2})?([,\.]\d{1,3})?")

# Слово, содержащее цифру; и пунктуация только
_DIGIT_WORD_RE = re.compile(r"\S*\d\S*")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

# ---------- Зависимости ----------
# faster-whisper используется для транскрипции/перевода
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Требуется пакет faster-whisper. Установите: pip install faster-whisper")
    sys.exit(1)


# ---------- Вспомогательные обёртки внешних утилит ----------

def check_ffmpeg() -> None:
    """Проверка наличия ffmpeg и ffprobe в PATH."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print(
            "Не найден ffmpeg/ffprobe. Установите и добавьте в PATH. "
            "(macOS: brew install ffmpeg)"
        )
        sys.exit(1)


def check_piper() -> None:
    """Проверка наличия утилиты piper в PATH."""
    if shutil.which("piper") is None:
        print(
            "Не найден 'piper'. Установите (macOS: brew install piper) "
            "и скачайте голос .onnx + .onnx.json."
        )
        sys.exit(1)


def run(cmd: list[str], err_msg: str) -> None:
    """
    Универсальный запуск внешней команды.
    Печать команды для отладки; при ошибке выводим понятный текст и выходим.
    """
    print("RUN:", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n{err_msg}")
        sys.exit(1)


# ---------- Работа с медиа через ffmpeg ----------

def extract_audio(input_video: Path, out_wav: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """
    Извлекает моно WAV с заданной частотой — удобно для транскрипции.
    Формат: PCM S16 LE (стандартный для Whisper).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    run(cmd, "Ошибка извлечения аудио с ffmpeg")


def probe_video_duration_ms(input_video: Path) -> int:
    """Получение длительности видео в миллисекундах через ffprobe."""
    try:
        out = (
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(input_video),
                ]
            )
            .decode()
            .strip()
        )
        seconds = float(out)
    except Exception:
        seconds = 0.0
    return int(round(seconds * 1000))


# ---------- Субтитры (SRT / VTT) ----------

def srt_timestamp(seconds: float) -> str:
    """Конвертация секунд в формат SRT HH:MM:SS,mmm."""
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    mm = (ms_total % 3_600_000) // 60_000
    ss = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def wrap_lines(text: str, line_max: int = DEFAULT_LINE_MAX) -> str:
    """
    Примитивный перенос строк по словам.
    Разбивает длинную строку на несколько строк <= line_max.
    """
    t = text.strip().replace("\n", " ")
    if len(t) <= line_max:
        return t
    words = t.split()
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = (1 if cur else 0) + len(w)
        if cur_len + add > line_max:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def write_srt(segments: list[tuple[float, float, str]], out_path: Path, line_max: int = DEFAULT_LINE_MAX) -> None:
    """Запись списка сегментов в формат .srt."""
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
            f.write(wrap_lines(text, line_max=line_max) + "\n\n")


def write_vtt(segments: list[tuple[float, float, str]], out_path: Path) -> None:
    """Запись сегментов в формат .vtt."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(VTT_HEADER)
        for start, end, text in segments:
            st = srt_timestamp(start).replace(",", ".")
            et = srt_timestamp(end).replace(",", ".")
            f.write(f"{st} --> {et}\n{text.strip()}\n\n")


# ---------- Whisper (транскрипция / перевод) ----------

def select_device(user_device: str) -> str:
    """Нормализация выбора устройства (учёт 'metal'/'mps')."""
    d = (user_device or "auto").lower()
    if d in ("metal", "mps"):
        print("Предупреждение: 'metal' не поддерживается. Использую CPU.")
        return "cpu"
    if d == "auto":
        return "cpu"
    return d


def select_compute_type(user_compute: str) -> str:
    """Нормализация типа вычислений ('auto' → 'default')."""
    c = (user_compute or "default").lower()
    return "default" if c == "auto" else c


def transcribe(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    translate_to_english: bool = False,
    vad_filter: bool = True,
    beam_size: int = DEFAULT_BEAM_SIZE,
) -> tuple[list[tuple[float, float, str]], object]:
    """
    Запуск модели faster-whisper: транскрипция или перевод.
    Возвращает список сегментов (start, end, text) и объект info.
    """
    device_arg = select_device(device)
    compute_arg = select_compute_type(compute_type)

    print(
        f"Загружаю модель faster-whisper: {model_size} "
        f"(device={device_arg}, compute={compute_arg})…"
    )
    model = WhisperModel(model_size, device=device_arg, compute_type=compute_arg)

    segments_iter, info = model.transcribe(
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


# ---------- Вшивание сабов (ffmpeg libass) ----------

def build_subtitles_filter(srt_path: Path, font: str | None) -> str:
    """
    Формирование фильтра -vf subtitles=... с экранированием символов.
    По умолчанию: белый текст снизу по центру с отступом.
    """
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")

    fn = srt_path.as_posix()
    vf = f"subtitles='{esc(fn)}'"

    styles: list[str] = []
    if font:
        styles.append("FontName=" + esc(font))
    styles += [
        "MarginV=50",
        "PrimaryColour=&H00FFFFFF&",
        "Alignment=2",
    ]
    if styles:
        vf += f":force_style='{','.join(styles)}'"
    return vf


def burn_subs(input_video: Path, srt_path: Path, output_video: Path, font: str | None) -> None:
    """
    Hardcode субтитров в видеопоток; аудиопоток копируем без изменений.
    """
    vf = build_subtitles_filter(srt_path, font)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-c:a",
        "copy",
        str(output_video),
    ]
    run(cmd, "Ошибка вшивания субтитров с ffmpeg")


# ---------- Подготовка текста для TTS (Piper) ----------

def clean_text_for_tts(text: str) -> str:
    """
    Жёсткая очистка текста перед передачей в TTS:
      - удаляем метки [..] и (..)
      - удаляем URL/стрелки/маркеры
      - нормализуем тире и ведущие символы
      - отбрасываем строки, содержащие только числа/таймкоды
      - удаляем слова, содержащие цифры
      - убираем одиночную пунктуацию
    Возвращаем пустую строку, если нет осмысленного текста.
    """
    t = text.strip()

    # удаляем содержимое в скобках/квадратных скобках
    t = re.sub(r"\[(?:[^\[\]]+)\]|\((?:[^()]+)\)", " ", t, flags=re.U)

    # удаляем URL, стрелки и маркеры
    t = re.sub(r"https?://\S+|www\.\S+|-->|->|=>|==>|«|»|►|•|·|▶", " ", t)

    # нормализуем разные тире в пробел
    t = t.replace("—", " ").replace("–", " ").replace("−", " ")

    # удаляем ведущие маркеры (пункты, буллеты и т.п.)
    t = re.sub(r"^[\-\—\–\:\;\,\.\·\•\*]+", " ", t)

    # если строка — таймкоды или чисто цифры — отбрасываем
    if _NUMERIC_ONLY_RE.match(t) or _TIMECODEY_RE.search(t):
        return ""

    # удаляем слова с цифрами (0.95, 95%, v2, prob=0.9)
    t = re.sub(_DIGIT_WORD_RE, " ", t)

    # убираем одиночную пунктуацию и пустые токены
    words = [w for w in t.split() if not _PUNCT_ONLY_RE.match(w)]
    t = " ".join(words)

    # схлопываем пробелы и обрезаем
    t = re.sub(r"\s+", " ", t).strip()

    # короткие/однобуквенные фразы не озвучиваем
    if len(t) <= 1:
        return ""

    return t


def _prepare_piper_config_with_length_scale(orig_config_path: Path | None, length_scale: float) -> Path | None:
    """
    При необходимости формируем временный JSON-конфиг для Piper,
    содержащий поле length_scale. Если исходный конфиг задан и валиден,
    дополняем его, иначе создаём минимальный конфиг.
    Возвращаем путь к временному файлу (или исходному, если length_scale=1.0).
    """
    if length_scale is None or length_scale == 1.0:
        return orig_config_path

    try:
        cfg: dict = {}
        if orig_config_path and orig_config_path.exists():
            with open(orig_config_path, "r", encoding="utf-8") as f:
                try:
                    cfg = json.load(f)
                except Exception:
                    cfg = {}

        cfg["length_scale"] = float(length_scale)

        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="piper_cfg_")
        tf_path = Path(tf.name)
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        # возвращаем путь к временному конфигу
        return tf_path
    except Exception:
        return orig_config_path


def tts_segment_piper(
    text: str,
    model_path: Path,
    config_path: Path | None,
    out_wav: Path,
    length_scale: float = 1.0,
) -> None:
    """
    Синтез одного сегмента через Piper.
      - текст передаётся через STDIN (не через -t)
      - если после очистки текста ничего не осталось — создаём 50 ms тишины
      - length_scale не передаётся как CLI-флаг; вместо этого используем .json конфиг
    """
    t = clean_text_for_tts(text)
    if not t:
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-t",
                "0.05",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ],
            "Ошибка создания пустого сегмента",
        )
        return

    # подготовка конфигурации с length_scale (при необходимости)
    temp_cfg: Path | None = None
    cfg_to_use: Path | None = config_path
    if length_scale is not None and length_scale != 1.0:
        temp_cfg = _prepare_piper_config_with_length_scale(config_path, length_scale)
        cfg_to_use = temp_cfg

    cmd: list[str] = ["piper", "-m", str(model_path), "-f", str(out_wav)]
    if cfg_to_use:
        cmd.extend(["-c", str(cfg_to_use)])

    # Текст передаётся через STDIN (чтобы Piper не получил лишних флагов как текст)
    print("RUN:", " ".join(str(c) for c in cmd), f'<<< "{t[:60]}{"..." if len(t) > 60 else ""}"')
    try:
        subprocess.run(cmd, input=(t + "\n").encode("utf-8"), check=True)
    except subprocess.CalledProcessError:
        print("Ошибка синтеза речи через Piper")
        sys.exit(1)
    finally:
        # чистим временный конфиг, если создали
        if temp_cfg and temp_cfg.exists():
            try:
                temp_cfg.unlink()
            except Exception:
                pass


# ---------- Сборка единой дорожки озвучки (ffmpeg) ----------

def render_voiceover_track_ffmpeg(
    seg_wavs: list[Path],
    starts_sec: list[float],
    duration_sec: float,
    out_wav: Path,
    normalize: bool = True,
    target_rate: int = 48000,
) -> None:
    """
    Собираем единую WAV-дорожку дубляжа:
      - создаём источник тишины по всей длительности;
      - для каждого сегмента применяем aformat и adelay;
      - смешиваем дорожки через amix и поднимаем громкость на 6 dB.
    """
    if len(seg_wavs) != len(starts_sec):
        print("Несогласованы списки seg_wavs и starts_sec.")
        sys.exit(1)

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "info",
        "-f",
        "lavfi",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        f"anullsrc=r={target_rate}:cl=mono",
    ]
    for w in seg_wavs:
        cmd += ["-i", str(w)]

    parts: list[str] = []
    labels: list[str] = ["[sil]"]
    parts.append(f"[0:a]aformat=sample_fmts=s16:sample_rates={target_rate}:channel_layouts=mono[sil]")

    for i, start in enumerate(starts_sec, start=1):
        delay_ms = max(0, int(round(start * 1000)))
        parts.append(
            f"[{i}:a]aformat=sample_fmts=s16:sample_rates={target_rate}:"
            f"channel_layouts=mono,adelay=delays={delay_ms}:all=1[a{i}]"
        )
        labels.append(f"[a{i}]")

    amix = "".join(labels) + f"amix=inputs={len(labels)}:normalize={'1' if normalize else '0'}[mix]"
    post = "[mix]volume=6dB[aout]"
    filter_complex = ";".join(parts + [amix, post])

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(target_rate),
        "-ac",
        "1",
        str(out_wav),
    ]
    run(cmd, "Ошибка рендера единой дорожки озвучки (ffmpeg)")
    print(f"Промежуточная дорожка озвучки: {out_wav}")


# ---------- Мультиплекс видео + новая аудиодорожка ----------

def mux_video_with_audio(
    input_video: Path,
    new_audio_wav: Path,
    output_video: Path,
    keep_original_audio: bool = False,
) -> None:
    """
    Собираем видео и новую аудиодорожку:
      - если keep_original_audio=False — заменяем оригинальный звук на TTS;
      - если True — TTS становится первым треком, оригинал остаётся вторым.
    """
    if not new_audio_wav.exists() or new_audio_wav.stat().st_size < 1000:
        # создаём 3 секунды тишины, если файл пуст/отсутствует
        print(f"Внимание: файл озвучки пустой: {new_audio_wav}. Подкладываю тишину.")
        tmp_sil = new_audio_wav.with_name("fallback_silence.wav")
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-t",
                "3",
                "-i",
                "anullsrc=r=48000:cl=mono",
                "-c:a",
                "pcm_s16le",
                str(tmp_sil),
            ],
            "Не удалось создать запасную тишину",
        )
        new_audio_wav = tmp_sil

    base = ["ffmpeg", "-y", "-loglevel", "info", "-i", str(input_video), "-i", str(new_audio_wav)]
    if keep_original_audio:
        cmd = base + [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-disposition:a:0",
            "default",
            "-metadata:s:a:0",
            "language=eng",
            "-shortest",
            str(output_video),
        ]
    else:
        cmd = base + [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-disposition:a:0",
            "default",
            "-metadata:s:a:0",
            "language=eng",
            "-shortest",
            str(output_video),
        ]
    run(cmd, "Ошибка мультиплексирования видео с английской озвучкой")


# ---------- Пайплайн: вспомогатели ----------

def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки для основного запуска."""
    parser = argparse.ArgumentParser(
        description="RU-сабы + EN-перевод/сабы + англ. дубляж (Piper) с фильтрацией мусорных реплик."
    )
    parser.add_argument("input", type=str, help="Путь к видеофайлу")
    parser.add_argument("-o", "--output", type=str, help="Базовое имя результата (без расширения). По умолчанию — имя видео.")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper модель: tiny/base/small/medium/large-v3")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Устройство: auto|cpu|cuda|rocm (metal не поддерживается)")
    parser.add_argument("--compute", default=DEFAULT_COMPUTE_TYPE, help="Тип вычислений: default|int8|int8_float16|float16|int16|auto")
    parser.add_argument("--lang", default=None, help="Код языка исходника (ru, en, ...). Если не задан — авто.")
    parser.add_argument("--format", choices=["srt", "vtt"], default="srt", help="Формат файлов сабов")
    parser.add_argument("--font", default=None, help="Имя шрифта/путь к .ttf для вшитых субтитров")
    # Piper
    parser.add_argument("--piper-voice", type=str, required=True, help="Путь к модели Piper .onnx (например voices/en_US-ryan-high.onnx)")
    parser.add_argument("--piper-voice-config", type=str, default=None, help="Путь к .onnx.json (если есть)")
    parser.add_argument("--piper-length-scale", type=float, default=1.0, help="Скорость речи Piper (0.9 быстрее, 1.1 медленнее)")
    parser.add_argument("--keep-original-audio", action="store_true", help="Оставить оригинальную дорожку вторым треком в EN-видео")
    return parser.parse_args()


def prepare_paths(input_path: str, output_base: str | None, subs_format: str) -> dict:
    """
    Формируем итоговые пути файлов в папке result/.
    Возвращаем словарь с путями к входному видео, субтам и результатам.
    """
    in_video = Path(input_path).expanduser().resolve()
    if not in_video.exists():
        print(f"Файл не найден: {in_video}")
        sys.exit(1)

    result_dir = in_video.parent / RESULT_DIR_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(output_base).stem if output_base else in_video.stem
    base = result_dir / base_name

    return {
        "in_video": in_video,
        "result_dir": result_dir,
        "ru_subs": base.with_suffix(f".ru.{subs_format}"),
        "en_subs": base.with_suffix(f".en.{subs_format}"),
        "ru_video": result_dir / f"{base_name}{RU_SUBBED_SUFFIX}{in_video.suffix}",
        "en_dubbed_temp": result_dir / f"{base_name}_en_dubbed_nosubs{in_video.suffix}",
        "en_video": result_dir / f"{base_name}{EN_DUBBED_SUFFIX}{in_video.suffix}",
        "en_voiceover_wav": result_dir / f"{base_name}_voiceover_en.wav",
    }


def transcribe_video(
    in_video: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    translate_to_english: bool,
    beam_size: int = DEFAULT_BEAM_SIZE,
) -> tuple[list[tuple[float, float, str]], object]:
    """
    Извлекаем аудио из видео и запускаем транскрипцию/перевод.
    Возвращаем сегменты и info.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / TEMP_AUDIO_NAME
        print("Извлекаю аудио…")
        extract_audio(in_video, tmp_wav)

        print("Распознаю речь… Это может занять время.")
        segments, info = transcribe(
            tmp_wav,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            translate_to_english=translate_to_english,
            beam_size=beam_size,
        )
    return segments, info


# ---------- Основной запуск пайплайна ----------

def main() -> None:
    """Точка входа: полный пайплайн распознавания, перевода, TTS и вшивания субтитров."""
    args = parse_args()
    check_ffmpeg()
    check_piper()

    paths = prepare_paths(args.input, args.output, args.format)
    in_video = paths["in_video"]

    # 1) RU-проход: распознаём и вшиваем русские субтитры (ориг. звук).
    ru_segments, ru_info = transcribe_video(
        in_video,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute,
        language=args.lang,
        translate_to_english=False,
        beam_size=DEFAULT_BEAM_SIZE,
    )
    print(f"RU: язык={ru_info.language} (prob={ru_info.language_probability:.2f})")

    if args.format == "srt":
        write_srt(ru_segments, paths["ru_subs"], line_max=DEFAULT_LINE_MAX)
    else:
        write_vtt(ru_segments, paths["ru_subs"])

    print("Вшиваю русские субтитры в видео (оригинальный звук)…")
    burn_subs(in_video, paths["ru_subs"], paths["ru_video"], font=args.font)

    # 2) EN-проход: переводим, генерируем сабы, синтезируем Piper, вшиваем EN-сабы.
    en_segments, en_info = transcribe_video(
        in_video,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute,
        language=args.lang,
        translate_to_english=True,
        beam_size=DEFAULT_BEAM_SIZE,
    )
    print(f"EN (translate): исходный язык={en_info.language} (prob={en_info.language_probability:.2f})")

    if args.format == "srt":
        write_srt(en_segments, paths["en_subs"], line_max=DEFAULT_LINE_MAX)
    else:
        write_vtt(en_segments, paths["en_subs"])

    # Проверяем файлы модели Piper и конфиг
    model_path = Path(args.piper_voice).expanduser().resolve()
    config_path = Path(args.piper_voice_config).expanduser().resolve() if args.piper_voice_config else None
    if not model_path.exists():
        print(f"Файл модели Piper не найден: {model_path}")
        sys.exit(1)
    if config_path and not config_path.exists():
        print(f"Файл конфигурации Piper не найден: {config_path}")
        sys.exit(1)

    # Генерация сегментов TTS и сборка единой дорожки
    print("Генерирую английскую озвучку (Piper)…")
    tmp_dir = Path(tempfile.mkdtemp(prefix="piper_segs_"))
    seg_wavs: list[Path] = []
    starts: list[float] = []
    for i, (start, _end, text) in enumerate(en_segments):
        seg_wav = tmp_dir / f"seg_{i:05d}.wav"
        tts_segment_piper(
            text,
            model_path=model_path,
            config_path=config_path,
            out_wav=seg_wav,
            length_scale=args.piper_length_scale,
        )
        if seg_wav.exists() and seg_wav.stat().st_size > 0:
            seg_wavs.append(seg_wav)
            starts.append(start)

    # Собираем единую дорожку озвучки
    duration_ms = probe_video_duration_ms(in_video)
    render_voiceover_track_ffmpeg(
        seg_wavs,
        starts,
        duration_sec=duration_ms / 1000.0,
        out_wav=paths["en_voiceover_wav"],
    )

    # Мультиплексируем видео + TTS (AAC)
    print("Собираю видео с английской озвучкой…")
    mux_video_with_audio(
        in_video,
        paths["en_voiceover_wav"],
        paths["en_dubbed_temp"],
        keep_original_audio=args.keep_original_audio,
    )

    # Вшиваем EN-сабы поверх дубляжа (аудио копируем)
    print("Вшиваю английские субтитры в видео с озвучкой…")
    burn_subs(paths["en_dubbed_temp"], paths["en_subs"], paths["en_video"], font=args.font)

    # Итог: вывод путей к полученным файлам
    print("\nГотово! Файлы в 'result/':")
    print(f"- RU сабы: {paths['ru_subs']}")
    print(f"- RU видео: {paths['ru_video']}")
    print(f"- EN сабы: {paths['en_subs']}")
    print(f"- WAV англ. озвучки: {paths['en_voiceover_wav']}")
    print(f"- EN видео (озвучка+сабы): {paths['en_video']}")


if __name__ == "__main__":
    main()
