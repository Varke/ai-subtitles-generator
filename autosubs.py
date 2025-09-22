#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --- модульные константы ---
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_LINE_MAX = 42
DEFAULT_MODEL_SIZE = "medium"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "auto"
DEFAULT_BEAM_SIZE = 5
NO_SPEECH_THRESHOLD = 0.6
SUBBED_SUFFIX = "_subbed"
VTT_HEADER = "WEBVTT\n\n"
TEMP_AUDIO_NAME = "audio.wav"
RESULT_DIR_NAME = "result"

# --- deps check ---
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Требуется пакет faster-whisper. Установите: pip install faster-whisper")
    sys.exit(1)


# ---------- helpers ----------

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("Не найден ffmpeg. Установите его и убедитесь, что он в PATH: https://ffmpeg.org/download.html")
        sys.exit(1)


def run(cmd, err_msg):
    """Выполнить внешнюю команду и завершить программу при ошибке."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # пробрасываем stderr, чтобы было видно, что не так
        sys.stderr.write(e.stderr.decode(errors="ignore"))
        print(f"\n{err_msg}")
        sys.exit(1)


def extract_audio(input_video: Path, out_wav: Path, sample_rate=DEFAULT_SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_wav)
    ]
    run(cmd, "Ошибка извлечения аудио с ffmpeg")


def srt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    mm = (ms_total % 3_600_000) // 60_000
    ss = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def wrap_lines(text: str, line_max=DEFAULT_LINE_MAX) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) <= line_max:
        return t
    words = t.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        add = (1 if cur else 0) + len(w)
        if cur_len + add > line_max:
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def write_srt(segments, out_path: Path, line_max=DEFAULT_LINE_MAX):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
            f.write(wrap_lines(text, line_max=line_max) + "\n\n")


def write_vtt(segments, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(VTT_HEADER)
        for (start, end, text) in segments:
            st = srt_timestamp(start).replace(",", ".")
            et = srt_timestamp(end).replace(",", ".")
            f.write(f"{st} --> {et}\n{text.strip()}\n\n")


def select_device(user_device: str) -> str:
    """
    faster-whisper/ctranslate2 поддерживают: 'cpu', 'cuda', 'rocm'
    'metal' НЕ поддерживается -> переводим на 'cpu' (для Mac).
    'auto' для стабильности сведём к 'cpu' (без доп. эвристик).
    """
    d = (user_device or "auto").lower()
    if d in ("metal", "mps"):
        print("Предупреждение: 'metal' не поддерживается в faster-whisper. Использую CPU.")
        return "cpu"
    if d == "auto":
        # можно было бы детектить CUDA, но без torch это ненадёжно; делаем предсказуемо
        return "cpu"
    return d


def select_compute_type(user_compute: str) -> str:
    """
    В ctranslate2 корректное дефолтное значение — 'default'.
    'auto' меняем на 'default'.
    """
    c = (user_compute or "default").lower()
    return "default" if c == "auto" else c


def transcribe(
    audio_path: Path,
    model_size=DEFAULT_MODEL_SIZE,
    device=DEFAULT_DEVICE,
    compute_type=DEFAULT_COMPUTE_TYPE,
    language=None,
    translate_to_english=False,
    vad_filter=True,
    beam_size=DEFAULT_BEAM_SIZE
):
    device_arg = select_device(device)
    compute_arg = select_compute_type(compute_type)

    print(f"Загружаю модель faster-whisper: {model_size} (device={device_arg}, compute={compute_arg})…")
    model = WhisperModel(model_size, device=device_arg, compute_type=compute_arg)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        task="translate" if translate_to_english else "transcribe",
        vad_filter=vad_filter,
        beam_size=beam_size,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        condition_on_previous_text=True
    )

    print(f"Определён язык: {info.language} (prob={info.language_probability:.2f})")
    segments = [(seg.start, seg.end, seg.text.strip()) for seg in segments_iter]
    return segments, info


def build_subtitles_filter(srt_path: Path, font: str | None) -> str:
    """
    Подготавливаем аргумент для -vf subtitles=...
    Экранируем символы, безопасные кавычки. Поддержка force_style для кириллицы/греческого.
    """
    # ffmpeg обычно понимает POSIX-путь
    fn = srt_path.as_posix()
    # экранируем ':' и '\'' внутри filtergraph
    fn_escaped = fn.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    result_str = f"subtitles='{fn_escaped}':force_style="
    styles = []
    if font:
        # если передали имя шрифта, пробрасываем как force_style=FontName=...
        font_escaped = font.replace("'", r"\'")
        styles.append(f"FontName={font_escaped}")

    ########### test other styles
    styles.append("MarginV=50")
    styles.append("PrimaryColour=&H0000FFFF&")
    params_as_str = ",".join(styles)
    
    result_str += f"'{params_as_str}'"
    return result_str


def burn_subs(input_video: Path, srt_path: Path, output_video: Path, font: str | None):
    vf = build_subtitles_filter(srt_path, font)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-c:a", "copy",
        str(output_video)
    ]
    run(cmd, "Ошибка вшивания субтитров с ffmpeg")


# ---------- high-level workflow helpers ----------

def parse_args() -> argparse.Namespace:
    """Сконфигурировать и разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Авто-субтитры (Whisper via faster-whisper) + вшивание в видео через ffmpeg."
    )
    parser.add_argument("input", type=str, help="Путь к видеофайлу")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Базовое имя результата (без расширения). По умолчанию — имя видео.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_SIZE,
        help="Модель: tiny/base/small/medium/large-v3 (по умолчанию medium)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Устройство: auto|cpu|cuda|rocm (metal не поддерживается)",
    )
    parser.add_argument(
        "--compute",
        default=DEFAULT_COMPUTE_TYPE,
        help="Тип вычислений: default|int8|int8_float16|float16|int16|auto",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Код языка (ru, el, en, ...). Если не указано — автоопределение.",
    )
    parser.add_argument(
        "--to-english",
        action="store_true",
        help="Переводить речь на английский (translate вместо transcribe)",
    )
    parser.add_argument(
        "--format",
        choices=["srt", "vtt"],
        default="srt",
        help="Формат файла субтитров (по умолчанию srt)",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Имя шрифта или путь к .ttf для корректного отображения (кириллица/греческий)",
    )
    parser.add_argument(
        "--line-max",
        type=int,
        default=DEFAULT_LINE_MAX,
        help="Макс. длина строки в субтитрах для переноса",
    )
    parser.add_argument(
        "--keep-srt-only",
        action="store_true",
        help="Только создать файл субтитров, без вшивания в видео",
    )
    return parser.parse_args()


def prepare_paths(input_path: str, output_base: str | None, subs_format: str) -> tuple[Path, Path, Path]:
    """Рассчитать и подготовить пути для входного видео и результатов в папке result."""
    in_video = Path(input_path).expanduser().resolve()
    if not in_video.exists():
        print(f"Файл не найден: {in_video}")
        sys.exit(1)

    result_dir = in_video.parent / RESULT_DIR_NAME
    result_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(output_base).stem if output_base else in_video.stem
    base = result_dir / base_name
    subs_path = base.with_suffix(f".{subs_format}")
    out_video = result_dir / f"{base_name}{SUBBED_SUFFIX}{in_video.suffix}"
    return in_video, subs_path, out_video


def transcribe_video(
    in_video: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    translate_to_english: bool,
    beam_size: int = DEFAULT_BEAM_SIZE,
):
    """Извлечь аудио и выполнить распознавание речи."""
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / TEMP_AUDIO_NAME
        print("Извлекаю аудио…")
        extract_audio(in_video, tmp_wav)

        print("Распознаю речь… Это может занять время в зависимости от длины видео и модели.")
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


def write_subtitle_file(segments, subs_path: Path, file_format: str, line_max: int):
    """Записать субтитры в нужном формате."""
    print(f"Пишу субтитры в {subs_path} …")
    if file_format == "srt":
        write_srt(segments, subs_path, line_max=line_max)
    else:
        write_vtt(segments, subs_path)


def handle_video_output(
    input_video: Path,
    subs_path: Path,
    output_video: Path,
    *,
    font: str | None,
    keep_srt_only: bool,
):
    """Вшить субтитры в видео или сообщить о готовности файла."""
    if keep_srt_only:
        print(f"Готово! Файл субтитров: {subs_path}")
        return

    print("Вшиваю субтитры в видео…")
    burn_subs(input_video, subs_path, output_video, font=font)
    print(f"Готово! Видео с вшитыми субтитрами: {output_video}")
    print(f"Файл субтитров также сохранён: {subs_path}")


# ---------- CLI ----------

def main():
    args = parse_args()
    check_ffmpeg()
    in_video, subs_path, out_video = prepare_paths(args.input, args.output, args.format)

    segments, _ = transcribe_video(
        in_video,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute,
        language=args.lang,
        translate_to_english=args.to_english,
        beam_size=DEFAULT_BEAM_SIZE,
    )

    write_subtitle_file(segments, subs_path, args.format, line_max=args.line_max)
    handle_video_output(
        in_video,
        subs_path,
        out_video,
        font=args.font,
        keep_srt_only=args.keep_srt_only,
    )

    print("Завершено.")


if __name__ == "__main__":
    main()
