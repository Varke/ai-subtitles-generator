from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple

from .config import (
    DEFAULT_BEAM_SIZE, DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_LINE_MAX,
    DEFAULT_MAX_LINES, DEFAULT_MODEL_SIZE, EN_DUBBED_SUFFIX, MIN_TTS_GAP_SEC,
    RESULT_DIR_NAME, RU_SUBBED_SUFFIX, TEMP_AUDIO_NAME, VTT_HEADER,
    BACKGROUND_MUSIC_DB, BACKGROUND_MUSIC_PATH,
    ASS_PLAYRES_X, ASS_PLAYRES_Y, ASS_FONT_SIZE, ASS_FADE_IN_MS, ASS_FADE_OUT_MS,
    MARGIN_V, ALIGNMENT, PRIMARY_COLOUR, BACK_COLOUR, BORDER_STYLE, OUTLINE, SHADOW
)
from .utils import check_ffmpeg, check_piper, probe_video_duration_ms, probe_media_duration_sec
from .utils import run  # если нужно локально ещё где-то использовать
from .subtitles import write_srt, write_vtt, resegment_for_readability, write_ass
from .whisperer import WhisperTranscriber
from .tts import PiperTTS
from .media import MediaProcessor

def _prepare_paths(input_path: str, output_base: str | None, subs_format: str) -> dict:
    in_video = Path(input_path).expanduser().resolve()
    if not in_video.exists():
        raise FileNotFoundError(f"Файл не найден: {in_video}")

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
        "en_voiceover_with_bg_wav": result_dir / f"{base_name}_voiceover_en_with_bg.wav",
    }

def _merge_segments_into_sentences(
    segments: List[Tuple[float, float, str]],
    max_gap: float = 0.6,  # до 600 мс между кусками считаем «одно предложение»
) -> List[Tuple[float, float, str]]:
    def terminal(t: str) -> bool:
        t = (t or "").rstrip()
        return bool(t) and t[-1] in (".", "!", "?", "…")

    if not segments:
        return []

    merged: List[Tuple[float, float, str]] = []
    cs, ce, ct = segments[0]

    for s, e, txt in segments[1:]:
        gap = s - ce
        if gap <= max_gap and not terminal(ct):
            joiner = "" if ct.endswith("-") else " "
            ct = (ct + joiner + (txt or "").strip()).strip()
            ce = e
        else:
            merged.append((cs, ce, ct))
            cs, ce, ct = s, e, txt
    merged.append((cs, ce, ct))
    return merged

def _align_segments_to_tts(
    base_segments: List[Tuple[float, float, str]],
    seg_wavs: List[Path],
    min_gap: float = MIN_TTS_GAP_SEC,
) -> tuple[List[Tuple[float, float, str]], List[float]]:
    """
    По каждому EN-кью берём длительность WAV и ставим его так, чтобы:
      - старт >= исходного старта и >= (конца предыдущего + min_gap)
      - конец = старт + фактическая длительность WAV
    """
    from .utils import probe_media_duration_sec
    if len(base_segments) != len(seg_wavs):
        raise RuntimeError("Несовпадение количества EN сегментов и WAV TTS.")

    aligned_segments: list[Tuple[float, float, str]] = []
    starts: list[float] = []
    prev_end = 0.0

    for (orig_start, _orig_end, text), wav in zip(base_segments, seg_wavs):
        dur = max(0.001, probe_media_duration_sec(wav))
        start = max(orig_start, prev_end + min_gap)
        end = start + dur
        aligned_segments.append((start, end, text))
        starts.append(start)
        prev_end = end

    return aligned_segments, starts

class AutosubsPipeline:
    def __init__(self,
                 model: str = DEFAULT_MODEL_SIZE,
                 device: str = DEFAULT_DEVICE,
                 compute: str = DEFAULT_COMPUTE_TYPE,
                 font: str | None = None,
                 piper_voice: Path | str = "",
                 piper_voice_config: Path | str | None = None,
                 piper_length_scale: float = 1.0,
                 keep_original_audio: bool = False,
                 subs_format: str = "srt") -> None:
        self.model = model
        self.device = device
        self.compute = compute
        self.font = font
        self.voice_path = Path(piper_voice).expanduser().resolve()
        self.cfg_path = Path(piper_voice_config).expanduser().resolve() if piper_voice_config else None
        self.length_scale = piper_length_scale
        self.keep_original_audio = keep_original_audio
        self.subs_format = subs_format

    def run(self, input_path: str, output_base: str | None, lang: str | None) -> None:
        check_ffmpeg()
        check_piper()

        paths = _prepare_paths(input_path, output_base, self.subs_format)
        in_video = paths["in_video"]

        # Транскрайбер
        transcriber = WhisperTranscriber(self.model, self.device, self.compute)

        # 1) RU
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / TEMP_AUDIO_NAME
            print("Извлекаю аудио…")
            MediaProcessor.extract_audio(in_video, tmp_wav)
            print("Распознаю речь (RU)…")
            ru_segments, ru_info = transcriber.transcribe(
                tmp_wav, language=lang, translate_to_english=False
            )

        print(f"RU: язык={ru_info.language} (prob={ru_info.language_probability:.2f})")
        ru_segments = resegment_for_readability(
            ru_segments, max_chars_per_line=DEFAULT_LINE_MAX, max_lines=DEFAULT_MAX_LINES
        )

        # --- RU SUBS ---
        if self.subs_format == "ass":
            write_ass(
                ru_segments, paths["ru_subs"],
                font=self.font,
                playres_x=ASS_PLAYRES_X, playres_y=ASS_PLAYRES_Y,
                font_size=ASS_FONT_SIZE, margin_v=MARGIN_V, alignment=ALIGNMENT,
                primary_colour=PRIMARY_COLOUR, back_colour=BACK_COLOUR,
                border_style=BORDER_STYLE, outline=OUTLINE, shadow=SHADOW,
                fade_in_ms=ASS_FADE_IN_MS, fade_out_ms=ASS_FADE_OUT_MS,
                line_max=DEFAULT_LINE_MAX,
            )
        elif self.subs_format == "srt":
            write_srt(ru_segments, paths["ru_subs"], line_max=DEFAULT_LINE_MAX)
        else:
            write_vtt(ru_segments, paths["ru_subs"], line_max=DEFAULT_LINE_MAX)

        print("Вшиваю русские субтитры в видео (оригинальный звук)…")
        MediaProcessor.burn_subs(in_video, paths["ru_subs"], paths["ru_video"], font=self.font)

        # 2) EN
        with tempfile.TemporaryDirectory() as td:
            tmp_wav = Path(td) / TEMP_AUDIO_NAME
            print("Извлекаю аудио…")
            MediaProcessor.extract_audio(in_video, tmp_wav)
            print("Перевожу и распознаю речь (EN)…")
            en_segments, en_info = transcriber.transcribe(
                tmp_wav, language=lang, translate_to_english=True
            )

        print(f"EN (translate): исходный язык={en_info.language} (prob={en_info.language_probability:.2f})")
        # стало: сначала склеиваем в предложения...
        en_segments = _merge_segments_into_sentences(en_segments, max_gap=0.6)
        # ...а уже потом, если хочешь, «красиво» разбиваем слишком длинные реплики на несколько кью:
        en_segments = resegment_for_readability(
            en_segments, max_chars_per_line=DEFAULT_LINE_MAX, max_lines=DEFAULT_MAX_LINES
        )

        # TTS для каждого EN-кью
        if not self.voice_path.exists():
            raise FileNotFoundError(f"Файл модели Piper не найден: {self.voice_path}")
        if self.cfg_path and not self.cfg_path.exists():
            raise FileNotFoundError(f"Файл конфигурации Piper не найден: {self.cfg_path}")
        tts = PiperTTS(self.voice_path, self.cfg_path)

        print("Генерирую английскую озвучку (Piper)…")
        tmp_dir = Path(tempfile.mkdtemp(prefix="piper_segs_"))
        seg_wavs: list[Path] = []

        for i, (seg_start, seg_end, text) in enumerate(en_segments):
            target_dur = max(0.35, seg_end - seg_start)  # целевая длительность сегмента
            seg_wav = tmp_dir / f"seg_{i:05d}.wav"

            # 1-я попытка — базовый length_scale
            ls = self.length_scale
            tts.synth_segment(text, seg_wav, length_scale=ls, trim_silence=True)
            dur = probe_media_duration_sec(seg_wav)

            # Если не влезает — ускоряем (уменьшаем length_scale) и пересинтезируем один раз
            # (можешь сделать до 3 попыток, но обычно хватает одной).
            if dur > target_dur * 1.02:
                ratio = (target_dur / max(0.01, dur)) * 0.98
                ls_fit = max(0.70, min(1.20, ls * ratio))   # не выходим за разумные пределы
                if abs(ls_fit - ls) > 1e-3:
                    tts.synth_segment(text, seg_wav, length_scale=ls_fit, trim_silence=True)
                    dur = probe_media_duration_sec(seg_wav)

            seg_wavs.append(seg_wav)

        # Выравниваем EN тайминги под фактические длительности TTS
        en_segments_aligned, en_starts_aligned = _align_segments_to_tts(
            en_segments, seg_wavs, min_gap=0.02  # было MIN_TTS_GAP_SEC, ставим 20 мс
        )

        # Пишем EN сабы
        if self.subs_format == "ass":
            write_ass(
                en_segments_aligned, paths["en_subs"],
                font=self.font,
                playres_x=ASS_PLAYRES_X, playres_y=ASS_PLAYRES_Y,
                font_size=ASS_FONT_SIZE, margin_v=MARGIN_V, alignment=ALIGNMENT,
                primary_colour=PRIMARY_COLOUR, back_colour=BACK_COLOUR,
                border_style=BORDER_STYLE, outline=OUTLINE, shadow=SHADOW,
                fade_in_ms=ASS_FADE_IN_MS, fade_out_ms=ASS_FADE_OUT_MS,
                line_max=DEFAULT_LINE_MAX,
            )
        elif self.subs_format == "srt":
            write_srt(en_segments_aligned, paths["en_subs"], line_max=DEFAULT_LINE_MAX)
        else:
            write_vtt(en_segments_aligned, paths["en_subs"], line_max=DEFAULT_LINE_MAX)

        # Собираем единую дорожку озвучки
        duration_ms = probe_video_duration_ms(in_video)
        MediaProcessor.render_voiceover_track(
            seg_wavs, en_starts_aligned, duration_sec=duration_ms / 1000.0, out_wav=paths["en_voiceover_wav"]
        )

        # Фоновая музыка (если указана в конфиге)
        bg_path_str = (BACKGROUND_MUSIC_PATH or "").strip()
        if bg_path_str:
            print("Миксую фоновую музыку…")
            MediaProcessor.mix_background_music(
                paths["en_voiceover_wav"],
                Path(bg_path_str).expanduser().resolve(),
                duration_sec=duration_ms / 1000.0,
                out_wav=paths["en_voiceover_with_bg_wav"],
            )
            final_voiceover_for_mux = paths["en_voiceover_with_bg_wav"]
        else:
            final_voiceover_for_mux = paths["en_voiceover_wav"]

        print("Собираю видео с английской озвучкой…")
        MediaProcessor.mux_video_with_audio(
            in_video, final_voiceover_for_mux, paths["en_dubbed_temp"],
            keep_original_audio=self.keep_original_audio,
            extend_to_audio=True,  # ← добавили
        )

        print("Вшиваю английские субтитры в видео с озвучкой…")
        MediaProcessor.burn_subs(paths["en_dubbed_temp"], paths["en_subs"], paths["en_video"], font=self.font)

        # Итоги
        print("\nГотово! Файлы в 'result/':")
        print(f"- RU сабы: {paths['ru_subs']}")
        print(f"- RU видео: {paths['ru_video']}")
        print(f"- EN сабы: {paths['en_subs']}")
        print(f"- WAV англ. озвучки: {paths['en_voiceover_wav']}")
        if bg_path_str:
            print(f"- WAV англ. озвучки с фоном: {paths['en_voiceover_with_bg_wav']}")
        print(f"- EN видео (озвучка+сабы): {paths['en_video']}")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RU/EN сабы + англ. дубляж (Piper), деление длинных реплик, авто-сдвиг EN таймингов и фолбэки."
    )
    p.add_argument("input", type=str, help="Путь к видеофайлу")
    p.add_argument("-o", "--output", type=str, help="Базовое имя результата (без расширения). По умолчанию — имя видео.")
    p.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper модель: tiny/base/small/medium/large-v3")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="Устройство: auto|cpu|cuda|rocm (metal не поддерживается)")
    p.add_argument("--compute", default=DEFAULT_COMPUTE_TYPE, help="Тип вычислений: default|int8|int8_float16|float16|int16|auto")
    p.add_argument("--lang", default=None, help="Код языка исходника (ru, en, ...). Если не задан — авто.")
    p.add_argument("--format", choices=["srt", "vtt", "ass"], default="srt", help="Формат файлов сабов")
    p.add_argument("--font", default=None, help="Имя шрифта/путь к .ttf для вшитых субтитров")
    # Piper
    p.add_argument("--piper-voice", type=str, required=True, help="Путь к модели Piper .onnx (например voices/en_US-ryan-high.onnx)")
    p.add_argument("--piper-voice-config", type=str, default=None, help="Путь к .onnx.json (если есть)")
    p.add_argument("--piper-length-scale", type=float, default=1.0, help="Скорость Piper (0.9 быстрее, 1.1 медленнее)")
    p.add_argument("--keep-original-audio", action="store_true", help="Оставить оригинальную дорожку вторым треком")


    return p
