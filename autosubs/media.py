from __future__ import annotations

from pathlib import Path
from typing import List

from .config import (
    ALIGNMENT, BACK_COLOUR, BORDER_STYLE, MARGIN_V, PRIMARY_COLOUR,
    VOLUME_BOOST_DB, DEFAULT_SAMPLE_RATE, BACKGROUND_MUSIC_DB, SHADOW, OUTLINE
)
from .utils import run, probe_media_duration_sec

class MediaProcessor:
    """FFmpeg-процедуры: извлечь аудио, смикшировать, вжечь сабы, замультиплексить."""

    @staticmethod
    def extract_audio(input_video: Path, out_wav: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """
        Извлекает моно WAV. Если у видео нет аудиодорожки или ffmpeg вернул ошибку —
        создаёт WAV-тишину той же длительности (фолбэк).
        """
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_video),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le", str(out_wav),
        ]
        print("RUN:", " ".join(str(c) for c in cmd))
        import subprocess
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0:
            return

        dur_sec = probe_media_duration_sec(input_video)
        if dur_sec <= 0:
            raise RuntimeError("Не удалось извлечь аудио и определить длительность видео.")

        print("Не удалось извлечь аудио (возможно, у видео нет звука). Подкладываю тишину…")
        cmd_sil = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-t", f"{dur_sec:.3f}",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-c:a", "pcm_s16le", str(out_wav),
        ]
        run(cmd_sil, "Ошибка создания WAV-тишины")

    @staticmethod
    def render_voiceover_track(seg_wavs: List[Path], starts_sec: List[float],
                               duration_sec: float, out_wav: Path,
                               normalize: bool = True, target_rate: int = 48000) -> None:
        if len(seg_wavs) != len(starts_sec):
            raise RuntimeError("Несогласованы списки seg_wavs и starts_sec.")

        cmd: list[str] = [
            "ffmpeg", "-y", "-loglevel", "info",
            "-f", "lavfi", "-t", f"{duration_sec:.3f}", "-i", f"anullsrc=r={target_rate}:cl=mono",
        ]
        for w in seg_wavs:
            cmd += ["-i", str(w)]

        parts: list[str] = []
        labels: list[str] = ["[sil]"]
        parts.append(f"[0:a]aformat=sample_fmts=s16:sample_rates={target_rate}:channel_layouts=mono[sil]")

        for i, start in enumerate(starts_sec, start=1):
            delay_ms = max(0, int(round(start * 1000)))
            parts.append(
                f"[{i}:a]aformat=sample_fmts=s16:sample_rates={target_rate}:channel_layouts=mono,"
                f"adelay=delays={delay_ms}:all=1[a{i}]"
            )
            labels.append(f"[a{i}]")

        amix = "".join(labels) + f"amix=inputs={len(labels)}:normalize={'1' if normalize else '0'}[mix]"
        post = f"[mix]volume={VOLUME_BOOST_DB}dB[aout]"
        filter_complex = ";".join(parts + [amix, post])

        cmd += ["-filter_complex", filter_complex, "-map", "[aout]",
                "-c:a", "pcm_s16le", "-ar", str(target_rate), "-ac", "1", str(out_wav)]
        run(cmd, "Ошибка рендера единой дорожки озвучки (ffmpeg)")

    @staticmethod
    def mix_background_music(voiceover_wav: Path, bg_music_path: Path,
                             duration_sec: float, out_wav: Path,
                             bg_gain_db: float = BACKGROUND_MUSIC_DB, target_rate: int = 48000) -> None:
        if not bg_music_path.exists() or not voiceover_wav.exists():
            print("Фоновая музыка не найдена или нет озвучки — пропускаю микс.")
            import shutil
            shutil.copyfile(voiceover_wav, out_wav)
            return
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(bg_music_path),
            "-i", str(voiceover_wav),
            "-t", f"{duration_sec:.3f}",
            "-filter_complex",
            (
                f"[0:a]aformat=sample_fmts=s16:sample_rates={target_rate}:channel_layouts=mono,volume={bg_gain_db}dB[bg];"
                f"[1:a]aformat=sample_fmts=s16:sample_rates={target_rate}:channel_layouts=mono[vo];"
                f"[vo][bg]amix=inputs=2:normalize=0:duration=first[aout]"
            ),
            "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", str(target_rate), "-ac", "1", str(out_wav),
        ]
        run(cmd, "Ошибка смешивания фоновой музыки с озвучкой (ffmpeg)")

    @staticmethod
    def build_subtitles_filter(srt_path: Path, font: str | None) -> str:
        """subtitles=...; для .ass — без force_style (всё задаётся в самом ASS)."""
        def esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")

        fn = srt_path.as_posix()

        # Если это уже .ass — вернём чистый фильтр без принудительных стилей
        if srt_path.suffix.lower() == ".ass":
            return f"subtitles='{esc(fn)}'"

        # иначе (srt/vtt) — применяем прежние стили
        vf = f"subtitles='{esc(fn)}'"

        styles: list[str] = []
        if font:
            styles.append("FontName=" + esc(font))
        styles += [
            f"MarginV={MARGIN_V}",
            f"PrimaryColour={PRIMARY_COLOUR}",
            f"BackColour={BACK_COLOUR}",
            f"BorderStyle={BORDER_STYLE}",
            f"Outline={OUTLINE}",
            f"Shadow={SHADOW}",
            f"Alignment={ALIGNMENT}",
        ]
        if styles:
            vf += f":force_style='{','.join(styles)}'"
        return vf


    @staticmethod
    def burn_subs(input_video: Path, srt_path: Path, output_video: Path, font: str | None) -> None:
        vf = MediaProcessor.build_subtitles_filter(srt_path, font)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_video),
            "-vf", vf, "-c:a", "copy", str(output_video),
        ]
        run(cmd, "Ошибка вшивания субтитров с ffmpeg")

    @staticmethod
    def mux_video_with_audio(
        input_video: Path,
        new_audio_wav: Path,
        output_video: Path,
        keep_original_audio: bool = False,
        extend_to_audio: bool = True,
        pad_tail_sec: float = 1.0,   # Принудительная задержка в конце, чтобы английский перевод влез, если он длиннее исходного
    ) -> None:
        if not new_audio_wav.exists() or new_audio_wav.stat().st_size < 1000:
            print(f"Внимание: файл озвучки пустой: {new_audio_wav}. Подкладываю тишину.")
            tmp_sil = new_audio_wav.with_name("fallback_silence.wav")
            run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "3",
                "-i", "anullsrc=r=48000:cl=mono", "-c:a", "pcm_s16le", str(tmp_sil)],
                "Не удалось создать запасную тишину",
            )
            new_audio_wav = tmp_sil

        video_dur = probe_media_duration_sec(input_video)
        audio_dur = probe_media_duration_sec(new_audio_wav)

        # Если нужно выровнять по аудио — берём разницу; плюс учитываем ручной паддинг
        extra_by_audio = max(0.0, audio_dur - video_dur) if extend_to_audio else 0.0
        pad = max(extra_by_audio, float(pad_tail_sec or 0.0))

        base = ["ffmpeg", "-y", "-loglevel", "info", "-i", str(input_video), "-i", str(new_audio_wav)]

        vf = []
        reencode_video = False
        if pad > 0.01:
            vf = ["-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}"]
            reencode_video = True  # есть -vf → перекодируем видео

        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"] if reencode_video else ["-c:v", "copy"]

        if keep_original_audio:
            cmd = base + vf + [
                "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
                *vcodec,
                "-c:a", "aac", "-b:a", "192k",
                "-disposition:a:0", "default",
                "-metadata:s:a:0", "language=eng",
                # БЕЗ -shortest — не режем хвост аудио
                str(output_video),
            ]
        else:
            cmd = base + vf + [
                "-map", "0:v:0", "-map", "1:a:0",
                *vcodec,
                "-c:a", "aac", "-b:a", "192k",
                "-disposition:a:0", "default",
                "-metadata:s:a:0", "language=eng",
                str(output_video),
            ]

        run(cmd, "Ошибка мультиплексирования видео с английской озвучкой")
        print(f"[mux] video={video_dur:.3f}s audio={audio_dur:.3f}s pad={pad:.3f}s reenc={reencode_video}")
