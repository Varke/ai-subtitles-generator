from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .config import (
    DIGIT_WORD_RE, NUMERIC_ONLY_RE, PUNCT_ONLY_RE, TIMECODEY_RE
)
from .utils import run

def _clean_text_for_tts(text: str) -> str:
    import re
    t = text.strip()
    t = re.sub(r"\[(?:[^\[\]]+)\]|\((?:[^()]+)\)", " ", t, flags=re.U)
    t = re.sub(r"https?://\S+|www\.\S+|-->|->|=>|==>|«|»|►|•|·|▶", " ", t)
    t = t.replace("—", " ").replace("–", " ").replace("−", " ")
    t = re.sub(r"^[\-\—\–\:\;\,\.\·\•\*]+", " ", t)
    if NUMERIC_ONLY_RE.match(t) or TIMECODEY_RE.search(t):
        return ""
    t = re.sub(DIGIT_WORD_RE, " ", t)
    words = [w for w in t.split() if not PUNCT_ONLY_RE.match(w)]
    t = " ".join(words)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= 1:
        return ""
    return t

def _prepare_piper_config_with_length_scale(orig_config_path: Path | None, length_scale: float) -> Path | None:
    if length_scale is None or length_scale == 1.0:
        return orig_config_path
    try:
        cfg: dict = {}
        if orig_config_path and orig_config_path.exists():
            with open(orig_config_path, "r", encoding="utf-8") as f:
                try: cfg = json.load(f)
                except Exception: cfg = {}
        cfg["length_scale"] = float(length_scale)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="piper_cfg_")
        tf_path = Path(tf.name)
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return tf_path
    except Exception:
        return orig_config_path

class PiperTTS:
    def __init__(self, model_path: Path, config_path: Path | None = None) -> None:
        self.model_path = model_path
        self.config_path = config_path

    def synth_segment(self, text: str, out_wav: Path, length_scale: float = 1.0,
                      trim_silence: bool = True) -> None:
        t = _clean_text_for_tts(text)
        if not t:
            run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "0.05",
                "-i", "anullsrc=r=48000:cl=mono", "-c:a", "pcm_s16le", str(out_wav)],
                "Ошибка создания пустого сегмента",
            )
            return

        temp_cfg: Path | None = None
        cfg_to_use: Path | None = self.config_path
        if length_scale is not None and length_scale != 1.0:
            temp_cfg = _prepare_piper_config_with_length_scale(self.config_path, length_scale)
            cfg_to_use = temp_cfg

        cmd: list[str] = ["piper", "-m", str(self.model_path), "-f", str(out_wav)]
        if cfg_to_use:
            cmd.extend(["-c", str(cfg_to_use)])

        print("RUN:", " ".join(str(c) for c in cmd), f'<<< "{t[:60]}{"..." if len(t) > 60 else ""}"')
        try:
            import subprocess
            subprocess.run(cmd, input=(t + "\n").encode("utf-8"), check=True)
        except Exception:
            print("Ошибка синтеза речи через Piper")
            raise
        finally:
            if temp_cfg and temp_cfg.exists():
                try: temp_cfg.unlink()
                except Exception: pass

        # Подрезаем тишину спереди/сзади — убирает паузы между соседними кусками
        if trim_silence and out_wav.exists():
            tmp = out_wav.with_suffix(".trim.wav")
            run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(out_wav),
                    "-af",
                    # обрезаем лидинг-тнишину, затем реверсим, обрезаем лидинг (бывший хвост), и обратно
                    "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.12,"
                    "areverse,"
                    "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.18,"
                    "areverse",
                    "-c:a", "pcm_s16le", str(tmp),
                ],
                "Ошибка удаления тишины на краях TTS",
            )
            tmp.replace(out_wav)

