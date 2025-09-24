from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("Не найден ffmpeg/ffprobe. Установите и добавьте в PATH. (macOS: brew install ffmpeg)")
        sys.exit(1)

def check_piper() -> None:
    if shutil.which("piper") is None:
        print("Не найден 'piper'. Установите (macOS: brew install piper) и скачайте голос .onnx + .onnx.json.")
        sys.exit(1)

def run(cmd: list[str], err_msg: str) -> None:
    print("RUN:", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n{err_msg}")
        sys.exit(1)

def probe_media_duration_sec(path: Path) -> float:
    """Длительность медиа в секундах через ffprobe (0.0 при ошибке)."""
    try:
        out = (
            subprocess.check_output(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 str(path)]
            ).decode().strip()
        )
        return float(out)
    except Exception:
        return 0.0

def probe_video_duration_ms(input_video: Path) -> int:
    return int(round(probe_media_duration_sec(input_video) * 1000))
