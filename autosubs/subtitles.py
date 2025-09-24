from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .config import DEFAULT_LINE_MAX, DEFAULT_MAX_LINES, VTT_HEADER

@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str

# --- ASS helpers ---

def ass_timestamp(seconds: float) -> str:
    """
    ASS время: H:MM:SS.cs (centiseconds). Пример: 0:01:02.34
    """
    total_cs = int(round(seconds * 100))  # сотые доли
    hh = total_cs // (100 * 3600)
    mm = (total_cs // (100 * 60)) % 60
    ss = (total_cs // 100) % 60
    cs = total_cs % 100
    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"

def _sanitize_ass_text(s: str) -> str:
    """
    Убираем/экраним то, что может сломать ASS-овские теги.
    Проще всего заменить фигурные скобки на круглые.
    """
    return (s or "").replace("{", "(").replace("}", ")").strip()

def write_ass(
    segments: list[tuple[float, float, str]],
    out_path: Path,
    *,
    font: str | None = None,
    playres_x: int = 1920,
    playres_y: int = 1080,
    font_size: int = 48,
    margin_v: int = 50,
    alignment: int = 2,              # 2 = bottom-center
    primary_colour: str = "&H00FFFFFF&",
    back_colour: str = "&H00000000&",
    border_style: int = 3,           # 3 = opaque box
    outline: int = 0,
    shadow: int = 0,
    fade_in_ms: int = 200,
    fade_out_ms: int = 120,
    line_max: int = DEFAULT_LINE_MAX,
) -> None:
    """
    Пишем полноценный ASS с преднастроенным стилем и fad(fade-in, fade-out)
    в каждой реплике. Перенос строк делаем wrap_lines(...), переносы -> N.
    """
    style_font = font or "Arial"

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {playres_x}",
        f"PlayResY: {playres_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{style_font},{font_size},{primary_colour},&H000000FF&,&H00000000&,{back_colour},"
            f"0,0,0,0,100,100,0,0,{border_style},{outline},{shadow},{alignment},10,10,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    lines: list[str] = []
    for (start, end, text) in segments:
        safe = _sanitize_ass_text(wrap_lines(text, line_max=line_max)).replace("\n", r"\N")
        t_start = ass_timestamp(start)
        t_end = ass_timestamp(end)
        override = r"{\fad(" + f"{max(0, int(fade_in_ms))},{max(0, int(fade_out_ms))}" + r")}"
        # Layer=0, Style=Default, Name=, Margins=0,0,margin_v (из стиля), Effect=
        lines.append(f"Dialogue: 0,{t_start},{t_end},Default,,0,0,0,,{override}{safe}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")


def srt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    hh = ms_total // 3_600_000
    mm = (ms_total % 3_600_000) // 60_000
    ss = (ms_total % 60_000) // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def wrap_lines(text: str, line_max: int = DEFAULT_LINE_MAX) -> str:
    """Перенос по словам + защита от сверхдлинных токенов (жёсткий рез)."""
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return ""
    # 1) жёстко режем сверхдлинные «слова»
    words: list[str] = []
    for w in t.split():
        if len(w) <= line_max:
            words.append(w)
        else:
            for i in range(0, len(w), line_max):
                words.append(w[i:i + line_max])
    # 2) собираем строки
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = (1 if cur else 0) + len(w)
        if cur_len + add > line_max:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

def _smart_chunk_text(text: str, hard_limit: int) -> list[str]:
    """Делит длинную реплику на куски <= hard_limit, стараясь резать «красиво»."""
    t = (text or "").strip()
    if not t:
        return []
    out: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        if n - i <= hard_limit:
            out.append(t[i:].strip())
            break
        window = t[i:i + hard_limit + 1]
        candidates = []
        for mark in (". ", "! ", "? ", "… ", "; ", ": ", ", ", " "):
            pos = window.rfind(mark)
            if pos != -1:
                candidates.append(pos + len(mark.strip()))
        cut = i + (max(candidates) if candidates else hard_limit)
        out.append(t[i:cut].strip())
        i = cut
    return [s for s in out if s]

def resegment_for_readability(
    segments: List[Tuple[float, float, str]],
    max_chars_per_line: int = DEFAULT_LINE_MAX,
    max_lines: int = DEFAULT_MAX_LINES,
) -> List[Tuple[float, float, str]]:
    """
    Если текст сегмента > max_chars_per_line*max_lines — делим на несколько кью.
    Тайминг частей распределяется пропорционально длине текста.
    """
    hard_limit = max(8, int(max_chars_per_line) * int(max_lines))
    new_segments: list[Tuple[float, float, str]] = []

    for start, end, text in segments:
        text = (text or "").strip()
        if not text:
            continue
        parts = _smart_chunk_text(text, hard_limit)
        if not parts:
            continue

        total_chars = sum(len(p) for p in parts) or 1
        t0 = start
        dur = max(0.001, end - start)
        acc = 0

        for part in parts:
            acc_next = acc + len(part)
            seg_start = t0 + dur * (acc / total_chars)
            seg_end = t0 + dur * (acc_next / total_chars)
            if seg_end <= seg_start:
                seg_end = seg_start + 0.001
            new_segments.append((seg_start, seg_end, part))
            acc = acc_next

    return new_segments

def write_srt(segments: List[Tuple[float, float, str]], out_path: Path,
              line_max: int = DEFAULT_LINE_MAX) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n")
            f.write(wrap_lines(text, line_max=line_max) + "\n\n")

def write_vtt(segments: List[Tuple[float, float, str]], out_path: Path,
              line_max: int = DEFAULT_LINE_MAX) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(VTT_HEADER)
        for start, end, text in segments:
            st = srt_timestamp(start).replace(",", ".")
            et = srt_timestamp(end).replace(",", ".")
            f.write(f"{st} --> {et}\n{wrap_lines(text, line_max=line_max)}\n\n")
