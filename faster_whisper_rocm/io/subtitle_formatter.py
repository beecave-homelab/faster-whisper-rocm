"""Subtitle formatting utilities with constraint enforcement.

This module builds and refines subtitle cues (SRT/VTT) from model segments
using environment-driven constraints configured in
`faster_whisper_rocm.utils.constant`.

It supports:
- Word-aware splitting when word timestamps are available
- Fallback splitting without word timings
- Merging short/fast cues within limits
- Enforcing minimum gaps
- Greedy line wrapping and 2-line clamping per cue
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from faster_whisper_rocm.io.timestamps import format_timestamp
from faster_whisper_rocm.utils.constant import (
    DEFAULT_SUB_BOUNDARY_CHARS,
    DEFAULT_SUB_DISPLAY_BUFFER_SEC,
    DEFAULT_SUB_INTERJECTION_WHITELIST,
    DEFAULT_SUB_MAX_BLOCK_CHARS,
    DEFAULT_SUB_MAX_CONSECUTIVE_REPEATS,
    DEFAULT_SUB_MAX_CPS,
    DEFAULT_SUB_MAX_LINE_CHARS,
    DEFAULT_SUB_MAX_SEGMENT_DURATION_SEC,
    DEFAULT_SUB_MIN_GAP_SEC,
    DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC,
    DEFAULT_SUB_SOFT_BOUNDARY_WORDS,
)


@dataclass
class Cue:
    """Lightweight subtitle cue with index, start/end times (seconds), and text."""

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Duration of the cue in seconds, clamped to be non-negative."""
        return max(0.0, float(self.end) - float(self.start))

    @property
    def chars(self) -> int:
        """Number of characters in the cue text."""
        return len(self.text or "")


# ---------
# Wrapping
# ---------


def _wrap_text(text: str, max_line_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add_len = (1 if cur else 0) + len(w)
        if cur_len + add_len <= max_line_chars:
            cur.append(w)
            cur_len += add_len
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def _count_wrapped_lines(text: str, max_line_chars: int) -> int:
    return len(_wrap_text(text, max_line_chars))


# -----------------
# Boundary heuristics
# -----------------


def _ends_with_boundary(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in DEFAULT_SUB_BOUNDARY_CHARS


def _is_interjection(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in DEFAULT_SUB_INTERJECTION_WHITELIST


# -------------------------------
# Text normalization (punctuation)
# -------------------------------


def _normalize_punctuation_spacing(text: str) -> str:
    """Fix common spacing issues.

    - Remove spaces before punctuation like , . ; : ? !
    - Tighten spaces around parentheses
    - Join compound words split around hyphen (e.g., "counter -clockwise" ->
      "counter-clockwise")
    - Collapse excessive spaces.

    Conservative heuristics aim to avoid altering em-dash usages (" - ") by
    only joining hyphens between letters.

    Returns:
        str: Normalized text with consistent punctuation spacing.
    """
    t = text or ""
    # No space before punctuation
    t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    # Parentheses spacing
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    # Hyphen in compound words: join when letters on both sides
    t = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", t)
    # Collapse multiple spaces
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# ---------------------------
# Text cleanup (repetition, orphans)
# ---------------------------


def _strip_orphan_punctuation(text: str) -> str:
    """Remove leading/trailing orphan punctuation and tidy commas.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with no leading commas/semicolons and no
            punctuation-only strings.
    """
    t = text or ""
    # Collapse multiple commas
    t = re.sub(r",{2,}", ",", t)
    # Remove leading commas/semicolon/colon
    t = re.sub(r"^[\s,;:]+", "", t)
    # Remove trailing commas
    t = re.sub(r"[\s,;:]+$", "", t)
    # If becomes punctuation-only, drop it
    if re.fullmatch(r"[\s\W]*", t or ""):
        return ""
    return t.strip()


def _collapse_repeated_words(text: str, limit: int) -> str:
    """Collapse runs of the same word separated by commas/spaces.

    Case-insensitive comparison; preserves the original casing of the
    first occurrence and the comma-space rhythm.

    Args:
        text (str): Input text.
        limit (int): Maximum allowed consecutive repeats of the same word.

    Returns:
        str: Text with repeated words clamped to the given limit.
    """
    if limit <= 0:
        return text or ""
    s = text or ""
    # Tokenize into word / whitespace / punctuation tokens
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\s+|[^\w\s]", s)
    out: list[str] = []
    cur_word: str | None = None
    cur_count = 0
    sep_buf: list[str] = []  # pending separators between repeated words

    def is_word(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok))

    def is_soft_sep(tok: str) -> bool:
        # Only spaces or commas count as soft separators inside a repetition run
        return bool(re.fullmatch(r"\s+|,", tok))

    for tok in tokens:
        if is_word(tok):
            low = tok.lower()
            if cur_word is None:
                # start new run
                out.append(tok)
                cur_word = low
                cur_count = 1
                sep_buf = []
            elif low == cur_word and all(is_soft_sep(x) for x in sep_buf):
                # same word separated only by commas/spaces => repetition
                if cur_count < limit:
                    out.extend(sep_buf)
                    out.append(tok)
                    cur_count += 1
                # else: drop this occurrence and its separators
                sep_buf = []
            else:
                # different word or hard separator seen in between
                out.extend(sep_buf)
                out.append(tok)
                cur_word = low
                cur_count = 1
                sep_buf = []
        elif tok.strip() == "":
            # whitespace: buffer it; not a hard break for repetition
            sep_buf.append(tok)
        elif tok == ",":
            # comma is a soft sep for repetition runs
            sep_buf.append(tok)
        else:
            # hard punctuation ends repetition grouping
            out.extend(sep_buf)
            out.append(tok)
            cur_word = None
            cur_count = 0
            sep_buf = []

    # flush remaining separators
    out.extend(sep_buf)
    return "".join(out)


def _sanitize_cue_text(text: str) -> str:
    """Normalize spacing, collapse duplicates, and strip orphan punctuation.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text suitable for cue rendering.
    """
    t = _normalize_punctuation_spacing(text)
    t = _collapse_repeated_words(t, DEFAULT_SUB_MAX_CONSECUTIVE_REPEATS)
    t = _strip_orphan_punctuation(t)
    return t


# -----------
# Core logic
# -----------


def _cps(chars: int, duration: float) -> float:
    if duration <= 0:
        return float("inf") if chars > 0 else 0.0
    return chars / duration


def _split_segment_by_words(
    seg: object,
    max_dur: float,
    max_cps: float,
    max_lines: int,
    max_line_chars: int,
) -> list[Cue]:
    """Split a segment into multiple cues using word timestamps.

    Args:
        seg (object): Segment with word-level timings; expects `.start`, `.end`,
            `.text`, and `words` iterable of objects each having `.start`, `.end`,
            and `.word` attributes.
        max_dur (float): Maximum cue duration in seconds.
        max_cps (float): Maximum characters per second.
        max_lines (int): Maximum number of wrapped lines per cue.
        max_line_chars (int): Maximum characters allowed per line.

    Returns:
        list[Cue]: Cues split using word timings. If no words are present,
        a single fallback cue is returned for the whole segment.
    """
    words = list(getattr(seg, "words", []) or [])
    text_fallback = str(getattr(seg, "text", "") or "").strip()

    if not words:
        # Fallback: single cue, will be further processed by non-word splitter
        return [
            Cue(
                index=0,
                start=float(seg.start),
                end=float(seg.end),
                text=text_fallback,
            )
        ]

    cues: list[Cue] = []
    cur_words: list[object] = []
    cur_start: float | None = None

    def flush() -> None:
        nonlocal cur_words, cur_start
        if not cur_words:
            return
        start = cur_start if cur_start is not None else float(seg.start)
        end = float(cur_words[-1].end)
        raw_text = (" ".join(w.word for w in cur_words)).strip()
        text = _sanitize_cue_text(raw_text)
        if text:
            cues.append(Cue(index=0, start=float(start), end=float(end), text=text))
        cur_words = []
        cur_start = None

    for w in words:
        if cur_start is None:
            cur_start = float(w.start)
        # Tentative values if we add w
        t_text = (" ".join([*(x.word for x in cur_words), w.word])).strip()
        t_start = cur_start
        t_end = float(w.end)
        t_dur = max(0.0, t_end - float(t_start))
        s_text = _sanitize_cue_text(t_text)
        t_cps = _cps(len(s_text), t_dur)
        t_lines = _count_wrapped_lines(s_text, max_line_chars)
        if t_dur > max_dur or t_cps > max_cps or t_lines > max_lines:
            # finalize current and start a new one with w
            flush()
            cur_start = float(w.start)
            cur_words = [w]
        else:
            cur_words.append(w)
    flush()

    # Post adjust: apply a small display buffer when beneficial
    for i in range(len(cues)):
        if i + 1 < len(cues):
            next_start = cues[i + 1].start
        else:
            next_start = float(seg.end)
        new_end = min(
            cues[i].end + DEFAULT_SUB_DISPLAY_BUFFER_SEC,
            next_start - DEFAULT_SUB_MIN_GAP_SEC,
        )
        if new_end > cues[i].end:
            cues[i].end = new_end
    return cues


def _split_segment_without_words(
    seg: object,
    max_dur: float,
    max_cps: float,
    max_lines: int,
    max_line_chars: int,
) -> list[Cue]:
    """Split when word timings are absent by approximating from text length.

    Args:
        seg (object): Segment with `.start`, `.end`, and `.text` attributes.
        max_dur (float): Maximum cue duration in seconds.
        max_cps (float): Maximum characters per second.
        max_lines (int): Maximum number of wrapped lines per cue.
        max_line_chars (int): Maximum characters allowed per line.

    Returns:
        list[Cue]: A list of cues built by approximating word timing from text length.
    """
    start = float(seg.start)
    end = float(seg.end)
    text = (str(getattr(seg, "text", "")) or "").strip()
    if not text:
        return [Cue(index=0, start=start, end=end, text="")]

    tokens = text.split()
    total_chars = sum(len(t) for t in tokens) + max(0, len(tokens) - 1)
    if total_chars == 0:
        return [Cue(index=0, start=start, end=end, text=text)]

    cues: list[Cue] = []
    cur_tokens: list[str] = []
    cur_chars = 0
    cur_start = start

    def cur_duration_for(chars_count: int) -> float:
        return (chars_count / total_chars) * max(0.0, end - start)

    def flush() -> None:
        nonlocal cur_tokens, cur_chars, cur_start
        if not cur_tokens:
            return
        dur = cur_duration_for(cur_chars)
        t_text = _sanitize_cue_text(" ".join(cur_tokens))
        new_end = min(end, cur_start + dur)
        if t_text:
            cues.append(
                Cue(
                    index=0,
                    start=cur_start,
                    end=new_end,
                    text=t_text,
                )
            )
        # Advance current start regardless to keep timing consistent
        cur_start = new_end
        cur_tokens = []
        cur_chars = 0

    for tok in tokens:
        t_text = (" ".join([*cur_tokens, tok])).strip()
        t_chars = len(t_text)
        t_dur = cur_duration_for(t_chars)
        s_text = _sanitize_cue_text(t_text)
        t_cps = _cps(len(s_text), t_dur)
        t_lines = _count_wrapped_lines(s_text, max_line_chars)
        if t_dur > max_dur or t_cps > max_cps or t_lines > max_lines:
            flush()
            cur_tokens = [tok]
            cur_chars = len(tok)
        else:
            cur_tokens.append(tok)
            cur_chars = t_chars
    flush()

    # display buffer & gap respect
    for i in range(len(cues)):
        next_start = cues[i + 1].start if i + 1 < len(cues) else end
        new_end = min(
            cues[i].end + DEFAULT_SUB_DISPLAY_BUFFER_SEC,
            next_start - DEFAULT_SUB_MIN_GAP_SEC,
        )
        if new_end > cues[i].end:
            cues[i].end = new_end
    return cues


def build_cues_from_segments(segments: Sequence[object]) -> list[Cue]:
    """Build initial cues by splitting segments according to constraints.

    Returns:
        list[Cue]: A list of cues created from the provided segments.
    """
    cues: list[Cue] = []
    for seg in segments:
        if getattr(seg, "words", None):
            parts = _split_segment_by_words(
                seg,
                DEFAULT_SUB_MAX_SEGMENT_DURATION_SEC,
                DEFAULT_SUB_MAX_CPS,
                max_lines=2,
                max_line_chars=DEFAULT_SUB_MAX_LINE_CHARS,
            )
        else:
            parts = _split_segment_without_words(
                seg,
                DEFAULT_SUB_MAX_SEGMENT_DURATION_SEC,
                DEFAULT_SUB_MAX_CPS,
                max_lines=2,
                max_line_chars=DEFAULT_SUB_MAX_LINE_CHARS,
            )
        cues.extend(parts)
    # assign indices
    for i, c in enumerate(cues, 1):
        c.index = i
    return cues


def _maybe_merge(a: Cue, b: Cue) -> Cue | None:
    gap = max(0.0, b.start - a.end)
    merged_text = _sanitize_cue_text((a.text + " " + b.text).strip())
    if not merged_text:
        return None
    merged_dur = max(0.0, b.end - a.start)
    merged_chars = len(merged_text)
    if (
        (
            a.duration < DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC
            or _cps(a.chars, a.duration) > DEFAULT_SUB_MAX_CPS
        )
        and gap <= max(DEFAULT_SUB_MIN_GAP_SEC, DEFAULT_SUB_MIN_GAP_SEC * 2)
        and merged_dur <= DEFAULT_SUB_MAX_SEGMENT_DURATION_SEC
        and merged_chars <= DEFAULT_SUB_MAX_BLOCK_CHARS
    ):
        return Cue(index=0, start=a.start, end=b.end, text=merged_text)
    return None


def refine_cues(cues: Sequence[Cue]) -> list[Cue]:
    """Refine cues by merging, enforcing gaps, and ensuring minimum durations.

    Returns:
        list[Cue]: The refined list of cues with updated timings and indices.
    """
    if not cues:
        return []
    # 1) Merge pass
    merged: list[Cue] = []
    i = 0
    while i < len(cues):
        if i + 1 < len(cues):
            m = _maybe_merge(cues[i], cues[i + 1])
            if m is not None:
                # consume both
                cues = list(cues[:i]) + [m] + list(cues[i + 2 :])
                continue
        merged.append(cues[i])
        i += 1

    # 2) Enforce minimum gaps (shift starts forward if necessary)
    for j in range(1, len(merged)):
        prev = merged[j - 1]
        cur = merged[j]
        if cur.start - prev.end < DEFAULT_SUB_MIN_GAP_SEC:
            cur.start = prev.end + DEFAULT_SUB_MIN_GAP_SEC
            if cur.end < cur.start:
                cur.end = cur.start + DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC

    # 3) Ensure minimum duration by extending end when possible
    for j in range(len(merged)):
        cur = merged[j]
        if cur.duration < DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC and (
            j + 1 < len(merged)
        ):
            max_end = merged[j + 1].start - DEFAULT_SUB_MIN_GAP_SEC
            desired = cur.start + DEFAULT_SUB_MIN_SEGMENT_DURATION_SEC
            cur.end = min(max_end, max(cur.end, desired))

    # 4) Reindex
    for k, c in enumerate(merged, 1):
        c.index = k
    return merged


# -------------
# Renderers
# -------------


def _format_cue_text(text: str) -> str:
    # Greedy wrap then clamp to 2 lines with boundary-aware mid split
    lines = _wrap_text(
        _sanitize_cue_text(text).strip(),
        DEFAULT_SUB_MAX_LINE_CHARS,
    )
    if len(lines) <= 2:
        return "\n".join(lines)
    # Clamp: keep best two lines; attempt punctuation/soft word split around midpoint
    joined = " ".join(lines)
    # Simple heuristic: split in the middle space nearest boundary char/soft-word
    mid = len(joined) // 2
    best_idx = None
    for delta in range(0, max(1, len(joined) // 2)):
        for idx in (mid - delta, mid + delta):
            if 0 < idx < len(joined) and joined[idx] == " ":
                left = joined[:idx].rstrip()
                right = joined[idx + 1 :].lstrip()
                if (
                    (
                        _ends_with_boundary(left)
                        or any(
                            right.startswith(w + " ")
                            for w in DEFAULT_SUB_SOFT_BOUNDARY_WORDS
                        )
                    )
                    and len(left) <= DEFAULT_SUB_MAX_LINE_CHARS
                    and len(right) <= DEFAULT_SUB_MAX_LINE_CHARS
                ):
                    best_idx = idx
                    break
        if best_idx is not None:
            break
    if best_idx is None:
        # fallback: naive split
        best_idx = joined.rfind(" ", 0, mid)
        if best_idx == -1:
            best_idx = joined.find(" ", mid)
        if best_idx == -1:
            best_idx = mid
    left = joined[:best_idx].rstrip()
    right = joined[best_idx + 1 :].lstrip()
    return "\n".join([
        left[:DEFAULT_SUB_MAX_LINE_CHARS],
        right[:DEFAULT_SUB_MAX_LINE_CHARS],
    ])


def format_srt(cues: Sequence[Cue]) -> str:
    """Format cues into SRT string.

    Returns:
        str: The SRT-formatted subtitle content ending with a newline.
    """
    parts: list[str] = []
    for idx, c in enumerate(cues, 1):
        start = format_timestamp(c.start)
        end = format_timestamp(c.end)
        body = _format_cue_text(c.text)
        parts.append(f"{idx}\n{start} --> {end}\n{body}\n")
    return "\n".join(parts).rstrip() + "\n"


def format_vtt(cues: Sequence[Cue]) -> str:
    """Format cues into WebVTT string.

    Returns:
        str: The VTT-formatted subtitle content ending with a newline.
    """
    parts: list[str] = ["WEBVTT", ""]
    for c in cues:
        start = format_timestamp(c.start).replace(",", ".")
        end = format_timestamp(c.end).replace(",", ".")
        body = _format_cue_text(c.text)
        parts.append(f"{start} --> {end}\n{body}\n")
    return "\n".join(parts).rstrip() + "\n"
