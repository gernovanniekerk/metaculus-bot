"""Deterministic parsers for the final answer lines the prompts ask for.

They read the LAST occurrence of each answer line, because the prompts require
the final answer to be the last thing written. If parsing fails, the bot makes
one cheap repair call (see bot.py) and parses again; if that fails too, the
ensemble member is dropped.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

PERCENTILES = (10, 20, 40, 60, 80, 90)

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"


class ParseError(ValueError):
    pass


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def parse_binary(text: str) -> float:
    """'Probability: 37%' -> 0.37, clamped to [0.01, 0.99]."""
    matches = re.findall(
        rf"probability\**\s*[:=]\s*\**\s*({_NUMBER})\s*(%?)", text, flags=re.IGNORECASE
    )
    if not matches:
        raise ParseError("No 'Probability: ZZ%' line found")
    number, percent = matches[-1]
    value = _to_float(number)
    if not percent and value <= 1.0:
        value *= 100.0
    if not 0.0 <= value <= 100.0 or math.isnan(value):
        raise ParseError(f"Probability {value} is outside 0-100")
    return max(0.01, min(0.99, value / 100.0))


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """One 'Option: PP%' line per option -> normalised probabilities."""
    raw: dict[str, float] = {}
    percent_flags: list[bool] = []
    for option in options:
        pattern = (
            r"^[\s>*\-•]*(?:option\s*)?[\"'“”*]*"
            + re.escape(option.strip())
            + rf"[\"'“”*]*\s*[:=\-–]\s*\**\s*({_NUMBER})\s*(%?)"
        )
        found = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not found:
            raise ParseError(f"No probability line for option '{option}'")
        number, percent = found[-1]
        raw[option] = _to_float(number)
        percent_flags.append(bool(percent))

    total = sum(raw.values())
    if not any(percent_flags) and 0.9 <= total <= 1.1:
        scaled = dict(raw)  # given as decimals
    else:
        scaled = {k: v / 100.0 for k, v in raw.items()}
    total = sum(scaled.values())
    if any(v < 0 for v in scaled.values()):
        raise ParseError("Negative option probability")
    if not 0.9 <= total <= 1.1:
        raise ParseError(f"Option probabilities sum to {total:.3f}, not about 1")
    floor = 0.005
    floored = {k: max(v / total, floor) for k, v in scaled.items()}
    norm = sum(floored.values())
    return {k: v / norm for k, v in floored.items()}


def _percentile_lines(text: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for match in re.finditer(
        r"^[\s>*\-•]*percentile\s*(\d{1,2})\**\s*[:=]\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        p = int(match.group(1))
        if p in PERCENTILES:
            found[p] = match.group(2)  # later lines overwrite earlier ones
    missing = [p for p in PERCENTILES if p not in found]
    if missing:
        raise ParseError(f"Missing percentile lines: {missing}")
    return found


def parse_numeric_percentiles(text: str) -> list[tuple[float, float]]:
    """-> [(0.10, value), ..., (0.90, value)], values non-decreasing."""
    lines = _percentile_lines(text)
    result: list[tuple[float, float]] = []
    for p in PERCENTILES:
        cleaned = lines[p].replace("$", "").replace("*", "").strip()
        m = re.search(_NUMBER, cleaned)
        if not m:
            raise ParseError(f"No number on percentile {p} line: '{lines[p]}'")
        result.append((p / 100.0, _to_float(m.group(0))))
    values = [v for _, v in result]
    if any(b < a for a, b in zip(values, values[1:])):
        raise ParseError(f"Percentile values are not increasing: {values}")
    return result


def parse_date_percentiles(text: str) -> list[tuple[float, datetime]]:
    """-> [(0.10, datetime_utc), ...], dates non-decreasing."""
    lines = _percentile_lines(text)
    result: list[tuple[float, datetime]] = []
    for p in PERCENTILES:
        m = re.search(
            r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?Z?)?", lines[p]
        )
        if not m:
            raise ParseError(f"No YYYY-MM-DD date on percentile {p} line: '{lines[p]}'")
        day = datetime.strptime(m.group(1), "%Y-%m-%d")
        hour = int(m.group(2) or 0)
        minute = int(m.group(3) or 0)
        second = int(m.group(4) or 0)
        result.append(
            (
                p / 100.0,
                day.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc),
            )
        )
    dates = [d for _, d in result]
    if any(b < a for a, b in zip(dates, dates[1:])):
        raise ParseError("Percentile dates are not in chronological order")
    return result
