"""
Bedrock — a simple shared calendar, persisted to ~/.ai4me/calendar.json.

Events are plain dicts: a date (YYYY-MM-DD), an optional time (HH:MM, 24h), a
title and optional notes. He edits them in the Bedrock view (under Magma); Aitha
sees what's coming up so she's schedule-aware, and can jot one down herself via an
<event> directive when he mentions something with a date.
"""

import json
import os
import time
import uuid
from datetime import date, datetime, timedelta

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
PATH = os.path.join(_DIR, "calendar.json")


def load() -> list:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [e for e in data if isinstance(e, dict) and e.get("date") and e.get("title")] \
            if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save(events: list) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PATH)
    except OSError as e:
        print(f"[calendar] save failed: {e}")


def _norm_date(s: str) -> str | None:
    """Accept YYYY-MM-DD (and a few forgiving variants) → canonical YYYY-MM-DD."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _norm_time(s: str) -> str:
    """Accept HH:MM (24h) or h:MM AM/PM → 'HH:MM', else '' (all-day)."""
    s = (s or "").strip()
    if not s:
        return ""
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
        try:
            return datetime.strptime(s.upper(), fmt).strftime("%H:%M")
        except ValueError:
            continue
    return ""


def add(date_str: str, title: str, time_str: str = "", notes: str = "") -> dict | None:
    d = _norm_date(date_str)
    title = (title or "").strip()
    if not d or not title:
        return None
    ev = {
        "id": "e_" + uuid.uuid4().hex[:8],
        "date": d,
        "time": _norm_time(time_str),
        "title": title,
        "notes": (notes or "").strip(),
        "created": time.time(),
    }
    events = load()
    events.append(ev)
    save(events)
    return ev


def delete(event_id: str) -> bool:
    events = load()
    kept = [e for e in events if e.get("id") != event_id]
    if len(kept) != len(events):
        save(kept)
        return True
    return False


def _sort_key(e: dict) -> tuple:
    # All-day events (no time) sort before timed ones on the same day.
    return (e.get("date", ""), e.get("time") or "00:00")


def for_month(year: int, month: int) -> list:
    prefix = f"{year:04d}-{month:02d}-"
    return sorted((e for e in load() if str(e.get("date", "")).startswith(prefix)), key=_sort_key)


def upcoming(days: int = 14, limit: int = 12) -> list:
    today = date.today()
    end = today + timedelta(days=days)
    out = []
    for e in load():
        try:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if today <= d <= end:
            out.append(e)
    out.sort(key=_sort_key)
    return out[:limit]


def _pretty(e: dict) -> str:
    try:
        d = datetime.strptime(e["date"], "%Y-%m-%d").date()
    except (ValueError, KeyError):
        return e.get("title", "")
    today = date.today()
    delta = (d - today).days
    when = d.strftime("%a %b %d")
    if delta == 0:
        when = "today"
    elif delta == 1:
        when = "tomorrow"
    elif 0 < delta < 7:
        when = d.strftime("%A")  # this-week → weekday name
    t = e.get("time")
    if t:
        try:
            when += " at " + datetime.strptime(t, "%H:%M").strftime("%-I:%M %p")
        except ValueError:
            try:
                when += " at " + datetime.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")
            except ValueError:
                pass
    return f"{when} — {e.get('title','')}"


def digest(days: int = 14) -> str:
    """A schedule snapshot for her prompt: today's date plus what's coming up, so she
    can naturally reference plans ('you've got the dentist tomorrow'). '' if empty."""
    up = upcoming(days=days)
    today_str = date.today().strftime("%A, %B %d, %Y")
    if not up:
        return f"Today is {today_str}. His calendar has nothing coming up in the next {days} days."
    lines = [f"Today is {today_str}. Coming up on his calendar (next {days} days):"]
    for e in up:
        lines.append(f"• {_pretty(e)}" + (f"  ({e['notes']})" if e.get("notes") else ""))
    return "\n".join(lines)
