"""
Obsidian-style linked notes, stored as plain markdown files in ~/.ai4me/notes/.

Notes link to each other with [[Note Title]] wikilinks. A note's "backlinks"
are every other note that links to it.
"""

import os
import re

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me", "notes")
_LINK = re.compile(r"\[\[([^\]\[]+?)\]\]")
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(title: str) -> str:
    name = _INVALID.sub("_", title).strip().strip(".")[:120]
    return name or "Untitled"


def _path(title: str) -> str:
    return os.path.join(_DIR, _safe_filename(title) + ".md")


def _ensure_dir():
    os.makedirs(_DIR, exist_ok=True)


def list_notes() -> list[dict]:
    _ensure_dir()
    out = []
    for fn in os.listdir(_DIR):
        if not fn.endswith(".md"):
            continue
        p = os.path.join(_DIR, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append({"title": fn[:-3], "modified": st.st_mtime, "size": st.st_size})
    out.sort(key=lambda n: -n["modified"])
    return out


def read_note(title: str) -> str | None:
    p = _path(title)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None
    return None


def write_note(title: str, content: str) -> bool:
    _ensure_dir()
    try:
        with open(_path(title), "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except OSError as e:
        print(f"[notes] write failed: {e}")
        return False


def delete_note(title: str) -> bool:
    p = _path(title)
    if os.path.exists(p):
        try:
            os.remove(p)
            return True
        except OSError:
            return False
    return False


def outgoing_links(content: str) -> list[str]:
    seen, out = set(), []
    for m in _LINK.findall(content or ""):
        key = m.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(m.strip())
    return out


def context_digest(char_budget: int = 16000) -> str:
    """A snapshot of all notes (titles + contents) so Aitha can reference and edit them.
    The title list is always complete; contents fill up to char_budget (DeepSeek's
    window is large, so she can see essentially everything she's written)."""
    notes = list_notes()
    if not notes:
        return "He has no notes yet."
    titles = ", ".join(n["title"] for n in notes)
    parts = [f"His notes ({len(notes)}): {titles}"]
    budget = char_budget
    for n in notes:
        content = read_note(n["title"]) or ""
        block = f'\n\n=== NOTE: "{n["title"]}" ===\n{content}'
        if len(block) > budget:
            block = block[:budget] + "\n…(truncated)"
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            parts.append("\n\n…(more notes not shown)")
            break
    return "".join(parts)


def notes_menu() -> str:
    """A cheap 'table of contents' — just the titles. Always safe to inline in a
    prompt; she pulls the actual bodies on demand with fetch_notes()."""
    notes = list_notes()
    if not notes:
        return "He has no notes yet."
    return f"His notes ({len(notes)}): " + ", ".join(n["title"] for n in notes)


def fetch_notes(titles: list[str], char_budget: int = 16000) -> str:
    """Full contents for the requested titles (case-insensitive). Pass a title of
    'all' (or 'everything'/'*') to get every note. Returns a formatted block."""
    all_notes = list_notes()
    wants_all = any(t.strip().lower() in ("all", "everything", "*") for t in titles)
    if wants_all:
        chosen = [n["title"] for n in all_notes]
    else:
        by_lower = {n["title"].lower(): n["title"] for n in all_notes}
        chosen = []
        for t in titles:
            real = by_lower.get(t.strip().lower())
            if real and real not in chosen:
                chosen.append(real)
    if not chosen:
        return "(no matching notes found — check the exact title from the list)"
    parts, budget = [], char_budget
    for title in chosen:
        content = read_note(title) or ""
        block = f'\n\n=== NOTE: "{title}" ===\n{content}'
        if len(block) > budget:
            block = block[:budget] + "\n…(truncated)"
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            parts.append("\n\n…(more notes not shown — ask for them by name)")
            break
    return "".join(parts)


def backlinks(title: str) -> list[str]:
    target = title.strip().lower()
    res = []
    for n in list_notes():
        if n["title"] == title:
            continue
        content = read_note(n["title"]) or ""
        links = {l.strip().lower() for l in _LINK.findall(content)}
        if target in links:
            res.append(n["title"])
    return res
