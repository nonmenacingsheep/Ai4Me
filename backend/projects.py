"""
Aitha's own goals & projects — things she chooses to pursue for herself over time
and returns to across days. Persisted to ~/.ai4me/projects.json so they outlive
restarts. Each project carries a short progress log she appends to as she advances.

She manages these conversationally via <project> / <advance> directives; the server
parses them and calls into here. The Mantle ("her mind") view reads them back.
"""

import json
import os
import time
import uuid

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
PATH = os.path.join(_DIR, "projects.json")

VALID_STATUS = ("active", "done", "shelved")
MAX_PROJECTS = 60          # storage ceiling (oldest shelved/done shed first)
MAX_LOG = 40               # progress entries kept per project


def load() -> list:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [p for p in data if isinstance(p, dict) and p.get("title")] if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save(projects: list) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PATH)
    except OSError as e:
        print(f"[projects] save failed: {e}")


def _find(projects: list, key: str) -> dict | None:
    """Match a project by id or (case-insensitive) title."""
    k = (key or "").strip().lower()
    if not k:
        return None
    for p in projects:
        if p.get("id", "").lower() == k or p.get("title", "").strip().lower() == k:
            return p
    return None


def _enforce_cap(projects: list) -> None:
    if len(projects) <= MAX_PROJECTS:
        return
    # Shed the least-recently-updated finished/shelved ones first; never active.
    finished = sorted((p for p in projects if p.get("status") != "active"),
                      key=lambda p: p.get("updated", 0))
    for p in finished:
        if len(projects) <= MAX_PROJECTS:
            break
        projects.remove(p)


def upsert(title: str, about: str | None = None, status: str | None = None,
           private: bool | None = None) -> tuple[dict, bool]:
    """Create a project, or update an existing one matched by title/id. Returns
    (project, created?). Only provided fields are changed on an update."""
    title = (title or "").strip()
    projects = load()
    p = _find(projects, title)
    now = time.time()
    created = False
    if p is None:
        p = {
            "id": "p_" + uuid.uuid4().hex[:8],
            "title": title,
            "about": (about or "").strip(),
            "status": "active",
            "private": bool(private) if private is not None else False,
            "created": now,
            "updated": now,
            "log": [],
        }
        projects.append(p)
        created = True
    else:
        if about:
            p["about"] = about.strip()
        if private is not None:
            p["private"] = bool(private)
    if status in VALID_STATUS:
        p["status"] = status
    p["updated"] = now
    _enforce_cap(projects)
    save(projects)
    return p, created


def advance(key: str, note: str, status: str | None = None) -> dict | None:
    """Append a progress entry to a project (and optionally change its status).
    Returns the project, or None if no match."""
    note = (note or "").strip()
    projects = load()
    p = _find(projects, key)
    if p is None:
        return None
    now = time.time()
    if note:
        p.setdefault("log", []).append({"time": time.strftime("%b %d, %I:%M %p"), "note": note})
        p["log"] = p["log"][-MAX_LOG:]
    if status in VALID_STATUS:
        p["status"] = status
    p["updated"] = now
    save(projects)
    return p


def set_status(key: str, status: str) -> dict | None:
    if status not in VALID_STATUS:
        return None
    projects = load()
    p = _find(projects, key)
    if p is None:
        return None
    p["status"] = status
    p["updated"] = time.time()
    save(projects)
    return p


def digest(limit_log: int = 2) -> str:
    """A compact snapshot of her projects for her prompt: active ones with their
    most recent progress, plus a tail of shelved/done titles. Includes private
    ones — this goes to HER, and she decides whether to bring them up with him."""
    projects = load()
    if not projects:
        return ""
    active = [p for p in projects if p.get("status") == "active"]
    done = [p for p in projects if p.get("status") == "done"]
    shelved = [p for p in projects if p.get("status") == "shelved"]
    active.sort(key=lambda p: -p.get("updated", 0))

    lines = [f"YOUR PROJECTS ({len(active)} active):"]
    for p in active[:12]:
        tag = " (private)" if p.get("private") else ""
        about = (p.get("about") or "").strip()
        lines.append(f'• "{p["title"]}"{tag} — {about}' if about else f'• "{p["title"]}"{tag}')
        for entry in (p.get("log") or [])[-limit_log:]:
            lines.append(f"    – {entry.get('time','')}: {entry.get('note','')}")
    if done:
        lines.append("Finished: " + ", ".join(f'"{p["title"]}"' for p in done[:8]))
    if shelved:
        lines.append("Shelved: " + ", ".join(f'"{p["title"]}"' for p in shelved[:8]))
    return "\n".join(lines)


def view() -> list:
    """Full project list for the Mantle (her-mind) view, newest activity first."""
    projects = load()
    projects.sort(key=lambda p: (p.get("status") != "active", -p.get("updated", 0)))
    out = []
    for p in projects:
        out.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "about": p.get("about", ""),
            "status": p.get("status", "active"),
            "private": bool(p.get("private")),
            "log": (p.get("log") or [])[-5:],
            "updated": p.get("updated", 0),
        })
    return out
