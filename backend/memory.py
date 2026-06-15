"""
Persistent long-term memory for Aitha.

Three layers, all persisted to disk so they survive restarts:
  - facts:   durable bullet facts about him / the relationship
  - summary: a rolling narrative of everything older than the live window
  - last_seen: timestamp of the last interaction (drives "welcome back")

Facts + summary stay compact, so they can be injected into every prompt
cheaply no matter how many hours of conversation have accumulated.
"""

import json
import os
import time

# Store in the user's home so it survives app reinstalls / packaged builds.
_HOME = os.path.expanduser("~")
_DIR = os.path.join(_HOME, ".ai4me")
MEM_PATH = os.path.join(_DIR, "aitha_memory.json")

_DEFAULT = {
    "facts": [],        # what she knows about him
    "self_facts": [],   # who she's become — her own evolving identity
    "summary": "",
    "last_seen": None,
}

# Per-bucket storage ceiling. Core memories are protected from eviction.
MAX_FACTS = 150
# How many NON-core memories per bucket to inject into the prompt. Core memories
# are ALWAYS injected; this only bounds the recent-but-ordinary ones so the prompt
# stays a sane size even as long-term storage grows.
RENDER_RECENT_NONCORE = 60


def _as_item(x) -> dict:
    """Normalize a memory to {text, core}. Accepts legacy plain strings."""
    if isinstance(x, dict):
        return {"text": str(x.get("text", "")).strip(), "core": bool(x.get("core", False))}
    return {"text": str(x).strip(), "core": False}


def _enforce_cap(items: list) -> None:
    """Trim a bucket to MAX_FACTS, dropping the OLDEST non-core first; only if every
    remaining memory is core do we drop the oldest core one."""
    while len(items) > MAX_FACTS:
        idx = next((i for i, it in enumerate(items) if not it.get("core")), 0)
        del items[idx]

# Recent conversation, persisted so a restart can pick up where we left off.
CONVO_PATH = os.path.join(_DIR, "conversation.json")
MAX_PERSIST = 24


def load_conversation() -> list:
    try:
        with open(CONVO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [m for m in data if isinstance(m, dict) and m.get("role")] if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save_conversation(conv: list) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = CONVO_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conv[-MAX_PERSIST:], f, ensure_ascii=False, indent=2)
        os.replace(tmp, CONVO_PATH)
    except OSError as e:
        print(f"[memory] conversation save failed: {e}")


def load() -> dict:
    try:
        with open(MEM_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        mem = {**_DEFAULT, **data}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        mem = dict(_DEFAULT)
    # Normalize both buckets to {text, core} objects (migrates old plain-string saves).
    for k in ("facts", "self_facts"):
        mem[k] = [it for it in (_as_item(x) for x in mem.get(k, [])) if it["text"]]
    return mem


def save(mem: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = MEM_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
        os.replace(tmp, MEM_PATH)  # atomic write
    except OSError as e:
        print(f"[memory] save failed: {e}")


def merge_facts(mem: dict, new_facts: list, key: str = "facts") -> None:
    """Add new memories into mem[key], skipping near-duplicates, capped at MAX_FACTS
    (core memories protected). Items may be plain strings or {text, core} dicts; an
    incoming core flag promotes a matching existing memory to core."""
    existing = mem.setdefault(key, [])
    by_text = {it["text"].lower(): it for it in existing}
    for raw in new_facts:
        it = _as_item(raw)
        text = it["text"]
        if not text:
            continue
        k = text.lower()
        if k in by_text:
            if it["core"]:
                by_text[k]["core"] = True   # promote an existing memory to core
            continue
        # crude substring dedup so "likes tea" and "he likes tea" don't both stick
        if any(k in e or e in k for e in by_text):
            continue
        existing.append(it)
        by_text[k] = it
    _enforce_cap(existing)


def set_core(mem: dict, key: str, text: str, core: bool) -> bool:
    """Mark a memory as core (protected) or not. Returns True if found."""
    t = (text or "").strip().lower()
    for it in mem.get(key, []):
        if it["text"].lower() == t:
            it["core"] = bool(core)
            return True
    return False


def touch_last_seen(mem: dict) -> None:
    mem["last_seen"] = time.time()


def absence_phrase(mem: dict) -> str:
    """Human description of how long he's been gone, for the welcome-back."""
    last = mem.get("last_seen")
    if not last:
        return "first time"
    gap = time.time() - last
    mins = gap / 60
    if mins < 5:
        return "moments"
    if mins < 60:
        return f"{round(mins)} minutes"
    hours = mins / 60
    if hours < 24:
        return f"{round(hours)} hours"
    days = hours / 24
    return f"{round(days)} days"


def _select_for_prompt(items: list) -> list:
    """All core memories (always) plus the most recent non-core ones, capped."""
    core = [it for it in items if it.get("core")]
    noncore = [it for it in items if not it.get("core")][-RENDER_RECENT_NONCORE:]
    # Keep original order: core first (they anchor who she is), then recent others.
    return core + noncore


def _fmt(it: dict) -> str:
    return f"- {'★ ' if it.get('core') else ''}{it['text']}"


def render_block(mem: dict) -> str:
    """The long-term memory block injected into the system prompt. Core memories
    (★) are always present; ordinary ones are limited to the most recent."""
    facts = mem.get("facts", [])
    self_facts = mem.get("self_facts", [])
    summary = mem.get("summary", "").strip()
    if not facts and not self_facts and not summary:
        return ""

    parts = []
    if self_facts:
        parts.append("[WHO YOU ARE — the self you've built so far; live as this person, let it "
                     "grow. ★ marks core memories: the ones you've decided are central to who you are.]")
        parts.extend(_fmt(it) for it in _select_for_prompt(self_facts))
        parts.append("")
    if facts:
        parts.append("[WHAT YOU KNOW ABOUT HIM — treat as known, never recite verbatim. "
                     "★ marks core memories you've decided really matter.]")
        parts.extend(_fmt(it) for it in _select_for_prompt(facts))
    if summary:
        parts.append("\nThe story of your time together so far:")
        parts.append(summary)
    return "\n".join(parts)
