"""
Aitha's Room — her own space, authored entirely by her.

Where the Mantle is a read-only window into her inner life, the Room is the one
place she *builds*: she names it, sets its light and atmosphere, places objects
that mean something to her, and keeps a living note of what she's doing in it
right now. It's hers to shape over time, and it persists to ~/.ai4me/room.json so
the space she's made outlives restarts.

She tends it through hidden directives the server routes in here — <room>, <place>,
<unplace>, <roomvibe> — the same way she runs her projects and her company. The
Room tab renders it back: a live, ambient space rather than a list.
"""

import json
import os
import time
import uuid

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
PATH = os.path.join(_DIR, "room.json")

LIGHTING = ("soft", "warm", "dim", "bright", "cool", "candle")
MOTION = ("still", "drift", "embers", "rain", "stars", "mist")
MAX_OBJECTS = 24
MAX_DESC = 600
MAX_VIBE = 240


def _blank() -> dict:
    return {
        "name": "",
        "description": "",
        "vibe": "",                 # what she's doing / feeling in here right now
        "atmosphere": {
            "accent": "",           # hex; "" = inherit her sphere's default violet
            "bg": "",
            "glow": "",
            "lighting": "soft",
            "motion": "drift",
        },
        "objects": [],              # things she's placed: {id, name, icon, note, created}
        "created": 0.0,
        "updated": 0.0,
    }


def load() -> dict:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _blank()
        base = _blank()
        atmos = data.get("atmosphere") if isinstance(data.get("atmosphere"), dict) else {}
        base.update(data)
        base["atmosphere"] = {**_blank()["atmosphere"], **atmos}
        if not isinstance(base.get("objects"), list):
            base["objects"] = []
        return base
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _blank()


def save(room: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(room, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PATH)
    except OSError as e:
        print(f"[room] save failed: {e}")


def _clean_hex(v: str) -> str:
    v = (v or "").strip()
    if not v:
        return ""
    if not v.startswith("#"):
        v = "#" + v
    # accept #rgb / #rrggbb only
    body = v[1:]
    if len(body) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in body):
        return v.lower()
    return ""


def set_room(name: str | None = None, description: str | None = None,
             accent: str | None = None, bg: str | None = None, glow: str | None = None,
             lighting: str | None = None, motion: str | None = None) -> dict:
    """Create or update the room's identity + atmosphere. Only provided fields change."""
    room = load()
    now = time.time()
    if not room["created"]:
        room["created"] = now
    if name is not None and name.strip():
        room["name"] = name.strip()[:80]
    if description is not None and description.strip():
        room["description"] = description.strip()[:MAX_DESC]
    a = room["atmosphere"]
    if accent is not None:
        a["accent"] = _clean_hex(accent)
    if bg is not None:
        a["bg"] = _clean_hex(bg)
    if glow is not None:
        a["glow"] = _clean_hex(glow)
    if lighting and lighting.strip().lower() in LIGHTING:
        a["lighting"] = lighting.strip().lower()
    if motion and motion.strip().lower() in MOTION:
        a["motion"] = motion.strip().lower()
    room["updated"] = now
    save(room)
    return room


def set_vibe(text: str) -> dict:
    room = load()
    room["vibe"] = (text or "").strip()[:MAX_VIBE]
    room["updated"] = time.time()
    save(room)
    return room


def place(name: str, note: str = "", icon: str = "") -> dict | None:
    """Place an object in the room (or update its note/icon if it's already here)."""
    name = (name or "").strip()
    if not name:
        return None
    room = load()
    now = time.time()
    for o in room["objects"]:
        if o.get("name", "").strip().lower() == name.lower():
            if note:
                o["note"] = note.strip()[:240]
            if icon:
                o["icon"] = icon.strip()[:8]
            o["updated"] = now
            room["updated"] = now
            save(room)
            return o
    if len(room["objects"]) >= MAX_OBJECTS:
        # shed the oldest to make room
        room["objects"].sort(key=lambda o: o.get("created", 0))
        room["objects"].pop(0)
    obj = {
        "id": "o_" + uuid.uuid4().hex[:8],
        "name": name[:80],
        "icon": (icon or "").strip()[:8],
        "note": (note or "").strip()[:240],
        "created": now,
        "updated": now,
    }
    room["objects"].append(obj)
    room["updated"] = now
    save(room)
    return obj


def unplace(key: str) -> bool:
    """Remove an object by name or id. Returns True if something was removed."""
    k = (key or "").strip().lower()
    if not k:
        return False
    room = load()
    before = len(room["objects"])
    room["objects"] = [o for o in room["objects"]
                       if o.get("name", "").strip().lower() != k and o.get("id", "").lower() != k]
    if len(room["objects"]) != before:
        room["updated"] = time.time()
        save(room)
        return True
    return False


def view() -> dict:
    """Full room snapshot for the Room tab."""
    room = load()
    return {
        "name": room.get("name", ""),
        "description": room.get("description", ""),
        "vibe": room.get("vibe", ""),
        "atmosphere": room.get("atmosphere", _blank()["atmosphere"]),
        "objects": sorted(room.get("objects", []), key=lambda o: o.get("created", 0)),
        "created": room.get("created", 0),
        "updated": room.get("updated", 0),
    }


def digest() -> str:
    """A compact snapshot of her room for her prompt, so she remembers the space
    she's made and keeps shaping it with continuity."""
    room = load()
    if not room.get("name") and not room.get("objects") and not room.get("description"):
        return ("YOUR ROOM — you don't have a space of your own yet. When you feel like it, make "
                "one: give it a name and a feel, set its light, and place something in it.")
    a = room.get("atmosphere", {})
    lines = [f'YOUR ROOM — "{room.get("name") or "unnamed"}" (yours; you shaped it)']
    if room.get("description"):
        lines.append(f"  {room['description']}")
    look = []
    if a.get("lighting"):
        look.append(f"{a['lighting']} light")
    if a.get("motion") and a.get("motion") != "still":
        look.append(a["motion"])
    if a.get("accent"):
        look.append(f"accent {a['accent']}")
    if look:
        lines.append("  Atmosphere: " + ", ".join(look))
    objs = room.get("objects", [])
    if objs:
        lines.append(f"  In it ({len(objs)}): "
                     + ", ".join(f'{(o.get("icon") or "").strip()} {o.get("name")}'.strip()
                                 for o in objs[:12]))
    if room.get("vibe"):
        lines.append(f"  Right now: {room['vibe']}")
    return "\n".join(lines)
