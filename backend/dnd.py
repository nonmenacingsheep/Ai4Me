"""
Hearth — the Dungeons & Dragons world shared by you, Aitha, and the DM (an AI).

Everything persists to ~/.ai4me/dnd.json:
  - a customizable DM (name + persona)
  - many campaigns, each with: a summary, two character sheets (you + Aitha),
    a session memory (enemies/locations/setting — toggle-hideable), Aitha's
    per-campaign session notes, a battle board, turn state, and a play log.
"""

import json
import os
import random
import re
import time
import uuid

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
_PATH = os.path.join(_DIR, "dnd.json")

DEFAULT_DM = {
    "name": "The Keeper",
    "persona": (
        "A warm, theatrical Dungeon Master with a storyteller's heart. Paints vivid, "
        "sensory scenes, voices NPCs with character, plays fair by the rules, keeps the "
        "pace moving, and never railroads. Cozy by the hearth, but the stakes are real."
    ),
}

# A condensed 5e SRD-style primer the DM keeps in mind. (Not the full rulebook —
# the core mechanics most play actually needs.)
RULES_PRIMER = """\
CORE RULES (D&D 5e, condensed):
• Ability checks: roll d20 + ability modifier (+ proficiency if proficient) vs a Difficulty Class (DC). Easy 10, Medium 15, Hard 20.
• Modifiers: score 10–11 = +0, then +1 per 2 above (12=+1,14=+2,16=+3,18=+4,20=+5); below 10 is negative.
• Advantage: roll 2d20, keep higher. Disadvantage: keep lower.
• Saving throws: d20 + relevant ability mod vs a DC (e.g. Dexterity save vs a trap).
• Attacks: d20 + attack mod vs target's Armor Class (AC). On hit, roll the weapon/spell damage dice + mod. Natural 20 = critical hit (double the damage dice).
• Hit Points (HP): damage reduces HP; at 0 HP a creature is dying (death saving throws: d20, 10+ success, 3 successes stabilize, 3 failures die; nat 20 = regain 1 HP).
• Combat round (~6 seconds): roll initiative (d20+Dex) to set order. On your turn you get: one Move (up to speed), one Action, one Bonus Action (if available), and one Reaction per round.
• Common actions: Attack, Cast a Spell, Dash (extra move), Disengage (no opportunity attacks), Dodge (attacks vs you have disadvantage), Help, Hide (Stealth check), Ready, Search, Use an Object.
• Conditions: blinded, charmed, frightened, grappled, incapacitated, invisible, paralyzed, poisoned, prone, restrained, stunned, unconscious.
• Skills (and their ability): Athletics(Str); Acrobatics, Sleight of Hand, Stealth(Dex); Arcana, History, Investigation, Nature, Religion(Int); Animal Handling, Insight, Medicine, Perception, Survival(Wis); Deception, Intimidation, Performance, Persuasion(Cha).
• Resting: short rest (~1hr, spend Hit Dice to heal); long rest (~8hr, restore HP and most resources).
• Spellcasting: spells use slots of their level; cantrips are free. Concentration: some spells end if you take damage and fail a Con save.
Call for rolls when an outcome is uncertain and meaningful. Narrate results dramatically."""

_ROLES = ("dm", "aitha", "me")


def _blank_sheet(name: str) -> dict:
    return {
        "name": name, "race": "", "class": "", "level": 1, "background": "",
        "stats": {"str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10},
        "hp": {"cur": 10, "max": 10, "temp": 0}, "ac": 10, "speed": 30,
        "prof_bonus": 2, "initiative": 0,
        "skills": "", "inventory": "", "features": "", "spells": "", "notes": "",
    }


def _blank_campaign(name: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name or "New Campaign",
        "summary": "",
        "created": time.time(),
        "updated": time.time(),
        "sheets": {"me": _blank_sheet("You"), "aitha": _blank_sheet("Aitha")},
        "memory": [],          # [{id, text, category, hidden}]
        "session_notes": [],   # [{id, ts, author, text}]  (Aitha's per-campaign reflections)
        "board": {"enabled": False, "w": 14, "h": 10, "tokens": []},
        "turn": {"order": list(_ROLES), "active": "dm", "round": 1},
        "log": [],             # [{id, ts, who, kind, text, roll}]
    }


_DEFAULT = {"dm": dict(DEFAULT_DM), "active": None, "campaigns": {}}


# ─── persistence ──────────────────────────────────────────────────────────
def load() -> dict:
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**_DEFAULT, **data}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return json.loads(json.dumps(_DEFAULT))


def save(state: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _PATH)
    except OSError as e:
        print(f"[dnd] save failed: {e}")


# ─── helpers ──────────────────────────────────────────────────────────────
def active_campaign(state: dict) -> dict | None:
    cid = state.get("active")
    return state.get("campaigns", {}).get(cid) if cid else None


def new_campaign(state: dict, name: str) -> dict:
    camp = _blank_campaign(name)
    state.setdefault("campaigns", {})[camp["id"]] = camp
    state["active"] = camp["id"]
    return camp


def append_log(camp: dict, who: str, kind: str, text: str, roll: dict | None = None) -> dict:
    entry = {"id": uuid.uuid4().hex[:8], "ts": time.time(), "who": who,
             "kind": kind, "text": text, "roll": roll}
    camp.setdefault("log", []).append(entry)
    if len(camp["log"]) > 400:
        del camp["log"][: len(camp["log"]) - 400]
    camp["updated"] = time.time()
    return entry


# ─── dice ─────────────────────────────────────────────────────────────────
DICE_SIDES = (4, 6, 8, 10, 12, 20, 100)
_DICE_RE = re.compile(r"(\d*)\s*d\s*(\d+)\s*([+-]\s*\d+)?", re.I)


def roll_expr(expr: str) -> dict | None:
    """Roll a dice expression like 'd20', '2d6+3', 'd20+5'. Returns details + total."""
    m = _DICE_RE.search(expr or "")
    if not m:
        return None
    n = int(m.group(1) or 1)
    sides = int(m.group(2))
    mod = int((m.group(3) or "0").replace(" ", ""))
    n = max(1, min(n, 50))
    if sides < 2 or sides > 1000:
        return None
    rolls = [random.randint(1, sides) for _ in range(n)]
    return {"expr": (expr or "").strip(), "n": n, "sides": sides, "mod": mod,
            "rolls": rolls, "total": sum(rolls) + mod}


# ─── context for the AIs ──────────────────────────────────────────────────
def _fmt_sheet(label: str, s: dict) -> str:
    st = s.get("stats", {})
    stat_line = " ".join(f"{k.upper()} {st.get(k, 10)}" for k in ("str", "dex", "con", "int", "wis", "cha"))
    hp = s.get("hp", {})
    parts = [
        f"{label}: {s.get('name','?')} — {s.get('race','')} {s.get('class','')} (lvl {s.get('level',1)})",
        f"  {stat_line}",
        f"  HP {hp.get('cur',0)}/{hp.get('max',0)} (temp {hp.get('temp',0)})  AC {s.get('ac',10)}  Speed {s.get('speed',30)}",
    ]
    for field in ("skills", "inventory", "features", "spells", "notes"):
        val = (s.get(field) or "").strip()
        if val:
            parts.append(f"  {field.capitalize()}: {val[:300]}")
    return "\n".join(parts)


def _fmt_memory(camp: dict, include_hidden: bool) -> str:
    items = [m for m in camp.get("memory", []) if include_hidden or not m.get("hidden")]
    if not items:
        return "(none yet)"
    by_cat: dict[str, list] = {}
    for m in items:
        by_cat.setdefault(m.get("category", "misc"), []).append(m.get("text", ""))
    out = []
    for cat, texts in by_cat.items():
        out.append(f"  [{cat}] " + "; ".join(t for t in texts if t))
    return "\n".join(out)


def _fmt_board(camp: dict) -> str:
    b = camp.get("board", {})
    if not b.get("enabled"):
        return "(battle board off)"
    toks = b.get("tokens", [])
    if not toks:
        return f"(battle board on, {b.get('w',14)}x{b.get('h',10)}, empty)"
    lines = [f"(battle board {b.get('w',14)}x{b.get('h',10)}; grid coords x,y)"]
    for t in toks:
        lines.append(f"  {t.get('label','?')} [{t.get('kind','npc')}] at ({t.get('x',0)},{t.get('y',0)})")
    return "\n".join(lines)


def recent_log_text(camp: dict, limit: int = 16) -> str:
    out = []
    for e in camp.get("log", [])[-limit:]:
        who = {"dm": "DM", "aitha": "Aitha", "me": "Player"}.get(e.get("who"), e.get("who", "?"))
        if e.get("kind") == "roll" and e.get("roll"):
            r = e["roll"]
            out.append(f"{who} rolled {r['expr']} = {r['total']} {r.get('rolls')}")
        else:
            out.append(f"{who}: {e.get('text','')}")
    return "\n".join(out) or "(the story hasn't begun yet)"


def build_world(camp: dict, include_hidden_memory: bool = True) -> str:
    """The shared scene description handed to the DM (and to Aitha on her turn)."""
    turn = camp.get("turn", {})
    return (
        f"CAMPAIGN: {camp.get('name','')}\n"
        f"SUMMARY: {camp.get('summary','') or '(no summary yet)'}\n\n"
        f"{_fmt_sheet('PLAYER (him)', camp['sheets']['me'])}\n\n"
        f"{_fmt_sheet('AITHA', camp['sheets']['aitha'])}\n\n"
        f"SESSION MEMORY (enemies/locations/setting):\n{_fmt_memory(camp, include_hidden_memory)}\n\n"
        f"BATTLE BOARD:\n{_fmt_board(camp)}\n\n"
        f"TURN: round {turn.get('round',1)}, active = {turn.get('active','dm')}\n\n"
        f"RECENT PLAY:\n{recent_log_text(camp)}"
    )
