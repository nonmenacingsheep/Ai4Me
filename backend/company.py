"""
Aitha's company — she's the CEO, you're the Chairman who guides her.

A single company she founds and runs over time: a mission, a roster of AI
"employees" she hires, a board of tasks she assigns, and a log of the decisions
she makes. Persisted to ~/.ai4me/company.json so the whole venture outlives
restarts, like the rest of her data.

She runs it conversationally through directives — <founded>, <hire>, <assign>,
<decision>, <costatus> — which the server parses and routes in here. The Helm
view reads it all back. (Phase 2 adds an autonomous engine where her employees
generate real work via her brain; this module is the data spine it builds on.)
"""

import json
import os
import time
import uuid

_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
PATH = os.path.join(_DIR, "company.json")

TASK_STATUS = ("backlog", "in_progress", "done", "blocked")
MAX_EMPLOYEES = 24
MAX_TASKS = 120            # storage ceiling (oldest done/blocked shed first)
MAX_DECISIONS = 80
MAX_TASK_LOG = 30
MAX_CHAT = 240             # group-chat messages kept


def _blank() -> dict:
    return {
        "founded": False,
        "name": "",
        "mission": "",
        "industry": "",
        "created": 0.0,
        "updated": 0.0,
        "heartbeat": False,   # autonomous timer: team acts on its own when True
        "employees": [],
        "tasks": [],
        "decisions": [],
        "chat": [],           # company group chat (CEO + employees + Chairman)
    }


def load() -> dict:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _blank()
        base = _blank()
        base.update(data)
        # Defensive: make sure the list fields are actually lists.
        for k in ("employees", "tasks", "decisions", "chat"):
            if not isinstance(base.get(k), list):
                base[k] = []
        base["heartbeat"] = bool(base.get("heartbeat", False))
        return base
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return _blank()


def save(co: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(co, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PATH)
    except OSError as e:
        print(f"[company] save failed: {e}")


def _find_employee(co: dict, key: str) -> dict | None:
    """Match an employee by id, exact name, or role (case-insensitive)."""
    k = (key or "").strip().lower()
    if not k:
        return None
    for e in co["employees"]:
        if e.get("id", "").lower() == k or e.get("name", "").strip().lower() == k \
           or e.get("role", "").strip().lower() == k:
            return e
    return None


def _find_task(co: dict, key: str) -> dict | None:
    """Match a task by id or (case-insensitive) title."""
    k = (key or "").strip().lower()
    if not k:
        return None
    for t in co["tasks"]:
        if t.get("id", "").lower() == k or t.get("title", "").strip().lower() == k:
            return t
    return None


def found(name: str, mission: str = "", industry: str = "") -> dict:
    """Establish (or rename/redefine) the company. There's only ever one."""
    co = load()
    name = (name or "").strip()
    if name:
        co["name"] = name
    if mission:
        co["mission"] = mission.strip()
    if industry:
        co["industry"] = industry.strip()
    now = time.time()
    if not co["founded"]:
        co["founded"] = True
        co["created"] = now
    co["updated"] = now
    save(co)
    return co


def hire(role: str, name: str = "", brief: str = "") -> dict | None:
    """Add an employee to the roster. Returns the employee, or None if no room /
    not founded yet. Re-hiring the same role+name updates their brief instead."""
    co = load()
    if not co["founded"]:
        return None
    role = (role or "").strip()
    name = (name or "").strip()
    if not role and not name:
        return None
    # Update in place if this person already exists.
    existing = _find_employee(co, name) or _find_employee(co, role)
    now = time.time()
    if existing:
        if role:
            existing["role"] = role
        if name:
            existing["name"] = name
        if brief:
            existing["brief"] = brief.strip()
        existing["updated"] = now
        co["updated"] = now
        save(co)
        return existing
    if len(co["employees"]) >= MAX_EMPLOYEES:
        return None
    emp = {
        "id": "e_" + uuid.uuid4().hex[:8],
        "name": name or role,
        "role": role or "team",
        "brief": (brief or "").strip(),
        "status": "active",
        "hired": now,
        "updated": now,
    }
    co["employees"].append(emp)
    co["updated"] = now
    save(co)
    return emp


def assign(to: str, title: str, detail: str = "") -> dict | None:
    """Create a task and assign it to an employee (by role/name/id). Returns the
    task, or None if not founded / no title."""
    co = load()
    if not co["founded"]:
        return None
    title = (title or "").strip()
    if not title:
        return None
    emp = _find_employee(co, to)
    now = time.time()
    task = {
        "id": "t_" + uuid.uuid4().hex[:8],
        "title": title,
        "detail": (detail or "").strip(),
        "assignee_id": emp.get("id") if emp else None,
        "assignee": (emp.get("name") if emp else (to or "").strip()) or "unassigned",
        "status": "in_progress" if emp else "backlog",
        "output": "",
        "log": [],
        "created": now,
        "updated": now,
    }
    co["tasks"].append(task)
    _enforce_task_cap(co)
    co["updated"] = now
    save(co)
    return task


def set_task_status(key: str, status: str, note: str = "") -> dict | None:
    """Move a task across the board and optionally log a note. Returns the task."""
    co = load()
    t = _find_task(co, key)
    if t is None:
        return None
    now = time.time()
    if status in TASK_STATUS:
        t["status"] = status
    if note:
        t.setdefault("log", []).append({"time": time.strftime("%b %d, %I:%M %p"),
                                        "note": note.strip()})
        t["log"] = t["log"][-MAX_TASK_LOG:]
    t["updated"] = now
    co["updated"] = now
    save(co)
    return t


def decide(summary: str, rationale: str = "") -> dict | None:
    """Log a CEO decision. Returns the decision entry."""
    co = load()
    if not co["founded"]:
        return None
    summary = (summary or "").strip()
    if not summary:
        return None
    now = time.time()
    d = {
        "id": "d_" + uuid.uuid4().hex[:8],
        "summary": summary,
        "rationale": (rationale or "").strip(),
        "time": time.strftime("%b %d, %I:%M %p"),
        "created": now,
    }
    co["decisions"].append(d)
    co["decisions"] = co["decisions"][-MAX_DECISIONS:]
    co["updated"] = now
    save(co)
    return d


def _enforce_task_cap(co: dict) -> None:
    if len(co["tasks"]) <= MAX_TASKS:
        return
    finished = sorted((t for t in co["tasks"] if t.get("status") in ("done", "blocked")),
                      key=lambda t: t.get("updated", 0))
    for t in finished:
        if len(co["tasks"]) <= MAX_TASKS:
            break
        co["tasks"].remove(t)


def set_heartbeat(on: bool) -> dict:
    """Turn the autonomous timer on/off. When on, the team works + talks on its own."""
    co = load()
    co["heartbeat"] = bool(on)
    co["updated"] = time.time()
    save(co)
    return co


def heartbeat_on() -> bool:
    return bool(load().get("heartbeat", False))


def post_message(author_id: str | None, author: str, role: str, text: str) -> dict | None:
    """Append a message to the company group chat. author_id is an employee id, or
    'ceo' (Aitha) / 'chairman' (him). Returns the message."""
    text = (text or "").strip()
    if not text:
        return None
    co = load()
    now = time.time()
    msg = {
        "id": "m_" + uuid.uuid4().hex[:8],
        "author_id": author_id,
        "author": author,
        "role": role,
        "text": text[:2000],
        "time": time.strftime("%I:%M %p"),
        "created": now,
    }
    co["chat"].append(msg)
    co["chat"] = co["chat"][-MAX_CHAT:]
    co["updated"] = now
    save(co)
    return msg


def recent_chat(n: int = 14) -> list[dict]:
    """The last n group-chat messages, oldest first."""
    return load().get("chat", [])[-n:]


def pick_work() -> tuple[dict, dict] | None:
    """Choose the next task for the autonomous engine to advance: the oldest
    in-progress task with an active assignee, or else a backlog task with an
    active assignee (to be picked up). Returns (employee, task) or None."""
    co = load()
    if not co["founded"]:
        return None
    emps = {e["id"]: e for e in co["employees"] if e.get("status", "active") == "active"}
    if not emps:
        return None
    in_prog = [t for t in co["tasks"]
               if t.get("status") == "in_progress" and t.get("assignee_id") in emps]
    in_prog.sort(key=lambda t: t.get("updated", 0))
    if in_prog:
        t = in_prog[0]
        return emps[t["assignee_id"]], t
    backlog = [t for t in co["tasks"]
               if t.get("status") == "backlog" and t.get("assignee_id") in emps]
    backlog.sort(key=lambda t: t.get("created", 0))
    if backlog:
        t = backlog[0]
        return emps[t["assignee_id"]], t
    return None


def record_work(task_id: str, output: str, status: str | None = None,
                note: str | None = None) -> dict | None:
    """Save an employee's work product onto a task: set its latest output, move its
    status, and log a short note. Returns the task."""
    co = load()
    t = _find_task(co, task_id)
    if t is None:
        return None
    now = time.time()
    if output:
        t["output"] = output.strip()[:6000]
    if status in TASK_STATUS:
        t["status"] = status
    if note:
        t.setdefault("log", []).append({"time": time.strftime("%b %d, %I:%M %p"),
                                        "note": note.strip()[:280]})
        t["log"] = t["log"][-MAX_TASK_LOG:]
    t["updated"] = now
    co["updated"] = now
    save(co)
    return t


def view() -> dict:
    """Full company snapshot for the Helm view."""
    co = load()
    tasks = sorted(co["tasks"], key=lambda t: -t.get("updated", 0))
    return {
        "founded": co["founded"],
        "name": co["name"],
        "mission": co["mission"],
        "industry": co["industry"],
        "created": co["created"],
        "heartbeat": bool(co.get("heartbeat", False)),
        "employees": sorted(co["employees"], key=lambda e: e.get("hired", 0)),
        "tasks": tasks,
        "decisions": list(reversed(co["decisions"]))[:30],
        "chat": co.get("chat", [])[-80:],
        "counts": {
            "employees": len(co["employees"]),
            "backlog": sum(1 for t in co["tasks"] if t.get("status") == "backlog"),
            "in_progress": sum(1 for t in co["tasks"] if t.get("status") == "in_progress"),
            "done": sum(1 for t in co["tasks"] if t.get("status") == "done"),
            "blocked": sum(1 for t in co["tasks"] if t.get("status") == "blocked"),
        },
    }


def digest() -> str:
    """A compact snapshot of the company for her prompt, so she runs it with
    continuity across turns. Empty string until she's founded it."""
    co = load()
    if not co["founded"]:
        return ""
    lines = [f'YOUR COMPANY — "{co["name"]}" (you are CEO)']
    if co.get("industry"):
        lines.append(f"  Industry: {co['industry']}")
    if co.get("mission"):
        lines.append(f"  Mission: {co['mission']}")
    emps = co["employees"]
    if emps:
        lines.append(f"  Team ({len(emps)}): "
                     + ", ".join(f'{e.get("name")} [{e.get("role")}]' for e in emps[:12]))
    active = [t for t in co["tasks"] if t.get("status") in ("in_progress", "backlog")]
    blocked = [t for t in co["tasks"] if t.get("status") == "blocked"]
    done = [t for t in co["tasks"] if t.get("status") == "done"]
    if active:
        lines.append(f"  Open tasks ({len(active)}):")
        for t in sorted(active, key=lambda t: -t.get("updated", 0))[:8]:
            lines.append(f'    – "{t["title"]}" → {t.get("assignee","unassigned")} [{t["status"]}]')
    if blocked:
        lines.append("  Blocked: " + ", ".join(f'"{t["title"]}"' for t in blocked[:5]))
    if done:
        lines.append(f"  Shipped: {len(done)} task(s)")
    chat = co.get("chat", [])
    if chat:
        lines.append("  Recent team group-chat:")
        for m in chat[-5:]:
            lines.append(f"    {m.get('author')}: {m.get('text','')[:120]}")
    return "\n".join(lines)
