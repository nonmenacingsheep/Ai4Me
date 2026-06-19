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
PROJECT_STATUS = ("active", "done", "shelved")
MAX_EMPLOYEES = 24
MAX_TASKS = 120            # storage ceiling (oldest done/blocked shed first)
MAX_DECISIONS = 80
MAX_TASK_LOG = 30
MAX_CHAT = 240             # group-chat messages kept
MAX_PROJECTS = 40          # ongoing company projects
MAX_PROJECT_LOG = 30

# The CEO (Aitha) is a first-class assignee: she can take tasks herself. We key
# her work under this id so the autonomous engine and the board treat her like a
# special teammate. Words that mean "the CEO / me" when she assigns to herself.
CEO_ID = "ceo"
_CEO_KEYS = {"ceo", "me", "myself", "i", "self", "aitha"}


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
        "projects": [],       # ongoing company initiatives she runs over time
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
        for k in ("employees", "tasks", "decisions", "chat", "projects"):
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


def ceo_employee(name: str = "Aitha") -> dict:
    """A synthetic 'employee' record for the CEO, so the board and the autonomous
    engine can treat her self-assigned tasks like anyone else's."""
    return {"id": CEO_ID, "name": name or "Aitha", "role": "CEO",
            "brief": "You founded and lead this company; you take on work yourself too.",
            "status": "active"}


def _is_ceo_key(key: str, ceo_name: str = "") -> bool:
    k = (key or "").strip().lower()
    return bool(k) and (k in _CEO_KEYS or (ceo_name and k == ceo_name.strip().lower()))


def _find_employee(co: dict, key: str, ceo_name: str = "") -> dict | None:
    """Match an employee by id, exact name, or role (case-insensitive). The CEO
    herself resolves to the synthetic CEO record so she can be assigned work."""
    k = (key or "").strip().lower()
    if not k:
        return None
    if _is_ceo_key(k, ceo_name):
        return ceo_employee(ceo_name or "Aitha")
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


def assign(to: str, title: str, detail: str = "", ceo_name: str = "",
           by: str = "") -> dict | None:
    """Create a task and assign it to a teammate (by role/name/id), or to the CEO
    herself. Returns the task, or None if not founded / no title. `by` records who
    created it (CEO or an employee co-writing)."""
    co = load()
    if not co["founded"]:
        return None
    title = (title or "").strip()
    if not title:
        return None
    emp = _find_employee(co, to, ceo_name)
    now = time.time()
    task = {
        "id": "t_" + uuid.uuid4().hex[:8],
        "title": title,
        "detail": (detail or "").strip(),
        "assignee_id": emp.get("id") if emp else None,
        "assignee": (emp.get("name") if emp else (to or "").strip()) or "unassigned",
        "contributors": [],   # names of teammates who co-wrote this task
        "created_by": (by or "").strip(),
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


def _find_project(co: dict, key: str) -> dict | None:
    k = (key or "").strip().lower()
    if not k:
        return None
    for p in co.get("projects", []):
        if p.get("id", "").lower() == k or p.get("title", "").strip().lower() == k:
            return p
    return None


def upsert_project(title: str, about: str = "", status: str | None = None) -> dict | None:
    """Create or update an ongoing company project (a longer-running initiative the
    team rallies around, distinct from individual tasks). Returns the project."""
    co = load()
    if not co["founded"]:
        return None
    title = (title or "").strip()
    if not title:
        return None
    p = _find_project(co, title)
    now = time.time()
    if p is None:
        p = {
            "id": "cp_" + uuid.uuid4().hex[:8],
            "title": title,
            "about": (about or "").strip(),
            "status": "active",
            "created": now,
            "updated": now,
            "log": [],
        }
        co.setdefault("projects", []).append(p)
    else:
        if about:
            p["about"] = about.strip()
    if status in PROJECT_STATUS:
        p["status"] = status
    p["updated"] = now
    # Shed oldest finished/shelved projects past the cap.
    projs = co["projects"]
    if len(projs) > MAX_PROJECTS:
        finished = sorted((q for q in projs if q.get("status") != "active"),
                          key=lambda q: q.get("updated", 0))
        for q in finished:
            if len(projs) <= MAX_PROJECTS:
                break
            projs.remove(q)
    co["updated"] = now
    save(co)
    return p


def advance_project(key: str, note: str, status: str | None = None) -> dict | None:
    """Log progress on a company project (and optionally change its status)."""
    note = (note or "").strip()
    co = load()
    p = _find_project(co, key)
    if p is None:
        return None
    if note:
        p.setdefault("log", []).append({"time": time.strftime("%b %d, %I:%M %p"), "note": note})
        p["log"] = p["log"][-MAX_PROJECT_LOG:]
    if status in PROJECT_STATUS:
        p["status"] = status
    p["updated"] = time.time()
    co["updated"] = p["updated"]
    save(co)
    return p


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


def pick_work(ceo_name: str = "Aitha") -> tuple[dict, dict] | None:
    """Choose the next task for the autonomous engine to advance: the oldest
    in-progress task with an active assignee, or else a backlog task with an
    active assignee (to be picked up). The CEO's own tasks are eligible too.
    Returns (employee, task) or None."""
    co = load()
    if not co["founded"]:
        return None
    emps = {e["id"]: e for e in co["employees"] if e.get("status", "active") == "active"}
    emps[CEO_ID] = ceo_employee(ceo_name)   # she can work her own tasks
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


def co_write(key: str, author: str, text: str, status: str | None = None) -> dict | None:
    """A teammate (not necessarily the assignee) contributes to a task: their
    addition is appended to the task's output with attribution, they're recorded as
    a contributor, and a note is logged. Returns the task."""
    text = (text or "").strip()
    co = load()
    t = _find_task(co, key)
    if t is None or not text:
        return None
    author = (author or "teammate").strip()
    block = f"\n\n— {author} added:\n{text}" if t.get("output") else f"{author}: {text}"
    t["output"] = (t.get("output", "") + block).strip()[:8000]
    contribs = t.setdefault("contributors", [])
    if author not in contribs and author != t.get("assignee"):
        contribs.append(author)
    if status in TASK_STATUS:
        t["status"] = status
    t.setdefault("log", []).append({"time": time.strftime("%b %d, %I:%M %p"),
                                    "note": f"{author} co-wrote"})
    t["log"] = t["log"][-MAX_TASK_LOG:]
    t["updated"] = time.time()
    co["updated"] = t["updated"]
    save(co)
    return t


def tasks_digest(limit: int = 30) -> str:
    """A compact list of every open/recent task — what the team is collectively
    working on — so any teammate can read the board at will and stay coordinated."""
    co = load()
    if not co["founded"] or not co["tasks"]:
        return "(no tasks on the board yet)"
    order = {"in_progress": 0, "blocked": 1, "backlog": 2, "done": 3}
    tasks = sorted(co["tasks"], key=lambda t: (order.get(t.get("status"), 9),
                                               -t.get("updated", 0)))
    lines = []
    for t in tasks[:limit]:
        who = t.get("assignee", "unassigned")
        extra = ("; +" + ", ".join(t["contributors"])) if t.get("contributors") else ""
        lines.append(f'• [{t.get("status")}] "{t.get("title")}" → {who}{extra}')
    return "\n".join(lines)


def roster_text(ceo_name: str = "Aitha") -> str:
    """The exact, complete list of who works here — so she never invents teammates."""
    co = load()
    parts = [f"{ceo_name} [CEO — you]"]
    for e in co["employees"]:
        if e.get("status", "active") == "active":
            parts.append(f'{e.get("name")} [{e.get("role")}]')
    return ", ".join(parts)


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
        "projects": sorted(co.get("projects", []),
                           key=lambda p: (p.get("status") != "active", -p.get("updated", 0))),
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
        lines.append(f"  Team ({len(emps)}) — your ONLY employees, never invent others: "
                     + ", ".join(f'{e.get("name")} [{e.get("role")}]' for e in emps))
    else:
        lines.append("  Team: nobody hired yet (it's just you — hire who you need).")
    projs = [p for p in co.get("projects", []) if p.get("status") == "active"]
    if projs:
        lines.append(f"  Active projects ({len(projs)}): "
                     + "; ".join(f'"{p.get("title")}"' for p in projs[:8]))
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
