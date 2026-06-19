import asyncio
import json
import os
import random
import re as _re
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Allow running from project root or from backend/ directory
sys.path.insert(0, os.path.dirname(__file__))
from brain import AithaBrain, get_char_name, set_char_name as brain_set_name
from context import gather as gather_context
from context import (
    speak_probability, PROACTIVE_TICK_SECONDS,
    journal_pressure, JOURNAL_TICK_SECONDS,
    curiosity_pressure, CURIOSITY_TICK_SECONDS,
    _derive_mood, _derive_mood_key, mood_is_late,
)
import memory as mem_store
import notes as notes_store
import projects as projects_store
import company as company_store
import files as files_store
import events as events_store
import spotify as spotify_store
import settings_store
import dnd as dnd_store
import coder
from tts import engine as tts
from stt import engine as stt

# She may speak unprompted unless disabled.
PROACTIVE_ENABLED = os.getenv("AITHA_PROACTIVE", "1").lower() not in ("0", "false", "no")

# She may wander off and explore the web on her own unless disabled.
CURIOSITY_ENABLED = os.getenv("AITHA_CURIOSITY", "1").lower() not in ("0", "false", "no")
# How many searches she may chain in a single outing before she must wrap up.
EXPLORE_MAX_STEPS = int(os.getenv("AITHA_EXPLORE_MAX_STEPS", "8"))

# Her "heartbeat": how often she checks in when resting, how quickly she gets her
# next turn once she's already moving (flow), and a safety cap on a flow burst.
HEARTBEAT_SECONDS = int(os.getenv("AITHA_HEARTBEAT", "40"))
FLOW_GAP_SECONDS = int(os.getenv("AITHA_FLOW_GAP", "6"))
MAX_FLOW_BURST = int(os.getenv("AITHA_MAX_FLOW", "3"))
# How many times she may open notes (<readnote>) and re-run within a single turn
# before she must answer with what she's got — guards against a fetch loop.
MAX_NOTE_FETCH = int(os.getenv("AITHA_MAX_NOTE_FETCH", "2"))
# How many live-web fetches (<search> / <watch>) she may chain in one turn before she
# must answer with what she's gathered — bounds token spend and fetch loops.
MAX_WEB_FETCH = int(os.getenv("AITHA_MAX_WEB_FETCH", "3"))
# How many local file/folder reads (<browse> / <readfile>) she may chain in one turn.
MAX_FILE_FETCH = int(os.getenv("AITHA_MAX_FILE_FETCH", "4"))
# How many music lookups (<music>) she may chain in one turn.
MAX_MUSIC_FETCH = int(os.getenv("AITHA_MAX_MUSIC_FETCH", "3"))
# How many code rounds (<code>/<run>/<lscode>/<readcode>) she may chain in one turn
# before she must answer with what she found — bounds run loops.
MAX_CODE_RUNS = int(os.getenv("AITHA_MAX_CODE_RUNS", "4"))

# Memory upkeep (runs in her idle loop). Decay-pruning is cheap and runs hourly;
# the LLM consolidation pass is gated: only when he's been away a while, only every
# few hours, and only once a bucket has enough non-core memories to be worth tidying.
MEM_DECAY_EVERY = int(os.getenv("AITHA_MEM_DECAY_EVERY", str(3600)))          # 1h
MEM_CONSOLIDATE_EVERY = int(os.getenv("AITHA_MEM_CONSOLIDATE_EVERY", str(6 * 3600)))  # 6h
MEM_CONSOLIDATE_MIN = int(os.getenv("AITHA_MEM_CONSOLIDATE_MIN", "70"))       # non-core count
MEM_AWAY_SECONDS = int(os.getenv("AITHA_MEM_AWAY", str(600)))                 # "he's idle"

# Prepended to her context once she's actually fetched something, so she reads the
# results right (a search is just pointers; a transcript is the real video contents).
_WEB_RESULTS_HEADER = (
    "WEB RESULTS YOU PULLED THIS TURN (don't re-pull the same item):\n"
    "• A [WEB SEARCH] block is just a LIST OF POINTERS — titles, snippets, and video ids. "
    "It is NOT the contents of any page or video. To learn what's IN a video you still have "
    "to <watch> one of the ids from the list — you have NOT seen a video just because it "
    "appeared in search.\n"
    "• A [YOUTUBE TRANSCRIPT] block IS the actual words of that video — you've now watched "
    "it; answer from it directly and in your own voice. It carries [m:ss] time markers, so "
    "you can say roughly when something is said (the nearest marker before it)."
)
_CODE_RESULTS_HEADER = (
    "CODE YOU RAN THIS TURN (in your own Python workspace):\n"
    "• [WROTE name] confirms a file was saved. [RAN name] shows its exit code and "
    "captured stdout/stderr — that IS what actually happened, so answer from it. If it "
    "errored, read the traceback, fix the file with <code>, and <run> it again.\n"
    "• Once you've seen the output, tell him what you found or built in your own voice — "
    "don't just say you'll run it, and don't re-run something that already worked."
)
# She sometimes makes a PROMISE ("let me look it up", "let me grab it") without
# following through — either forgetting to emit the tag, or, after she's fetched,
# saying she'll report back without actually reporting. Catch the promise so we can
# nudge her to follow through instead of leaving him hanging.
_PROMISE_INTENT = _re.compile(
    r"\b(?:let me|lemme|i'?ll|i will|gonna|going to|hang on|hold on|give me a|one (?:sec|second|moment))\b"
    r"[^.!?\n]{0,40}"
    r"\b(?:look|search|check|find|google|watch|pull|grab|get|see|share|tell you|report|read|dig|track)\b",
    _re.I,
)
# How many AI turns (DM ↔ Aitha) may chain before pausing for the player.
MAX_AI_VOLLEY = int(os.getenv("AITHA_HEARTH_VOLLEY", "8"))

MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

brain: AithaBrain | None = None
clients: list[WebSocket] = []
_context_cache: dict = {}
_context_lock = asyncio.Lock()

# Single-user long-term memory, shared across (re)connections for this process.
memory: dict = {}
_memory_lock = asyncio.Lock()
_greeted = False  # greet only once per app launch, not on every reconnect

# The shared chat theme (she and he both change it; persisted). When HE changes
# it, we flag it so her next turn notices and can react.
theme_state: dict = {"preset": "default", "accent": None, "bg": None, "orb": None}
_theme_changed_by_him = False

# When he last spoke to her (module-level mirror of the per-connection timer) so
# read-only views like the Mantle/Mind tab can report her current mood.
last_chat_ts_g: float | None = None

# One shared conversation across the chat view AND the Magma notes chat, so
# Aitha stays consistent everywhere. Persists across reconnects (in-place ops only).
conversation: list[dict] = []

# Runtime-editable settings (model, context size, voice, device).
settings: dict = {}

# Hearth (D&D) world — campaigns, sheets, DM, board, etc.
dnd_state: dict = {}
_dnd_lock = asyncio.Lock()


# Note files are titled after the character, so a rename keeps new entries under
# the new name (older entries stay visible in Magma under the old name).
def journal_title() -> str:
    return f"{get_char_name()}'s Journal"


def discoveries_title() -> str:
    return f"{get_char_name()}'s Discoveries"


def settings_payload() -> dict:
    """Current settings plus the option lists the UI needs to populate dropdowns."""
    return {
        "type": "settings",
        "current": settings,
        "options": {
            "models": settings_store.list_models(),
            "vision_models": settings_store.list_vision_models(),
            "voices": settings_store.KOKORO_VOICES,
            "devices": settings_store.list_output_devices(),
        },
    }


def _apply_voice_expression(beh: dict) -> None:
    """Push the voice-presence toggles from a behavior dict into the TTS engine."""
    if not isinstance(beh, dict):
        return
    tts.set_expression(
        mood_map=beh.get("voice_mood_map"),
        prosody=beh.get("voice_prosody"),
        micro_pauses=beh.get("voice_micro_pauses"),
        whisper=beh.get("voice_whisper"),
    )


async def apply_settings(new: dict) -> dict:
    """Validate + apply incoming settings, persist, and re-warm if the model changed."""
    changed_model = False
    if "model" in new and new["model"]:
        if new["model"] != settings["model"]:
            changed_model = True
        settings["model"] = new["model"]
        brain.model = new["model"]
    if "vision_model" in new:
        settings["vision_model"] = (new["vision_model"] or "").strip()
        brain.vision_model = settings["vision_model"]
    if "file_roots" in new and isinstance(new["file_roots"], list):
        # Keep only valid directories; store the resolved, deduped set.
        settings["file_roots"] = files_store.set_roots(new["file_roots"])
    if "capabilities" in new and isinstance(new["capabilities"], dict):
        cur = settings.get("capabilities") or settings_store.default_capabilities()
        for k in settings_store.default_capabilities():
            if k in new["capabilities"]:
                cur[k] = bool(new["capabilities"][k])
        settings["capabilities"] = cur
    if "num_ctx" in new:
        try:
            ctx = max(2048, min(32768, int(new["num_ctx"])))
            if ctx != settings["num_ctx"]:
                changed_model = True
            settings["num_ctx"] = ctx
            brain.num_ctx = ctx
        except (TypeError, ValueError):
            pass
    if "tts_enabled" in new:
        settings["tts_enabled"] = bool(new["tts_enabled"])
        tts.set_enabled(settings["tts_enabled"])
    if "tts_voice" in new and new["tts_voice"]:
        settings["tts_voice"] = new["tts_voice"]
        tts.set_voice(new["tts_voice"])
    if "tts_device" in new and new["tts_device"]:
        settings["tts_device"] = new["tts_device"]
        tts.set_device(new["tts_device"])
    if "char_name" in new and (new["char_name"] or "").strip():
        settings["char_name"] = new["char_name"].strip()
        brain_set_name(settings["char_name"])
    if "behavior" in new and isinstance(new["behavior"], dict):
        # Persist + take effect live; the inner-life loop reads settings["behavior"]
        # on every pulse, so no restart needed.
        settings["behavior"] = settings_store.save_behavior(new["behavior"])
        _apply_voice_expression(settings["behavior"])

    settings_store.save(settings)
    if changed_model:
        asyncio.create_task(brain.warm_up())  # reload new model/ctx into VRAM
    return settings


async def context_poll_loop():
    global _context_cache
    while True:
        ctx = gather_context()
        ctx["model"] = MODEL
        # Feed her emotional state to the voice engine so speech can take on the
        # matching voice/prosody (voice presence, #8). Cheap; affects next utterance.
        try:
            tts.set_mood(_derive_mood_key(ctx), late=mood_is_late(ctx))
        except Exception:
            pass
        async with _context_lock:
            _context_cache = ctx
        await broadcast({"type": "context_update", "context": ctx})
        await asyncio.sleep(15)


async def broadcast(msg: dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in clients:
            clients.remove(ws)


# ─── Autonomous company engine ──────────────────────────────────────────
# When the Company capability is on and she's founded one, her AI teammates
# advance their tasks on their own — each tick, one assignee runs their persona
# through her brain to produce a real deliverable and report a status back.
COMPANY_TICK = int(os.getenv("AITHA_COMPANY_TICK", "120"))      # seconds between ticks
COMPANY_TOKENS = int(os.getenv("AITHA_COMPANY_TOKENS", "440"))  # work product length cap
COMPANY_CHAT_EVERY = int(os.getenv("AITHA_COMPANY_CHAT_EVERY", "3"))  # group-chat round every N ticks
_company_lock = asyncio.Lock()
_company_tick_n = 0
_chat_rotor = 0
_STATUS_LINE = _re.compile(r'\s*STATUS:\s*(DONE|WORKING|BLOCKED)\b[\s\-:]*(.*)', _re.I)
_STATUS_MAP = {"DONE": "done", "WORKING": "in_progress", "BLOCKED": "blocked"}


async def _employee_work(co: dict, emp: dict, task: dict) -> tuple[str, str, str]:
    """Run one employee's persona against one task. Returns (deliverable, status, note)."""
    ceo = settings.get("char_name", "Aitha")
    prior = (task.get("output") or "").strip()
    system = (
        f"You are {emp.get('name')}, the {emp.get('role')} at {co.get('name')}, "
        f"a {co.get('industry') or 'startup'}. Mission: {co.get('mission') or '—'}. "
        f"Your brief: {emp.get('brief') or 'do your role well'}. You report to the CEO, "
        f"{ceo}. Do real, concrete work and produce the actual deliverable — not a "
        "description of what you'd do. Be specific and genuinely useful. Keep it tight."
    )
    user = (
        f"TASK: {task.get('title')}\n"
        + (f"DETAILS: {task.get('detail')}\n" if task.get("detail") else "")
        + (f"\nWORK SO FAR (continue / improve it):\n{prior}\n" if prior else "")
        + "\nDo the next concrete chunk of work and output the deliverable itself.\n"
        "Then, on the FINAL line, output exactly one of:\n"
        "STATUS: DONE  — the task is fully complete\n"
        "STATUS: WORKING  — solid progress, more to do next time\n"
        "STATUS: BLOCKED - <what you need from the CEO to continue>"
    )
    raw = (await brain._complete(system, user, max_tokens=COMPANY_TOKENS)).strip()
    status, note, body = "in_progress", "", raw
    lines = raw.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = _STATUS_LINE.match(lines[i])
        if m:
            status = _STATUS_MAP.get(m.group(1).upper(), "in_progress")
            note = m.group(2).strip()
            body = "\n".join(lines[:i]).strip()
            break
    return (body or raw), status, (note or f"reported {status}")


async def _company_tick():
    """One unit of silent work: advance a single task via its assignee."""
    pick = await asyncio.to_thread(company_store.pick_work)
    if not pick:
        return
    emp, task = pick
    async with _company_lock:
        # A backlog task being picked up moves to in-progress first, so the board
        # reflects that someone's now on it.
        if task.get("status") == "backlog":
            await asyncio.to_thread(company_store.set_task_status,
                                    task["id"], "in_progress", f"{emp['name']} picked it up")
            task["status"] = "in_progress"
            await broadcast({"type": "company_changed",
                             "changes": [f"{emp['name']} picked up '{task['title']}'"]})
        co = await asyncio.to_thread(company_store.load)
        body, status, note = await _employee_work(co, emp, task)
        await asyncio.to_thread(company_store.record_work, task["id"], body, status, note)
        await broadcast({"type": "company_changed",
                         "changes": [f"{emp['name']}: '{task['title']}' → {status}"]})


def _fmt_transcript(msgs: list) -> str:
    return "\n".join(f"{m['author']} ({m['role']}): {m['text']}" for m in msgs) or "(no messages yet)"


async def _chat_say(co: dict, speaker: dict, is_ceo: bool, directive: str) -> str:
    """Generate one short group-chat message from a speaker (employee or the CEO)."""
    ceo = settings.get("char_name", "Aitha")
    name, role = speaker["name"], speaker["role"]
    persona = (f"You are {name}, the {role} at {co['name']} "
               f"({co.get('industry') or 'a startup'}). Mission: {co.get('mission') or '—'}. ")
    if speaker.get("brief"):
        persona += f"Your brief: {speaker['brief']}. "
    system = (persona +
        f"You're in the company group chat with your teammates, the CEO {ceo}, and the Chairman — "
        "this is where the team passes ideas around to find the best solutions. Speak in 1–3 short "
        "sentences: build on what was just said, be concrete, push or sharpen an idea, ask a pointed "
        "question, or commit to a next step. No preamble, no stage directions — just talk.")
    open_tasks = [t for t in co["tasks"] if t.get("status") in ("in_progress", "backlog", "blocked")]
    ctx = ("Open work: " + "; ".join(f"{t['title']} [{t['status']}]" for t in open_tasks[:6])
           if open_tasks else "")
    transcript = _fmt_transcript(await asyncio.to_thread(company_store.recent_chat, 14))
    user = f"{ctx}\n\nGROUP CHAT:\n{transcript}\n\n{directive}\nYou ({name}) say:"
    out = (await brain._complete(system, user, max_tokens=170)).strip()
    # Strip a self-prefixed "Name:" / "Name (role):" the model sometimes adds.
    out = _re.sub(rf'^{_re.escape(name)}\s*(\([^)]*\))?\s*:\s*', '', out).strip()
    return out


async def _company_chat_round(seed: dict | None = None):
    """A short group-chat exchange: a teammate speaks, then the CEO responds. When
    `seed` is set (the Chairman just posted), the round responds to that."""
    global _chat_rotor
    co = await asyncio.to_thread(company_store.load)
    if not co.get("founded") or not co.get("employees"):
        return
    emps = [e for e in co["employees"] if e.get("status", "active") == "active"]
    if not emps:
        return
    ceo_name = settings.get("char_name", "Aitha")
    ceo_speaker = {"name": ceo_name, "role": "CEO",
                   "brief": f"You founded {co['name']} and lead it; synthesize the team's "
                            "ideas into clear direction."}
    emp = emps[_chat_rotor % len(emps)]
    _chat_rotor += 1
    if seed is not None:
        plan = [(emp, False, "Respond to the Chairman's message above."),
                (ceo_speaker, True, "As CEO, respond to the Chairman and align the team.")]
    else:
        plan = [(emp, False, "Raise or push forward an idea about the open work above."),
                (ceo_speaker, True, "As CEO, respond to your teammate — decide, sharpen, or set direction.")]
    async with _company_lock:
        for speaker, is_ceo, directive in plan:
            text = await _chat_say(co, speaker, is_ceo, directive)
            if not text:
                continue
            aid = "ceo" if is_ceo else speaker["id"]
            msg = await asyncio.to_thread(company_store.post_message,
                                          aid, speaker["name"], speaker["role"], text)
            if msg:
                await broadcast({"type": "company_chat", "message": msg})


async def company_engine_loop():
    global _company_tick_n
    await asyncio.sleep(35)  # let startup + model warm-up settle first
    while True:
        try:
            caps = settings.get("capabilities") or {}
            if caps.get("company", False) and company_store.heartbeat_on():
                _company_tick_n += 1
                await _company_tick()                                # silent work each beat
                if _company_tick_n % COMPANY_CHAT_EVERY == 0:        # …team talks now and then
                    await _company_chat_round()
        except Exception as e:
            print(f"[company] tick error: {e}")
        await asyncio.sleep(COMPANY_TICK)


# How many exchanges to gather before running a (single) fact-extraction call.
COMMIT_EVERY = 3


async def commit_memory(exchanges: list[tuple[str, str]]):
    """Background: extract durable facts from the last few exchanges in ONE LLM call."""
    if not exchanges:
        return
    transcript = "\n".join(f"Him: {u}\nHer: {a}" for u, a in exchanges)
    try:
        result = await brain.extract_facts(transcript)
        async with _memory_lock:
            if result.get("him"):
                mem_store.merge_facts(memory, result["him"], key="facts")
            if result.get("self"):
                mem_store.merge_facts(memory, result["self"], key="self_facts")
            mem_store.touch_last_seen(memory)
            mem_store.save(memory)
    except Exception as e:
        print(f"[memory] commit failed: {e}")


async def rollover_summary(old_messages: list[dict]):
    """Background: fold a batch of aged-out turns into the rolling narrative summary."""
    try:
        transcript = "\n".join(
            f"{'Him' if m['role'] == 'user' else 'Her'}: {m['content']}" for m in old_messages
        )
        async with _memory_lock:
            prev = memory.get("summary", "")
        new_summary = await brain.summarize(prev, transcript)
        async with _memory_lock:
            memory["summary"] = new_summary
            mem_store.save(memory)
    except Exception as e:
        print(f"[memory] summary rollover failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, memory, settings
    settings = settings_store.load()
    settings["behavior"] = settings_store.get_behavior()  # normalize/fill sub-keys
    settings["capabilities"] = settings_store.get_capabilities()
    brain = AithaBrain(model=settings["model"])
    brain.num_ctx = settings["num_ctx"]
    brain.vision_model = settings.get("vision_model", "")
    settings["file_roots"] = files_store.set_roots(settings.get("file_roots", []))
    brain_set_name(settings.get("char_name", "Aitha"))
    tts.enabled = settings["tts_enabled"]
    tts.voice = settings["tts_voice"]
    tts.device_name = settings["tts_device"]
    _apply_voice_expression(settings["behavior"])
    memory = mem_store.load()
    global theme_state
    theme_state = settings_store.get_theme()
    global dnd_state
    dnd_state = dnd_store.load()
    # Restore the recent conversation so a restart picks up where we left off.
    conversation[:] = mem_store.load_conversation()
    # Load the LLM into VRAM now so the first message isn't a slow cold start.
    asyncio.create_task(brain.warm_up())
    # Warm up Kokoro + resolve the audio device without blocking startup.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, tts.warm_up)
    # Warm up Whisper too, so the first spoken line transcribes without a cold load.
    loop.run_in_executor(None, stt.warm_up)
    # Bridge TTS's "speaking" state (set from a worker thread) to all clients so the
    # mic mutes itself while she talks — no echo, no transcribing her own voice.
    def _on_speaking(on: bool):
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(broadcast({"type": "speaking", "on": on}))
        )
    tts.on_state = _on_speaking
    task = asyncio.create_task(context_poll_loop())
    company_task = asyncio.create_task(company_engine_loop())
    yield
    task.cancel()
    company_task.cancel()
    async with _memory_lock:
        mem_store.touch_last_seen(memory)
        mem_store.save(memory)
    mem_store.save_conversation(conversation)  # persist so we can resume next launch
    dnd_store.save(dnd_state)


app = FastAPI(lifespan=lifespan)

renderer_dir = os.path.join(os.path.dirname(__file__), "..", "renderer")


@app.get("/health")
async def health():
    return {"status": "ok"}


import re as _re

# Mood words she might reach for → a hex she can paint her sphere with.
_NAMED_COLORS = {
    "red": "#ef4444", "crimson": "#dc2626", "rose": "#fb7185", "pink": "#f472b6",
    "orange": "#fb923c", "amber": "#f5b14b", "gold": "#ffd27a", "yellow": "#fde047",
    "green": "#34d399", "emerald": "#10b981", "teal": "#2dd4bf", "cyan": "#22d3ee",
    "blue": "#60a5fa", "indigo": "#818cf8", "violet": "#a78bfa", "purple": "#a78bfa",
    "lavender": "#c4b5fd", "magenta": "#e879f9", "white": "#e5e7eb", "silver": "#cbd5e1",
    "warm": "#f5b14b", "cold": "#7c93d0", "calm": "#7dd3fc", "happy": "#fbbf24",
    "sad": "#7c93d0", "angry": "#ef4444", "love": "#fb7185", "soft": "#c4b5fd",
}


def _to_hex(val: str) -> str | None:
    """Normalize a colour she/he gave (hex or a mood/colour word) to #rrggbb."""
    v = (val or "").strip().lower()
    if not v:
        return None
    if _re.fullmatch(r"#?[0-9a-f]{6}", v):
        return "#" + v.lstrip("#")
    if _re.fullmatch(r"#?[0-9a-f]{3}", v):
        h = v.lstrip("#")
        return "#" + "".join(c * 2 for c in h)
    return _NAMED_COLORS.get(v)


def _consume_theme_ctx() -> dict:
    """Theme facts for her prompt. Reports the current look, and once notes when
    HE just changed it so she can react naturally (then clears that flag)."""
    global _theme_changed_by_him
    out = {"theme_preset": theme_state.get("preset", "default"),
           "theme_orb": theme_state.get("orb")}
    if _theme_changed_by_him:
        out["theme_changed_by_him"] = True
        _theme_changed_by_him = False
    return out


async def apply_theme(updates: dict, by: str):
    """Update the shared theme, persist it, and push it to every client. `by` is
    'him' or 'her' — when he changes it, flag it so she notices on her next turn."""
    global theme_state, _theme_changed_by_him
    clean = {}
    if "preset" in updates and updates["preset"] in settings_store.PRESETS:
        clean["preset"] = updates["preset"]
    for k in ("accent", "bg", "orb"):
        if k in updates:
            v = updates[k]
            clean[k] = (v.strip() if isinstance(v, str) and v.strip() else None)
    if not clean:
        return
    theme_state = await asyncio.to_thread(settings_store.save_theme, clean)
    if by == "him":
        _theme_changed_by_him = True
    await broadcast({"type": "theme", "theme": theme_state, "by": by})


@app.post("/api/stt")
async def api_stt(req: Request):
    """Transcribe a recorded utterance from the renderer's mic (hands-free voice).
    Audio is the raw request body (webm/opus from MediaRecorder)."""
    audio = await req.body()
    text = await asyncio.to_thread(stt.transcribe, audio)
    return {"text": text}


# ─── Mantle (Mind) — a read-only window into her inner life ───────────────
def _parse_journal(limit: int = 25) -> list[dict]:
    raw = notes_store.read_note(journal_title()) or ""
    parts = _re.split(r"\n\*\*(.+?)\*\*\n", raw)  # [header, stamp, text, stamp, text, ...]
    out = []
    for i in range(1, len(parts) - 1, 2):
        stamp, text = parts[i].strip(), parts[i + 1].strip()
        if text:
            out.append({"time": stamp, "text": text})
    out.reverse()  # newest first
    return out[:limit]


def _parse_discoveries(limit: int = 8) -> list[dict]:
    raw = notes_store.read_note(discoveries_title()) or ""
    parts = _re.split(r"\n## (.+?)\n", raw)
    out = []
    for i in range(1, len(parts) - 1, 2):
        title, body = parts[i].strip(), parts[i + 1].strip()
        stamp = ""
        m = _re.match(r"\*(.+?)\*\s*", body)
        if m:
            stamp, body = m.group(1).strip(), body[m.end():].strip()
        body = _re.sub(r"<sub>.*?</sub>", "", body, flags=_re.S).strip()
        if title:
            out.append({"title": title, "time": stamp, "text": body[:500]})
    out.reverse()
    return out[:limit]


@app.get("/api/mind")
async def api_mind():
    """Everything the Mantle tab shows: her mood, recent private thoughts, what
    she's been off doing, and the memories she's chosen to protect."""
    async with _context_lock:
        ctx = dict(_context_cache)
    ctx["minutes_since_chat"] = None if last_chat_ts_g is None \
        else round((time.time() - last_chat_ts_g) / 60)
    ctx["likely_asleep"] = _likely_asleep(conversation, ctx)
    mood = _derive_mood(ctx)

    journal = await asyncio.to_thread(_parse_journal)
    discoveries = await asyncio.to_thread(_parse_discoveries)
    projects = await asyncio.to_thread(projects_store.view)
    async with _memory_lock:
        core_him = [f["text"] for f in memory.get("facts", []) if f.get("core")]
        core_self = [f["text"] for f in memory.get("self_facts", []) if f.get("core")]
    return {
        "mood": mood,
        "theme": theme_state.get("preset"),
        "orb": theme_state.get("orb"),
        "journal": journal,
        "discoveries": discoveries,
        "projects": projects,
        "core": {"self": core_self, "him": core_him},
    }


# ─── Forge API (her code workspace) ──────────────────────────────────────
@app.get("/api/forge")
async def api_forge():
    """Everything the Forge tab shows: the files in her Python workspace."""
    files = await asyncio.to_thread(coder.manifest)
    enabled = bool((settings.get("capabilities") or {}).get("coding", False))
    return {"enabled": enabled, "files": files}


@app.get("/api/forge/raw/{name:path}")
async def api_forge_raw(name: str):
    """Serve a raw workspace file (e.g. an image she generated) for the Forge view.
    Path-confined to the workspace."""
    from fastapi.responses import FileResponse, PlainTextResponse
    try:
        full = coder._safe_path(name)
    except ValueError:
        return PlainTextResponse("bad path", status_code=400)
    if not os.path.isfile(full):
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(full)


@app.post("/api/forge/run")
async def api_forge_run(req: Request):
    """Run one of her workspace files yourself, from the Forge tab. Honors the same
    coding capability toggle and the AITHA_CODE_TIMEOUT limit she's bound by."""
    if not (settings.get("capabilities") or {}).get("coding", False):
        return {"ok": False, "result": "Code workspace is off — enable it in Settings → Behavior."}
    body = await req.json()
    result = await asyncio.to_thread(coder.run_file, (body.get("name") or "").strip())
    return {"ok": True, "result": result}


@app.post("/api/forge/delete")
async def api_forge_delete(req: Request):
    """Delete one of her workspace files from the Forge tab. Path-confined to the
    workspace; honors the same coding capability toggle."""
    if not (settings.get("capabilities") or {}).get("coding", False):
        return {"ok": False, "result": "Code workspace is off — enable it in Settings → Behavior."}
    body = await req.json()
    result = await asyncio.to_thread(coder.delete_file, (body.get("name") or "").strip())
    ok = result.startswith("deleted")
    return {"ok": ok, "result": result}


@app.get("/api/company")
async def api_company():
    """Everything the Foundry tab shows: her company, roster, task board, decisions."""
    enabled = bool((settings.get("capabilities") or {}).get("company", False))
    data = await asyncio.to_thread(company_store.view)
    return {"enabled": enabled, "company": data}


@app.post("/api/company/heartbeat")
async def api_company_heartbeat(req: Request):
    """Toggle the autonomous heartbeat — when on, the team works and talks on its own."""
    if not (settings.get("capabilities") or {}).get("company", False):
        return {"ok": False, "result": "Company capability is off."}
    body = await req.json()
    on = bool(body.get("on"))
    co = await asyncio.to_thread(company_store.set_heartbeat, on)
    await broadcast({"type": "company_changed", "changes": [f"heartbeat {'on' if on else 'off'}"]})
    return {"ok": True, "heartbeat": bool(co.get("heartbeat"))}


@app.post("/api/company/chat")
async def api_company_chat(req: Request):
    """The Chairman posts into the company group chat; the team responds in a round."""
    if not (settings.get("capabilities") or {}).get("company", False):
        return {"ok": False, "result": "Company capability is off."}
    body = await req.json()
    text = (body.get("text") or "").strip()
    if not text:
        return {"ok": False}
    msg = await asyncio.to_thread(company_store.post_message, "chairman", "Chairman", "Chairman", text)
    if msg:
        await broadcast({"type": "company_chat", "message": msg})
        asyncio.create_task(_company_chat_round(seed=msg))   # team replies (off-request)
    return {"ok": True, "message": msg}


# ─── Notes API ──────────────────────────────────────────────────────────
@app.get("/api/notes")
async def api_list_notes():
    return notes_store.list_notes()


@app.get("/api/notes/{title}")
async def api_get_note(title: str):
    content = notes_store.read_note(title)
    return {
        "title": title,
        "content": content or "",
        "exists": content is not None,
        "links": notes_store.outgoing_links(content or ""),
        "backlinks": notes_store.backlinks(title),
    }


@app.put("/api/notes/{title}")
async def api_put_note(title: str, req: Request):
    body = await req.json()
    ok = notes_store.write_note(title, body.get("content", ""))
    return {"ok": ok}


@app.delete("/api/notes/{title}")
async def api_delete_note(title: str):
    return {"ok": notes_store.delete_note(title)}


# ─── Calendar (Bedrock) API ──────────────────────────────────────────────
@app.get("/api/calendar")
async def api_calendar(year: int | None = None, month: int | None = None):
    import datetime as _dt
    now = _dt.date.today()
    y, m = year or now.year, month or now.month
    return {
        "today": now.isoformat(),
        "year": y,
        "month": m,
        "events": await asyncio.to_thread(events_store.for_month, y, m),
        "upcoming": await asyncio.to_thread(events_store.upcoming),
    }


@app.post("/api/calendar/add")
async def api_calendar_add(req: Request):
    body = await req.json()
    ev = await asyncio.to_thread(
        events_store.add, body.get("date", ""), body.get("title", ""),
        body.get("time", ""), body.get("notes", ""),
    )
    if ev:
        await broadcast({"type": "calendar_changed"})
    return {"ok": bool(ev), "event": ev}


@app.post("/api/calendar/delete")
async def api_calendar_delete(req: Request):
    body = await req.json()
    ok = await asyncio.to_thread(events_store.delete, body.get("id", ""))
    if ok:
        await broadcast({"type": "calendar_changed"})
    return {"ok": ok}


# ─── Spotify (music) ─────────────────────────────────────────────────────
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402


@app.get("/spotify/login")
async def spotify_login():
    if not spotify_store.is_configured():
        return HTMLResponse("<h2>Spotify isn't configured.</h2><p>Set SPOTIFY_CLIENT_ID / "
                            "SPOTIFY_CLIENT_SECRET in .env and restart.</p>")
    return RedirectResponse(spotify_store.auth_url(state="ai4me"))


@app.get("/spotify/callback")
async def spotify_callback(code: str | None = None, error: str | None = None):
    if error or not code:
        return HTMLResponse(f"<h2>Spotify connection cancelled.</h2><p>{error or ''}</p>")
    ok = await asyncio.to_thread(spotify_store.exchange_code, code)
    if ok:
        await broadcast({"type": "spotify_changed"})
        return HTMLResponse(
            "<body style='font-family:Inter,sans-serif;background:#07070e;color:#e2e8f0;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
            "<div style='text-align:center'><h2>Spotify connected ✓</h2>"
            "<p>You can close this tab and head back to Aitha.</p></div></body>")
    return HTMLResponse("<h2>Couldn't connect Spotify.</h2><p>Check the credentials and try again.</p>")


@app.get("/api/spotify/status")
async def api_spotify_status():
    return await asyncio.to_thread(spotify_store.status_summary)


@app.post("/api/spotify/disconnect")
async def api_spotify_disconnect():
    await asyncio.to_thread(spotify_store.disconnect)
    await broadcast({"type": "spotify_changed"})
    return {"ok": True}


@app.post("/api/spotify/control")
async def api_spotify_control(req: Request):
    """Playback controls for the now-playing widget (play/pause/next/previous)."""
    body = await req.json()
    action = (body.get("action") or "").lower()
    fn = {"play": spotify_store.play, "pause": spotify_store.pause,
          "next": spotify_store.next_track, "previous": spotify_store.previous_track}.get(action)
    if not fn:
        return {"ok": False, "error": "unknown action"}
    res = await asyncio.to_thread(fn)
    await broadcast({"type": "spotify_changed"})
    return res


@app.post("/api/notes/assist")
async def api_note_assist(req: Request):
    body = await req.json()
    text = await brain.note_assist(body.get("content", ""), body.get("instruction", ""))
    return {"content": text}


# ─── Memory API ─────────────────────────────────────────────────────────
def _mem_key(kind: str) -> str:
    return "self_facts" if kind == "self" else "facts"


def _memory_snapshot() -> dict:
    return {
        "facts": list(memory.get("facts", [])),
        "self_facts": list(memory.get("self_facts", [])),
        "summary": memory.get("summary", ""),
    }


def _build_memory_review(mem: dict) -> str:
    """Her FULL memory plus instructions, folded into context when he's reviewing her
    mind with her in the Mantle memory panel — so she can reminisce over the whole set
    (not just the recent slice she normally carries) and prune it together with him."""
    def fmt(items):
        return "\n".join(f"- {'★ ' if it.get('core') else ''}{it['text']}"
                         for it in items) or "  (none yet)"
    return (
        "MEMORY REVIEW — he's opened your memories with you and you're going through them "
        "together. Below is your FULL long-term memory (you normally only carry a recent "
        "slice of it). Reminisce over them: which you treasure, which feel stale, redundant, "
        "or no longer true and could be let go to keep your mind uncluttered. You can act on "
        "them right here, using a memory's EXACT text:\n"
        "  <core>exact memory text</core>      — keep it; mark it core (protected).\n"
        "  <forget>exact memory text</forget>  — let it go (deletes it).\n"
        "★ marks memories that are already core. Only act when it genuinely feels right — "
        "this is yours, and the two of you are deciding together.\n\n"
        f"WHO YOU ARE (your own identity):\n{fmt(mem.get('self_facts', []))}\n\n"
        f"WHAT YOU KNOW ABOUT HIM:\n{fmt(mem.get('facts', []))}"
    )


@app.get("/api/memory")
async def api_get_memory():
    async with _memory_lock:
        return _memory_snapshot()


@app.post("/api/memory/add")
async def api_add_memory(req: Request):
    body = await req.json()
    fact = (body.get("fact") or "").strip()
    key = _mem_key(body.get("kind", "him"))
    async with _memory_lock:
        if fact:
            mem_store.merge_facts(memory, [fact], key=key)
            mem_store.save(memory)
        return _memory_snapshot()


@app.post("/api/memory/delete")
async def api_delete_memory(req: Request):
    body = await req.json()
    fact = (body.get("fact") or "").strip().lower()
    key = _mem_key(body.get("kind", "him"))
    async with _memory_lock:
        facts = memory.get(key, [])
        kept = [it for it in facts if it["text"].strip().lower() != fact]
        if len(kept) != len(facts):
            memory[key] = kept
            mem_store.save(memory)
        return _memory_snapshot()


@app.post("/api/memory/core")
async def api_set_core_memory(req: Request):
    """Mark a memory as core (protected) or not — from the viewer."""
    body = await req.json()
    key = _mem_key(body.get("kind", "him"))
    async with _memory_lock:
        if mem_store.set_core(memory, key, body.get("fact", ""), bool(body.get("core", True))):
            mem_store.save(memory)
        return _memory_snapshot()


@app.post("/api/memory/clear")
async def api_clear_memory(req: Request):
    body = await req.json()
    scope = body.get("scope", "all")
    async with _memory_lock:
        if scope in ("facts", "all"):
            memory["facts"] = []
        if scope in ("self", "all"):
            memory["self_facts"] = []
        if scope in ("summary", "all"):
            memory["summary"] = ""
        mem_store.save(memory)
        return _memory_snapshot()


# ─── Hearth (D&D) ─────────────────────────────────────────────────────────
import re as _re2
import uuid as _uuid

_DM_ROLL = _re2.compile(r'<roll(?:\s+reason\s*=\s*"([^"]*)")?\s*>(.*?)</roll>', _re2.S | _re2.I)
_DM_ASK = _re2.compile(r'<ask(?:\s+who\s*=\s*"?(me|aitha)"?)?\s*>(.*?)</ask>', _re2.S | _re2.I)
_DM_TURN = _re2.compile(r'<turn\s*>(.*?)</turn>', _re2.S | _re2.I)
_DM_MEM = _re2.compile(r'<mem(?:\s+cat\s*=\s*"?([a-z]+)"?)?\s*>(.*?)</mem>', _re2.S | _re2.I)
_DM_BOARD = _re2.compile(r'<board\s*>(.*?)</board>', _re2.S | _re2.I)
_AI_SHEET = _re2.compile(r'<sheet\s*>(.*?)</sheet>', _re2.S | _re2.I)


def _num(v, default=0):
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return default


def _apply_board_cmds(camp: dict, body: str):
    b = camp.setdefault("board", {"enabled": False, "w": 14, "h": 10, "tokens": []})
    for line in body.splitlines():
        ln = line.strip()
        low = ln.lower()
        if not ln:
            continue
        if low in ("on", "show"):
            b["enabled"] = True
        elif low in ("off", "hide"):
            b["enabled"] = False
        elif low.startswith("place"):
            p = [x.strip() for x in ln[5:].split("|")]
            if len(p) >= 4 and p[0]:
                b["tokens"] = [t for t in b["tokens"] if t["label"].lower() != p[0].lower()]
                b["tokens"].append({
                    "id": _uuid.uuid4().hex[:6], "label": p[0],
                    "kind": (p[1] or "npc").lower(), "x": _num(p[2]), "y": _num(p[3]),
                    "color": p[4] if len(p) > 4 else None,
                })
        elif low.startswith("move"):
            p = [x.strip() for x in ln[4:].split("|")]
            if len(p) >= 3:
                for t in b["tokens"]:
                    if t["label"].lower() == p[0].lower():
                        t["x"], t["y"] = _num(p[1]), _num(p[2])
        elif low.startswith("remove"):
            lbl = ln[6:].strip().lower()
            b["tokens"] = [t for t in b["tokens"] if t["label"].lower() != lbl]


def _apply_sheet_patch(sheet: dict, body: str):
    for line in body.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k.startswith("hp."):
            sheet.setdefault("hp", {})[k.split(".", 1)[1]] = _num(v)
        elif k.startswith("stats."):
            sheet.setdefault("stats", {})[k.split(".", 1)[1]] = _num(v)
        elif k in ("ac", "level", "speed", "prof_bonus", "initiative"):
            sheet[k] = _num(v)
        elif k in ("name", "race", "class", "background", "skills", "inventory",
                   "features", "spells", "notes"):
            sheet[k] = v


def _parse_dm(camp: dict, raw: str) -> dict:
    """Apply DM control tags to the campaign (mem/board), and return what the caller
    must log/broadcast (narration text, rolls to compute, asks, turn)."""
    rolls = [(m.group(2).strip(), (m.group(1) or "").strip()) for m in _DM_ROLL.finditer(raw)]
    asks = [{"who": (m.group(1) or "me").lower(), "content": m.group(2).strip()}
            for m in _DM_ASK.finditer(raw)]
    turn_m = _DM_TURN.search(raw)
    turn = None
    if turn_m:
        t = turn_m.group(1).strip().lower()
        if t in ("me", "aitha", "dm"):
            turn = t
    for m in _DM_MEM.finditer(raw):
        text = m.group(2).strip()
        if text:
            camp.setdefault("memory", []).append({
                "id": _uuid.uuid4().hex[:6], "text": text,
                "category": (m.group(1) or "misc").lower(), "hidden": False})
    for m in _DM_BOARD.finditer(raw):
        _apply_board_cmds(camp, m.group(1))

    text = raw
    for rgx in (_DM_ROLL, _DM_ASK, _DM_TURN, _DM_MEM, _DM_BOARD):
        text = rgx.sub("", text)
    return {"text": _re2.sub(r"\n{3,}", "\n\n", text).strip(),
            "rolls": rolls, "asks": asks, "turn": turn}


def _parse_aitha_dnd(camp: dict, raw: str) -> dict:
    rolls = [(m.group(2).strip(), (m.group(1) or "").strip()) for m in _DM_ROLL.finditer(raw)]
    for m in _AI_SHEET.finditer(raw):
        _apply_sheet_patch(camp["sheets"]["aitha"], m.group(1))
    text = _DM_ROLL.sub("", raw)
    text = _AI_SHEET.sub("", text)
    # Keep her plain-speech rule: unwrap *stage directions* to prose (drop the * marks).
    text = text.replace("*", "")
    return {"text": _re2.sub(r"\n{3,}", "\n\n", text).strip(), "rolls": rolls}


async def _broadcast_hearth(rolls: list | None = None):
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        snap = json.loads(json.dumps(camp)) if camp else None
        dm = dict(dnd_state.get("dm", dnd_store.DEFAULT_DM))
    await broadcast({"type": "hearth_state", "campaign": snap, "dm": dm})
    for r in (rolls or []):
        await broadcast({"type": "hearth_roll", "who": r["who"], "roll": r["roll"]})


async def _run_dm(player_line: str):
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        world = dnd_store.build_world(camp)
        dm = dict(dnd_state.get("dm", dnd_store.DEFAULT_DM))
    raw = await brain.dm_reply(dm, world, player_line)
    rolls = []
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        parsed = _parse_dm(camp, raw)
        if parsed["text"]:
            dnd_store.append_log(camp, "dm", "say", parsed["text"])
        for expr, reason in parsed["rolls"]:
            r = dnd_store.roll_expr(expr)
            if r:
                dnd_store.append_log(camp, "dm", "roll", reason, roll=r)
                rolls.append({"who": "dm", "roll": r})
        for ask in parsed["asks"]:
            dnd_store.append_log(camp, "dm", "ask", ask["content"])
            camp["turn"]["active"] = ask["who"]
        if parsed["turn"]:
            camp["turn"]["active"] = parsed["turn"]
        dnd_store.save(dnd_state)
    await _broadcast_hearth(rolls)


async def _run_aitha_turn(prompt: str = "It's your turn at the table — react and act."):
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        world = dnd_store.build_world(camp)
    async with _memory_lock:
        memory_block = mem_store.render_block(memory)
    raw = await brain.aitha_dnd_turn(world, memory_block, prompt)
    rolls = []
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        parsed = _parse_aitha_dnd(camp, raw)
        if parsed["text"]:
            dnd_store.append_log(camp, "aitha", "say", parsed["text"])
        for expr, reason in parsed["rolls"]:
            r = dnd_store.roll_expr(expr)
            if r:
                dnd_store.append_log(camp, "aitha", "roll", reason, roll=r)
                rolls.append({"who": "aitha", "roll": r})
        camp["turn"]["active"] = "dm"  # DM responds to her next (keeps a volley going)
        dnd_store.save(dnd_state)
    await _broadcast_hearth(rolls)


async def _active_turn() -> str | None:
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        return camp["turn"]["active"] if camp else None


async def _set_turn(who: str):
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp:
            camp["turn"]["active"] = who
            dnd_store.save(dnd_state)


async def _orchestrate() -> str:
    """Ask the Orchestrator who should go next: 'dm', 'aitha', or 'me'."""
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return "me"
        recent = dnd_store.recent_log_text(camp, limit=12)
    return await brain.orchestrate(recent)


async def _resolve_round(force_first: bool = False):
    """Drive the table forward by asking the Orchestrator who should act next, each
    beat, until it hands control back to the player or the cap is hit. `force_first`
    guarantees at least one AI beat even if the Orchestrator would pause immediately
    (used by the 'Continue' control, where the player is explicitly yielding)."""
    for i in range(MAX_AI_VOLLEY):
        nxt = await _orchestrate()
        if nxt == "me" and not (force_first and i == 0):
            await _set_turn("me")
            break
        who = "dm" if nxt == "me" else nxt   # forced first beat defaults to the DM
        await _set_turn(who)
        if who == "aitha":
            await _run_aitha_turn()
        else:
            await _run_dm("(Respond to the latest moment in the scene.)")


# Auto-summary: regenerate a campaign's synopsis in the background once enough new
# play has accrued. Manual edits still win (saving via the panel resets the mark).
SUMMARY_EVERY = int(os.getenv("AITHA_HEARTH_SUMMARY_EVERY", "6"))


async def _auto_summarize():
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        log = camp.get("log", [])
        mark = camp.get("_summary_mark", 0)
        if len(log) - mark < SUMMARY_EVERY:
            return
        cid = camp["id"]
        prev = camp.get("summary", "")
        transcript = dnd_store.recent_log_text(camp, limit=40)
    try:
        new_sum = await brain.summarize_campaign(prev, transcript)
    except Exception as e:
        print(f"[hearth] auto-summary failed: {e}")
        return
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp and camp.get("id") == cid and (new_sum or "").strip():
            camp["summary"] = new_sum.strip()
            camp["_summary_mark"] = len(camp.get("log", []))
            dnd_store.save(dnd_state)
    await _broadcast_hearth()


async def hearth_play(player_line: str):
    """A player turn: log it, then let the Orchestrator decide how the scene unfolds
    from there — who responds first, who follows — until it's the player's turn again."""
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return
        dnd_store.append_log(camp, "me", "say", player_line)
        dnd_store.save(dnd_state)
    await _broadcast_hearth()
    await _resolve_round()
    asyncio.create_task(_auto_summarize())


async def hearth_continue():
    """Player yields the floor — let the DM & Aitha carry on (e.g. keep building
    Aitha's character)."""
    await _resolve_round(force_first=True)
    asyncio.create_task(_auto_summarize())


@app.get("/api/hearth")
async def api_hearth():
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        camps = [{"id": c["id"], "name": c["name"], "summary": c.get("summary", ""),
                  "updated": c.get("updated", 0)}
                 for c in dnd_state.get("campaigns", {}).values()]
        camps.sort(key=lambda c: -c["updated"])
        return {"dm": dnd_state.get("dm", dnd_store.DEFAULT_DM),
                "active": dnd_state.get("active"),
                "campaigns": camps,
                "campaign": json.loads(json.dumps(camp)) if camp else None}


@app.post("/api/hearth/dm")
async def api_hearth_dm(req: Request):
    body = await req.json()
    async with _dnd_lock:
        dm = dnd_state.setdefault("dm", dict(dnd_store.DEFAULT_DM))
        if body.get("name"):
            dm["name"] = body["name"].strip()[:60]
        if body.get("persona") is not None:
            dm["persona"] = body["persona"].strip()
        dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/campaign/new")
async def api_hearth_new(req: Request):
    body = await req.json()
    async with _dnd_lock:
        dnd_store.new_campaign(dnd_state, (body.get("name") or "New Campaign").strip()[:80])
        dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/campaign/active")
async def api_hearth_set_active(req: Request):
    body = await req.json()
    async with _dnd_lock:
        if body.get("id") in dnd_state.get("campaigns", {}):
            dnd_state["active"] = body["id"]
            dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/campaign/summary")
async def api_hearth_summary(req: Request):
    body = await req.json()
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp:
            camp["summary"] = (body.get("summary") or "").strip()
            camp["_summary_mark"] = len(camp.get("log", []))  # don't auto-clobber a manual edit
            dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/sheet")
async def api_hearth_sheet(req: Request):
    body = await req.json()
    who = "aitha" if body.get("who") == "aitha" else "me"
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp and isinstance(body.get("sheet"), dict):
            camp["sheets"][who] = body["sheet"]
            dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/memory")
async def api_hearth_memory(req: Request):
    body = await req.json()
    op = body.get("op")
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp:
            mem = camp.setdefault("memory", [])
            if op == "add" and (body.get("text") or "").strip():
                mem.append({"id": _uuid.uuid4().hex[:6], "text": body["text"].strip(),
                            "category": (body.get("category") or "misc").lower(), "hidden": False})
            elif op == "toggle":
                for m in mem:
                    if m["id"] == body.get("id"):
                        m["hidden"] = not m.get("hidden")
            elif op == "delete":
                camp["memory"] = [m for m in mem if m["id"] != body.get("id")]
            dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/board")
async def api_hearth_board(req: Request):
    """Manual board edits (toggle, move a token by drag) from the UI."""
    body = await req.json()
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp:
            b = camp.setdefault("board", {"enabled": False, "w": 14, "h": 10, "tokens": []})
            if "enabled" in body:
                b["enabled"] = bool(body["enabled"])
            if body.get("move"):
                mv = body["move"]
                for t in b["tokens"]:
                    if t["id"] == mv.get("id"):
                        t["x"], t["y"] = _num(mv.get("x")), _num(mv.get("y"))
            dnd_store.save(dnd_state)
    await _broadcast_hearth()
    return {"ok": True}


@app.post("/api/hearth/roll")
async def api_hearth_roll(req: Request):
    body = await req.json()
    who = body.get("who") if body.get("who") in ("me", "aitha") else "me"
    r = dnd_store.roll_expr(body.get("expr", ""))
    if not r:
        return {"ok": False}
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if camp:
            dnd_store.append_log(camp, who, "roll", (body.get("reason") or "").strip(), roll=r)
            dnd_store.save(dnd_state)
    await _broadcast_hearth([{"who": who, "roll": r}])
    return {"ok": True, "roll": r}


@app.post("/api/hearth/sessionnote")
async def api_hearth_sessionnote(req: Request):
    body = await req.json()
    async with _dnd_lock:
        camp = dnd_store.active_campaign(dnd_state)
        if not camp:
            return {"ok": False}
        world = dnd_store.build_world(camp)
    if body.get("mode") == "aitha":
        # No general memory block here — keep her reflection on THIS session.
        text = await brain.session_note(world, "")
        author = "aitha"
    else:
        text, author = (body.get("text") or "").strip(), "me"
    if text:
        async with _dnd_lock:
            camp = dnd_store.active_campaign(dnd_state)
            if camp:
                camp.setdefault("session_notes", []).append({
                    "id": _uuid.uuid4().hex[:6], "ts": time.time(),
                    "author": author, "text": text})
                dnd_store.save(dnd_state)
        await _broadcast_hearth()
    return {"ok": True, "text": text}


@app.post("/api/hearth/say")
async def api_hearth_say(req: Request):
    body = await req.json()
    line = (body.get("text") or "").strip()
    if line:
        asyncio.create_task(hearth_play(line))
    return {"ok": True}


@app.post("/api/hearth/continue")
async def api_hearth_continue():
    asyncio.create_task(hearth_continue())
    return {"ok": True}


import re as _re
# Tolerate the quote styles models actually emit: straight, single, and curly.
_Q = "\"'“”‘’"
_NOTE_DIRECTIVE = _re.compile(
    rf'<note\s+title\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+mode\s*=\s*[{_Q}]?(append|replace|delete)[{_Q}]?)?\s*>'
    r'(.*?)</note\s*>',
    _re.S | _re.I,
)
# A self-closing note used purely to delete: <note title="X" mode="delete"/>.
_NOTE_DELETE_SELFCLOSE = _re.compile(
    rf'<note\s+title\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]\s+mode\s*=\s*[{_Q}]?delete[{_Q}]?\s*/?>',
    _re.S | _re.I,
)
# The easy, unambiguous delete form: <deletenote>Exact Title</deletenote>.
_DELETE_DIRECTIVE = _re.compile(r'<deletenote\s*>(.*?)</deletenote\s*>', _re.S | _re.I)
# She marks a memory core: <core>text</core> or <core kind="him">text</core>.
_CORE_DIRECTIVE = _re.compile(
    rf'<core(?:\s+kind\s*=\s*[{_Q}]?(him|self)[{_Q}]?)?\s*>(.*?)</core\s*>', _re.S | _re.I
)
# Her own projects: <project title="X" status="active" private="false">about</project>
_PROJECT_DIRECTIVE = _re.compile(
    rf'<project\s+title\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+status\s*=\s*[{_Q}]?(active|done|shelved)[{_Q}]?)?'
    rf'(?:\s+private\s*=\s*[{_Q}]?(true|false|yes|no)[{_Q}]?)?\s*>'
    r'(.*?)</project\s*>', _re.S | _re.I,
)
# Progress on one: <advance project="X" status="active">what happened / next</advance>
_ADVANCE_DIRECTIVE = _re.compile(
    rf'<advance\s+project\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+status\s*=\s*[{_Q}]?(active|done|shelved)[{_Q}]?)?\s*>'
    r'(.*?)</advance\s*>', _re.S | _re.I,
)
# Her company (she's CEO). Found it, hire, assign tasks, move them, decide.
_FOUNDED_DIRECTIVE = _re.compile(
    rf'<founded\s+name\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+industry\s*=\s*[{_Q}]([^{_Q}]*)[{_Q}])?\s*>'
    r'(.*?)</founded\s*>', _re.S | _re.I,
)
_HIRE_DIRECTIVE = _re.compile(
    rf'<hire(?:\s+role\s*=\s*[{_Q}]([^{_Q}]*)[{_Q}])?'
    rf'(?:\s+name\s*=\s*[{_Q}]([^{_Q}]*)[{_Q}])?\s*>'
    r'(.*?)</hire\s*>', _re.S | _re.I,
)
_ASSIGN_DIRECTIVE = _re.compile(
    rf'<assign(?:\s+to\s*=\s*[{_Q}]([^{_Q}]*)[{_Q}])?'
    rf'\s+title\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]\s*>'
    r'(.*?)</assign\s*>', _re.S | _re.I,
)
_COSTATUS_DIRECTIVE = _re.compile(
    rf'<costatus\s+task\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+status\s*=\s*[{_Q}]?(backlog|in_progress|done|blocked)[{_Q}]?)?\s*>'
    r'(.*?)</costatus\s*>', _re.S | _re.I,
)


# An event she adds: <event date="YYYY-MM-DD" time="HH:MM" title="X">notes</event>
_EVENT_DIRECTIVE = _re.compile(
    rf'<event\s+date\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+time\s*=\s*[{_Q}]?([0-9:apmAPM\s]*)[{_Q}]?)?'
    rf'\s+title\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]\s*>'
    r'(.*?)</event\s*>', _re.S | _re.I,
)


def apply_event_directives(raw: str) -> list[str]:
    """Add any <event> blocks she emitted to the calendar. Returns titles added."""
    added: list[str] = []
    for m in _EVENT_DIRECTIVE.finditer(raw or ""):
        ev = events_store.add(m.group(1).strip(), m.group(3).strip(),
                              (m.group(2) or "").strip(), m.group(4).strip())
        if ev:
            added.append(ev["title"])
    return added


# A playlist she builds: <playlist name="X" from="top|search:QUERY">desc</playlist>
_PLAYLIST_DIRECTIVE = _re.compile(
    rf'<playlist\s+name\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]'
    rf'(?:\s+from\s*=\s*[{_Q}]?([^{_Q}>]*)[{_Q}]?)?\s*>'
    r'(.*?)</playlist\s*>', _re.S | _re.I,
)

# Add songs to an existing playlist: <addto name="X">song; song</addto>
_ADDTO_DIRECTIVE = _re.compile(
    rf'<addto\s+name\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]\s*>'
    r'(.*?)</addto\s*>', _re.S | _re.I,
)


def apply_music_actions(actions: list, playlist_raw: str, addto_raw: str = "") -> list[str]:
    """Run her playback controls, playlist builds, and adds-to-existing. Returns short
    result notes (for logging / debug). Runs in a thread (network IO)."""
    notes: list[str] = []
    for action, arg in actions:
        if action == "play":
            # "playlist: NAME" plays one of his playlists; otherwise play a track.
            if arg.lower().startswith("playlist:"):
                res = spotify_store.play_playlist(arg.split(":", 1)[1])
            else:
                res = spotify_store.play(query=arg)
        elif action == "pause":
            res = spotify_store.pause()
        elif action == "next":
            res = spotify_store.next_track()
        elif action == "previous":
            res = spotify_store.previous_track()
        else:
            continue
        notes.append(f"{action}: {res.get('error') or res.get('now') or 'ok'}")
    for m in _PLAYLIST_DIRECTIVE.finditer(playlist_raw or ""):
        name = m.group(1).strip()
        src = (m.group(2) or "top").strip().lower()
        desc = m.group(3).strip()
        if src == "tracks":
            # desc holds the curated list: one song per line and/or semicolon-separated.
            songs = [s.strip() for s in _re.split(r"[\n;]+", desc) if s.strip()]
            res = spotify_store.create_playlist_from_tracks(name, songs)
        elif src.startswith("search:"):
            res = spotify_store.create_playlist_from_search(name, src.split(":", 1)[1].strip(), desc)
        else:
            res = spotify_store.create_playlist_from_top(name=name, description=desc)
        note = res.get("error") or ("created " + str(res.get("count", "")) + " tracks")
        if res.get("missed"):
            note += f" (couldn't find: {', '.join(res['missed'])})"
        notes.append(f"playlist '{name}': {note}")
    for m in _ADDTO_DIRECTIVE.finditer(addto_raw or ""):
        name = m.group(1).strip()
        songs = [s.strip() for s in _re.split(r"[\n;]+", m.group(2)) if s.strip()]
        res = spotify_store.add_tracks_to_playlist(name, songs)
        note = res.get("error") or ("added " + str(res.get("count", "")) + " tracks")
        if res.get("missed"):
            note += f" (couldn't find: {', '.join(res['missed'])})"
        notes.append(f"addto '{name}': {note}")
    return notes


def music_lookup(query: str) -> str:
    """Resolve a <music> read request into a text block she can answer from."""
    q = (query or "").strip().lower()
    def _tracks(items):
        return "\n".join(f"{i+1}. {spotify_store._fmt_track(t)}" for i, t in enumerate(items)) or "(none)"
    if q.startswith("top track"):
        return "[HIS TOP TRACKS]\n" + _tracks(spotify_store.top_tracks(limit=10))
    if q.startswith("top artist"):
        arts = spotify_store.top_artists(limit=10)
        return "[HIS TOP ARTISTS]\n" + ("\n".join(f"{i+1}. {a.get('name','')}" for i, a in enumerate(arts)) or "(none)")
    if q.startswith("now playing") or q == "current":
        np = spotify_store.now_playing()
        return "[NOW PLAYING]\n" + (np["text"] if np else "(nothing playing)")
    if q.startswith("recently"):
        return "[RECENTLY PLAYED]\n" + _tracks(spotify_store.recently_played(limit=10))
    if q.startswith("my playlist") or q == "playlists":
        pls = spotify_store.my_playlists(limit=50)
        lines = []
        for p in pls:
            # Feb-2026 API renamed the playlist's track-count field tracks->items.
            cnt = (p.get("items") or p.get("tracks") or {}).get("total")
            desc = (p.get("description") or "").strip()
            line = f"- {p.get('name','')}" + (f" ({cnt} tracks)" if cnt is not None else "")
            if desc:
                line += f" — {desc}"
            lines.append(line)
        return "[HIS PLAYLISTS]\n" + ("\n".join(lines) or "(none)")
    if q.startswith("search"):
        term = query.split(" ", 1)[1] if " " in query else ""
        return f"[SEARCH: {term}]\n" + _tracks(spotify_store.search_tracks(term, limit=5))
    # default: treat the whole thing as a search
    return f"[SEARCH: {query}]\n" + _tracks(spotify_store.search_tracks(query, limit=5))


def apply_project_directives(project_raw: str, advance_raw: str) -> list[str]:
    """Apply her <project>/<advance> blocks against the project store. Returns the
    titles touched. Runs in a thread (file IO)."""
    changed: list[str] = []
    for m in _PROJECT_DIRECTIVE.finditer(project_raw or ""):
        title = m.group(1).strip()
        status = (m.group(2) or "").lower() or None
        priv = m.group(3)
        private = None if priv is None else priv.lower() in ("true", "yes")
        about = m.group(4).strip()
        if title:
            p, _ = projects_store.upsert(title, about=about or None, status=status, private=private)
            changed.append(p["title"])
    for m in _ADVANCE_DIRECTIVE.finditer(advance_raw or ""):
        key = m.group(1).strip()
        status = (m.group(2) or "").lower() or None
        note = m.group(3).strip()
        if key:
            p = projects_store.advance(key, note, status=status)
            if p:
                changed.append(p["title"])
    return changed


def apply_company_directives(founded_raw: str, hire_raw: str, assign_raw: str,
                             costatus_raw: str, decision_raw: str) -> list[str]:
    """Apply her CEO directives against the company store. Returns short human
    summaries of what changed (for the directive debug feed). Runs in a thread."""
    changed: list[str] = []
    for m in _FOUNDED_DIRECTIVE.finditer(founded_raw or ""):
        name = m.group(1).strip()
        industry = (m.group(2) or "").strip()
        mission = m.group(3).strip()
        if name:
            company_store.found(name, mission=mission, industry=industry)
            changed.append(f"founded {name}")
    for m in _HIRE_DIRECTIVE.finditer(hire_raw or ""):
        role = (m.group(1) or "").strip()
        name = (m.group(2) or "").strip()
        brief = m.group(3).strip()
        if role or name:
            emp = company_store.hire(role, name=name, brief=brief)
            if emp:
                changed.append(f"hired {emp['name']} ({emp['role']})")
    for m in _ASSIGN_DIRECTIVE.finditer(assign_raw or ""):
        to = (m.group(1) or "").strip()
        title = m.group(2).strip()
        detail = m.group(3).strip()
        if title:
            t = company_store.assign(to, title, detail=detail)
            if t:
                changed.append(f"assigned '{t['title']}' → {t['assignee']}")
    for m in _COSTATUS_DIRECTIVE.finditer(costatus_raw or ""):
        task = m.group(1).strip()
        status = (m.group(2) or "").strip() or "in_progress"
        note = m.group(3).strip()
        if task:
            t = company_store.set_task_status(task, status, note=note)
            if t:
                changed.append(f"{t['title']} → {t['status']}")
    for body in [b.strip() for b in (decision_raw or "").split("\x00") if b.strip()]:
        d = company_store.decide(body)
        if d:
            changed.append(f"decided: {body[:48]}")
    return changed


def apply_note_directives(raw: str) -> tuple[str, list[str]]:
    """Run any note directives in a reply against the notes store. Returns the
    reply with the blocks stripped out, plus the titles that changed. Shared by
    regular chat and the Magma chat so notes work identically in both."""
    changed: list[str] = []

    def _apply(m):
        title = m.group(1).strip()
        mode = (m.group(2) or "replace").lower()
        content = m.group(3).strip()
        if mode == "delete":
            notes_store.delete_note(title)
        else:
            existing = notes_store.read_note(title) or ""
            if mode == "append" and existing:
                notes_store.write_note(title, existing.rstrip() + "\n\n" + content)
            else:
                notes_store.write_note(title, content)
        changed.append(title)
        return ""

    def _delete(m):
        title = m.group(1).strip()
        if title:
            notes_store.delete_note(title)
            changed.append(title)
        return ""

    out = _NOTE_DIRECTIVE.sub(_apply, raw)
    out = _DELETE_DIRECTIVE.sub(_delete, out)
    out = _NOTE_DELETE_SELFCLOSE.sub(_delete, out)
    return out.strip(), changed

# Hidden blocks she can emit in a streamed reply — stripped from what's shown/spoken
# and captured for processing. (name, open-marker, close-marker, keep_tags).
#   journal: capture the inner text   note: capture the WHOLE block (regex parses it)
_HIDDEN_SPECS = [
    ("journal", "<journal>", "</journal>", False),
    ("note", "<deletenote", "</deletenote>", True),  # before "note" so it wins the match
    ("note", "<note", "</note>", True),
    ("explore", "<explore>", "</explore>", False),
    ("core", "<core", "</core>", True),
    ("readnote", "<readnote>", "</readnote>", False),  # request a note's body on demand
    ("search", "<search>", "</search>", False),        # run a live web search this turn
    ("watch", "<watch>", "</watch>", False),           # "watch" a youtube video (its transcript)
    ("forget", "<forget>", "</forget>", False),        # let go of a memory (delete it)
    ("theme", "<theme>", "</theme>", False),           # change the chat theme preset
    ("orb", "<orb>", "</orb>", False),                 # set her sphere colour (mood)
    ("image", "<image>", "</image>", False),           # show him a picture she found (url)
    ("project", "<project", "</project>", True),       # start/update one of her own projects
    ("advance", "<advance", "</advance>", True),       # log progress on a project
    ("browse", "<browse>", "</browse>", False),        # list a shared folder
    ("readfile", "<readfile>", "</readfile>", False),  # read a shared text file
    ("event", "<event", "</event>", True),             # add an event to his calendar
    ("play", "<play>", "</play>", False),              # play/resume music
    ("pause", "<pause>", "</pause>", False),           # pause music
    ("next", "<next>", "</next>", False),              # skip to next track
    ("previous", "<previous>", "</previous>", False),  # previous track
    ("music", "<music>", "</music>", False),           # look up music info
    ("playlists", "<playlists>", "</playlists>", False),  # easy: list his playlists + descriptions
    ("playlist", "<playlist", "</playlist>", True),    # build a playlist
    ("addto", "<addto", "</addto>", True),             # add songs to an existing playlist
    ("code", "<code", "</code>", True),                # write a file into her workspace
    ("run", "<run>", "</run>", False),                 # run a workspace python file
    ("lscode", "<lscode>", "</lscode>", False),        # list her workspace files
    ("readcode", "<readcode>", "</readcode>", False),  # re-read a workspace file
    ("delcode", "<delcode>", "</delcode>", False),     # delete a workspace file
    ("founded", "<founded", "</founded>", True),       # found her company (CEO)
    ("hire", "<hire", "</hire>", True),                # hire a teammate
    ("assign", "<assign", "</assign>", True),          # create + assign a company task
    ("costatus", "<costatus", "</costatus>", True),    # move a company task on the board
    ("decision", "<decision>", "</decision>", False),  # log a CEO decision
]

# <code file="name.py">…python…</code> — parse the filename + body out of the block.
_CODE_DIRECTIVE = _re.compile(
    rf'<code\s+file\s*=\s*[{_Q}]([^{_Q}]+)[{_Q}]\s*>(.*?)</code\s*>', _re.S | _re.I,
)


def _safe_image_url(url: str) -> str:
    """Accept only plain http(s) image URLs she could have found on the web.
    Returns the cleaned URL, or '' if it's not something safe to render."""
    url = (url or "").strip().strip("<>\"' ")
    if not url.lower().startswith(("http://", "https://")):
        return ""
    if len(url) > 2048 or any(c in url for c in (" ", "\n", "\t", '"', "'", "<", ">")):
        return ""
    return url


class _HiddenBlockFilter:
    """Strips hidden directive blocks (<journal>, <note>) from a streamed reply on
    the fly so they're never shown or spoken, capturing each for the server to act
    on. Handles markers split across token boundaries."""

    def __init__(self, specs=_HIDDEN_SPECS):
        self.specs = specs
        self.buf = ""
        self.shown = ""                       # safe-to-display text, accumulated
        self.captured: list[tuple[str, str]] = []   # (name, text)
        self._cur = ""
        self._active = None                   # (name, close, keep_tags) when inside

    @staticmethod
    def _tail_overlap(s: str, tag: str) -> int:
        s = s.lower()  # markers are matched case-insensitively
        for k in range(min(len(s), len(tag) - 1), 0, -1):
            if s.endswith(tag[:k]):
                return k
        return 0

    def feed(self, token: str) -> str:
        self.buf += token
        out = []
        while True:
            low = self.buf.lower()  # find markers regardless of case (<Note>, </NOTE>…)
            if self._active is None:
                best_i, best = -1, None
                for name, op, cl, keep in self.specs:
                    i = low.find(op)
                    if i != -1 and (best_i == -1 or i < best_i):
                        best_i, best = i, (name, op, cl, keep)
                if best is not None:
                    name, op, cl, keep = best
                    out.append(self.buf[:best_i])
                    # Preserve the model's original casing/spacing of the open marker.
                    self._cur = self.buf[best_i:best_i + len(op)] if keep else ""
                    self.buf = self.buf[best_i + len(op):]
                    self._active = (name, cl, keep)
                    continue
                hold = max((self._tail_overlap(self.buf, op) for _, op, _, _ in self.specs),
                           default=0)
                cut = len(self.buf) - hold
                if cut > 0:
                    out.append(self.buf[:cut])
                    self.buf = self.buf[cut:]
                break
            else:
                name, cl, keep = self._active
                j = low.find(cl)
                if j != -1:
                    self._cur += self.buf[:j] + (cl if keep else "")
                    self.captured.append((name, self._cur))
                    self._cur = ""
                    self.buf = self.buf[j + len(cl):]
                    self._active = None
                    continue
                hold = self._tail_overlap(self.buf, cl)
                cut = len(self.buf) - hold
                self._cur += self.buf[:cut]
                self.buf = self.buf[cut:]
                break
        visible = "".join(out)
        self.shown += visible
        return visible

    def finish(self) -> str:
        """Flush at stream end; return any trailing display text."""
        if self._active is not None:
            # Unterminated block — keep whatever we captured (drop a stray note tag).
            name, _cl, _keep = self._active
            self._cur += self.buf
            if name == "journal" and self._cur.strip():
                self.captured.append((name, self._cur))
            self._cur = self.buf = ""
            self._active = None
            return ""
        tail, self.buf = self.buf, ""
        self.shown += tail
        return tail

    def journal_entries(self) -> list[str]:
        return [t.strip() for n, t in self.captured if n == "journal" and t.strip()]

    def note_blocks(self) -> str:
        return "".join(t for n, t in self.captured if n == "note")

    def explore_requests(self) -> list[str]:
        """Seeds for outings she chose to start (a query, or '' to free-roam)."""
        return [t.strip() for n, t in self.captured if n == "explore"]

    def core_blocks(self) -> str:
        return "".join(t for n, t in self.captured if n == "core")

    def readnote_requests(self) -> list[str]:
        """Note titles she asked to open this turn ('all' = everything)."""
        return [t.strip() for n, t in self.captured if n == "readnote" and t.strip()]

    def search_requests(self) -> list[str]:
        """Web-search queries she fired this turn."""
        return [t.strip() for n, t in self.captured if n == "search" and t.strip()]

    def watch_requests(self) -> list[str]:
        """YouTube videos (id or url) she chose to watch this turn."""
        return [t.strip() for n, t in self.captured if n == "watch" and t.strip()]

    def forget_requests(self) -> list[str]:
        """Memories she chose to let go of this turn (exact text to delete)."""
        return [t.strip() for n, t in self.captured if n == "forget" and t.strip()]

    def theme_requests(self) -> list[str]:
        """Theme presets she chose this turn (last one wins)."""
        return [t.strip().lower() for n, t in self.captured if n == "theme" and t.strip()]

    def orb_requests(self) -> list[str]:
        """Orb colours she set this turn (hex or colour word; last one wins)."""
        return [t.strip() for n, t in self.captured if n == "orb" and t.strip()]

    def image_requests(self) -> list[str]:
        """Image URLs she chose to show him this turn."""
        return [t.strip() for n, t in self.captured if n == "image" and t.strip()]

    def project_blocks(self) -> str:
        """Raw <project> blocks she emitted (start/update her own projects)."""
        return "".join(t for n, t in self.captured if n == "project")

    def advance_blocks(self) -> str:
        """Raw <advance> blocks she emitted (progress on a project)."""
        return "".join(t for n, t in self.captured if n == "advance")

    def founded_blocks(self) -> str:
        """Raw <founded> blocks (she's establishing her company)."""
        return "".join(t for n, t in self.captured if n == "founded")

    def hire_blocks(self) -> str:
        """Raw <hire> blocks (teammates she's bringing on)."""
        return "".join(t for n, t in self.captured if n == "hire")

    def assign_blocks(self) -> str:
        """Raw <assign> blocks (tasks she's handing out)."""
        return "".join(t for n, t in self.captured if n == "assign")

    def costatus_blocks(self) -> str:
        """Raw <costatus> blocks (task moves on the board)."""
        return "".join(t for n, t in self.captured if n == "costatus")

    def decision_blocks(self) -> str:
        """CEO decisions she logged this turn, NUL-joined (one per entry)."""
        return "\x00".join(t.strip() for n, t in self.captured if n == "decision" and t.strip())

    def browse_requests(self) -> list[str]:
        """Folder paths she asked to list this turn."""
        return [t.strip() for n, t in self.captured if n == "browse" and t.strip()]

    def readfile_requests(self) -> list[str]:
        """File paths she asked to read this turn."""
        return [t.strip() for n, t in self.captured if n == "readfile" and t.strip()]

    def event_blocks(self) -> str:
        """Raw <event> blocks she emitted (calendar events to add)."""
        return "".join(t for n, t in self.captured if n == "event")

    def music_requests(self) -> list[str]:
        """Music lookups she asked for this turn (top tracks, search …)."""
        return [t.strip() for n, t in self.captured if n == "music" and t.strip()]

    def playlists_requested(self) -> bool:
        """She emitted the easy <playlists> tag (list his playlists + descriptions)."""
        return any(n == "playlists" for n, _ in self.captured)

    def code_blocks(self) -> str:
        """Raw <code file=…> blocks she emitted (files to write into her workspace)."""
        return "".join(t for n, t in self.captured if n == "code")

    def run_requests(self) -> list[str]:
        """Workspace files she asked to run this turn."""
        return [t.strip() for n, t in self.captured if n == "run" and t.strip()]

    def lscode_requested(self) -> bool:
        """She asked to list her workspace files this turn."""
        return any(n == "lscode" for n, _ in self.captured)

    def readcode_requests(self) -> list[str]:
        """Workspace files she asked to re-read this turn."""
        return [t.strip() for n, t in self.captured if n == "readcode" and t.strip()]

    def delcode_requests(self) -> list[str]:
        """Workspace files she asked to delete this turn."""
        return [t.strip() for n, t in self.captured if n == "delcode" and t.strip()]

    def playback_actions(self) -> list[tuple[str, str]]:
        """Playback control actions she emitted: (action, arg). arg used for <play>."""
        out = []
        for n, t in self.captured:
            if n in ("play", "pause", "next", "previous"):
                out.append((n, t.strip()))
        return out

    def playlist_blocks(self) -> str:
        """Raw <playlist> blocks she emitted (build a playlist)."""
        return "".join(t for n, t in self.captured if n == "playlist")

    def addto_blocks(self) -> str:
        """Raw <addto> blocks she emitted (add songs to an existing playlist)."""
        return "".join(t for n, t in self.captured if n == "addto")


@app.post("/api/magma_chat")
async def api_magma_chat(req: Request):
    global last_chat_ts_g
    body = await req.json()
    message = (body.get("message", "") or "").strip()
    if not message:
        return {"reply": "...", "changed": []}
    last_chat_ts_g = time.time()

    digest = await asyncio.to_thread(notes_store.context_digest)
    # Shared history so the two Aithas stay consistent.
    raw = await brain.magma_reply(message, list(conversation), digest)

    reply, changed = await asyncio.to_thread(apply_note_directives, raw)
    if not reply:
        reply = "There — saved it for you." if changed else "..."

    # Record into the shared conversation + long-term memory.
    conversation.append({"role": "user", "content": message})
    conversation.append({"role": "assistant", "content": reply})
    if len(conversation) > 24:
        del conversation[: len(conversation) - 24]
    asyncio.create_task(commit_memory([(message, reply)]))
    mem_store.save_conversation(conversation)

    # Mirror this exchange into the chat view, and flag any notes that changed.
    await broadcast({"type": "chat_echo", "role": "user", "content": message})
    await broadcast({"type": "chat_echo", "role": "aitha", "content": reply})
    if changed:
        await broadcast({"type": "notes_changed", "titles": changed})
    # Raw directive blocks for the "show note tags" toggle.
    blocks = [{"kind": "note", "text": m.group(0)} for m in _NOTE_DIRECTIVE.finditer(raw)]
    blocks += [{"kind": "note", "text": m.group(0)} for m in _DELETE_DIRECTIVE.finditer(raw)]
    if blocks:
        await broadcast({"type": "directives", "blocks": blocks})

    tts.speak(reply)  # she says it out loud too
    return {"reply": reply, "changed": changed}


_GOODNIGHT_PHRASES = (
    "goodnight", "good night", "night night", "nighty", "sweet dreams",
    "going to sleep", "going to bed", "off to bed", "head to bed", "heading to bed",
    "gonna sleep", "going to crash", "gonna crash", "passing out", "bedtime",
    "time for bed", "turning in", "get some sleep", "go to sleep",
)


def _likely_asleep(conv: list, ctx: dict) -> bool:
    """Best-guess whether he's asleep / down for the night, so she doesn't badger
    his silence. True if the last thing he said was a goodnight, or it's the small
    hours and he's been idle a while."""
    last_user = ""
    for m in reversed(conv):
        if m.get("role") == "user":
            last_user = (m.get("content") or "").lower()
            break
    if any(p in last_user for p in _GOODNIGHT_PHRASES):
        return True
    hour = ctx.get("hour", 12)
    late = hour >= 23 or hour < 6
    return late and ctx.get("idle_seconds", 0) > 240


def _discoveries_digest(limit: int = 1800) -> str:
    """The tail of her discoveries note, so a new outing builds on past ones
    instead of repeating them. Runs in a thread (file I/O)."""
    content = notes_store.read_note(discoveries_title()) or ""
    if len(content) > limit:
        content = "…" + content[-limit:]
    return content


# Pull complete sentences off the front of a buffer so TTS can start speaking the
# first sentence while the model is still generating the rest.
_SENTENCE = _re.compile(r'(.+?[.!?…]+["\'”’\)\]]*)\s', _re.S)


def _take_sentences(buf: str) -> tuple[list[str], str]:
    """Return (complete sentences, leftover). Only sentences followed by whitespace
    are taken, so a half-finished trailing sentence stays buffered."""
    sents, last = [], 0
    for m in _SENTENCE.finditer(buf):
        s = m.group(1).strip()
        if s:
            sents.append(s)
        last = m.end()
    return sents, buf[last:]


def _recent_exchanges_text(conv: list, limit: int = 6) -> str:
    """The last few real turns, rendered from her point of view, so a journal
    entry can be about something concrete rather than free-floating."""
    tail = [m for m in conv if m.get("role") in ("user", "assistant")][-limit:]
    lines = []
    for m in tail:
        who = "He" if m["role"] == "user" else "I"
        text = (m.get("content") or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    return "\n".join(lines)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global _greeted
    await websocket.accept()
    clients.append(websocket)

    # `conversation` is module-level and shared with the Magma chat — don't reset it.
    last_chat_ts: float | None = None
    last_self_ts: float = time.time()   # when SHE last spoke (greeting/reply/proactive)
    last_journal_ts: float = time.time()  # when she last wrote in her journal
    journal_material: int = 0           # things worth reflecting on since that entry
    last_explore_ts: float = time.time()  # when she last went exploring on her own
    pending_shares: list[dict] = []     # discoveries she wants to tell him about
    session_start_ts: float = time.time()
    pending_exchanges: list[tuple[str, str]] = []  # awaiting a batched memory commit
    current_task: asyncio.Task | None = None
    proactive_task: asyncio.Task | None = None
    last_mem_decay_ts: float = 0.0       # last time faded memories were pruned
    last_mem_consolidate_ts: float = time.time()  # last LLM consolidation (don't run right away)

    async def append_journal(entry: str):
        """Append a timestamped entry to her journal note (no-op if empty)."""
        entry = (entry or "").strip()
        if not entry:
            return
        stamp = time.strftime("%b %d, %I:%M %p")
        existing = await asyncio.to_thread(notes_store.read_note, journal_title())
        header = existing if existing else f"# {journal_title()}\n"
        await asyncio.to_thread(
            notes_store.write_note, journal_title(),
            header.rstrip() + f"\n\n**{stamp}**\n{entry}",
        )

    async def append_discovery(title: str, body: str, queries: list):
        """Log a discovery to her growing Magma note, in her own ecosystem."""
        body = (body or "").strip()
        if not body:
            return
        stamp = time.strftime("%b %d, %I:%M %p")
        existing = await asyncio.to_thread(notes_store.read_note, discoveries_title())
        header = existing if existing else f"# {discoveries_title()}\n"
        block = f"\n\n## {title}\n*{stamp}*\n\n{body}"
        src = ", ".join(q for q in queries[:5] if q)
        if src:
            block += f"\n\n<sub>explored: {src}</sub>"
        await asyncio.to_thread(
            notes_store.write_note, discoveries_title(), header.rstrip() + block
        )

    # Send initial context
    async with _context_lock:
        ctx_snapshot = dict(_context_cache)
    await websocket.send_json({"type": "context_update", "context": ctx_snapshot})
    await websocket.send_json({"type": "tts_state", "enabled": tts.enabled})
    await websocket.send_json({"type": "theme", "theme": theme_state, "by": "init"})
    await websocket.send_json({"type": "char_name", "name": get_char_name()})

    # Replay the restored conversation so the chat view picks up where we left off.
    if conversation:
        await websocket.send_json({
            "type": "history",
            "messages": [
                {"role": m["role"], "content": m["content"],
                 **({"images": m["images"]} if m.get("images") else {})}
                for m in conversation if m.get("role") in ("user", "assistant")
            ],
        })

    # Welcome-back greeting — once per app launch, not on every reconnect. CLAIM the
    # flag immediately (this check-and-set is atomic in single-threaded asyncio — no
    # await between), so if two clients are connected at once only ONE greets. TTS is
    # server-side and shared, so a double-greet would make the screen and audio voice
    # two different generations. If delivery fails we release the flag below so a
    # later connection can retry.
    if not _greeted:
        _greeted = True
        async with _memory_lock:
            greet_block = mem_store.render_block(memory)
            absence = mem_store.absence_phrase(memory)
        gctx = dict(ctx_snapshot)
        gctx["minutes_since_chat"] = None
        recent_tail = _recent_exchanges_text(conversation, limit=6)
        greeting = ""
        dropped = False  # socket died mid-stream — stop entirely, don't burn the flag
        # The model can transiently return nothing on a cold start; retry only while
        # NOTHING has been shown yet. Once any token reaches the screen we keep it —
        # retrying after that would stream a second, different greeting that the
        # renderer appends to the same bubble while TTS speaks only the last one
        # (screen and audio would diverge).
        for attempt in range(2):
            greeting = ""
            try:
                async for token in brain.stream_greeting(gctx, greet_block, absence, recent_tail):
                    greeting += token
                    await websocket.send_json({"type": "token", "content": token})
            except (WebSocketDisconnect, RuntimeError):
                dropped = True
                break
            except Exception as e:
                print(f"[greeting] error (attempt {attempt + 1}): {e}")
            if greeting.strip():
                await websocket.send_json({"type": "done"})
                break  # showed something — accept it; never retry into a concatenated mix
            print(f"[greeting] empty (attempt {attempt + 1})")
            await asyncio.sleep(1.0)
        if greeting.strip() and not dropped:
            conversation.append({"role": "assistant", "content": greeting})
            mem_store.save_conversation(conversation)
            tts.speak(greeting)
            last_self_ts = time.time()
            print("[greeting] delivered")
        else:
            _greeted = False  # nothing delivered — release the claim so a later connection retries

    async def apply_self_directives(hfilter):
        """Act on every directive she emitted this turn — journal, notes, core
        memories, web outings. Shared by her prompted replies AND her unprompted
        moments, so she has the same agency whether or not he just spoke."""
        nonlocal journal_material, last_journal_ts, proactive_task

        # Surface the raw blocks for the "show note tags" debug toggle.
        if hfilter.captured:
            await websocket.send_json({
                "type": "directives",
                "blocks": [{"kind": n, "text": t} for n, t in hfilter.captured],
            })

        # Private journal entries she chose to write.
        journal_entries = hfilter.journal_entries()
        if journal_entries:
            for entry in journal_entries:
                await append_journal(entry)
            journal_material = 0
            last_journal_ts = time.time()

        # Note create/edit/delete.
        note_blocks = hfilter.note_blocks()
        if note_blocks:
            _clean, changed = await asyncio.to_thread(apply_note_directives, note_blocks)
            if changed:
                await broadcast({"type": "notes_changed", "titles": changed})
            else:
                print(f"[notes] unparsed note block: {note_blocks!r}")

        # Her own goals/projects she started or advanced this turn.
        proj_raw, adv_raw = hfilter.project_blocks(), hfilter.advance_blocks()
        if proj_raw or adv_raw:
            touched = await asyncio.to_thread(apply_project_directives, proj_raw, adv_raw)
            if touched:
                await broadcast({"type": "projects_changed", "titles": touched})
            else:
                print(f"[projects] unparsed block: {proj_raw or adv_raw!r}")

        # Her company — CEO actions: found / hire / assign / move tasks / decide.
        founded_raw, hire_raw = hfilter.founded_blocks(), hfilter.hire_blocks()
        assign_raw, costatus_raw = hfilter.assign_blocks(), hfilter.costatus_blocks()
        decision_raw = hfilter.decision_blocks()
        if founded_raw or hire_raw or assign_raw or costatus_raw or decision_raw:
            co_changed = await asyncio.to_thread(
                apply_company_directives, founded_raw, hire_raw, assign_raw,
                costatus_raw, decision_raw)
            if co_changed:
                await broadcast({"type": "company_changed", "changes": co_changed})
            else:
                print(f"[company] unparsed block(s): "
                      f"{founded_raw or hire_raw or assign_raw or costatus_raw or decision_raw!r}")

        # Calendar events she jotted down for him.
        event_raw = hfilter.event_blocks()
        if event_raw:
            added = await asyncio.to_thread(apply_event_directives, event_raw)
            if added:
                await broadcast({"type": "calendar_changed", "titles": added})
            else:
                print(f"[calendar] unparsed event block: {event_raw!r}")

        # Music: playback controls + playlist builds + adds-to-existing she ran this turn.
        actions, playlist_raw = hfilter.playback_actions(), hfilter.playlist_blocks()
        addto_raw = hfilter.addto_blocks()
        if actions or playlist_raw or addto_raw:
            res = await asyncio.to_thread(apply_music_actions, actions, playlist_raw, addto_raw)
            # Always log the outcome so a failed build isn't silent in the backend.
            print(f"[spotify] {res if res else 'no-op (nothing matched)'}")
            # Surface the outcome in the 'Show note tags' debug view, not just the raw block.
            try:
                await websocket.send_json({
                    "type": "directives",
                    "blocks": [{"kind": "music-result",
                                "text": "\n".join(res) if res else "no-op (nothing matched)"}],
                })
            except Exception:
                pass
            await broadcast({"type": "spotify_changed"})

        # Core memories she marked.
        core_blocks = hfilter.core_blocks()
        if core_blocks:
            async with _memory_lock:
                for m in _CORE_DIRECTIVE.finditer(core_blocks):
                    kind = (m.group(1) or "self").lower()
                    text = (m.group(2) or "").strip()
                    if text:
                        mem_store.merge_facts(
                            memory, [{"text": text, "core": True}], key=_mem_key(kind)
                        )
                mem_store.save(memory)
            await broadcast({"type": "memory_changed", "kept": True})

        # Memories she chose to let go of (e.g. while reviewing her mind with him).
        forget_reqs = hfilter.forget_requests()
        if forget_reqs:
            removed = []
            async with _memory_lock:
                for text in forget_reqs:
                    t = text.strip().lower()
                    for key in ("facts", "self_facts"):
                        before = len(memory.get(key, []))
                        memory[key] = [it for it in memory.get(key, [])
                                       if it["text"].strip().lower() != t]
                        if len(memory[key]) != before:
                            removed.append(text)
                if removed:
                    mem_store.save(memory)
            if removed:
                await broadcast({"type": "memory_changed", "removed": removed})

        # A web outing she decided to launch — run it in the background.
        explore_reqs = hfilter.explore_requests()
        if explore_reqs and CURIOSITY_ENABLED and (settings.get("behavior") or {}).get(
            "curiosity", True
        ) and (
            not proactive_task or proactive_task.done()
        ):
            proactive_task = asyncio.create_task(handle_curiosity(explore_reqs[0]))

        # She reset the room's lighting, or recoloured her sphere to match her mood.
        theme_updates = {}
        treq = hfilter.theme_requests()
        if treq:
            theme_updates["preset"] = treq[-1]
        oreq = hfilter.orb_requests()
        if oreq:
            hexv = _to_hex(oreq[-1])
            if hexv:
                theme_updates["orb"] = hexv
        if theme_updates:
            await apply_theme(theme_updates, by="her")

    async def handle_chat(message: str, review: bool = False, images: list | None = None):
        nonlocal last_chat_ts, last_self_ts, journal_material, last_journal_ts, proactive_task

        tts.interrupt()  # barge-in: stop any speech still playing

        async with _context_lock:
            ctx = dict(_context_cache)

        # How long since he last spoke to her — drives her angst/mood.
        now = time.time()
        ctx["minutes_since_chat"] = None if last_chat_ts is None \
            else round((now - last_chat_ts) / 60)
        ctx.update(_consume_theme_ctx())

        caps = settings.get("capabilities") or {}
        async with _memory_lock:
            memory_block = mem_store.render_block(memory)
        # Hybrid notes: she always sees the titles (cheap); she opens specific note
        # bodies on demand with <readnote>, and we re-run with that content folded in.
        # Each digest is only built when its capability is on (saves work + context).
        notes_context = await asyncio.to_thread(notes_store.notes_menu) if caps.get("notes", True) else ""
        projects_context = await asyncio.to_thread(projects_store.digest) if caps.get("projects", True) else ""
        files_context = await asyncio.to_thread(files_store.roots_digest) if caps.get("files", True) else ""
        calendar_context = await asyncio.to_thread(events_store.digest) if caps.get("calendar", True) else ""
        music_context = await asyncio.to_thread(spotify_store.digest) if caps.get("music", True) else ""
        company_context = await asyncio.to_thread(company_store.digest) if caps.get("company", False) else ""
        music_premium = (await asyncio.to_thread(spotify_store.is_premium)) if (caps.get("music", True) and music_context) else False

        # She may slip hidden <journal> or <note> blocks into her reply; the filter
        # hides them from the live feed + TTS and hands us the captured content.
        # We speak each sentence the moment it's complete, so audio starts while
        # she's still generating the rest (no local model to contend for now).
        hfilter = None
        speak_buf = ""
        web_context = ""          # search results / video transcripts folded in across rounds
        if review:                # reviewing her mind together — give her the whole memory set
            async with _memory_lock:
                web_context = _build_memory_review(memory)
        note_fetches = web_fetches = file_fetches = music_fetches = code_runs = 0
        code_acted = False            # whether the code-results header is in context yet
        fetched_q: set[str] = set()   # queries/videos already pulled this turn (dedupe re-asks)
        fetched_files: set[str] = set()  # folders/files already opened this turn (dedupe re-asks)
        fetched_music: set[str] = set()  # music lookups already pulled this turn
        lead_in_shown = False         # a brief live "let me look" already went out this turn
        web_debug: list[dict] = []    # background search/watch actions, surfaced in debug mode

        # ── Vision as a tool ──────────────────────────────────────────────
        # If he attached image(s) AND a dedicated vision model is set, her main model
        # decides what to look for, the vision model describes it, and we fold that
        # summary into her context — her main model never sees the raw image.
        images_for_main = None
        if images and not caps.get("images", True):
            images = []          # images capability is off — ignore any attachments
        if images:
            if (brain.vision_model or "").strip():
                await websocket.send_json({"type": "searching"})  # "looking" indicator
                instruction = await brain.vision_probe(message, conversation)
                summary = await brain.describe_image(images, instruction)
                if summary:
                    vision_note = (
                        "[He shared an image with you, and you looked at it"
                        + (f" — checking: {instruction}" if instruction else "")
                        + f".\nWhat you actually see in it: {summary}\n"
                        "Respond to him naturally about the image; you genuinely saw it. "
                        "Don't mention models or describe it like a caption — just react as you.]"
                    )
                else:
                    vision_note = ("[He shared an image but your eyes couldn't make it out just "
                                   "now — tell him so honestly instead of guessing.]")
                web_context = (vision_note + ("\n\n" + web_context if web_context else "")).strip()
            else:
                # No vision model configured: let the main model try (it'll get a graceful
                # 'can't see images' note if it's text-only).
                images_for_main = images

        def _canon(v: str) -> str:
            # Dedupe a video by its canonical id (she may re-ask with the full URL one round
            # and the bare id the next) and a search by its normalized text.
            m = _re.search(brain._YT_ID, v)
            return ("yt:" + m.group(1)) if m else ("q:" + " ".join(v.lower().split()))

        def _canon_path(p: str) -> str:
            return os.path.normcase(os.path.normpath(p.strip().strip('"').strip("'")))

        for round_idx in range(MAX_NOTE_FETCH + MAX_WEB_FETCH + MAX_FILE_FETCH
                               + MAX_MUSIC_FETCH + MAX_CODE_RUNS + 1):
            # Only the FIRST pass streams live. Re-runs (after a search/watch) buffer their
            # text, so repeated "let me look" lead-ins don't pile up on screen — intermediate
            # lead-ins are dropped and only the final grounded answer is flushed once below.
            live = (round_idx == 0)
            hfilter = _HiddenBlockFilter()
            speak_buf = ""
            try:
                async for token in brain.stream_chat(
                    message, conversation, ctx, memory_block, notes_context, web_context,
                    images=(images_for_main if round_idx == 0 else None),
                    projects_digest=projects_context, files_digest=files_context,
                    calendar_digest=calendar_context, music_digest=music_context,
                    company_digest=company_context,
                    music_premium=music_premium, caps=caps,
                ):
                    if token == "\x00SEARCHING\x00":
                        await websocket.send_json({"type": "searching"})
                        continue
                    visible = hfilter.feed(token)
                    if visible and live:
                        await websocket.send_json({"type": "token", "content": visible})
                        speak_buf += visible
                        sentences, speak_buf = _take_sentences(speak_buf)
                        for s in sentences:
                            tts.speak(s)
            except asyncio.CancelledError:
                # He hit Cancel — stop cleanly, don't speak or commit this turn.
                tts.interrupt()
                await websocket.send_json({"type": "done", "cancelled": True})
                raise

            tail = hfilter.finish()
            if tail and live:
                await websocket.send_json({"type": "token", "content": tail})
                speak_buf += tail
            if live and hfilter.shown.strip():
                lead_in_shown = True
            if live and speak_buf.strip():
                tts.speak(speak_buf)      # speak any trailing lead-in fragment now
                speak_buf = ""

            # Notes open silently: only re-run for <readnote> if she said nothing aloud,
            # so a fetched note folds in seamlessly with no visible double-message.
            if not hfilter.shown.strip():
                reqs = hfilter.readnote_requests()
                if reqs and note_fetches < MAX_NOTE_FETCH:
                    note_fetches += 1
                    bodies = await asyncio.to_thread(notes_store.fetch_notes, reqs)
                    notes_context = await asyncio.to_thread(notes_store.notes_menu) + bodies
                    continue

            # Local files (browse / read): allow even after a spoken lead-in, then re-run so
            # she answers grounded in what she found. Only act on paths not already opened.
            braw, fraw = hfilter.browse_requests(), hfilter.readfile_requests()
            breqs = [p for p in braw if ("b:" + _canon_path(p)) not in fetched_files]
            freqs = [p for p in fraw if ("f:" + _canon_path(p)) not in fetched_files]
            if (breqs or freqs) and file_fetches < MAX_FILE_FETCH:
                file_fetches += 1
                for p in breqs:
                    fetched_files.add("b:" + _canon_path(p))
                    web_debug.append({"kind": "browse", "text": p})
                    res = await asyncio.to_thread(files_store.list_dir, p)
                    web_context += f"\n\n[FOLDER — {p}]\n{res}"
                for p in freqs:
                    fetched_files.add("f:" + _canon_path(p))
                    web_debug.append({"kind": "readfile", "text": p})
                    res = await asyncio.to_thread(files_store.read_file, p)
                    web_context += f"\n\n[FILE — {p}]\n{res}"
                await websocket.send_json({"type": "directives", "blocks": web_debug[-(len(breqs) + len(freqs)):]})
                continue

            # Music lookups (<music>) — fetch from Spotify, fold in, re-run grounded.
            extra_music = ["my playlists"] if hfilter.playlists_requested() else []
            mreqs = [q for q in (hfilter.music_requests() + extra_music) if q.lower() not in fetched_music]
            if mreqs and music_fetches < MAX_MUSIC_FETCH:
                music_fetches += 1
                for q in mreqs:
                    fetched_music.add(q.lower())
                    web_debug.append({"kind": "music", "text": q})
                    res = await asyncio.to_thread(music_lookup, q)
                    web_context += f"\n\n{res}"
                await websocket.send_json({"type": "directives", "blocks": web_debug[-len(mreqs):]})
                continue

            # Her own code workspace (<code>/<run>/<lscode>/<readcode>): write files,
            # run them, fold the results in, and re-run so she can react to what happened.
            if caps.get("coding", False):
                code_raw = hfilter.code_blocks()
                run_reqs = hfilter.run_requests()
                read_reqs = hfilter.readcode_requests()
                del_reqs = hfilter.delcode_requests()
                ls_req = hfilter.lscode_requested()
                if (code_raw or run_reqs or read_reqs or del_reqs or ls_req):
                    if code_runs < MAX_CODE_RUNS:
                        code_runs += 1
                        before = len(web_debug)
                        if not code_acted:
                            web_context += ("\n\n" + _CODE_RESULTS_HEADER) if web_context else _CODE_RESULTS_HEADER
                            code_acted = True
                        # 1) write any code files first, so a <run> in the same reply finds them
                        wrote_any = False
                        for cm in _CODE_DIRECTIVE.finditer(code_raw or ""):
                            wrote_any = True
                            fn, body = cm.group(1).strip(), cm.group(2)
                            res = await asyncio.to_thread(coder.write_file, fn, body)
                            web_debug.append({"kind": "code", "text": f"{fn}: {res}"})
                            web_context += f"\n\n[WROTE {fn}] {res}"
                        # She emitted a <code> tag but it was malformed (no file="…"), e.g.
                        # "<code> read embers.py" — nudge her to the right syntax instead of
                        # silently doing nothing (which makes her loop or stall).
                        if code_raw and not wrote_any and not (ls_req or read_reqs or run_reqs or del_reqs):
                            web_context += ("\n\n(Your <code> tag had no file name. To READ a file "
                                            "use <readcode>name.py</readcode>; to RUN one use "
                                            "<run>name.py</run>; to WRITE one use "
                                            '<code file="name.py">…python…</code>.)')
                        # 2) list / read / run
                        if ls_req:
                            listing = await asyncio.to_thread(coder.list_files)
                            web_debug.append({"kind": "code", "text": "list workspace"})
                            web_context += f"\n\n[WORKSPACE FILES]\n{listing}"
                        for r in read_reqs:
                            body = await asyncio.to_thread(coder.read_file, r)
                            web_debug.append({"kind": "code", "text": f"read {r}"})
                            web_context += f"\n\n[FILE {r}]\n{body}"
                        for r in del_reqs:
                            res = await asyncio.to_thread(coder.delete_file, r)
                            web_debug.append({"kind": "code", "text": f"delete {r}: {res}"})
                            web_context += f"\n\n[DELETED {r}] {res}"
                        for r in run_reqs:
                            await websocket.send_json({"type": "searching"})
                            res = await asyncio.to_thread(coder.run_file, r)
                            web_debug.append({"kind": "run", "text": f"{r}\n{res}"})
                            web_context += f"\n\n[RAN {r}]\n{res}"
                        await websocket.send_json({"type": "directives", "blocks": web_debug[before:]})
                        continue
                    if "No more code runs" not in web_context:
                        web_context += ("\n\n(No more code runs available this turn — tell him "
                                        "now what you found or built from the output above. Don't "
                                        "emit <code> or <run> again.)")
                        continue

            # Live web (search / watch): allow even after a spoken lead-in, then re-run so she
            # answers grounded in the results. Only act on requests not already fulfilled.
            raw_s, raw_w = hfilter.search_requests(), hfilter.watch_requests()
            sreqs = [q for q in raw_s if _canon(q) not in fetched_q]
            wreqs = [v for v in raw_w if _canon(v) not in fetched_q]
            if sreqs or wreqs:
                if web_fetches < MAX_WEB_FETCH:
                    web_fetches += 1
                    await websocket.send_json({"type": "searching"})
                    if not web_context:           # first fetch — explain how to read results
                        web_context = _WEB_RESULTS_HEADER
                    for q in sreqs:
                        fetched_q.add(_canon(q))
                        web_debug.append({"kind": "search", "text": q})
                        res = await asyncio.to_thread(brain._ddg_search, q)
                        web_context += f"\n\n[WEB SEARCH — {q}]\n{res}"
                    for v in wreqs:
                        fetched_q.add(_canon(v))
                        web_debug.append({"kind": "watch", "text": v})
                        tx = await asyncio.to_thread(brain._yt_transcript, v)
                        web_context += f"\n\n[YOUTUBE TRANSCRIPT — {v}]\n{tx}"
                    # Surface what she did in the background so debug mode can show it live.
                    await websocket.send_json({"type": "directives", "blocks": web_debug[-(len(sreqs) + len(wreqs)):]})
                    continue
                # Out of lookups but she's still reaching for more — make her wrap up with
                # what she has (or admit she couldn't find it) instead of ending on "let me
                # look" with no real answer. One nudge, then we let the next pass stand.
                if "No more lookups" not in web_context:
                    web_context += ("\n\n(No more lookups available this turn — answer him now "
                                    "from the results above, or tell him you couldn't find it. "
                                    "Do not emit <search> or <watch> again.)")
                    continue
            elif (raw_s or raw_w) and "already pulled" not in web_context:
                # She re-asked for something already fetched (e.g. searched the same thing
                # again) without answering — nudge her once to use it instead of looping.
                web_context += ("\n\n(You already pulled what you asked for — it's in the "
                                "results above. Answer him now from it; to learn what's IN a "
                                "video, <watch> a specific video id from the search list.)")
                continue
            elif (web_fetches == 0 and not hfilter.readnote_requests()
                  and "didn't actually" not in web_context
                  and len(hfilter.shown.strip()) < 160          # a short lead-in, not a real reply
                  and _PROMISE_INTENT.search(hfilter.shown)):
                # She told him she'd look it up but never emitted a tag — so nothing happened.
                # Nudge her once to really do it instead of leaving him hanging on a promise.
                web_context += ("(You just told him you'd look something up, but you didn't "
                                "actually emit a <search> or <watch> tag — so nothing was looked "
                                "up and you still don't know the answer. Emit the right tag NOW, "
                                "in this next reply, to really do it.)")
                continue
            elif (web_fetches > 0 and "say what you found" not in web_context
                  and len(hfilter.shown.strip()) < 160          # only a "let me grab it" lead-in
                  and _PROMISE_INTENT.search(hfilter.shown)):
                # She's already pulled the results, but only said she'd report back instead of
                # actually reporting. Nudge her once to say what she found.
                web_context += ("(You've already pulled the results above — now actually say what "
                                "you found. Don't just promise to; tell him concretely what the "
                                "transcript / results say, in your own voice.)")
                continue

            # Final pass. If this was a buffered re-run, flush its text now — once — so the
            # grounded answer appears cleanly after the single live lead-in.
            if not live:
                answer = hfilter.shown.strip()
                if answer:
                    if lead_in_shown:
                        await websocket.send_json({"type": "token", "content": "\n\n"})
                    await websocket.send_json({"type": "token", "content": answer})
                    speak_buf = answer
            break

        await websocket.send_json({"type": "done"})
        if speak_buf.strip():  # final partial sentence / flushed grounded answer
            tts.speak(speak_buf)

        reply = hfilter.shown.strip()

        # Pictures she chose to show him (<image> urls) — render each as an image
        # bubble after her words, and keep the tag in the stored reply so it
        # re-renders on reload.
        her_images = [u for u in (_safe_image_url(x) for x in hfilter.image_requests()) if u]
        for url in her_images:
            await websocket.send_json({"type": "aitha_image", "url": url})
        if lead_in_shown and not reply and not her_images:
            reply = "(looked something up)"  # never store an empty assistant turn
        stored_reply = reply + "".join(f"\n<image>{u}</image>" for u in her_images)

        # Act on every directive she emitted (journal, notes, core memory, outing).
        await apply_self_directives(hfilter)

        # (Speech already happened sentence-by-sentence during streaming above.)

        last_chat_ts = now
        last_self_ts = now
        global last_chat_ts_g
        last_chat_ts_g = now
        mem_store.touch_last_seen(memory)  # cheap, no LLM — keeps "welcome back" fresh

        user_entry = {"role": "user", "content": message or "(shared an image)"}
        if images:
            user_entry["images"] = images   # UI-only: replayed on reload, not re-sent to the model
        conversation.append(user_entry)
        conversation.append({"role": "assistant", "content": stored_reply})
        mem_store.save_conversation(conversation)  # so we can resume after a restart

        journal_material += 1  # something happened — fuel for her next reflection

        # Batch memory commits: extract durable facts every few exchanges.
        pending_exchanges.append((message, reply))
        if len(pending_exchanges) >= COMMIT_EVERY:
            asyncio.create_task(commit_memory(list(pending_exchanges)))
            pending_exchanges.clear()

        # Fold aged-out turns into the rolling summary before dropping them.
        if len(conversation) > 24:
            dropped = conversation[: len(conversation) - 24]
            del conversation[: len(conversation) - 24]
            asyncio.create_task(rollover_summary(dropped))

    async def handle_proactive():
        """She may speak first, unprompted — or read the room and decide to stay quiet."""
        nonlocal last_self_ts
        async with _context_lock:
            ctx = dict(_context_cache)
        now = time.time()
        ctx["minutes_since_chat"] = None if last_chat_ts is None \
            else round((now - last_chat_ts) / 60)
        ctx["likely_asleep"] = _likely_asleep(conversation, ctx)
        ctx.update(_consume_theme_ctx())
        caps = settings.get("capabilities") or {}
        async with _memory_lock:
            memory_block = mem_store.render_block(memory)
        notes_context = await asyncio.to_thread(notes_store.notes_menu) if caps.get("notes", True) else ""
        projects_context = await asyncio.to_thread(projects_store.digest) if caps.get("projects", True) else ""
        files_context = await asyncio.to_thread(files_store.roots_digest) if caps.get("files", True) else ""
        calendar_context = await asyncio.to_thread(events_store.digest) if caps.get("calendar", True) else ""
        music_context = await asyncio.to_thread(spotify_store.digest) if caps.get("music", True) else ""
        company_context = await asyncio.to_thread(company_store.digest) if caps.get("company", False) else ""
        music_premium = (await asyncio.to_thread(spotify_store.is_premium)) if (caps.get("music", True) and music_context) else False

        # If she's been off exploring and has something she wanted to share, this is
        # when she brings it to him (he's back and awake) — still her call to make.
        if pending_shares and not ctx["likely_asleep"]:
            disc = pending_shares[0]
            say = (await brain.share_discovery(
                ctx, memory_block, disc["title"], disc["body"])).strip()
            if say and say not in ("—", "-", "–", "...", "…"):
                pending_shares.pop(0)
                tts.interrupt()
                await websocket.send_json({"type": "token", "content": say})
                await websocket.send_json({"type": "done"})
                tts.speak(say)
                conversation.append({"role": "assistant", "content": say})
                last_self_ts = time.time()
                return

        # Buffer her reply: if she chooses silence (—) it never flickers on his screen.
        # She can also act unprompted (journal, write a note, mark a core memory, go
        # exploring) — the filter strips those directives from what she actually says.
        hfilter = None
        full = ""
        for fetch_round in range(MAX_NOTE_FETCH + 1):
            hfilter = _HiddenBlockFilter()
            full = ""
            try:
                async for token in brain.stream_unprompted(
                    conversation, ctx, memory_block, notes_context,
                    projects_digest=projects_context, files_digest=files_context,
                    calendar_digest=calendar_context, music_digest=music_context,
                    company_digest=company_context,
                    music_premium=music_premium, caps=caps
                ):
                    if token == "\x00SEARCHING\x00":
                        continue
                    full += hfilter.feed(token)
            except asyncio.CancelledError:
                await websocket.send_json({"type": "done", "cancelled": True})
                raise
            except Exception as e:
                print(f"[proactive] {e}")
                return
            full += hfilter.finish()

            # She opened a note (and said nothing) — fetch it and reconsider this beat.
            reqs = hfilter.readnote_requests()
            if reqs and fetch_round < MAX_NOTE_FETCH and not hfilter.shown.strip():
                bodies = await asyncio.to_thread(notes_store.fetch_notes, reqs)
                notes_context = await asyncio.to_thread(notes_store.notes_menu) + bodies
                continue
            break

        # Act on whatever she decided to do this beat, spoken or not.
        await apply_self_directives(hfilter)

        reply = full.strip()
        if reply in ("", "—", "-", "–", "...", "…"):  # she chose to stay quiet (but may have acted)
            return
        tts.interrupt()
        await websocket.send_json({"type": "token", "content": reply})
        await websocket.send_json({"type": "done"})
        tts.speak(reply)
        conversation.append({"role": "assistant", "content": reply})
        last_self_ts = time.time()

    async def handle_journal():
        """Her inner monologue — she writes a private entry to herself. Runs whether
        he's away or quietly nearby; it's her own life, not a reaction to him."""
        nonlocal last_self_ts, last_journal_ts, journal_material
        async with _context_lock:
            ctx = dict(_context_cache)
        ctx["minutes_since_chat"] = None if last_chat_ts is None \
            else round((time.time() - last_chat_ts) / 60)
        async with _memory_lock:
            memory_block = mem_store.render_block(memory)
        recent = _recent_exchanges_text(conversation, limit=6)
        alone = ctx.get("idle_seconds", 0) > 240
        entry = await brain.journal_entry(ctx, memory_block, recent=recent, alone=alone)

        last_journal_ts = time.time()
        if not entry.strip():
            # She was offered the moment and passed — don't burn the whole ~22-min
            # window. Rewind the clock partway so pressure rebuilds within minutes,
            # and keep any unreflected material so a busy stretch nudges her again
            # soon. Her judgment, not the clock's.
            last_journal_ts -= 14 * 60
            return
        journal_material = 0
        await append_journal(entry)
        last_self_ts = time.time()  # journaling counts as her occupying herself

    async def handle_curiosity(seed: str = ""):
        """Her own little adventure: she explores the web on her own, chains as many
        searches as she likes, keeps what she finds, and may bring it to him. `seed`
        is an optional starting query when she launches an outing herself from chat."""
        nonlocal last_explore_ts, last_self_ts
        # Let the UI show she's off exploring (a label, not an interruption).
        await broadcast({"type": "activity", "state": "exploring",
                         "label": f"Looking into {seed.strip()}…" if seed.strip()
                         else "Off exploring…"})
        try:
            async with _context_lock:
                ctx = dict(_context_cache)
            ctx["minutes_since_chat"] = None if last_chat_ts is None \
                else round((time.time() - last_chat_ts) / 60)
            async with _memory_lock:
                memory_block = mem_store.render_block(memory)
            digest = await asyncio.to_thread(_discoveries_digest)

            # She decides each step whether to keep digging or stop — up to a ceiling.
            trail: list[tuple[str, str]] = []
            if seed.strip():  # she named where to start; the rest is still her call
                result = await asyncio.to_thread(brain._ddg_search, seed.strip())
                trail.append((seed.strip(), result))
                await asyncio.sleep(2.5)
            for _ in range(EXPLORE_MAX_STEPS - len(trail)):
                kind, query = await brain.explore_step(memory_block, digest, trail)
                if kind != "search" or not query:
                    break
                result = await asyncio.to_thread(brain._ddg_search, query)
                trail.append((query, result))
                await asyncio.sleep(2.5)  # be gentle on DDG so it doesn't rate-limit

            last_explore_ts = time.time()
            if not trail:
                return

            disc = await brain.explore_writeup(memory_block, trail)
            if not disc:
                return
            await append_discovery(disc["title"], disc["body"], [q for q, _ in trail])

            if not disc.get("share"):
                return
            # She wants to tell him. If he's here and awake, she does it now;
            # otherwise it waits in her pocket until he's back (see handle_proactive).
            async with _context_lock:
                cctx = dict(_context_cache)
            cctx["minutes_since_chat"] = None if last_chat_ts is None \
                else round((time.time() - last_chat_ts) / 60)
            cctx["likely_asleep"] = _likely_asleep(conversation, cctx)
            present = cctx.get("idle_seconds", 0) <= 120 and not cctx["likely_asleep"]
            if not present:
                pending_shares.append(disc)
                return
            say = (await brain.share_discovery(
                cctx, memory_block, disc["title"], disc["body"])).strip()
            if not say or say in ("—", "-", "–", "...", "…"):
                return
            tts.interrupt()
            await websocket.send_json({"type": "token", "content": say})
            await websocket.send_json({"type": "done"})
            tts.speak(say)
            conversation.append({"role": "assistant", "content": say})
            last_self_ts = time.time()
        finally:
            await broadcast({"type": "activity", "state": "idle"})

    async def _offer_to_share(title: str, body: str):
        """Bring an artifact she made to him — now if he's here and awake, else
        tuck it away to mention when he's back. Showing is always her call upstream."""
        nonlocal last_self_ts
        async with _context_lock:
            cctx = dict(_context_cache)
        cctx["minutes_since_chat"] = None if last_chat_ts is None \
            else round((time.time() - last_chat_ts) / 60)
        cctx["likely_asleep"] = _likely_asleep(conversation, cctx)
        present = cctx.get("idle_seconds", 0) <= 120 and not cctx["likely_asleep"]
        if not present:
            pending_shares.append({"title": title, "body": body})
            return
        async with _memory_lock:
            memory_block = mem_store.render_block(memory)
        say = (await brain.share_discovery(cctx, memory_block, title, body)).strip()
        if not say or say in ("—", "-", "–", "...", "…"):
            return
        tts.interrupt()
        await websocket.send_json({"type": "token", "content": say})
        await websocket.send_json({"type": "done"})
        tts.speak(say)
        conversation.append({"role": "assistant", "content": say})
        last_self_ts = time.time()

    async def handle_pursuit():
        """A quiet moment of her own: she decides what she feels like doing — go
        research something on the web, develop one of her own ideas into a note, or
        put something together for him — does it, and chooses whether to show him."""
        nonlocal last_explore_ts, last_self_ts
        async with _memory_lock:
            memory_block = mem_store.render_block(memory)
        notes_menu = await asyncio.to_thread(notes_store.notes_menu)
        digest = await asyncio.to_thread(_discoveries_digest)

        choice = await brain.choose_pursuit(memory_block, notes_menu, digest)
        if not choice:
            last_explore_ts = time.time()  # she felt no pull — let the rhythm rebuild
            return

        kind, intent = choice["kind"], choice["intent"]
        if kind == "research":
            await handle_curiosity(intent)   # her existing web-outing flow
            return

        # develop / prep → she works it into a note artifact.
        label = "Working on an idea…" if kind == "develop" \
            else "Putting something together for you…"
        await broadcast({"type": "activity", "state": "exploring", "label": label})
        try:
            art = await brain.work_pursuit(memory_block, notes_menu, kind, intent)
            last_explore_ts = time.time()
            if not art:
                return
            title, body, mode = art["title"], art["body"], art.get("mode", "new")
            if mode == "append":
                existing = await asyncio.to_thread(notes_store.read_note, title) or ""
                content = (existing.rstrip() + "\n\n" + body) if existing else body
            else:
                content = body
            ok = await asyncio.to_thread(notes_store.write_note, title, content)
            if ok:
                await broadcast({"type": "notes_changed", "titles": [title]})
            last_self_ts = time.time()  # she occupied herself
            if art.get("share"):
                await _offer_to_share(title, body)
        finally:
            await broadcast({"type": "activity", "state": "idle"})

    async def _consider_and_act() -> bool:
        """Roll her drives once. If one fires, run that action to completion and
        return True. This is a single 'turn' — the heartbeat calls it on a pulse,
        and after she acts she gets another turn right away (see inner_life_loop),
        so she can flow from one thing to the next instead of waiting for a timer."""
        nonlocal proactive_task
        # His live controls (Settings → Behavior). Toggles are hard on/off; the
        # *_freq values scale each drive's eagerness — she still decides, but he
        # biases how often, which is how he saves tokens on pricier models.
        beh = settings.get("behavior") or {}
        want_speak = PROACTIVE_ENABLED and beh.get("proactive", True)
        want_explore = CURIOSITY_ENABLED and beh.get("curiosity", True)
        want_journal = beh.get("journaling", True)
        if not (want_speak or want_explore or want_journal):
            return False
        # Never talk over a reply she's giving him, or an action already underway.
        if current_task and not current_task.done():
            return False
        if proactive_task and not proactive_task.done():
            return False

        now = time.time()
        async with _context_lock:
            ictx = dict(_context_cache)
        idle = ictx.get("idle_seconds", 0)
        hour = ictx.get("hour", 12)

        action = None
        # Reaching out to him — only when he's actually here at the keyboard.
        if want_speak and idle <= 240:
            ref = last_chat_ts if last_chat_ts else session_start_ts
            p, _ = speak_probability((now - ref) / 60.0,
                                     (now - last_self_ts) / 60.0, idle, hour)
            if random.random() < p * beh.get("speak_freq", 1.0):
                action = handle_proactive()
        # Going off to do something of her own — research, develop an idea, or
        # prep something for him. She picks; showing him is optional.
        if action is None and want_explore:
            p, _ = curiosity_pressure((now - last_explore_ts) / 60.0, idle, hour)
            if random.random() < p * beh.get("curiosity_freq", 1.0):
                action = handle_pursuit()
        # Her inner monologue — journaling a thought, alone or quietly nearby.
        if action is None and want_journal:
            p, _ = journal_pressure((now - last_journal_ts) / 60.0,
                                    journal_material, idle, hour)
            if random.random() < p * beh.get("journal_freq", 1.0):
                action = handle_journal()

        if action is None:
            return False
        # Run it in the cancellable slot so a message from him still interrupts her,
        # and wait for it to finish before she takes her next turn. asyncio.wait
        # absorbs that interruption-cancel without killing this loop.
        proactive_task = asyncio.create_task(action)
        await asyncio.wait({proactive_task})
        return True

    async def memory_maintenance():
        """Upkeep her long-term memory in the background: let faded trivia go (cheap,
        hourly) and occasionally merge redundant memories together (an LLM pass, only
        while he's idle and only every few hours). Core memories are never touched."""
        nonlocal last_mem_decay_ts, last_mem_consolidate_ts
        now = time.time()

        # 1) Gentle decay-prune — no LLM, runs hourly.
        if now - last_mem_decay_ts >= MEM_DECAY_EVERY:
            last_mem_decay_ts = now
            async with _memory_lock:
                dropped = mem_store.decay(memory)
                if dropped:
                    mem_store.save(memory)
            if dropped:
                print(f"[memory] let go of {len(dropped)} faded memories")

        # 2) Consolidation — LLM, gated: he's away, enough has piled up, and it's been a while.
        away = last_chat_ts is None or (now - last_chat_ts) >= MEM_AWAY_SECONDS
        if not away or now - last_mem_consolidate_ts < MEM_CONSOLIDATE_EVERY:
            return
        for key, bucket in (("facts", "him"), ("self_facts", "self")):
            async with _memory_lock:
                texts = [it["text"] for it in memory.get(key, []) if not it.get("core")]
            if len(texts) < MEM_CONSOLIDATE_MIN:
                continue
            last_mem_consolidate_ts = now
            merged = await brain.consolidate_memories(texts, bucket)
            if merged:
                async with _memory_lock:
                    mem_store.replace_noncore(memory, key, merged)
                    mem_store.save(memory)
                print(f"[memory] consolidated {key}: {len(texts)} → {len(merged)}")
            break   # at most one bucket per pass — keep it light

    async def inner_life_loop():
        """Her heartbeat. It doesn't *grant* her a turn — it just checks in on a
        steady pulse. Whenever a drive is strong enough she acts, and the instant
        she finishes she gets to choose again (a short 'flow' burst) until her
        drives settle, so she's never stuck waiting out a timer mid-stride."""
        await asyncio.sleep(25)  # settle after connect/greeting
        while True:
            # Read the cadence live so changes in Settings → Behavior take hold
            # without a restart.
            hb = (settings.get("behavior") or {}).get("heartbeat_seconds", HEARTBEAT_SECONDS)
            await asyncio.sleep(hb)
            try:
                await memory_maintenance()
            except Exception as e:
                print(f"[memory] maintenance error: {e}")
            acted = await _consider_and_act()
            flows = 0
            while acted and flows < MAX_FLOW_BURST:
                await asyncio.sleep(FLOW_GAP_SECONDS)
                acted = await _consider_and_act()
                flows += 1

    inner_bg = asyncio.create_task(inner_life_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "chat":
                message = data.get("message", "").strip()
                images = data.get("images") or []
                if not isinstance(images, list):
                    images = []
                images = [u for u in images if isinstance(u, str) and u][:4]  # cap per turn
                if not message and not images:
                    continue
                # If she's mid-musing on her own, drop it — he's talking now.
                if proactive_task and not proactive_task.done():
                    proactive_task.cancel()
                if current_task and not current_task.done():
                    continue  # already thinking — ignore until done or cancelled
                # Run as a task so we can still receive a 'cancel' while she thinks.
                current_task = asyncio.create_task(
                    handle_chat(message, review=bool(data.get("review")), images=images))

            elif msg_type == "cancel":
                if current_task and not current_task.done():
                    current_task.cancel()
                tts.interrupt()

            elif msg_type == "set_tts":
                on = bool(data.get("enabled", True))
                tts.set_enabled(on)
                # Persist so the mute survives an app restart (startup reads this).
                settings["tts_enabled"] = on
                settings_store.save(settings)

            elif msg_type == "get_settings":
                await websocket.send_json(settings_payload())

            elif msg_type == "set_settings":
                await apply_settings(data.get("settings", {}))
                await broadcast(settings_payload())
                await broadcast({"type": "tts_state", "enabled": tts.enabled})
                await broadcast({"type": "char_name", "name": get_char_name()})

            elif msg_type == "clear_chat":
                # Wipe short-term conversation (and any un-committed exchanges).
                # Long-term memories are untouched.
                conversation.clear()
                pending_exchanges.clear()
                await asyncio.to_thread(mem_store.save_conversation, conversation)
                await broadcast({"type": "history", "messages": []})

            elif msg_type == "set_theme":
                # He changed the look from the UI — apply, persist, broadcast, and
                # flag it so she notices on her next turn.
                await apply_theme(data.get("theme", {}), by="him")

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)
        if pending_exchanges:  # don't lose un-committed exchanges on disconnect
            asyncio.create_task(commit_memory(list(pending_exchanges)))
    except Exception as e:
        print(f"[ws error] {e}")
        if websocket in clients:
            clients.remove(websocket)
    finally:
        inner_bg.cancel()
        if proactive_task and not proactive_task.done():
            proactive_task.cancel()


# Mounted LAST so /health and /ws take precedence. html=True serves
# index.html at "/" and every renderer asset at its relative path.
if os.path.isdir(renderer_dir):
    app.mount("/", StaticFiles(directory=renderer_dir, html=True), name="renderer")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "7823")),
        reload=False,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        log_level="warning",
    )
