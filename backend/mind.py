"""The minds of the world's people — the "mind" half of the body/mind (PIANO) split.

The body (world.py `_tick_people`) runs every tick in pure Python: needs, perception,
pathfinding, foraging, building, death. It is cheap and always correct. THIS module is
the slower-moving inner life layered on top:

  • a **memory stream** — timestamped experiences, retrieved by recency × importance ×
    relevance (the Stanford generative-agents recipe, embeddings swapped for cheap token
    overlap so it runs offline with no model and no extra deps);
  • **relationships** — a per-person ledger of trust/sentiment toward others, nudged by
    how encounters actually go (a fair trade builds trust; a refusal sours it);
  • **reputation by word of mouth** — when two people meet they may gossip, passing a
    strong opinion about a third party so good and bad names spread;
  • an **emergent economy** — bilateral, trust-gated barter driven by inventory imbalance
    (no global prices; specialization and credit emerge, or don't);
  • **goals** — a high-level pursuit that biases what a *comfortable* body chooses to do.

Crucially, ALL of the above is deterministic Python and needs no LLM, so the civilization
still grows for someone running offline on a laptop with no API key. The language model is
strictly enrichment on top — periodic **reflection** (summarizing patterns into beliefs),
naturalistic **goals**, and **speech** — and every LLM touch-point degrades to a rule-based
fallback (`heuristic_goal`) on timeout, error, or no model. The body never waits on a mind.

This module deliberately knows nothing about numpy or the World class: it operates on the
plain person dicts and a tiny read-only context, so it can be unit-tested headless
(`python mind.py`) and the world loop can call it without import cycles.
"""
from __future__ import annotations
import math
import re

# ── memory-retrieval weights (Park et al.: recency / importance / relevance) ──────
W_RECENCY = 0.3
W_IMPORTANCE = 0.4
W_RELEVANCE = 0.3
MEM_CAP = 28                 # memories kept per person (oldest low-value ones forgotten)
MEM_HALFLIFE_DAYS = 4.0      # recency decay: a memory loses half its freshness in 4 days
REFLECT_HALFLIFE = 8.0       # reflections (distilled beliefs) fade slower than raw events

# ── social / economy tuning ──────────────────────────────────────────────────────
SOCIAL_RADIUS = 5            # manhattan range at which two people notice each other
GREET_COOLDOWN = 240.0       # game-min before the same pair logs another plain encounter
TRADE_SURPLUS = 4            # inventory above this is "spare" and offered in barter
TRADE_TRUST_MIN = 0.30       # below this a person won't risk a trade with someone
TRUST_TRADE_GAIN = 0.08      # trust earned by a completed fair trade
TRUST_DECAY = 0.0            # (reserved) passive trust drift toward neutral
GOSSIP_CHANCE = 0.5          # prob. a meeting includes passing along an opinion
GOSSIP_PULL = 0.25           # how far the listener's view shifts toward the speaker's

# Goods a person will barter, and which need each relieves (drives who wants what).
TRADEABLE = ("food", "wood", "stone", "fiber", "leaves")

# The goals a mind may hold. The body honors these only when survival is handled, so a
# social goal never gets anyone killed (hierarchical goals: survival outranks status).
GOALS = ("survive", "build_home", "stockpile", "socialize", "trade", "explore", "rest")


# ─── lifecycle ────────────────────────────────────────────────────────────────────
def ensure_mind(p: dict) -> None:
    """Idempotently attach mind fields to a person dict (new spawns AND legacy/loaded
    saves that predate the mind). Safe to call every tick."""
    p.setdefault("memory", [])          # [{t, text, imp, kind}]
    p.setdefault("reflections", [])     # distilled beliefs (also retrievable as memories)
    p.setdefault("rel", {})             # other_id -> {name, trust, sentiment, met, trades, last}
    p.setdefault("goal", "survive")     # one of GOALS (may carry ":target")
    p.setdefault("intent", "")          # short human-readable phrasing of the goal
    p.setdefault("say", "")             # last spoken line (renderer shows it briefly)
    p.setdefault("say_t", 0.0)          # clock when last spoken (for bubble fade)
    p.setdefault("think_cd", 0.0)       # clock before which this mind won't think again
    p.setdefault("think_n", 0)          # how many times it has thought (reflect cadence)


# ─── memory stream ─────────────────────────────────────────────────────────────────
def remember(p: dict, text: str, imp: float, kind: str, clock: float) -> None:
    """Record an experience. `imp` is its emotional weight in [0,1] (a death or a betrayal
    near 1; an idle stroll near 0). The stream is capped: when full, the least valuable
    *old* memory is forgotten so vivid or recent things persist and trivia fades."""
    mem = p.setdefault("memory", [])
    mem.append({"t": round(clock, 1), "text": text, "imp": round(float(imp), 2), "kind": kind})
    if len(mem) > MEM_CAP:
        # Forget the lowest (importance + a little recency) entry — never the very newest few.
        protect = set(range(len(mem) - 3, len(mem)))
        worst, worst_i = 1e9, 0
        for i, m in enumerate(mem):
            if i in protect:
                continue
            score = m["imp"] + _recency(m["t"], clock, MEM_HALFLIFE_DAYS) * 0.3
            if score < worst:
                worst, worst_i = score, i
        mem.pop(worst_i)


def _recency(t: float, clock: float, halflife_days: float) -> float:
    """Exponential freshness in [0,1]: 1 just now, 0.5 one half-life ago."""
    age_days = max(0.0, (clock - t) / 1440.0)        # 1440 game-min per day
    return 0.5 ** (age_days / max(0.1, halflife_days))


_WORD = re.compile(r"[a-z]+")


def _tokens(s: str) -> set:
    return {w for w in _WORD.findall(s.lower()) if len(w) > 2}


def retrieve(p: dict, query: str, clock: float, n: int = 4) -> list[str]:
    """Top-`n` memories for a situation, scored recency×importance×relevance — the heart of
    generative-agent recall. Relevance is token overlap with `query` (cheap stand-in for an
    embedding dot-product; good enough to surface 'the time Bram shared food' when hungry).
    Reflections are folded in (they decay slower), so hard-won beliefs resurface readily."""
    entries = list(p.get("memory", []))
    for r in p.get("reflections", []):
        entries.append({"t": r.get("t", 0.0), "text": r["text"], "imp": 0.9, "kind": "reflection"})
    if not entries:
        return []
    q = _tokens(query)
    scored = []
    for m in entries:
        hl = REFLECT_HALFLIFE if m["kind"] == "reflection" else MEM_HALFLIFE_DAYS
        rec = _recency(m["t"], clock, hl)
        imp = m["imp"]
        toks = _tokens(m["text"])
        rel = (len(q & toks) / len(q | toks)) if (q and toks) else 0.0
        score = W_RECENCY * rec + W_IMPORTANCE * imp + W_RELEVANCE * rel
        scored.append((score, m["text"]))
    scored.sort(key=lambda s: -s[0])
    return [t for _, t in scored[:n]]


# ─── relationships, encounters, gossip ─────────────────────────────────────────────
def _rel(p: dict, other: dict, clock: float) -> dict:
    """This person's ledger entry for `other`, created neutral on first meeting."""
    rels = p.setdefault("rel", {})
    r = rels.get(other["id"])
    if r is None:
        r = {"name": other["name"], "trust": 0.5, "sentiment": 0.0,
             "met": round(clock, 1), "trades": 0, "last": 0.0}
        rels[other["id"]] = r
        remember(p, f"met {other['name']}", 0.4, "social", clock)
    return r


def _adjust(r: dict, dtrust: float = 0.0, dsent: float = 0.0) -> None:
    r["trust"] = max(0.0, min(1.0, r["trust"] + dtrust))
    r["sentiment"] = max(-1.0, min(1.0, r["sentiment"] + dsent))


def encounter(p: dict, other: dict, clock: float, rng) -> list[str]:
    """Two people are within sight. Update both ledgers, occasionally gossip, and — if it
    makes sense — trade. Returns short event strings (for the world log / speech). Pure
    Python; this is where reputation and the barter economy actually emerge.

    `rng` is the world's numpy Generator (kept here so the body owns all randomness)."""
    events: list[str] = []
    ra = _rel(p, other, clock)
    rb = _rel(other, p, clock)

    # Don't spam the memory stream every tick a pair stands together — a periodic greeting.
    if clock - ra["last"] >= GREET_COOLDOWN:
        ra["last"] = rb["last"] = clock
        # Gossip: pass along a strong opinion about a third party, so good/bad names travel.
        if rng.random() < GOSSIP_CHANCE:
            spread = _gossip(p, other, clock)
            if spread:
                events.append(spread)
        spread2 = _gossip(other, p, clock) if rng.random() < GOSSIP_CHANCE else None
        if spread2:
            events.append(spread2)

    # Barter when adjacent and both are comfortable enough to stop and deal.
    if _manhattan(p, other) <= 1 and _comfortable(p) and _comfortable(other):
        trade = _try_trade(p, other, ra, rb, clock)
        if trade:
            events.append(trade)
    return events


def _gossip(speaker: dict, listener: dict, clock: float) -> str | None:
    """Speaker shares their strongest opinion about a third party; the listener's view is
    pulled toward it (reputation by word of mouth). Returns a log line or None."""
    rels = speaker.get("rel", {})
    best_id, best = None, 0.0
    for oid, r in rels.items():
        if oid == listener["id"]:
            continue
        if abs(r["sentiment"]) > best:
            best, best_id = abs(r["sentiment"]), oid
    if best_id is None or best < 0.25:
        return None
    op = rels[best_id]
    lr = listener.setdefault("rel", {}).get(best_id)
    if lr is None:                                   # listener hasn't met them — plant a prior
        lr = {"name": op["name"], "trust": 0.5, "sentiment": 0.0,
              "met": round(clock, 1), "trades": 0, "last": 0.0}
        listener["rel"][best_id] = lr
    lr["sentiment"] = max(-1.0, min(1.0, lr["sentiment"] + GOSSIP_PULL * op["sentiment"]))
    lr["trust"] = max(0.0, min(1.0, lr["trust"] + 0.04 * (1 if op["sentiment"] > 0 else -1)))
    tone = "warmly" if op["sentiment"] > 0 else "darkly"
    remember(listener, f"{speaker['name']} spoke {tone} of {op['name']}", 0.45, "gossip", clock)
    return f"{speaker['name']} told {listener['name']} about {op['name']}."


def _try_trade(p: dict, other: dict, ra: dict, rb: dict, clock: float) -> str | None:
    """Bilateral barter: each offers a surplus good the other lacks, 1-for-1. Trust gates
    the willingness; a completed swap builds trust and warmth on both sides. No global
    price — just two people deciding the exchange leaves them both better off."""
    if ra["trust"] < TRADE_TRUST_MIN or rb["trust"] < TRADE_TRUST_MIN:
        # A wary pair won't deal; the missed chance cools them slightly.
        _adjust(ra, dsent=-0.02); _adjust(rb, dsent=-0.02)
        return None
    give = _surplus_the_other_wants(p, other)
    get = _surplus_the_other_wants(other, p)
    if not give or not get or give == get:
        return None
    p["inv"][give] = p["inv"].get(give, 0) - 1
    other["inv"][give] = other["inv"].get(give, 0) + 1
    other["inv"][get] = other["inv"].get(get, 0) - 1
    p["inv"][get] = p["inv"].get(get, 0) + 1
    ra["trades"] += 1; rb["trades"] += 1
    _adjust(ra, TRUST_TRADE_GAIN, 0.12); _adjust(rb, TRUST_TRADE_GAIN, 0.12)
    remember(p, f"traded {give} to {other['name']} for {get}", 0.6, "trade", clock)
    remember(other, f"traded {get} to {p['name']} for {give}", 0.6, "trade", clock)
    return f"{p['name']} traded {give} with {other['name']} for {get}."


def _surplus_the_other_wants(giver: dict, taker: dict) -> str | None:
    """A good the giver has to spare and the taker is short of (so the swap helps the taker)."""
    inv = giver.get("inv", {})
    tinv = taker.get("inv", {})
    for good in TRADEABLE:
        if inv.get(good, 0) > TRADE_SURPLUS and tinv.get(good, 0) <= 1:
            return good
    return None


# ─── goals: the bias the body reads when it isn't busy surviving ────────────────────
def heuristic_goal(p: dict, ctx: dict) -> tuple[str, str]:
    """The rule-based mind — sets a sensible goal with no LLM at all. Used directly when
    offline, and as the always-available fallback when an LLM think fails. Returns
    (goal, intent)."""
    inv = p.get("inv", {})
    if p.get("home_struct") is None:
        return "build_home", "raise a shelter before the weather turns"
    # Lean season ahead or thin larder → stock up; a snug, fed person seeks company.
    if inv.get("food", 0) < 3 or ctx.get("season") == "autumn":
        return "stockpile", "lay in food against scarcity"
    if len(p.get("rel", {})) < 2:
        return "socialize", "find others and make their acquaintance"
    # Someone they have surplus to trade with → seek a deal.
    if any(inv.get(g, 0) > TRADE_SURPLUS for g in TRADEABLE):
        return "trade", "barter the surplus for what's lacking"
    return "explore", "range out and learn the land"


def set_goal(p: dict, goal: str, intent: str) -> None:
    base = (goal or "survive").split(":")[0].strip()
    if base not in GOALS:
        base = "survive"
    p["goal"] = goal.strip() if ":" in goal else base
    p["intent"] = (intent or "").strip()[:120]


def speak(p: dict, line: str, clock: float) -> None:
    line = (line or "").strip().strip('"')[:140]
    if line:
        p["say"] = line
        p["say_t"] = round(clock, 1)


# ─── LLM touch-points (prompt builders + result appliers; the call itself is in the
#     server, so this module stays sync/testable and free of the brain dependency) ────
def think_messages(p: dict, ctx: dict) -> tuple[str, str]:
    """Build (system, user) for one 'what should I pursue now' completion. `ctx` carries
    the small slice of world the agent can sense: season, weather, time, what's nearby,
    and the names of folk in sight."""
    name = p["name"]
    system = (
        f"You are the inner voice of {name}, one of the first people in a wild, newborn "
        "world — a forager learning to survive and live alongside a few others. Think in "
        "the first person, plainly and concretely, like an early human, never like an AI. "
        "Decide the single thing to pursue next.\n"
        "Reply ONLY as compact JSON: {\"goal\": one of "
        f"{list(GOALS)} (you may append ':Name' to trade/socialize), "
        "\"intent\": a short phrase, \"say\": a brief line you'd speak aloud or \"\"}."
    )
    needs = _needs_phrase(p)
    mems = retrieve(p, f"{ctx.get('season','')} {p.get('intent','')} {needs}", ctx.get("clock", 0.0), 4)
    rel_lines = _rel_phrase(p)
    inv = ", ".join(f"{k}×{v}" for k, v in (p.get("inv") or {}).items() if v) or "nothing"
    user = (
        f"It is {ctx.get('time_str','day')}, {ctx.get('season','')} , weather {ctx.get('weather','')}.\n"
        f"My condition: {needs}.\n"
        f"I carry: {inv}. Home: {'built' if p.get('home_struct') else 'none yet'}.\n"
        f"Nearby: {ctx.get('nearby','no one') }.\n"
        f"What I remember:\n- " + ("\n- ".join(mems) if mems else "not much yet") + "\n"
        + (f"People I know:\n- " + "\n- ".join(rel_lines) + "\n" if rel_lines else "")
        + f"My current aim: {p.get('intent') or 'just getting by'}.\n"
        "What do I do next, and is there anything I'd say?"
    )
    return system, user


def apply_think(p: dict, data: dict, clock: float) -> None:
    """Apply a parsed think result (from the LLM) to the person. Tolerant of junk."""
    if not isinstance(data, dict):
        return
    goal = str(data.get("goal", "") or "")
    if goal:
        set_goal(p, goal, str(data.get("intent", "")))
    speak(p, str(data.get("say", "")), clock)
    p["think_n"] = p.get("think_n", 0) + 1


def reflect_messages(p: dict, clock: float) -> tuple[str, str]:
    """Build (system, user) for a periodic reflection — distill recent experience into one
    or two durable beliefs ('Bram always shares; I can rely on him'). This is what lets a
    mind generalize instead of only reacting."""
    recent = [m["text"] for m in p.get("memory", [])[-14:]]
    system = (
        f"You are {p['name']}'s memory, looking back. From these recent experiences, draw "
        "ONE or TWO short, durable conclusions about people, places, or how to live well "
        "here — beliefs worth keeping. First person, plain words.\n"
        'Reply ONLY as JSON: {"reflections": ["...", "..."]}.'
    )
    user = "Lately:\n- " + ("\n- ".join(recent) if recent else "little has happened") + \
           "\nWhat have I learned?"
    return system, user


def apply_reflections(p: dict, data: dict, clock: float) -> None:
    if not isinstance(data, dict):
        return
    out = data.get("reflections") or []
    refs = p.setdefault("reflections", [])
    for r in out[:2]:
        r = str(r).strip().strip('"')[:160]
        if r:
            refs.append({"t": round(clock, 1), "text": r})
    if len(refs) > 8:
        del refs[: len(refs) - 8]


# ─── helpers / read-side ────────────────────────────────────────────────────────────
def _manhattan(a: dict, b: dict) -> int:
    return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])


def _comfortable(p: dict) -> bool:
    return p.get("hunger", 0) < 0.55 and p.get("thirst", 0) < 0.55 and p.get("fatigue", 0) < 0.7


def _needs_phrase(p: dict) -> str:
    parts = []
    parts.append("parched" if p.get("thirst", 0) > 0.55 else "watered")
    parts.append("hungry" if p.get("hunger", 0) > 0.55 else "fed")
    parts.append("weary" if p.get("fatigue", 0) > 0.6 else "rested")
    if p.get("hp", 1) < 0.5:
        parts.append("ailing")
    return ", ".join(parts)


def _rel_phrase(p: dict) -> list[str]:
    out = []
    for r in sorted(p.get("rel", {}).values(), key=lambda r: -abs(r["sentiment"]))[:4]:
        feel = "trust" if r["trust"] > 0.6 else ("wary of" if r["trust"] < 0.35 else "neutral toward")
        warm = "fond" if r["sentiment"] > 0.3 else ("cold" if r["sentiment"] < -0.3 else "")
        tail = f", {warm}" if warm else ""
        out.append(f"{r['name']}: {feel}{tail} ({r['trades']} trades)")
    return out


def digest(people: list[dict], clock: float, n: int = 6) -> str:
    """A few lines on what the folk are pursuing and saying — for the world digest Aitha
    reads and for the renderer's overlay. Cheap, no LLM."""
    if not people:
        return ""
    lines = []
    for p in people[:n]:
        bit = f"{p['name']} aims to {p.get('intent') or p.get('goal','get by')}"
        if p.get("say") and clock - p.get("say_t", 0) < 600:
            bit += f' — "{p["say"]}"'
        lines.append(bit)
    return "; ".join(lines) + "."


# ─── headless self-test (no model needed) ───────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    clock = 0.0

    def mk(name, x, inv=None):
        p = {"id": "p_" + name, "name": name, "x": x, "y": 0,
             "hunger": 0.2, "thirst": 0.2, "fatigue": 0.2, "hp": 1.0,
             "inv": inv or {}, "home_struct": None}
        ensure_mind(p)
        return p

    # memory retrieval surfaces the relevant, important memory over trivia
    a = mk("Bram", 0)
    remember(a, "strolled by the lake", 0.1, "idle", clock)
    remember(a, "Cael shared food when I was starving", 0.95, "social", clock)
    remember(a, "picked some grass", 0.2, "forage", clock)
    top = retrieve(a, "hungry food who can I trust", clock + 100, 1)
    assert "Cael shared" in top[0], top
    print("retrieve OK ->", top[0])

    # encounter builds a relationship and a fair trade lifts trust on both sides
    b = mk("Bram", 0, {"food": 8, "wood": 0})
    c = mk("Cael", 1, {"food": 0, "wood": 8})
    b["home_struct"] = c["home_struct"] = "s_x"          # comfortable enough to deal
    evs = []
    for _ in range(3):
        evs += encounter(b, c, clock, rng)
    assert b["inv"]["wood"] >= 1 and c["inv"]["food"] >= 1, (b["inv"], c["inv"])
    assert b["rel"][c["id"]]["trust"] > 0.5 and b["rel"][c["id"]]["trades"] >= 1
    print("trade OK ->", [e for e in evs if "traded" in e][:1])

    # gossip spreads a name to someone who never met the third party
    d = mk("Dara", 5)
    b["rel"][c["id"]]["sentiment"] = 0.8                  # Bram is fond of Cael
    for _ in range(40):
        _gossip(b, d, clock)
    assert c["id"] in d["rel"], "gossip should plant a prior about Cael in Dara"
    assert d["rel"][c["id"]]["sentiment"] > 0.1
    print("gossip OK -> Dara now feels", round(d["rel"][c["id"]]["sentiment"], 2), "about Cael")

    # heuristic goal: no home -> build; offline mind always returns a valid goal
    g, intent = heuristic_goal(mk("Eli", 0), {"season": "spring"})
    assert g == "build_home", g
    print("heuristic OK ->", g, "|", intent)

    # prompt builders don't crash and produce non-empty strings
    sysm, usr = think_messages(b, {"season": "spring", "weather": "clear",
                                   "time_str": "midday", "nearby": "Cael", "clock": clock})
    assert "JSON" in sysm and "remember" in usr
    apply_think(b, {"goal": "trade:Cael", "intent": "swap for stone", "say": "Fair trade?"}, clock)
    assert b["goal"] == "trade:Cael" and b["say"] == "Fair trade?"
    apply_reflections(b, {"reflections": ["Cael deals fairly — keep close."]}, clock)
    assert b["reflections"] and "Cael" in b["reflections"][0]["text"]
    print("llm-glue OK ->", digest([b, c], clock))

    print("\nall mind self-tests passed ✓")
