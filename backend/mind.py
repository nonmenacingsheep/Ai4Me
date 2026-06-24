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
import itertools
import re

import crafting   # the recipe registry / discovery matcher (pure data, no cycle)

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
STORY_RENOWN = 0.25          # standing at which a soul's tales start to carry to others
STORY_CHANCE = 0.35          # prob. a meeting with a soul of standing passes on one of their tales

# Goods a person will barter, and which need each relieves (drives who wants what).
TRADEABLE = ("food", "wood", "stone", "fiber", "leaves")

# ─── the thinking-civilization core: drives, temperament, intentions ─────────────────
# This is a "mind-first" sim: survival is not a hardcoded priority chain but ONE drive
# among many, competing in a utility arbiter against belonging, status, curiosity and
# fear. Whatever wins becomes the person's INTENTION, and the body merely actuates it.
# The arbiter is pure Python so a full inner life runs offline; the LLM (when present)
# supplies what a utility function cannot — novel intentions, reasons, beliefs, identity
# and speech — at a slow, human cadence.

# A person's standing intention is one of these kinds (some carry a target person id):
INTENT_KINDS = ("drink", "eat", "rest", "build", "provide", "provision", "socialize",
                "befriend", "explore", "avoid", "tend", "tinker", "ply", "flee", "guard", "help",
                "forage", "whittle", "marvel", "aspire")

PROVISION_TARGET = 12       # a settled soul lays in food at home up to this before it eases off
# Stockpiling scales with the season — lay in heavily through autumn against the lean winter,
# ease off in spring's plenty (mirrors world.SEASON_STOCK_MULT; kept here so the mind stays
# free of any world import).
SEASON_STOCK_MULT = {"spring": 1.0, "summer": 1.15, "autumn": 2.1, "winter": 1.6}

# Vocations — a soul's calling, emerging from temperament (division of labour). A driven soul
# becomes the band's BUILDER, a restless curious one its TOOLMAKER, and the steady rest its
# FORAGERS. Each plies a different trade in idle hours, producing a different surplus — which
# is what gives the barter economy something to move and gifting something to share.
VOCATIONS = ("forager", "builder", "toolmaker")


def vocation(p: dict) -> str:
    """The soul's calling, read from its (lived-bent) temperament. Ambition builds, curiosity
    makes; everyone else forages — the steady backbone that keeps the band fed."""
    amb, cur = _trait(p, "ambition"), _trait(p, "curiosity")
    if amb >= 0.55 and amb >= cur:
        return "builder"
    if cur >= 0.55 and cur > amb:
        return "toolmaker"
    return "forager"

# The vocabulary a tinkering mind brainstorms from when guessing how to make a make-shift
# craft (raws it can gather plus rope, which every band knows). Offline discovery is honest
# trial-and-error over this small space; the LLM reasons straight to the answer.
CRAFT_VOCAB = ("leaves", "fiber", "rope", "wood", "stone")

# Loose names → canonical material ids, so an LLM saying "cord", "vines" or "branches"
# still lands on the right ingredient.
MATERIAL_SYNONYMS = {
    "cord": "rope", "cordage": "rope", "twine": "rope", "string": "rope", "vine": "rope",
    "vines": "rope", "plant fiber": "fiber", "fibre": "fiber", "fibres": "fiber",
    "grass": "fiber", "reeds": "fiber", "straw": "fiber", "leaf": "leaves",
    "foliage": "leaves", "frond": "leaves", "fronds": "leaves", "branch": "wood",
    "branches": "wood", "log": "wood", "logs": "wood", "timber": "wood", "stick": "wood",
    "rock": "stone", "rocks": "stone", "stones": "stone", "pebble": "stone",
}


def canon_material(name: str) -> str | None:
    """Map a free-text material name to a known ingredient id, or None if unrecognized."""
    n = (name or "").strip().lower()
    if n in MATERIAL_SYNONYMS:
        return MATERIAL_SYNONYMS[n]
    if n in crafting.ITEMS:
        return n
    if n.endswith("s") and n[:-1] in crafting.ITEMS:   # singular fallback (stones→stone)
        return n[:-1]
    return None

# Temperament — fixed-ish character rolled at birth. It weights the drives, so two people
# in the same circumstance want different things. This is where individuality (and, in
# aggregate, culture) comes from. Values (below) nudge these over a life from experience.
TRAITS = ("sociability", "ambition", "curiosity", "caution")

# A survival drive now reads TWO signals (the body/comfort split): the COMFORT/desire (how
# much the soul wants relief — rises early, drives ordinary behaviour) and the physiological
# RESERVE (how much the body can still take — the true survival clock). Urgency = a strong
# but BOUNDED desire term + a STEEP danger term that only ignites as the reserve runs out.
# So a comfortable-but-thirsty soul is firmly pulled toward water (desire), while a soul whose
# body is actually failing is dragged there no matter what else it wanted (danger override).
DESIRE_GAIN = 0.80          # CONCAVE (<1): desire peaks EARLY & strong, so a moderately thirsty
                            # soul already feels a firm pull (comfort 0.7 → desire ~0.75) and tops
                            # up proactively rather than tolerating near-empty comfort. (A convex
                            # >1 curve peaks late — souls would rest through rising thirst and only
                            # scramble once comfort pinned, stranding the inland-settled ones.)
# Two reserve terms shape survival. A gentle ALWAYS-ON tilt (proportional to how drawn-down the
# reserve is) is the discriminator: among three needs all sitting at peak desire, it nudges the
# soul to service the MOST-depleted reserve first, so they round-robin their upkeep and none
# quietly craters while another is tended. A STEEP ramp below ~half-full is the true override —
# overwhelming as the body actually nears failure. In steady state both stay small (routine
# relief keeps reserves high), so meaning and projects still get their turn.
RESERVE_TILT = 0.40         # gentle, always-on pull toward the lowest reserve (the discriminator)
DANGER_RESERVE = 0.50       # reserve level below which the steep danger ramp begins
DANGER_SLOPE = 2.5          # steepness of that ramp — overwhelming as the reserve nears empty


def need_urgency(comfort: float, reserve: float = 1.0) -> float:
    """Urgency of relieving a need — the survival drive's value, also used by the body's
    emergency-interrupt so the two agree on when survival must take over. `comfort` is the
    0..1 desire; `reserve` is the 0..1 physiological store (1 = full, 0 = the body failing)."""
    desire = comfort ** DESIRE_GAIN
    danger = (1.0 - reserve) * RESERVE_TILT + max(0.0, DANGER_RESERVE - reserve) * DANGER_SLOPE
    return desire + danger
HYSTERESIS = 0.12            # an intention must be beaten by this margin to be dropped
REST_BY_DAY_DAMP = 0.35      # daytime: ordinary drowsiness is damped to this fraction (keeps daylight hours)
REST_BY_DAY_RESERVE = 0.35   # …but a stamina reserve at/below this is true exhaustion — rest pull stands
BUILD_MOMENTUM = 0.35       # sunk-cost pull: a fully-underway build adds this much to its drive,
                            # so a half-raised home is FINISHED rather than abandoned mid-wall for
                            # some lesser whim (the user's "houses never get done" complaint)
SOCIAL_FORGET = 1440.0      # a day alone fully "lonelies" a sociable soul
VALUE_CAP = 0.25            # how far lived experience can bend a trait


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


# ─── lifecycle ────────────────────────────────────────────────────────────────────
def ensure_mind(p: dict, rng=None) -> None:
    """Idempotently attach mind fields to a person dict (new spawns AND legacy/loaded
    saves that predate the mind). Safe to call every tick."""
    p.setdefault("memory", [])          # [{t, text, imp, kind}]
    p.setdefault("reflections", [])     # distilled beliefs (also retrievable as memories)
    p.setdefault("rel", {})             # other_id -> {name, trust, sentiment, met, trades, last}
    if "traits" not in p:               # temperament, rolled once (0.2..0.8 each)
        if rng is not None:
            p["traits"] = {t: round(float(rng.random()) * 0.6 + 0.2, 2) for t in TRAITS}
        else:
            p["traits"] = {t: 0.5 for t in TRAITS}
    p.setdefault("values", {t: 0.0 for t in TRAITS})  # lived drift on traits (identity)
    p.setdefault("renown", 0.0)         # social standing — earned by visible deeds, fades slowly
    p.setdefault("intention", None)     # {kind, target, u, why} — what they're set on now
    p.setdefault("goal", "tend")        # back-compat label for UI (kind, maybe :target)
    p.setdefault("intent", "")          # human phrasing of the current intention's "why"
    p.setdefault("say", "")             # last spoken line (renderer shows it briefly)
    p.setdefault("say_t", 0.0)          # clock when last spoken (for bubble fade)
    p.setdefault("delib_cd", 0.0)       # clock of next scheduled deliberation
    p.setdefault("think_cd", 0.0)       # clock cursor for the LLM round-robin
    p.setdefault("think_n", 0)          # how many LLM thoughts (reflect cadence)
    p.setdefault("last_social_t", 0.0)  # last meaningful contact (drives loneliness)
    p.setdefault("last_explore_t", 0.0) # last venture into the unknown (drives curiosity)
    p.setdefault("recipes", [])         # survival crafts THIS person personally knows
    p.setdefault("tinker_cd", 0.0)      # clock of next make-shift-craft experiment
    p.setdefault("learn_cd", 0.0)       # clock before they can be taught another craft
    p.setdefault("tried", [])           # material combos already experimented with (dead ends)


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


def _pick(rng, options: list[str]) -> str:
    """Choose one line (rng is the world's numpy Generator)."""
    return options[int(rng.integers(len(options)))]


def _topic_line(m: dict, name: str, rng) -> str:
    """Turn a vivid memory into something a soul would actually bring up to a neighbour, so the
    chatter reads as ABOUT something they lived (#2)."""
    k = m.get("kind", "")
    if k == "danger":
        return _pick(rng, [f"A wolf gave me a fright, {name}.", "I'll not wander far alone again.",
                           f"Did you hear the howling, {name}?"])
    if k == "trade":
        return _pick(rng, [f"That swap served me well, {name}.", f"We should deal again, {name}."])
    if k in ("discovery", "craft"):
        return _pick(rng, [f"I worked something out, {name} — I'll show you.",
                           f"My hands have learned a new trick, {name}."])
    if k == "gossip":
        return _pick(rng, [f"You'll have heard the talk, {name}?", f"Folk are saying things, {name}."])
    if k in ("loss", "death"):
        return _pick(rng, [f"I still feel the loss, {name}.", f"It's been hard since, {name}."])
    if k == "social":
        return _pick(rng, [f"It's good to have kin about, {name}.", f"You've been kind, {name}."])
    return f"{name}, {m.get('text','')}"


def _chat_line(sp: dict, ot: dict, clock: float, rng) -> str:
    """What `sp` says to `ot` on meeting: warm/curt by how they feel, and now and then about
    something recent on the mind rather than an empty hello (#3 greetings, #2 topics)."""
    sent = sp.get("rel", {}).get(ot["id"], {}).get("sentiment", 0.0)
    name = ot["name"]
    vivid = [m for m in sp.get("memory", []) if m.get("imp", 0) >= 0.5
             and m.get("kind") in ("danger", "trade", "discovery", "craft", "gossip",
                                    "loss", "death", "social")]
    if vivid and rng.random() < 0.45:
        m = max(vivid, key=lambda mm: mm["imp"] + _recency(mm["t"], clock, MEM_HALFLIFE_DAYS))
        return _topic_line(m, name, rng)
    if sent > 0.3:
        return _pick(rng, [f"Good to see you, {name}.", f"{name}! Come sit a while.",
                           f"Well met, {name} — how do you fare?"])
    if sent < -0.3:
        return _pick(rng, [f"...{name}.", f"I've little to say to you, {name}.",
                           f"Hmph. {name}."])
    return _pick(rng, [f"Well met, {name}.", f"How goes it, {name}?",
                       f"A fair day, {name}.", f"Greetings, {name}."])


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
        p["last_social_t"] = other["last_social_t"] = clock   # contact eases loneliness
        # They greet — each says a line, warm or curt by how they feel and now and then about
        # something on their mind. Spoken bubbles make a gathering read as a conversation.
        speak(p, _chat_line(p, other, clock, rng), clock)
        speak(other, _chat_line(other, p, clock, rng), clock)
        # Gossip: pass along a strong opinion about a third party, so good/bad names travel.
        if rng.random() < GOSSIP_CHANCE:
            spread = _gossip(p, other, clock)
            if spread:
                events.append(spread)
        spread2 = _gossip(other, p, clock) if rng.random() < GOSSIP_CHANCE else None
        if spread2:
            events.append(spread2)
        # Storytelling: a soul of standing passes a vivid tale of something they lived into a
        # listener's memory — how lore (a great hunt, a death, a hard-won craft) travels the band
        # and outlives the one who lived it. The seed of an abstract, shared culture.
        for teller, listener in ((p, other), (other, p)):
            tale = _tell_tale(teller, listener, clock, rng)
            if tale:
                events.append(tale)

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
    # Deference: a speaker of standing sways opinion further — the band weights a renowned voice
    # more heavily than a nobody's (#15).
    pull = GOSSIP_PULL * (1.0 + min(1.0, speaker.get("renown", 0.0)))
    lr["sentiment"] = max(-1.0, min(1.0, lr["sentiment"] + pull * op["sentiment"]))
    lr["trust"] = max(0.0, min(1.0, lr["trust"] + 0.04 * (1 if op["sentiment"] > 0 else -1)))
    tone = "warmly" if op["sentiment"] > 0 else "darkly"
    remember(listener, f"{speaker['name']} spoke {tone} of {op['name']}", 0.45, "gossip", clock)
    return f"{speaker['name']} told {listener['name']} about {op['name']}."


def _tell_tale(teller: dict, listener: dict, clock: float, rng) -> str | None:
    """A soul of some standing recounts a vivid thing they lived; the listener takes it into
    their own memory as something *heard* (not lived). This is how lore spreads and persists —
    abstract culture riding the same channel as gossip."""
    if teller.get("renown", 0.0) < STORY_RENOWN or rng.random() >= STORY_CHANCE:
        return None
    # Only first-hand, vivid, share-worthy memories make tales — not hearsay already passed on.
    tales = [m for m in teller.get("memory", []) if m.get("imp", 0) >= 0.7
             and m.get("kind") in ("danger", "discovery", "craft", "death", "trade", "illness")]
    if not tales:
        return None
    m = tales[int(rng.integers(len(tales)))]
    text = m["text"]
    heard = f"I heard tell from {teller['name']}: {text}"
    if any(mm.get("text") == heard for mm in listener.get("memory", [])):
        return None                                      # they've already heard this one
    remember(listener, heard, min(0.7, m["imp"] * 0.8), "tale", clock)
    speak(teller, _pick(rng, ["Let me tell you of something...", "There's a tale in that...",
                              "I'll not forget the day..."]), clock)
    return f"{teller['name']} told {listener['name']} a tale."


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


# ─── the drive arbiter: the always-on, model-free mind ──────────────────────────────
def _trait(p: dict, name: str) -> float:
    """A trait, bent by the values a life has crystallized (identity)."""
    return _clamp01(p.get("traits", {}).get(name, 0.5) + p.get("values", {}).get(name, 0.0))


def drives(p: dict, ctx: dict) -> list[tuple[str, str | None, float, str]]:
    """Score every pull on this person right now as (kind, target, utility, why). Survival,
    shelter, belonging, status, curiosity and fear all compete on one scale — this is the
    whole point of a thinking-first world: the agent *weighs what matters*, it doesn't run
    a fixed survival script. `ctx` is the small slice of world the body hands in."""
    soc, amb, cur, cau = (_trait(p, "sociability"), _trait(p, "ambition"),
                          _trait(p, "curiosity"), _trait(p, "caution"))
    inv = p.get("inv", {})
    clock = ctx.get("clock", 0.0)
    out: list[tuple[str, str | None, float, str]] = []

    # Survival — desire pulls early, the body's danger ramp overrides everything as a reserve fails.
    out.append(("drink", None, need_urgency(p.get("thirst", 0), p.get("hydration", 1.0)), "my throat is parched"))
    hunger_u = need_urgency(p.get("hunger", 0), p.get("satiety", 1.0))
    if p.get("hunger", 0) < 0.3 and inv.get("food", 0) > 0:
        hunger_u *= 0.5                                  # a fed person with food in the pack isn't driven to eat
    out.append(("eat", None, hunger_u, "hunger gnaws at me"))
    fatigue_u = need_urgency(p.get("fatigue", 0), p.get("stamina", 1.0))
    night = ctx.get("night")
    exposed = _clamp01(ctx.get("exposed", 0.0))
    if night:
        # People sleep at night, non-negotiably: fatigue rises fast (~a full day's worth daily),
        # and being up in the dark cold (exposure) and abroad (wolves) is deadly. A soul stays
        # bedded down at home through the night — even fully rested, resting in the warm is far
        # safer than wandering, so "asleep at 0% fatigue" at night is correct, not a bug. (Only a
        # pressing thirst/hunger, whose danger ramp out-scores this, wakes them.)
        fatigue_u = max(fatigue_u, 0.85)
    elif exposed <= 0.1 and p.get("stamina", 1.0) > REST_BY_DAY_RESERVE:
        # By DAY a soul pushes through ordinary drowsiness instead of napping — only genuine
        # exhaustion (the stamina reserve actually running low, whose danger ramp survives this
        # damp) sends them to rest. This is what makes them keep daylight hours rather than
        # sleeping at all times: the comfort-desire part of the fatigue pull is suppressed,
        # the body's true-exhaustion ramp is not.
        fatigue_u *= REST_BY_DAY_DAMP
    # Injury — a hurt body wants to lie low and mend rather than carry on into harm. Scales with
    # how badly wounded; keeps the soul resting (where hp heals) until it has recovered (#9).
    hp_deficit = 1.0 - _clamp01(p.get("hp", 1.0))
    if hp_deficit > 0.3:
        fatigue_u = max(fatigue_u, 0.45 + 0.4 * hp_deficit)
    # Exposure — caught out in the cold/wet, a soul is pulled to get under cover (rest at home).
    if exposed > 0.1:
        fatigue_u = max(fatigue_u, 0.4 + 0.45 * exposed)
    out.append(("rest", None, fatigue_u, "I should get out of this weather" if exposed > 0.3
                else "weariness drags at me"))

    # Fear — a prowling predator nearby overrides ordinary wants: get to safety, the home or the
    # band (a roof and company are protection). Keenest in cautious souls; scales with how close
    # the danger is. Pitched to out-score everyday projects/socializing but yield to acute survival.
    danger = _clamp01(ctx.get("danger", 0.0))
    if danger > 0.05:
        out.append(("flee", None, (0.5 + 0.55 * danger) * (0.7 + 0.6 * cau),
                    "a wolf is near — get to safety"))

    # Guardianship — a bold soul whose neighbour is in a wolf's sights doesn't bolt: it puts
    # itself between them and the threat. Scales with the ward's peril and the guardian's NERVE
    # (the inverse of caution), and only fires for a settled soul with a roof of its own to
    # spare the watch. Pitched to out-score a bold soul's own flee — so courage actually shows —
    # while a timid soul's flee still wins, and acute personal survival out-scores both.
    ward = _clamp01(ctx.get("ward_threat", 0.0))
    if ward > 0.2 and p.get("home_struct") is not None:
        nerve = 1.0 - cau
        out.append(("guard", None, (0.45 + 0.5 * ward) * (0.35 + 0.9 * nerve),
                    "a wolf threatens my own — I'll stand watch"))

    # Shelter & ambition — a home of one's own is half survival, half pride. A build already
    # underway adds sunk-cost MOMENTUM so it gets finished rather than dropped mid-wall.
    momentum = BUILD_MOMENTUM * _clamp01(ctx.get("build_progress", 0.0))
    if p.get("home_struct") is None:
        # A roofless soul out in foul weather feels the lack keenly — hurry the shelter up.
        out.append(("build", None, 0.5 + 0.18 * amb + 0.25 * exposed + momentum,
                    "I must raise a roof of my own"))
    elif ctx.get("needs_gear"):
        # A roof is up but the band has worked out make-shift gear this soul still lacks
        # (a water flask, say) — worth the effort to fashion it.
        out.append(("build", None, 0.42, "I should fashion the gear we've worked out"))
    elif ctx.get("needs_hearth"):
        # They've learned raw water sickens but have no hearth to boil it — raise one.
        out.append(("build", None, 0.44, "I must raise a hearth to boil my water clean"))
    else:
        # Survival & first shelter are behind them — now a standing LIFE-PROJECT pulls,
        # so a comfortable soul climbs toward a finer life instead of standing idle. The
        # world hands in the next worthwhile project (a snugger home, …); having a real
        # target makes this a genuine pull, keenest in ambitious & curious souls and a
        # little stronger once the body's needs are quiet. Still below survival's danger
        # ramp, so it never starves anyone — it just fills the empty hours with purpose.
        proj = ctx.get("project")
        if proj:
            # Comfort GATES the project (multiplies it), so the pull fades as any need rises:
            # a content soul throws themselves into building, but the moment thirst/hunger/
            # fatigue creep up the project shrinks and survival reclaims the wheel well before
            # crisis — a soul must never march off to forage timber while quietly dying of
            # thirst inland, out of reach of a passing sip.
            comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
            u = (0.22 + 0.22 * amb + 0.12 * cur) * comfort + momentum
            out.append(("build", None, u, proj.get("why") or "I'll make my home finer"))
        else:
            out.append(("build", None, 0.10 * amb, "I could make my home finer"))
        needy_id, needy_name = ctx.get("needy_id"), ctx.get("needy_name")
        if needy_id and inv.get("food", 0) > TRADE_SURPLUS:
            out.append(("provide", needy_id, 0.25 * amb + 0.2 * soc,
                        f"I have plenty — {needy_name} does not"))
        # ASPIRATION — a settled, content soul authors its OWN project beyond survival: tidy the
        # ground round its home, plant a garden to look on. A life with taste and self-expression,
        # not just upkeep. Comfort-gated (a needy soul has no time for it), keenest in the curious
        # and ambitious. The OPEN-VOCABULARY drive: the body seeds the project, the mind can enrich.
        if ctx.get("can_aspire"):
            comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
            if comfort > 0.55:
                # Pitched so a content, curious/ambitious soul will sometimes choose to beautify
                # over merely polishing its home — beauty as a real rival to utility — and momentum
                # carries a started project to completion.
                u = (0.30 + 0.20 * cur + 0.12 * amb) * comfort
                if ctx.get("aspiring"):
                    u += 0.20
                out.append(("aspire", None, u, ctx.get("aspire_why", "I'd make my home a finer place")))
        # Done with their own roof and carrying materials to spare — a soul lends a hand on a
        # band-mate's unfinished build. Comfort-gated (a needy soul tends itself first); keenest
        # in sociable, ambitious folk. The shared labour is a thread of interdependence.
        help_site = ctx.get("help_site")
        if help_site:
            comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
            if comfort > 0.4:
                # Pitched to OUT-rank a comfortable soul's own home-polishing project (~0.22), so a
                # neighbour mid-build draws a willing hand instead of everyone puttering solo — but
                # still below survival and below a roofless soul's own urgent first shelter.
                out.append(("help", help_site, (0.34 + 0.20 * soc + 0.12 * amb) * comfort,
                            "I'll lend a hand on their build"))

    # Belonging — loneliness grows the longer since real contact; sociable souls feel it most.
    # Hushed at night, when the world sleeps.
    night_damp = 0.3 if night else 1.0
    lonely = _clamp01((clock - p.get("last_social_t", 0.0)) / SOCIAL_FORGET)
    if ctx.get("others_exist"):
        fav_id, fav_name = ctx.get("fav_id"), ctx.get("fav_name")
        if fav_id:
            out.append(("befriend", fav_id, (0.18 + soc * lonely) * night_damp,
                        f"I'd seek out {fav_name}"))
        out.append(("socialize", None, (0.14 + soc * lonely * 0.85) * night_damp, "I want for company"))

    # Curiosity — the unknown tugs at restless minds that have lingered too long.
    bored = _clamp01((clock - p.get("last_explore_t", 0.0)) / SOCIAL_FORGET)
    out.append(("explore", None, (0.08 + cur * bored * 0.75) * night_damp, "the far country calls"))

    # AWE — a machine far beyond the band's craft is in sight (the sublime). It pulls hard while
    # NOVEL — the curious to approach and study, the cautious to recoil — then fades as the soul
    # comes to understand it and returns to ordinary life. Below survival, but above idle work.
    wonder = ctx.get("wonder")
    if wonder:
        nov = _clamp01(wonder.get("novelty", 1.0))
        out.append(("marvel", None, (0.42 + 0.30 * cur + 0.18 * cau) * nov,
                    "what manner of thing is THAT?"))

    # Apprenticeship — a soul still short on craft seeks out a markedly more-skilled band-mate to
    # learn from, keenest in the curious (the teaching itself happens when they're together).
    mentor_id = ctx.get("mentor_id")
    if mentor_id and not night:
        out.append(("befriend", mentor_id, (0.15 + 0.15 * cur) * night_damp,
                    f"I'd learn the crafts {ctx.get('mentor_name')} knows"))

    # Invention — when the band still has problems it hasn't solved (no way to carry water,
    # say), a curious mind tinkers toward a make-shift fix. Sharpened by hardship recently
    # felt: a soul that has known thirst is keener to crack the water problem.
    if ctx.get("unsolved") and not night:
        hardship = _clamp01(ctx.get("hardship", 0.0))
        out.append(("tinker", None, (0.16 + cur * 0.5 + hardship * 0.3) * night_damp,
                    "I'll puzzle out something to make"))

    # Fear / grudge — keep distance from someone who has wronged or unsettled them.
    foe_id, foe_name, foe_mag = ctx.get("foe_id"), ctx.get("foe_name"), ctx.get("foe_mag", 0.0)
    if foe_id and foe_mag > 0.3:
        out.append(("avoid", foe_id, cau * foe_mag, f"I'll keep clear of {foe_name}"))

    # Care — a well, settled soul goes to nurse a band-mate laid low by sickness (#11), bringing
    # food to them if it can (#10 rescue). Keenest in the sociable; comfort-gated so a soul tends
    # its own needs first, and below survival's danger ramp so nursing never costs a life.
    ail_id = ctx.get("ail_id")
    if ail_id and p.get("home_struct") is not None:
        comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
        out.append(("tend", ail_id, (0.55 + 0.30 * soc) * comfort,
                    f"{ctx.get('ail_name')} is ill — I'll tend them"))

    # Provisioning — a settled soul lays in a food reserve at home against lean days: survival
    # FORESIGHT, not just an idle-hours flourish, so every housed soul feels it (not only the
    # forager). Comfort-gated like any project and keenest when the larder runs low, but always
    # below survival's danger ramp — a hungry soul eats now and stocks later, never the reverse.
    if p.get("home_struct") is not None and not night:
        larder = p.get("store", {}).get("food", 0)
        target = PROVISION_TARGET * SEASON_STOCK_MULT.get(ctx.get("season", "spring"), 1.0)
        if larder < target:
            comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
            want = 1.0 - larder / target                    # keener when the cupboard's bare
            # The pull itself stiffens in autumn — winter is coming and the larder must be deep.
            urgency = 1.0 + 0.25 * (SEASON_STOCK_MULT.get(ctx.get("season", "spring"), 1.0) - 1.0)
            why = ("lay in food before winter" if ctx.get("season") == "autumn"
                   else "lay in food against lean days")
            out.append(("provision", None, (0.17 + 0.13 * want) * comfort * urgency, why))

    # Vocation — a settled soul plies its trade in the hours survival and projects leave free,
    # producing the surplus that division of labour runs on (a forager's full larder, a
    # woodcutter's wood pile, a toolmaker's spare gear). Comfort-gated like any project, and
    # pitched to fill genuine idle time — it yields to company, building and every real need.
    voc = ctx.get("vocation")
    if voc and p.get("home_struct") is not None and not night:
        comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
        ply_u = {"forager": 0.20 + 0.10 * (1.0 - amb),
                 "builder": 0.15 + 0.13 * amb,
                 "toolmaker": 0.15 + 0.16 * cur}.get(voc, 0.18) * comfort
        # Stop at enough (#14): a forager whose own larder is already brimming doesn't keep
        # mindlessly hauling food — the trade pull eases right off, freeing the hours for the
        # band (company, helping, building) instead of an ever-growing pile that only spoils.
        if voc == "forager":
            larder = p.get("store", {}).get("food", 0)
            target = PROVISION_TARGET * SEASON_STOCK_MULT.get(ctx.get("season", "spring"), 1.0)
            if larder >= target:
                ply_u *= 0.25
        why = {"forager": "lay in food while I can", "builder": "stock good timber",
               "toolmaker": "make spare gear for us all"}.get(voc, "ply my trade")
        out.append(("ply", None, ply_u, why))

    # A low baseline of just tending one's own patch, so an idle mind has somewhere to rest.
    out.append(("tend", None, 0.07, "tend to my own"))

    # YOUNGLINGS — a child's hands aren't ready for the hard or dangerous work (raising buildings,
    # crafting gear, standing a watch, plying a trade). Strip those pulls and give them a child's
    # life instead: forage what they can, whittle and practice, and learn from their elders —
    # growing the very skills that will make them capable adults. Survival, play, wonder and fear
    # all still apply (those drives are left untouched above).
    if ctx.get("is_child"):
        BLOCKED = {"build", "ply", "guard", "provide", "provision", "help", "tinker"}
        out = [d for d in out if d[0] not in BLOCKED]
        comfort = 1.0 - _clamp01(max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0)))
        # Forage what little hands can — and learn by doing (grows their foraging skill). The pull
        # EASES once their pack holds food, so a fed child turns to practising a craft instead of
        # endlessly gathering.
        forage_u = (0.22 + 0.12 * cur) * comfort
        if inv.get("food", 0) >= 3:
            forage_u *= 0.35
        out.append(("forage", None, forage_u, "I'll gather what I can and learn how"))
        # Whittle arrows and practice — small, safe handiwork that grows a young crafter's skill.
        out.append(("whittle", None, (0.20 + 0.14 * cur) * comfort, "I'll whittle arrows and get better"))
    return out


def deliberate(p: dict, ctx: dict, rng=None) -> dict:
    """Weigh the drives and (re)set the standing intention. Hysteresis keeps a person from
    flip-flopping every beat — the current aim must be clearly out-competed to be dropped.
    Returns the intention dict. This is the model-free mind; it always yields a choice."""
    cand = drives(p, ctx)
    cand.sort(key=lambda c: -c[2])
    best = cand[0]
    cur = p.get("intention")
    if cur and _intention_valid(cur, ctx):
        cur_u = next((c[2] for c in cand if c[0] == cur["kind"] and c[1] == cur.get("target")), 0.0)
        if cur_u + HYSTERESIS >= best[2]:
            cur["u"] = round(float(cur_u), 3)
            return cur
    kind, target, u, why = best
    changed = (not cur) or cur.get("kind") != kind or cur.get("target") != target
    inten = {"kind": kind, "target": target, "u": round(float(u), 3), "why": why}
    _set_intention(p, inten, ctx)
    # Choosing a *deliberate* aim (not just answering thirst) is itself a remembered moment.
    if changed and kind not in ("drink", "eat", "rest", "tend"):
        remember(p, f"resolved to {p['intent']}", 0.4, "intent", ctx.get("clock", 0.0))
    return inten


def _intention_valid(inten: dict, ctx: dict) -> bool:
    """An intention is stale if it aimed at someone no longer present/alive."""
    tgt = inten.get("target")
    if tgt and tgt not in ctx.get("alive_ids", ()):
        return False
    return True


def _set_intention(p: dict, inten: dict, ctx: dict) -> None:
    p["intention"] = inten
    kind, target = inten["kind"], inten.get("target")
    label = kind if not target else f"{kind}:{ctx.get('_names', {}).get(target, target)}"
    p["goal"] = label
    p["intent"] = (inten.get("why") or kind)[:120]


def set_goal(p: dict, goal: str, intent: str) -> None:
    """Back-compat shim: coerce a bare goal string into an intention (no target resolution).
    Kept so older call-sites and the LLM-glue path keep working."""
    kind = (goal or "tend").split(":")[0].strip()
    if kind not in INTENT_KINDS:
        kind = "tend"
    p["intention"] = {"kind": kind, "target": None, "u": 0.5, "why": intent}
    p["goal"] = goal.strip() if ":" in goal else kind
    p["intent"] = (intent or "").strip()[:120]


def heuristic_goal(p: dict, ctx: dict) -> tuple[str, str]:
    """Compatibility wrapper: run the drive arbiter and report (goal, intent). The real
    decision lives in `deliberate`; this just exposes it in the old (goal, intent) shape."""
    inten = deliberate(p, ctx)
    return p.get("goal", inten["kind"]), inten.get("why", "")


def crystallize_values(p: dict) -> None:
    """Identity forms from what a person actually does: the kind of memory they accrue most
    nudges the matching trait, so a habitual builder becomes ambitious, a habitual trader
    sociable. This is culture in miniature — character emerging from history, no LLM needed."""
    counts: dict[str, int] = {}
    for m in p.get("memory", []):
        counts[m["kind"]] = counts.get(m["kind"], 0) + 1
    lean = {"build": "ambition", "craft": "ambition", "trade": "sociability",
            "social": "sociability", "gossip": "sociability", "explore": "curiosity",
            "death": "caution", "whisper": "curiosity"}
    bump: dict[str, float] = {}
    for kind, n in counts.items():
        tr = lean.get(kind)
        if tr:
            bump[tr] = bump.get(tr, 0.0) + n
    if not bump:
        return
    top = max(bump, key=bump.get)
    vals = p.setdefault("values", {t: 0.0 for t in TRAITS})
    vals[top] = round(min(VALUE_CAP, vals.get(top, 0.0) + 0.03), 3)


def speak(p: dict, line: str, clock: float) -> None:
    line = (line or "").strip().strip('"')[:140]
    if line:
        p["say"] = line
        p["say_t"] = round(clock, 1)


# ─── discovery: a people works out its own make-shift crafts ────────────────────────
def experiment(p: dict, candidates: list[str], rng) -> tuple[list[str] | None, str | None]:
    """The OFFLINE inventor: brainstorm an untried small combination of materials and see if
    it amounts to anything (honest trial and error, no model). Returns (guess, recipe_id|None);
    recipe_id is set only when the hunch hits an as-yet-undiscovered craft. Tried combos are
    remembered so a mind doesn't keep banging the same two rocks together."""
    if not candidates:
        return None, None
    tried = {tuple(t) for t in p.setdefault("tried", [])}
    combos = [c for k in (2, 3) for c in itertools.combinations(CRAFT_VOCAB, k)]
    untried = [c for c in combos if c not in tried]
    if not untried:
        return None, None
    guess = list(untried[int(rng.integers(len(untried)))])
    p["tried"].append(list(guess))
    return guess, crafting.identify(set(guess), candidates)


def learn_recipe(p: dict, world_known: set, rid: str, clock: float, via: str = "worked out") -> bool:
    """Record a discovery: add it to the band's shared knowledge, burn it into memory as a
    proud moment, voice it, and let the breakthrough deepen a curious self-image. Returns
    True if it was genuinely new."""
    if not rid or rid in world_known:
        return False
    world_known.add(rid)
    if rid not in p.setdefault("recipes", []):    # personal knowledge, too
        p["recipes"].append(rid)
    name = rid.replace("_", " ")
    problem = crafting.SURVIVAL_DISCOVERIES.get(rid, "")
    remember(p, f"I {via} how to make a {name}" + (f" — for {problem}" if problem else ""),
             0.92, "discovery", clock)
    speak(p, f"I've worked out how to make a {name}!", clock)
    vals = p.setdefault("values", {t: 0.0 for t in TRAITS})
    vals["curiosity"] = round(min(VALUE_CAP, vals.get("curiosity", 0.0) + 0.05), 3)
    return True


def discover_messages(p: dict, ctx: dict) -> tuple[str, str]:
    """Build (system, user) asking the mind to REASON OUT a make-shift craft: given a problem
    it's struggling with and the materials at hand, guess the thing and what it's made of.
    A correct guess of the ingredients unlocks it — invention by insight, not by luck."""
    problems = ctx.get("unsolved_problems") or ["surviving out here"]
    system = (
        f"You are {p['name']}, an early human facing a hard problem with only raw materials "
        "and your wits. Invent a simple make-shift thing you could fashion to solve it, and "
        "say plainly what it is MADE OF. Think like a resourceful forager.\n"
        'Reply ONLY as JSON: {"make": short name of the thing, "ingredients": '
        "[2-3 materials from " + str(list(CRAFT_VOCAB)) + '], "say": a brief line or ""}.'
    )
    user = (
        f"My trouble: {problems[0]}.\n"
        f"Materials I can gather: {', '.join(CRAFT_VOCAB)}.\n"
        "What simple thing could I make, and from which materials?"
    )
    return system, user


def apply_discovery(p: dict, data: dict, ctx: dict, world_known: set, clock: float) -> str | None:
    """Check an LLM craft-hypothesis: canonicalize the guessed materials and, if they match a
    real undiscovered craft, learn it. Returns the discovered recipe id, or None."""
    if not isinstance(data, dict):
        return None
    raw = data.get("ingredients") or []
    guess = {m for m in (canon_material(str(x)) for x in raw) if m}
    say = str(data.get("say", "") or "")
    if say:
        speak(p, say, clock)
    rid = crafting.identify(guess, ctx.get("unsolved") or [])
    if rid and learn_recipe(p, world_known, rid, clock, via="reasoned out"):
        return rid
    # A wrong guess is still a lesson — remember the dead end so it isn't tried forever.
    if guess:
        p.setdefault("tried", []).append(sorted(guess))
    return None


# ─── LLM touch-points (prompt builders + result appliers; the call itself is in the
#     server, so this module stays sync/testable and free of the brain dependency) ────
def deliberate_messages(p: dict, ctx: dict) -> tuple[str, str]:
    """Build (system, user) for a deliberation — the LLM reasons over the SAME drives the
    arbiter weighs, plus memory, relationships and hard-won values, and chooses what to set
    its mind on. The drive scores are shown so the model grounds in the body's reality but
    is free to follow meaning over mere calories. This is the heart of a thinking-first
    world: the agent decides what its life is *for*, moment to moment."""
    name = p["name"]
    system = (
        f"You are the mind of {name}, one of the first people in a wild, newborn world. You "
        "are not only surviving — you want company, standing, discovery, a place that is "
        "yours; hunger and thirst are just some of the pulls you weigh. Think in the first "
        "person, plainly, like an early human, never like an AI. Choose ONE thing to set "
        "your mind on now, and why it matters to you.\n"
        f"Reply ONLY as compact JSON: {{\"intention\": one of {list(INTENT_KINDS)} "
        "(append ':Name' for befriend/provide/avoid to aim at a person), "
        '"why": a short first-person reason, "say": a brief line you\'d speak aloud or ""'
        + (", and IF you set your mind on bettering your own home (intention 'aspire'), add "
           f"\"project\": {{\"kind\": one of {ctx.get('aspire_kinds', [])}, \"goal\": a short "
           "first-person description of what you'll make and why}"
           if ctx.get("can_aspire") else "")
        + "}."
    )
    needs = _needs_phrase(p)
    pulls = "; ".join(f"{k}{':'+ctx.get('_names',{}).get(t,t) if t else ''} {u:.2f}"
                      for k, t, u, _ in sorted(drives(p, ctx), key=lambda c: -c[2])[:5])
    mems = retrieve(p, f"{ctx.get('season','')} {p.get('intent','')} {needs}", ctx.get("clock", 0.0), 4)
    rel_lines = _rel_phrase(p)
    vals = _values_phrase(p)
    inv = ", ".join(f"{k}×{v}" for k, v in (p.get("inv") or {}).items() if v) or "nothing"
    danger = _clamp01(ctx.get("danger", 0.0))
    exposed = _clamp01(ctx.get("exposed", 0.0))
    peril = ""
    if danger > 0.4:
        peril += " A WOLF is prowling close — I am not safe out here."
    elif danger > 0.1:
        peril += " I sense a wolf somewhere near."
    if exposed > 0.4:
        peril += " The weather is harsh and I have no roof over me — I'm chilled to the bone."
    user = (
        f"It is {ctx.get('time_str','day')}, {ctx.get('season','')}, weather {ctx.get('weather','')}.\n"
        f"My body: {needs}. I carry: {inv}. Home: {'built' if p.get('home_struct') else 'none yet'}.\n"
        f"Nearby: {ctx.get('nearby','no one')}.{peril}\n"
        f"What pulls at me (and how strongly): {pulls}.\n"
        + (f"Who I am becoming: {vals}.\n" if vals else "")
        + "What I remember:\n- " + ("\n- ".join(mems) if mems else "not much yet") + "\n"
        + ("People I know:\n- " + "\n- ".join(rel_lines) + "\n" if rel_lines else "")
        + f"My current aim: {p.get('intent') or 'drifting'}.\n"
        "What do I set my mind on, why, and is there anything I'd say aloud?"
    )
    return system, user


def apply_deliberation(p: dict, data: dict, ctx: dict, clock: float) -> None:
    """Apply a parsed LLM deliberation: override the standing intention with the reasoned
    one (validated; target names resolved to ids), and maybe speak. Junk is ignored, in
    which case the arbiter's own intention stands — the body is never left without one."""
    p["think_n"] = p.get("think_n", 0) + 1
    if not isinstance(data, dict):
        return
    raw = str(data.get("intention", "") or "").strip()
    why = str(data.get("why", "") or "").strip()
    speak(p, str(data.get("say", "")), clock)
    if not raw:
        return
    kind = raw.split(":")[0].strip().lower()
    if kind not in INTENT_KINDS:
        return
    target = None
    if ":" in raw:
        want = raw.split(":", 1)[1].strip().lower()
        for tid, nm in ctx.get("_names", {}).items():
            if nm.lower() == want and tid in ctx.get("alive_ids", ()):
                target = tid
                break
    inten = {"kind": kind, "target": target, "u": 0.6, "why": why or kind}
    _set_intention(p, inten, ctx)
    # The mind may AUTHOR its own home-bettering project in words — a goal beyond the fixed verbs.
    # We record the chosen (executable) kind + its free-text goal; the body grounds it into a plan
    # of skills (world._form_aspiration). Invalid/unknown kinds are ignored — the rule taste-pick
    # then stands, so the model's reasoning enriches but never breaks the offline path.
    proj = data.get("project")
    if isinstance(proj, dict) and str(proj.get("kind", "")).strip().lower() in set(ctx.get("aspire_kinds", ())):
        p["llm_project"] = {"kind": str(proj["kind"]).strip().lower(),
                            "goal": str(proj.get("goal", "") or "")[:120]}


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
    user = ("Lately:\n- " + ("\n- ".join(recent) if recent else "little has happened") +
            "\nWhat have I learned, and in a word, who am I becoming "
            f"({', '.join(TRAITS)}, or none)?")
    return system, user


def apply_reflections(p: dict, data: dict, clock: float) -> None:
    """Fold reflections into durable beliefs, and let a named self-image bend a trait — so
    a mind doesn't just remember, it becomes someone (identity feeding back into drives)."""
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
    ident = str(data.get("identity", "") or "").strip().lower()
    if ident in TRAITS:
        vals = p.setdefault("values", {t: 0.0 for t in TRAITS})
        vals[ident] = round(min(VALUE_CAP, vals.get(ident, 0.0) + 0.06), 3)


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


def _values_phrase(p: dict) -> str:
    """The traits a life has most strongly bent — a person's emerging self-image."""
    vals = {t: v for t, v in (p.get("values") or {}).items() if v >= 0.06}
    if not vals:
        return ""
    return ", ".join(t for t, _ in sorted(vals.items(), key=lambda kv: -kv[1]))


def give(p: dict, other: dict, good: str, clock: float) -> str | None:
    """A one-way gift — generosity, not barter. It costs the giver and builds a bond and the
    giver's standing; both remember it warmly. The social glue a trading economy alone lacks."""
    if p.get("inv", {}).get(good, 0) <= 0:
        return None
    p["inv"][good] -= 1
    other.setdefault("inv", {})[good] = other["inv"].get(good, 0) + 1
    ra, rb = _rel(p, other, clock), _rel(other, p, clock)
    _adjust(ra, 0.05, 0.18); _adjust(rb, 0.10, 0.25)        # the receiver warms most
    p["renown"] = p.get("renown", 0.0) + 0.05               # generosity is seen — it builds standing
    remember(p, f"gave {good} to {other['name']}, freely", 0.6, "social", clock)
    remember(other, f"{p['name']} gave me {good} when I had none", 0.75, "social", clock)
    other.setdefault("owes", {})[p["id"]] = round(clock, 1)   # a kindness remembered, to be repaid (gratitude)
    p["last_social_t"] = other["last_social_t"] = clock
    return f"{p['name']} gave {good} to {other['name']}."


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
        why = p.get("intent") or p.get("goal", "getting by")
        bit = f"{p['name']} ({why})"
        if p.get("say") and clock - p.get("say_t", 0) < 600:
            bit += f' says "{p["say"]}"'
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

    # gift: one-way generosity builds a strong bond, more than a trade
    g1 = mk("Finn", 0, {"food": 5}); g2 = mk("Orla", 1, {"food": 0})
    msg = give(g1, g2, "food", clock)
    assert g2["inv"]["food"] == 1 and g1["inv"]["food"] == 4 and msg
    assert g2["rel"][g1["id"]]["sentiment"] > 0.2
    print("gift OK ->", msg)

    # DRIVE ARBITER — survival is one drive among many. A thirsty soul drinks; a sated one
    # with no home builds; a settled, fed loner (alone a while) reaches for company.
    base = {"clock": 5000.0, "season": "spring", "weather": "clear", "night": False,
            "others_exist": True, "alive_ids": (), "_names": {}}
    thirsty = mk("Pell", 0); thirsty["thirst"] = 0.9
    assert deliberate(thirsty, base)["kind"] == "drink", thirsty["intention"]
    homeless = mk("Rua", 0)                     # comfortable, no roof
    assert deliberate(homeless, base)["kind"] == "build", homeless["intention"]
    settled = mk("Sefa", 0); settled["home_struct"] = "s"; settled["traits"]["sociability"] = 0.8
    settled["last_social_t"] = 0.0              # lonely for a long while
    ctx_soc = {**base, "fav_id": c["id"], "fav_name": "Cael", "_names": {c["id"]: "Cael"},
               "alive_ids": (c["id"],)}
    k = deliberate(settled, ctx_soc)["kind"]
    assert k in ("socialize", "befriend"), settled["intention"]
    print("arbiter OK -> thirsty=drink, homeless=build, lonely=", k)

    # FEAR & EXPOSURE — a wolf prowling close makes even a comfortable, housed soul flee; and a
    # roofless soul out in foul weather hurries its shelter (exposure boosts the first-home build).
    afraid = mk("Veyan", 0); afraid["home_struct"] = "s"
    assert deliberate(afraid, {**base, "danger": 0.9})["kind"] == "flee", afraid["intention"]
    cold = mk("Wyn", 0)                          # no roof yet, caught in a storm
    dry = drives(cold, {**base, "exposed": 0.0})
    wet = drives(cold, {**base, "exposed": 0.9})
    bu_dry = next(u for kd, _, u, _ in dry if kd == "build")
    bu_wet = next(u for kd, _, u, _ in wet if kd == "build")
    assert bu_wet > bu_dry, (bu_dry, bu_wet)
    print("fear/exposure OK -> wolf=flee, exposure lifts the roofless build", round(bu_dry, 2), "->", round(bu_wet, 2))

    # BUILD MOMENTUM — a half-raised project pulls harder than an unstarted one, so it gets
    # finished instead of abandoned mid-wall. A settled soul mid-build should out-pull socializing.
    bm = mk("Tove", 0); bm["home_struct"] = "s"; bm["traits"]["ambition"] = 0.5
    proj_ctx = {**base, "project": {"why": "raise a snug hut"}}
    fresh = next(u for kd, _, u, _ in drives(bm, {**proj_ctx, "build_progress": 0.0}) if kd == "build")
    underway = next(u for kd, _, u, _ in drives(bm, {**proj_ctx, "build_progress": 0.9}) if kd == "build")
    assert underway > fresh + 0.2, (fresh, underway)
    print("momentum OK -> mid-build pull", round(fresh, 2), "->", round(underway, 2))

    # identity crystallizes from what you do: a habitual builder grows ambitious
    builder = mk("Tam", 0)
    for _ in range(20):
        remember(builder, "laid another wall", 0.3, "build", clock)
    for _ in range(5):
        crystallize_values(builder)
    assert builder["values"]["ambition"] > 0, builder["values"]
    print("identity OK -> Tam's ambition drifted to", builder["values"]["ambition"])

    # LLM glue: deliberation prompt builds, and a reasoned override sets the intention
    sysm, usr = deliberate_messages(settled, ctx_soc)
    assert "JSON" in sysm and "pulls at me" in usr
    apply_deliberation(settled, {"intention": "befriend:Cael", "why": "I miss good company",
                                 "say": "Cael! Walk with me?"}, ctx_soc, clock)
    assert settled["intention"]["kind"] == "befriend" and settled["intention"]["target"] == c["id"]
    assert settled["say"].startswith("Cael")
    apply_reflections(settled, {"reflections": ["Cael deals fairly — keep close."],
                                "identity": "sociability"}, clock)
    assert settled["values"]["sociability"] > 0
    print("llm-glue OK ->", digest([settled, c], clock))

    # discovery: offline trial-and-error finds the hidden flask; an LLM-style ingredient
    # guess ("leaves + cord") reasons straight to it; both unlock it for the band.
    inventor = mk("Wren", 0)
    band = set(crafting.STARTER_RECIPES)
    for _ in range(80):
        _, rid = experiment(inventor, crafting.discoverable(band), rng)
        if rid:
            learn_recipe(inventor, band, rid, clock)
        if "leaf_flask" in band:
            break
    assert "leaf_flask" in band, "offline experimentation should eventually find the flask"
    assert any(m["kind"] == "discovery" for m in inventor["memory"])
    band2 = set(crafting.STARTER_RECIPES)
    got = apply_discovery(mk("Sage", 0), {"make": "water holder", "ingredients": ["leaves", "cord"]},
                          {"unsolved": crafting.discoverable(band2)}, band2, clock)
    assert got == "leaf_flask" and "leaf_flask" in band2, "leaves+cord should reason to a flask"
    print("discovery OK -> offline found gear; 'leaves+cord' reasoned to the flask")

    print("\nall mind self-tests passed ✓")
