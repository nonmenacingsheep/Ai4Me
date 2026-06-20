"""
The World — a living, top-down god-sim that Aitha and he preside over together.

This module is the *data layer*: the world's terrain, its ecology, and its clock,
with no rendering and (for now) no people. It owns a 128×128 tile grid whose tiles
carry elevation, biome, soil, moisture and growing vegetation, plus a population of
wildlife that wanders, grazes, hunts, breeds and dies — all driven by cheap, rule-
based simulation (no LLM). On top of that sits a god-action API: the same calls the
UI's brush tools make AND the calls Aitha's <world>/<sculpt>/<spawn> directives route
into, so either god can sculpt land, paint biomes, lay water, and seed life.

Design notes that shape everything here:
  • Body/mind split (Project Sid): the world ticks in pure Python every beat for free;
    the only LLM cost (later) is the occasional "mind" — and, even now, Aitha's god
    heartbeat. Nothing in this file calls a model.
  • Time is accelerated: 1 game-day ≈ 1 real hour. The server calls step(dt_real_sec)
    on its tick; we convert to game-time and run heavy ecology on a slow (per game-hour)
    cadence so 16k tiles stay cheap. The world pauses when the app closes and resumes
    from ~/.ai4me/world.npz on next launch.
  • Vectorized: tile fields are numpy arrays, so growth/spread/moisture run as whole-
    grid array ops rather than per-tile loops.
"""

import base64
import heapq
import json
import os
import time
import uuid

import numpy as np

import crafting   # item/recipe registry (content for gods, UI & the future mind)

# ─── Dimensions & clock ───────────────────────────────────────────────────
# A large world (2048×2048 ≈ 4.2M tiles) that still runs smoothly because we never
# touch the whole grid on a tick. The cheap per-tick work (movement) is bounded by
# the entity count, and the expensive work (ecology) runs only inside an "active
# region" around the people — chunks no one is near go dormant and fast-forward
# their growth when life returns (catch-up-on-revisit). See _active_region /
# _tick_ecology_active below.
W = 2048
H = 2048

# Bumped whenever the saved-world layout changes in a way older saves can't
# satisfy. A save stamped with a different (or missing) schema is treated as
# incompatible and regenerated on load, so a world written by a broken/older
# build self-heals on the next launch instead of staying frozen forever.
SCHEMA = 6
CHUNK = 64                      # tiles per chunk → (W//CHUNK)² dormancy bookkeeping cells
NCHUNK = W // CHUNK
SEA_LEVEL = 0.36                # elevation below this is ocean
MOUNTAIN_LEVEL = 0.78           # elevation above this reads as mountain/rock

# Ecology only ever runs inside a box around the people, capped to this many tiles
# on a side so a scattered population can't blow the cost up; dormant chunks that
# re-enter the box fast-forward up to this many game-hours of growth at once.
ACTIVE_MAX = 768
ECO_CATCHUP_CAP = 48
OVERVIEW_MAX = 256              # the whole-world snapshot is downsampled to ≤ this per side

# Accelerated time: 1 game-day == 1 real hour  →  1 real second == 24 game-seconds.
GAME_SEC_PER_REAL_SEC = 24.0
DAYS_PER_SEASON = 15
SEASONS = ("spring", "summer", "autumn", "winter")
DAYS_PER_YEAR = DAYS_PER_SEASON * len(SEASONS)

def _json_safe(o):
    """JSON fallback: numpy scalars/arrays → native Python (entity dicts can pick up
    np.float32 from the tile arrays during simulation)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


_DIR = os.path.join(os.path.expanduser("~"), ".ai4me")
PATH_GRID = os.path.join(_DIR, "world.npz")     # tile arrays (compressed)
PATH_META = os.path.join(_DIR, "world.json")    # clock, weather, entities, log

# ─── Biomes ────────────────────────────────────────────────────────────────
# index → (name, base map colour the renderer can start from). Water is a separate
# layer, so OCEAN here just means "submerged land cell".
BIOMES = (
    "ocean", "beach", "grassland", "forest", "rainforest", "desert",
    "savanna", "tundra", "snow", "swamp", "mountain", "rock",
    "shingle",       # gravel/pebble shore — steep, wave-eroded, sediment-starved coasts
)
B = {name: i for i, name in enumerate(BIOMES)}

# ─── Water layer ─────────────────────────────────────────────────────────────
WATER_NONE, WATER_RIVER, WATER_LAKE, WATER_OCEAN = 0, 1, 2, 3

# ─── Vegetation species ──────────────────────────────────────────────────────
# Each plant defines where it thrives: the biomes it tolerates, its temperature and
# moisture comfort band, how fast it grows, and how readily it spreads to neighbours.
VEG_NONE = 0
PLANTS = {
    1: dict(name="grass",     biomes={"grassland", "savanna", "beach"},
            t=(0.35, 0.85), m=(0.20, 0.80), grow=0.060, spread=0.020, icon="🌱"),
    2: dict(name="shrub",     biomes={"grassland", "savanna", "tundra", "desert"},
            t=(0.25, 0.90), m=(0.10, 0.60), grow=0.030, spread=0.010, icon="🌿"),
    3: dict(name="oak",       biomes={"forest", "grassland"},
            t=(0.35, 0.75), m=(0.45, 0.95), grow=0.012, spread=0.004, icon="🌳"),
    4: dict(name="pine",      biomes={"forest", "tundra", "mountain"},
            t=(0.10, 0.55), m=(0.35, 0.90), grow=0.010, spread=0.004, icon="🌲"),
    5: dict(name="cactus",    biomes={"desert"},
            t=(0.55, 1.00), m=(0.00, 0.25), grow=0.008, spread=0.003, icon="🌵"),
    6: dict(name="reeds",     biomes={"swamp", "beach"},
            t=(0.35, 0.85), m=(0.70, 1.00), grow=0.040, spread=0.012, icon="🌾"),
    7: dict(name="palm",      biomes={"rainforest", "beach"},
            t=(0.65, 1.00), m=(0.60, 1.00), grow=0.014, spread=0.005, icon="🌴"),
}
PLANT_BY_NAME = {v["name"]: k for k, v in PLANTS.items()}

# ─── Wildlife species ────────────────────────────────────────────────────────
# diet: "graze" eats vegetation off its tile; "hunt" eats nearby prey species.
# All rates are per game-MINUTE (the sim scales them by elapsed game-time each step,
# so behaviour is identical regardless of how often the server ticks). Ages and
# lifespans are in game-DAYS. Energy is an abstract reserve (≈ days of fuel).
ANIMALS = {
    # Prey are r-strategists: mature fast, breed often, find mates from afar.
    "rabbit": dict(diet="graze", speed=1, vision=4, maturity=3, max_age=50,
                   drain=0.0013, graze=0.06, eat_gain=4.0, repro_at=7.0,
                   repro_cost=3.0, repro_cd=2.0, pop_cap=150, icon="🐇"),
    "deer":   dict(diet="graze", speed=1, vision=6, maturity=6, max_age=120,
                   drain=0.0015, graze=0.06, eat_gain=4.5, repro_at=10.0,
                   repro_cost=5.0, repro_cd=4.0, pop_cap=90, icon="🦌"),
    # Predators are few, slow-breeding, and digest for over a game-day between kills.
    "wolf":   dict(diet="hunt",  speed=2, vision=8, maturity=14, max_age=100,
                   drain=0.0024, eat_gain=16.0, repro_at=24.0, repro_cost=12.0,
                   repro_cd=16.0, feed_cd=1800.0, kill_chance=0.6,
                   prey={"rabbit", "deer"}, pop_cap=40, icon="🐺"),
}
ENERGY_CAP_MULT = 1.6           # max reserve = repro_at × this
MATE_RADIUS = 6                 # a mate within this range is close enough to breed

# ─── People ────────────────────────────────────────────────────────────────
# Phase 2 — the "body". People are driven by the same cheap, rule-based loop as
# wildlife (needs decay → perception → greedy pathfinding → forage/drink/rest →
# death) with NO LLM. The "mind" (goals, memory, speech via DeepSeek) and the
# <whisper> nudge arrive in a later step; social/reproduction is Phase 4. Needs
# run 0.0 (sated) .. 1.0 (dire). Rates are per game-MINUTE so behaviour is the
# same regardless of how often the server ticks; ages/lifespans are in game-DAYS
# (1 game-year == DAYS_PER_YEAR == 60 game-days).
PERSON = dict(
    vision=8, speed=1, max_age=70 * DAYS_PER_YEAR,
    # Survival spans are calibrated in game-DAYS (1 day = 1440 game-min): a healthy
    # person lasts ~3 days without water and ~3 weeks without food. A need rises to
    # 1.0 over that span (thirst ~2.3 days, hunger ~21 days), after which starve_dmg
    # erodes hp over a further ~0.9 day → sated→dead ≈ 3 days (thirst) / ~22 (hunger).
    hunger_rate=0.000033, thirst_rate=0.0003, fatigue_rate=0.00069,
    t_hunger=0.50, t_thirst=0.45, t_rest=0.70,        # need thresholds to act on
    eat_bite=0.05, food_value=2.5,                     # graze speed / hunger restored per unit
    drink_rate=0.05, rest_rate=0.04,                   # thirst/fatigue relieved per min
    inv_cap=8, gather_min=0.30,                         # carry capacity / tile richness to gather
    starve_dmg=0.0008, heal=0.0006,                    # hp lost when a need maxes / regained when sated
)
EDIBLE_PLANTS = {"grass", "oak", "reeds", "palm", "shrub"}   # plants people can forage
EDIBLE_IDS = [sp for sp, info in PLANTS.items() if info["name"] in EDIBLE_PLANTS]
NAMES_M = ("Aren", "Bram", "Cael", "Doran", "Eli", "Finn", "Garreth", "Holt",
           "Ivo", "Joss", "Korin", "Lugh", "Mato", "Niall", "Osric", "Pell")
NAMES_F = ("Ada", "Bel", "Cyra", "Dara", "Esme", "Fern", "Greta", "Hana",
           "Isla", "Juno", "Kira", "Lena", "Maeve", "Nara", "Orla", "Petra")

# ─── Crafting & building (Phase 3) ───────────────────────────────────────────
# Still pure body, no LLM: once survival needs are comfortable, a person gathers
# raw materials, crafts a crude axe (their first tool — it speeds wood-getting),
# then raises a shelter at home (a sheltered person rests faster). Wood comes from
# felling trees, stone from bare rock/mountain. Stone houses / workbench are a
# later slice. Tools/materials live in the person's inv dict alongside food.
WOOD_PLANTS = {"oak", "pine", "palm"}       # trees that yield wood when felled
WOOD_IDS = [sp for sp, info in PLANTS.items() if info["name"] in WOOD_PLANTS]
BUILD = dict(
    chop_growth_min=0.35, chop_take=0.5,    # a chop needs a grown tree; removes this much growth
    chop_yield=2, axe_bonus=2,              # wood gained per chop; an axe adds this much (→4 w/ axe)
    mine_yield=1, stone_stock=4,            # stone per mine; how much to stockpile once sheltered
    axe_wood=2,                             # crude axe recipe
    shelter_wood=6,                         # a brush/wood shelter
    rest_sheltered_mult=2.0,                # rest this much faster in your own shelter
)
STRUCT_KINDS = ("shelter",)

# ─── Tile building (Phase 3.5) — top-down "blocks" ───────────────────────────
# People raise REAL buildings the Minecraft way: a building is a footprint of
# individual placed tiles — walls, a door, a floor, and a thatch roof — laid one
# block per build-action from a BLUEPRINT. Blueprints are the "design"; built-ins
# live here, but the format is plain data so the future LLM "mind" can author new
# ones (a chicken coop, a smithy) and drop them into this same library — the
# Voyager "skill library" idea applied to architecture. The rule-body just walks
# whatever blueprint it's given. Blocks live in a sparse layer (self.blocks);
# roofs in self.roofs; an in-progress building is a "site" with per-tile tasks.
BLOCK_EMPTY, BLOCK_FLOOR, BLOCK_WALL, BLOCK_DOOR, BLOCK_WINDOW, BLOCK_FENCE, BLOCK_LEAF = 0, 1, 2, 3, 4, 5, 6
BLOCK_NAMES = {BLOCK_FLOOR: "floor", BLOCK_WALL: "wall", BLOCK_DOOR: "door",
               BLOCK_WINDOW: "window", BLOCK_FENCE: "fence", BLOCK_LEAF: "leaves"}
# Blueprint glyphs → block codes (rows read top-down, like a tiny map). "C" is a
# special core: no block is placed, but the tile is roofed and becomes the home —
# used for the open-fronted leaf lean-to whose centre you walk into.
BLOCK_CHARS = {".": BLOCK_EMPTY, "F": BLOCK_FLOOR, "W": BLOCK_WALL,
               "D": BLOCK_DOOR, "O": BLOCK_WINDOW, "#": BLOCK_FENCE, "L": BLOCK_LEAF}
GLYPH_CORE = "C"
# Material a single block costs (item, qty). Wood/fiber/leaf based so a band can
# build straight from what it forages, no workshop required.
BLOCK_COST = {BLOCK_FLOOR: ("wood", 1), BLOCK_WALL: ("wood", 2), BLOCK_DOOR: ("wood", 2),
              BLOCK_WINDOW: ("wood", 1), BLOCK_FENCE: ("wood", 1), BLOCK_LEAF: ("leaves", 1)}
ROOF_COST = ("fiber", 2)                 # default thatch over each sheltered tile
SOLID_BLOCKS = {BLOCK_WALL, BLOCK_LEAF}  # tiles people can't walk through (doors/cores are open)

# Built-in blueprints. layout = rows of glyphs; roof=True thatches every interior
# (floor/door) tile once the shell is up. "insulation" (0..1) is how well it holds
# heat/cold — the leaf lean-to is a quick, draughty starter (poor insulation, so a
# tiny rest bonus), a timber hut/cabin is snug. The format is plain data, so the
# future LLM "mind" can author new ones (a chicken coop, a smithy) and drop them in.
BLUEPRINTS = {
    # A leaf lean-to: three leaf panels in a triangle (apex + two sides), open at the
    # base to walk into, with a leaf roof over the core. Cheap, fast, barely insulating.
    "leaf_shelter": dict(name="Leaf Shelter", roof=False, insulation=0.12,
                         roof_cost=("leaves", 1), layout=[
        ".L.",
        "LCL",
    ]),
    "hut": dict(name="Hut", roof=True, insulation=1.0, layout=[
        "WDW",
        "WFW",
        "WWW",
    ]),
    "cabin": dict(name="Cabin", roof=True, insulation=1.0, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
}

# ─── Live-sim material sourcing (Phase 3.5) ──────────────────────────────────
# The raws the crafting registry (crafting.py) consumes are drawn from the living
# map, each gated by the right tool (see crafting.RAW): fiber from grasses, clay
# from waterside lowland, sand from beach/desert, flint from bare rock, and metal
# ORES from scattered deposits in hill/mountain rock. Wood (logs) already comes
# from felling trees. Ore deposits are a small sparse node list so a 4M-tile map
# costs nothing; the band must explore the highlands to find them.
FIBER_PLANTS = {"grass", "shrub", "reeds"}
FIBER_IDS = [sp for sp, info in PLANTS.items() if info["name"] in FIBER_PLANTS]
# Leaves come off trees & shrubs in armfuls — gathered fast, several per pull, and a
# person can lug a big bundle. They build the quick leaf shelter (but insulate poorly).
LEAF_PLANTS = {"oak", "pine", "palm", "shrub"}
LEAF_IDS = [sp for sp, info in PLANTS.items() if info["name"] in LEAF_PLANTS]
LEAF_GATHER = 3                          # leaves gained per pull (vs. 1 for fiber)
LEAF_CAP = 24                            # a person can carry a big bundle of leaves
ORE_KINDS = ("copper_ore", "tin_ore", "iron_ore", "gold_ore", "coal")
ORE_WEIGHTS = (0.30, 0.18, 0.30, 0.07, 0.15)
ORE_NODES_PER = 1_000_000               # ~1 deposit per this many tiles


# ════════════════════════════════════════════════════════════════════════════
#  Small numpy helpers
# ════════════════════════════════════════════════════════════════════════════
def _resize_bilinear(src: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinearly upsample a small grid to (h, w) — used to build smooth noise."""
    sh, sw = src.shape
    ys = np.linspace(0, sh - 1, h)
    xs = np.linspace(0, sw - 1, w)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1); x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    a = src[y0][:, x0]; b = src[y0][:, x1]
    c = src[y1][:, x0]; d = src[y1][:, x1]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def _fractal_noise(h: int, w: int, octaves: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-octave value noise in [0,1] — coarse rolling shapes plus finer detail."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 2 ** (o + 2)                      # 4, 8, 16, 32, ...
        grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
        out += _resize_bilinear(grid, h, w) * amp
        total += amp
        amp *= 0.55
    out /= total
    out -= out.min()
    out /= (out.max() + 1e-9)
    return out


def _smooth(a: np.ndarray, passes: int = 1) -> np.ndarray:
    """Cheap separable box blur (5-point) — used to keep coastlines coherent instead of
    a pixelly half-water marsh. Edges wrap (fine for a sea-framed map)."""
    a = a.astype(np.float32)
    for _ in range(passes):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
             + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """1.0 inside [lo,hi], falling linearly to 0 within a margin outside it.
    Used as a suitability curve for temperature / moisture comfort bands."""
    margin = 0.18
    left = np.clip((x - (lo - margin)) / margin, 0, 1)
    right = np.clip(((hi + margin) - x) / margin, 0, 1)
    return np.minimum(left, right).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  The World
# ════════════════════════════════════════════════════════════════════════════
class World:
    def __init__(self):
        self.seed = 0
        self.clock = 0.0                 # game-minutes since creation
        self._last_eco = 0.0             # game-minute of last ecology pass
        self.weather = "clear"           # clear | cloudy | rain | storm | snow
        self.weather_intensity = 0.0     # 0..1
        self._weather_until = 0.0        # game-minute the current weather expires
        self.animals: list[dict] = []
        self.people: list[dict] = []     # Phase 2 — simulated folk (body only, no LLM yet)
        self.structures: list[dict] = [] # Phase 3 — things people build (shelters, …)
        self.blocks: dict[tuple[int, int], int] = {}   # sparse placed tiles (x,y)->block code
        self.roofs: set[tuple[int, int]] = set()       # tiles thatched over (rendered above)
        self.sites: list[dict] = []      # buildings under construction (blueprint + per-tile tasks)
        self.ore_nodes: list[dict] = []  # scattered metal/coal deposits to mine
        self._ore_index: dict[tuple[int, int], dict] = {}  # (x,y)->node, rebuilt on load/seed
        self.log: list[dict] = []        # recent god actions / notable events
        self.version = 0                 # bumped on any mutation, for render diffing
        self.rng = np.random.default_rng()
        self._origin = (W // 2, H // 2)  # founding-valley centre
        self._chunk_eco = None           # per-chunk game-min of last ecology pass (dormancy)
        # Tile fields (allocated in generate()/load()).
        self.elevation = self.biome = self.soil = self.moisture = None
        self.water = self.veg_sp = self.veg_growth = None

    # ── generation ──────────────────────────────────────────────────────────
    def generate(self, seed: int | None = None) -> "World":
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
        self.rng = np.random.default_rng(self.seed)
        rng = self.rng

        # Elevation = a low-frequency CONTINENTAL base (big landmasses, gulfs, islands)
        # plus a little mid-frequency relief, then smoothed so coastlines are coherent.
        # (The old 9-octave field had fine noise straddling sea level everywhere, which
        # fragmented the coast into a pixelly half-water marsh — the look we're fixing.)
        base = _fractal_noise(H, W, 4, rng)
        relief = _fractal_noise(H, W, 7, np.random.default_rng(self.seed ^ 0x51ED2C9F))
        elev = _smooth(base * 0.74 + relief * 0.26, 2)
        # Ocean frame: the map sits in sea, falling toward the borders, but land can still
        # reach the edge as peninsulas and interior lows become bays/inland seas, so islands
        # large and small form naturally rather than one blobby continent.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        edge = np.minimum.reduce([xx, yy, (W - 1 - xx), (H - 1 - yy)]) / (min(W, H) * 0.5)
        elev = elev * (0.52 + 0.48 * np.clip(edge, 0, 1))
        elev -= elev.min(); elev /= (elev.max() + 1e-9)
        # A gentle steepen around sea level for a crisp shoreline (no swampy half-drowned
        # shelf), then a final light smooth. Smoothing — not fine noise — keeps coasts clean.
        elev = np.clip((elev - SEA_LEVEL) * 1.12 + SEA_LEVEL, 0, 1)
        self.elevation = _smooth(elev, 1).astype(np.float32)
        elev = self.elevation

        moist = _smooth(_fractal_noise(H, W, 6, np.random.default_rng(self.seed ^ 0x9E3779B9)), 1)
        # Latitudinal wetness like Earth's: a wet tropical belt at the equator (→ hot, wet
        # rainforest), a dry belt around the 30° horse latitudes (→ deserts), damp temperate
        # mid-latitudes (→ forests), drying again toward the poles.
        lat = np.abs(yy / (H - 1) - 0.5) * 2                     # 0 at equator … 1 at poles
        equator_wet = np.exp(-(lat / 0.30) ** 2) * 0.48
        midlat_wet = np.exp(-((lat - 0.70) / 0.26) ** 2) * 0.30
        horse_dry = np.exp(-((lat - 0.36) / 0.13) ** 2) * 0.34   # the ~30° desert belt
        moist = np.clip(moist * 0.60 + equator_wet + midlat_wet - horse_dry, 0, 1)
        self.moisture = moist.astype(np.float32)

        self.water = np.where(elev < SEA_LEVEL, WATER_OCEAN, WATER_NONE).astype(np.uint8)
        self._carve_hydrology()
        # Damp ground near any water.
        self._dampen_near_water()

        self.biome = self._classify_biomes()
        self._shape_beaches()
        # Soil: fertile where moist & low, poor on rock/desert/snow; plus noise.
        soil = np.clip(0.35 + self.moisture * 0.5 - elev * 0.25, 0, 1)
        soil += (_fractal_noise(H, W, 6, np.random.default_rng(self.seed ^ 0x12345)) - 0.5) * 0.2
        for bad in ("rock", "mountain", "snow", "desert", "ocean", "shingle"):
            soil[self.biome == B[bad]] *= 0.35
        self.soil = np.clip(soil, 0, 1).astype(np.float32)

        self.veg_sp = np.zeros((H, W), np.uint8)
        self.veg_growth = np.zeros((H, W), np.float32)
        self._seed_initial_vegetation()
        # On a vast map, life starts clustered in one hospitable valley (the rest of the
        # world is generated but unpopulated, awaiting exploration / a god's hand).
        self._origin = self._choose_origin()
        self._seed_grove(self._origin)          # a copse so the band has wood within reach
        self._seed_initial_wildlife(count=340, center=self._origin, radius=600)
        self._seed_ore_nodes()
        self._seed_initial_people(count=7, center=self._origin)

        self.clock = 8 * 60.0            # start at 08:00 on day 0
        self._last_eco = self.clock
        # Every chunk starts "freshly grown"; dormant ones fast-forward on revisit.
        self._chunk_eco = np.full((NCHUNK, NCHUNK), self.clock, np.float32)
        self.version += 1
        self._note("world", f"A new world took shape (seed {self.seed}).")
        return self

    def _seed_grove(self, center, r: int = 7):
        """Scatter mature, biome-appropriate trees around the founding valley so the band
        can immediately gather wood (without it, a treeless valley = no building)."""
        cx, cy = center
        ob = BIOMES[self.biome[cy, cx]]
        name = ("palm" if ob in ("beach", "rainforest")
                else "pine" if ob in ("tundra", "mountain", "snow", "rock")
                else "oak")
        sp = PLANT_BY_NAME[name]
        sl = (slice(max(0, cy - r), min(H, cy + r + 1)), slice(max(0, cx - r), min(W, cx + r + 1)))
        land = self.water[sl] == WATER_NONE
        place = land & (self.rng.random(land.shape) < 0.33)
        self.veg_sp[sl][place] = sp
        gr = self.veg_growth[sl]
        gr[place] = np.maximum(gr[place], 0.6)         # mature → choppable right away

    def _seed_ore_nodes(self):
        """Scatter mineable metal/coal deposits across hill & mountain rock. A small
        sparse list (≈ W*H / ORE_NODES_PER nodes) so the vast map costs nothing; the
        band has to explore the highlands to strike copper or iron."""
        self.ore_nodes = []
        rock = np.argwhere(np.isin(self.biome, [B["mountain"], B["rock"]])
                           & (self.water == WATER_NONE))
        if not len(rock):
            self._rebuild_ore_index()
            return
        n = max(8, (W * H) // ORE_NODES_PER)
        picks = self.rng.choice(len(rock), size=min(n, len(rock)), replace=False)
        for idx in picks:
            ry, rx = rock[idx]
            kind = str(self.rng.choice(ORE_KINDS, p=ORE_WEIGHTS))
            self.ore_nodes.append({"x": int(rx), "y": int(ry), "kind": kind,
                                   "amount": int(self.rng.integers(8, 25))})
        self._rebuild_ore_index()

    def _rebuild_ore_index(self):
        self._ore_index = {(n["x"], n["y"]): n for n in self.ore_nodes}

    def _choose_origin(self):
        """Pick a hospitable founding spot near both water AND woodland, so the band can
        drink and also gather wood to build. Returns (cx, cy)."""
        # Beach is excluded: it's effectively treeless, so a beach founding valley can't
        # supply wood to build. Grassland/forest/savanna have trees natively + a grove.
        habitable = np.isin(self.biome, [B["grassland"], B["forest"], B["savanna"]])
        land = self.water == WATER_NONE

        def dilate(mask, n):
            out = mask.copy()
            for _ in range(n):
                acc = out.copy()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    acc |= np.roll(np.roll(out, dy, 0), dx, 1)
                out = acc
            return out

        near_water = dilate(self.water != WATER_NONE, 5)
        # Tree-bearing ground within a short range, so a crude axe & shelter are reachable.
        near_wood = dilate(np.isin(self.biome, [B["forest"], B["rainforest"], B["tundra"]]), 14)
        for cond in (land & habitable & near_water & near_wood,
                     land & habitable & near_water,
                     land & near_water,
                     land):
            good = np.argwhere(cond)
            if len(good):
                cy, cx = good[self.rng.integers(len(good))]
                return (int(cx), int(cy))
        return (W // 2, H // 2)

    # ── hydrology: a real drainage network (flow accumulation) ──────────────────
    _DIRS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    def _carve_hydrology(self):
        """Carve lakes and rivers from a proper drainage model instead of tracing springs.
        Working on a coarse grid (for speed on 4M tiles): fill depressions so every land
        cell drains to the sea (Barnes priority-flood), accumulate each cell's rainfall +
        snowmelt downhill, then the rivers are wherever enough water has gathered — so they
        bend naturally, grow THICKER downstream (width ∝ √volume), pool into lakes at filled
        basins, and always reach the ocean (a lake simply overflows onward to the sea)."""
        DS = 4
        rh, rw = H // DS, W // DS
        ce = self.elevation[:rh * DS, :rw * DS].reshape(rh, DS, rw, DS).mean((1, 3)).astype(np.float64)
        cm = self.moisture[:rh * DS, :rw * DS].reshape(rh, DS, rw, DS).mean((1, 3))
        sea = ce < SEA_LEVEL
        # The displayed terrain is heavily smoothed (clean coasts), but a near-flat surface
        # makes D8 flow snap into long parallel diagonals. Give the DRAINAGE surface real
        # fractal valleys (generated at the coarse resolution so they survive) on land only:
        # flow then converges down these valleys into branching, dendritic river networks.
        valleys = _fractal_noise(rh, rw, 7, np.random.default_rng(self.seed ^ 0x2B1D77C4))
        ce = ce + np.where(sea, 0.0, (valleys - 0.5) * 0.14)

        # Priority-flood depression fill (+ε so flats still drain): fe = hydrologically
        # corrected elevation that descends monotonically from every land cell to the sea.
        INF = 1e18
        fe = np.where(sea, ce, INF)
        closed = sea.copy()
        eps = 1e-6
        pq = [(float(ce[y, x]), int(y), int(x)) for y, x in zip(*np.nonzero(sea))]
        heapq.heapify(pq)
        while pq:
            e, y, x = heapq.heappop(pq)
            for dy, dx in self._DIRS8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rh and 0 <= nx < rw and not closed[ny, nx]:
                    closed[ny, nx] = True
                    fe[ny, nx] = ce[ny, nx] if ce[ny, nx] > e else e + eps
                    heapq.heappush(pq, (float(fe[ny, nx]), ny, nx))

        # D8 receiver (steepest-descent neighbour on the filled surface).
        bestv = fe.copy()
        rdy = np.zeros((rh, rw), np.int64); rdx = np.zeros((rh, rw), np.int64)
        for dy, dx in self._DIRS8:
            s = np.roll(np.roll(fe, -dy, 0), -dx, 1)
            m = s < bestv
            bestv[m] = s[m]; rdy[m] = dy; rdx[m] = dx
        yy = np.arange(rh)[:, None]; xx = np.arange(rw)[None, :]
        ry = np.clip(yy + rdy, 0, rh - 1); rx = np.clip(xx + rdx, 0, rw - 1)
        recv = (ry * rw + rx).ravel()

        # Flow accumulation: each cell sources rainfall (∝ moisture) plus snowmelt (cold,
        # high ground) and passes its running total to its receiver, processed high→low.
        lat = np.abs(yy / (rh - 1) - 0.5) * 2
        ctemp = np.clip(0.92 - lat * 0.62 - ce * 0.42, 0, 1)
        rain = 0.35 + cm + np.clip(0.32 - ctemp, 0, 1) * 1.6     # +snowmelt where cold
        acc = np.where(sea, 0.0, rain).ravel().astype(np.float64)
        seaf = sea.ravel()
        order = np.argsort(fe, axis=None)[::-1]
        for idx in order:
            if seaf[idx]:
                continue
            r = recv[idx]
            if r != idx:
                acc[r] += acc[idx]
        acc = acc.reshape(rh, rw)

        # Lakes = basins the flood actually filled (water would pond there).
        lake_c = (~sea) & (fe > ce + 5e-4)
        # Rivers = land cells carrying more than a threshold volume; width grows with √flow.
        land_acc = acc[~sea]
        thresh = np.quantile(land_acc, 0.93) if land_acc.size else 1e9
        river_c = (~sea) & (acc > thresh)

        # Paint lakes (upscaled blocks) onto the full-res water grid.
        if lake_c.any():
            big_lake = np.repeat(np.repeat(lake_c, DS, 0), DS, 1)
            self.water[big_lake & (self.water == WATER_NONE)] = WATER_LAKE

        # Paint rivers: stamp variable-width channels from each river cell to its receiver,
        # so the line is continuous and thickens downstream. Carve the bed down a touch too.
        rmask = np.zeros((H, W), bool)
        wfac = np.sqrt(np.clip(acc / max(thresh, 1e-9), 1.0, 400.0))   # 1..20
        ys, xs = np.nonzero(river_c)
        for cy, cx in zip(ys.tolist(), xs.tolist()):
            r_tiles = float(np.clip(wfac[cy, cx] * 0.9, 1.0, 9.0))
            fy0, fx0 = cy * DS + DS // 2, cx * DS + DS // 2
            ny, nx = int(ry[cy, cx]), int(rx[cy, cx])
            fy1, fx1 = ny * DS + DS // 2, nx * DS + DS // 2
            steps = max(1, int(max(abs(fy1 - fy0), abs(fx1 - fx0))))
            for s in range(steps + 1):
                t = s / steps
                self._stamp_disc(rmask, int(fy0 + (fy1 - fy0) * t), int(fx0 + (fx1 - fx0) * t), r_tiles)
        place = rmask & (self.water == WATER_NONE) & (self.elevation >= SEA_LEVEL)
        self.water[place] = WATER_RIVER
        self.elevation[place] = np.maximum(SEA_LEVEL - 0.005,
                                           self.elevation[place] - 0.02).astype(np.float32)

    def _stamp_disc(self, mask, fy, fx, r):
        r = max(1, int(round(r)))
        y0, y1 = max(0, fy - r), min(H, fy + r + 1)
        x0, x1 = max(0, fx - r), min(W, fx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - fx) ** 2 + (yy - fy) ** 2 <= r * r

    def _dampen_near_water(self):
        # A narrow, gentle damp fringe by the water — wide/strong dampening was turning
        # every coast into swamp. Two soft rings, modest boost.
        wet = (self.water != WATER_NONE).astype(np.float32)
        near = wet.copy()
        for _ in range(2):
            acc = near.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                acc = np.maximum(acc, np.roll(np.roll(near, dy, 0), dx, 1) * 0.6)
            near = acc
        self.moisture = np.clip(self.moisture + near * 0.18, 0, 1).astype(np.float32)

    def _annual_temperature(self, reg=None) -> np.ndarray:
        """Mean annual temperature field in [0,1] from latitude and elevation. With a
        region (y0,y1,x0,x1) it computes only that slice (ecology stays off the full map)."""
        if reg is None:
            y0, y1, x0, x1 = 0, H, 0, W
        else:
            y0, y1, x0, x1 = reg
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        lat = np.abs(yy / (H - 1) - 0.5) * 2          # 0 at equator-ish middle, 1 at poles
        temp = 0.92 - lat * 0.62 - self.elevation[y0:y1, x0:x1] * 0.42
        return np.clip(temp, 0, 1).astype(np.float32)

    def _classify_biomes(self) -> np.ndarray:
        elev, moist, temp = self.elevation, self.moisture, self._annual_temperature()
        bm = np.full((H, W), B["grassland"], np.uint8)
        bm[temp < 0.30] = B["tundra"]
        bm[(temp < 0.18)] = B["snow"]
        hot = temp >= 0.62
        bm[hot & (moist < 0.30)] = B["desert"]
        bm[hot & (moist >= 0.30) & (moist < 0.55)] = B["savanna"]
        bm[hot & (moist >= 0.55)] = B["rainforest"]
        temperate = (temp >= 0.30) & (temp < 0.62)
        bm[temperate & (moist >= 0.55)] = B["forest"]
        bm[temperate & (moist < 0.55) & (moist >= 0.30)] = B["grassland"]
        # Swamp only in genuinely wet, very low interior pockets (not the whole coast).
        bm[(elev < SEA_LEVEL + 0.04) & (moist > 0.82) & (self.water == WATER_NONE)] = B["swamp"]
        # Heights override climate.
        bm[elev >= MOUNTAIN_LEVEL] = B["mountain"]
        bm[(elev >= MOUNTAIN_LEVEL) & (temp < 0.30)] = B["rock"]
        # Coast.
        ocean = self.water == WATER_OCEAN
        coast = np.zeros((H, W), bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            coast |= np.roll(np.roll(ocean, dy, 0), dx, 1)
        bm[coast & ~ocean & (elev < SEA_LEVEL + 0.05)] = B["beach"]
        bm[ocean] = B["ocean"]
        return bm

    def _shape_beaches(self):
        """Give shores realistic variety instead of a uniform one-tile sand ring. Real
        beaches depend on: SEDIMENT washed down by rivers (sandy, wide deltas at river
        mouths); coastal SLOPE (gentle → broad sand, steep cliffs → narrow gravel/pebble
        'shingle'); and COASTLINE SHAPE (sand piles up in sheltered bays; exposed headlands
        are scoured to shingle). We score each coastal tile on these and grow the beach
        inland by a width that follows the score, choosing sand vs. shingle by exposure.
        (Seasonal/tidal widening is a future dynamic layer; this is the static geography.)"""
        ocean = self.water == WATER_OCEAN
        land = self.water == WATER_NONE
        if not ocean.any():
            return
        adj = np.zeros((H, W), bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj |= np.roll(np.roll(ocean, dy, 0), dx, 1)
        coast = land & adj
        if not coast.any():
            return
        elev = self.elevation
        # Coastal slope (how fast the land rises behind the shore): steep ⇒ cliff/shingle.
        gy = np.roll(elev, -1, 0) - np.roll(elev, 1, 0)
        gx = np.roll(elev, -1, 1) - np.roll(elev, 1, 1)
        slope = np.sqrt(gy * gy + gx * gx)
        # Exposure: how much open sea surrounds a spot — high on jutting headlands, low in
        # sheltered bays. Sediment: a soft halo around river mouths (sand supply).
        exposure = _smooth(ocean.astype(np.float32), 6)
        sediment = _smooth((self.water == WATER_RIVER).astype(np.float32), 5)
        # Width budget per coastal tile: sediment & shelter widen it, slope narrows it.
        wf = np.clip(1.6 + sediment * 14.0 - slope * 22.0 - exposure * 2.2, 0.0, 6.0)
        # Shingle where steep, exposed and starved of river sand; sand everywhere else.
        # Thresholds are relative to THIS coast's own distribution so a sensible fraction
        # of the steep, wave-exposed shore turns to gravel regardless of overall relief.
        cs, cx_ = slope[coast], exposure[coast]
        s_hi = np.quantile(cs, 0.55) if cs.size else 1.0
        e_hi = np.quantile(cx_, 0.50) if cx_.size else 1.0
        shingle = coast & (slope > s_hi) & (exposure > e_hi) & (sediment < 0.03)
        mat = np.zeros((H, W), np.uint8)
        mat[coast] = B["beach"]
        mat[shingle] = B["shingle"]
        budget = np.where(coast, wf, 0.0).astype(np.float32)
        inbeach = coast.copy()
        for _ in range(6):                      # grow the beach inland, width ∝ budget
            best = np.zeros((H, W), np.float32)
            bestmat = np.zeros((H, W), np.uint8)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = np.roll(np.roll(budget, dy, 0), dx, 1) - 1.0
                nm = np.roll(np.roll(mat, dy, 0), dx, 1)
                take = nb > best
                best = np.where(take, nb, best)
                bestmat = np.where(take, nm, bestmat)
            grow = land & ~inbeach & (best > 0)
            budget = np.where(grow, best, budget)
            mat = np.where(grow, bestmat, mat)
            inbeach |= grow
        set_tiles = inbeach & (mat != 0) & land
        self.biome[set_tiles] = mat[set_tiles]

    def _suitability(self, species: int, reg=None) -> np.ndarray:
        """Per-tile growth suitability in [-1,1] for a plant *right now* (season-aware).
        Optionally restricted to a region slice so ecology stays off the full map."""
        if reg is None:
            sl = (slice(None), slice(None))
        else:
            y0, y1, x0, x1 = reg
            sl = (slice(y0, y1), slice(x0, x1))
        p = PLANTS[species]
        temp = self.temperature_field(reg)
        biome_ok = np.isin(self.biome[sl], [B[b] for b in p["biomes"]]).astype(np.float32)
        t_ok = _band(temp, *p["t"])
        m_ok = _band(self.moisture[sl], *p["m"])
        suit = biome_ok * np.minimum(t_ok, m_ok) * (0.4 + 0.6 * self.soil[sl])
        # No land plants in water.
        suit[self.water[sl] != WATER_NONE] = 0
        # Below 0 means actively dying conditions (out of comfort band on a live tile).
        return suit.astype(np.float32) * 2 - (1 - np.minimum(t_ok, m_ok)) * 0.15

    def _seed_initial_vegetation(self):
        for sp in PLANTS:
            suit = self._suitability(sp)
            chance = np.clip(suit, 0, 1) * PLANTS[sp]["spread"] * 8
            place = (self.rng.random((H, W)) < chance) & (self.veg_sp == 0)
            self.veg_sp[place] = sp
            self.veg_growth[place] = self.rng.random((H, W))[place].astype(np.float32) * 0.6 + 0.2

    def _seed_initial_wildlife(self, count: int, center=None, radius: int = 600):
        """Seed wildlife on land, clustered within `radius` of `center` so the founding
        valley is alive (the rest of the vast map is left to be discovered / stocked)."""
        if center is None:
            y0, y1, x0, x1 = 0, H, 0, W
        else:
            cx, cy = center
            x0, x1 = max(0, cx - radius), min(W, cx + radius)
            y0, y1 = max(0, cy - radius), min(H, cy + radius)
        land = np.argwhere((self.water[y0:y1, x0:x1] == WATER_NONE)
                           & (self.biome[y0:y1, x0:x1] != B["ocean"]))
        if not len(land):
            return
        weights = {"rabbit": 0.62, "deer": 0.31, "wolf": 0.07}
        for _ in range(count):
            sp = self.rng.choice(list(weights), p=list(weights.values()))
            ly, lx = land[self.rng.integers(len(land))]
            self._add_animal(sp, int(x0 + lx), int(y0 + ly), age=self.rng.integers(0, 60))

    def _add_animal(self, species: str, x: int, y: int, age: float = 0):
        species = str(species)
        a = ANIMALS[species]
        self.animals.append({
            "id": "a_" + uuid.uuid4().hex[:8], "sp": species,
            "x": int(x), "y": int(y), "age": float(age),
            "energy": float(a["repro_at"]) * 0.8,
            "sex": int(self.rng.integers(0, 2)),
            "repro_cd": 0.0,            # game-minute the animal may breed again
            "feed_next": 0.0,          # game-minute a predator may eat again (digestion)
        })

    # ── time ──────────────────────────────────────────────────────────────────
    def day(self) -> int:
        return int(self.clock // (24 * 60))

    def time_of_day(self) -> float:
        return (self.clock % (24 * 60)) / 60.0          # hours 0..24

    def season(self) -> str:
        return SEASONS[(self.day() // DAYS_PER_SEASON) % len(SEASONS)]

    def season_phase(self) -> float:
        """0..1 through the current season (for smooth seasonal transitions)."""
        return ((self.day() % DAYS_PER_SEASON) + self.time_of_day() / 24) / DAYS_PER_SEASON

    def temperature_field(self, reg=None) -> np.ndarray:
        """Current temperature field (whole grid, or just a region slice): annual mean
        shifted by season and time of day. Region form keeps ecology off the full map."""
        base = self._annual_temperature(reg)
        season_off = {"spring": 0.0, "summer": 0.18, "autumn": -0.02, "winter": -0.22}[self.season()]
        diurnal = np.cos((self.time_of_day() / 24 - 0.5) * 2 * np.pi) * -0.06  # cool nights
        cold = -0.10 if self.weather in ("rain", "storm", "snow") else 0.0
        return np.clip(base + season_off + diurnal + cold, 0, 1).astype(np.float32)

    # ── stepping ────────────────────────────────────────────────────────────────
    def step(self, dt_real_sec: float):
        """Advance the world by `dt_real_sec` of wall-clock time. Movement/wildlife run
        every call (bounded by entity count); ecology runs once per game-hour and ONLY
        inside the active region around the people, so the 4M-tile map stays cheap."""
        dt_game_min = dt_real_sec * GAME_SEC_PER_REAL_SEC / 60.0
        self.clock += dt_game_min
        self._update_weather()
        self._tick_wildlife(dt_game_min)
        self._tick_people(dt_game_min)
        if self.clock - self._last_eco >= 60.0:                    # one game-hour elapsed
            self._tick_ecology_active()
            self._last_eco = self.clock
        self.version += 1

    # ── active region & dormancy (the key to a huge, smooth world) ───────────────
    def _active_region(self):
        """Tile box (y0,y1,x0,x1) around the living population — the only place ecology
        runs. Snapped to chunk edges and capped to ACTIVE_MAX per side. None if empty."""
        pts = self.people or self.animals
        if not pts:
            return None
        xs = [int(e["x"]) for e in pts]; ys = [int(e["y"]) for e in pts]
        m = 2 * CHUNK                                   # margin so nearby grazing stays fresh
        x0, x1 = max(0, min(xs) - m), min(W, max(xs) + m + 1)
        y0, y1 = max(0, min(ys) - m), min(H, max(ys) + m + 1)
        if x1 - x0 > ACTIVE_MAX:
            cx = (x0 + x1) // 2; x0 = max(0, cx - ACTIVE_MAX // 2); x1 = min(W, x0 + ACTIVE_MAX)
        if y1 - y0 > ACTIVE_MAX:
            cy = (y0 + y1) // 2; y0 = max(0, cy - ACTIVE_MAX // 2); y1 = min(H, y0 + ACTIVE_MAX)
        x0 = (x0 // CHUNK) * CHUNK; y0 = (y0 // CHUNK) * CHUNK
        x1 = min(W, -(-x1 // CHUNK) * CHUNK); y1 = min(H, -(-y1 // CHUNK) * CHUNK)
        return (y0, y1, x0, x1)

    def _tick_ecology_active(self):
        reg = self._active_region()
        if reg is None:
            return
        y0, y1, x0, x1 = reg
        cy0, cy1 = y0 // CHUNK, (y1 - 1) // CHUNK
        cx0, cx1 = x0 // CHUNK, (x1 - 1) // CHUNK
        # Fast-forward by however stale the most-dormant chunk in the box is (capped).
        lag_hours = (self.clock - float(self._chunk_eco[cy0:cy1 + 1, cx0:cx1 + 1].min())) / 60.0
        passes = int(min(max(lag_hours, 1), ECO_CATCHUP_CAP))
        for _ in range(passes):
            self._tick_ecology(reg)
        self._chunk_eco[cy0:cy1 + 1, cx0:cx1 + 1] = self.clock

    def _update_weather(self):
        if self.clock < self._weather_until:
            return
        season = self.season()
        roll = self.rng.random()
        if season == "winter":
            self.weather = "snow" if roll < 0.4 else ("cloudy" if roll < 0.7 else "clear")
        elif season in ("spring", "autumn"):
            self.weather = "rain" if roll < 0.35 else ("cloudy" if roll < 0.6 else "clear")
        else:  # summer — drier, the odd storm
            self.weather = "storm" if roll < 0.08 else ("rain" if roll < 0.2 else "clear")
        self.weather_intensity = float(self.rng.random() * 0.6 + 0.4) if self.weather != "clear" else 0.0
        self._weather_until = self.clock + self.rng.random() * 240 + 60   # 1–5 game-hours

    def _tick_ecology(self, reg):
        """One game-hour of plant growth, spread, soil and moisture dynamics — confined
        to the region slice `reg` (y0,y1,x0,x1). Sub-array views write back in place."""
        y0, y1, x0, x1 = reg
        sl = (slice(y0, y1), slice(x0, x1))
        moist = self.moisture[sl]; vsp = self.veg_sp[sl]
        vgr = self.veg_growth[sl]; soil = self.soil[sl]
        # Rain refills moisture; sun/heat dries it.
        if self.weather in ("rain", "storm"):
            np.clip(moist + 0.04 * self.weather_intensity, 0, 1, out=moist)
        elif self.weather == "clear":
            dry = 0.012 + 0.02 * np.clip(self.temperature_field(reg) - 0.5, 0, 1)
            np.clip(moist - dry, 0, 1, out=moist)
        self._dampen_tick(reg)

        for sp in PLANTS:
            mask = vsp == sp
            if not mask.any():
                continue
            suit = self._suitability(sp, reg)
            rate = PLANTS[sp]["grow"]
            delta = np.where(suit > 0, suit * rate, suit * 0.05)   # thrive vs. wither
            vgr[mask] = vgr[mask] + delta[mask]
            dead = mask & (vgr <= 0)                                # withered to nothing
            vsp[dead] = VEG_NONE
            vgr[dead] = 0
            np.clip(vgr, 0, 1, out=vgr)
            mature = (vsp == sp) & (vgr > 0.75)
            if mature.any():
                self._spread(sp, mature, suit, reg)
            soil[vsp == sp] = np.clip(soil[vsp == sp] - 0.0008, 0, 1)
        fallow = vsp == VEG_NONE
        soil[fallow] = np.clip(soil[fallow] + 0.0004, 0, 1)

    def _spread(self, sp: int, mature: np.ndarray, suit: np.ndarray, reg):
        y0, y1, x0, x1 = reg
        vsp = self.veg_sp[y0:y1, x0:x1]; vgr = self.veg_growth[y0:y1, x0:x1]
        wat = self.water[y0:y1, x0:x1]
        empty = (vsp == VEG_NONE) & (suit > 0.15) & (wat == WATER_NONE)
        chance = PLANTS[sp]["spread"]
        h, w = mature.shape
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = np.roll(np.roll(mature, dy, 0), dx, 1)
            cand = src & empty & (self.rng.random((h, w)) < chance)
            vsp[cand] = sp
            vgr[cand] = 0.05
            empty &= ~cand

    def _dampen_tick(self, reg):
        y0, y1, x0, x1 = reg
        moist = self.moisture[y0:y1, x0:x1]
        wet = self.water[y0:y1, x0:x1] != WATER_NONE
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj = np.roll(np.roll(wet, dy, 0), dx, 1)
            moist[adj] = np.maximum(moist[adj], 0.6)

    def _tick_wildlife(self, dt_game_min: float):
        if not self.animals:
            return
        dt_day = dt_game_min / (24 * 60)
        births = []
        eaten: set[int] = set()
        # Spatial index: species → tile → animals. Powers cheap prey/mate lookups
        # so the tick is ~O(n) instead of O(n²) as populations grow.
        index: dict[str, dict[tuple[int, int], list[dict]]] = {}
        per_sp: dict[str, int] = {}
        for a in self.animals:
            index.setdefault(a["sp"], {}).setdefault((a["x"], a["y"]), []).append(a)
            per_sp[a["sp"]] = per_sp.get(a["sp"], 0) + 1
        for a in self.animals:
            if id(a) in eaten:
                continue
            spec = ANIMALS[a["sp"]]
            a["age"] += dt_day
            a["energy"] -= spec["drain"] * dt_game_min
            cap = spec["repro_at"] * ENERGY_CAP_MULT
            # Feed.
            if spec["diet"] == "graze":
                g = self.veg_growth[a["y"], a["x"]]
                if g > 0.02:
                    bite = float(min(g, spec["graze"] * dt_game_min))  # keep energy a py-float
                    self.veg_growth[a["y"], a["x"]] = g - bite
                    a["energy"] = min(cap, a["energy"] + bite * spec["eat_gain"])
                    target = None                      # well-fed: graze in place / wander
                else:
                    target = self._nearest_food_dir(a)  # barren tile: go find greener grass
            else:
                target = self._nearest_prey_dir(a, index, spec)
                if self.clock >= a.get("feed_next", 0):     # hungry again (digested)
                    here = self._prey_here(a, index, spec, eaten)
                    if here and self.rng.random() < spec["kill_chance"]:   # prey may escape
                        eaten.add(id(here[0]))
                        a["energy"] = min(cap, a["energy"] + spec["eat_gain"])
                        a["feed_next"] = self.clock + spec["feed_cd"]
            # Move (speed is tiles per tick; server ticks ≈ once/real-sec).
            self._move_animal(a, target, spec["speed"])
            # Reproduce — gate on the cheap checks first (so the bounded mate search,
            # which is the only non-trivial cost, almost never runs), then look nearby.
            if (a["energy"] >= spec["repro_at"] and a["age"] >= spec["maturity"]
                    and self.clock >= a.get("repro_cd", 0)
                    and per_sp.get(a["sp"], 0) < spec.get("pop_cap", 200)
                    and self._has_mate(a, index)):
                a["energy"] -= spec["repro_cost"]
                a["repro_cd"] = self.clock + spec["repro_cd"] * 24 * 60
                per_sp[a["sp"]] = per_sp.get(a["sp"], 0) + 1
                births.append((a["sp"], a["x"], a["y"]))
        # Cull the dead (starved, old, or eaten).
        self.animals = [
            a for a in self.animals
            if id(a) not in eaten and a["energy"] > 0 and a["age"] <= ANIMALS[a["sp"]]["max_age"]
        ]
        for sp, x, y in births:
            self._add_animal(sp, x, y, age=0)

    def _nearest_food_dir(self, a):
        v = ANIMALS[a["sp"]]["vision"]
        y0, y1 = max(0, a["y"] - v), min(H, a["y"] + v + 1)
        x0, x1 = max(0, a["x"] - v), min(W, a["x"] + v + 1)
        patch = self.veg_growth[y0:y1, x0:x1]
        if patch.max() < 0.1:
            return None
        ly, lx = np.unravel_index(np.argmax(patch), patch.shape)
        return (x0 + lx - a["x"], y0 + ly - a["y"])

    def _prey_here(self, a, index, spec, eaten) -> list:
        """Live prey sharing this predator's tile."""
        tile = (a["x"], a["y"])
        out = []
        for psp in spec["prey"]:
            for p in index.get(psp, {}).get(tile, ()):
                if id(p) not in eaten:
                    out.append(p)
        return out

    def _nearest_prey_dir(self, a, index, spec):
        """Direction toward the nearest visible prey (scans only occupied prey tiles)."""
        v = spec["vision"]
        best = None; bd = 1e9
        for psp in spec["prey"]:
            for (px, py) in index.get(psp, {}):
                d = abs(px - a["x"]) + abs(py - a["y"])
                if d < bd and d <= v:
                    bd, best = d, (px - a["x"], py - a["y"])
        return best

    def _move_animal(self, a, target, speed):
        if target and (target[0] or target[1]):
            sx = int(np.sign(target[0])); sy = int(np.sign(target[1]))
        else:
            sx = int(self.rng.integers(-1, 2)); sy = int(self.rng.integers(-1, 2))
        for _ in range(speed):
            nx, ny = a["x"] + sx, a["y"] + sy
            if 0 <= nx < W and 0 <= ny < H and self.water[ny, nx] != WATER_OCEAN:
                a["x"], a["y"] = nx, ny
            else:
                break

    def _has_mate(self, a, index) -> bool:
        """A mature same-species mate within MATE_RADIUS, via the spatial index. Sex is
        tracked (it matters for people later) but wildlife breeding doesn't gate on it —
        that would make sparse prey non-viable. Scans only the small radius box."""
        mat = ANIMALS[a["sp"]]["maturity"]
        tiles = index.get(a["sp"], {})
        for dy in range(-MATE_RADIUS, MATE_RADIUS + 1):
            for dx in range(-MATE_RADIUS, MATE_RADIUS + 1):
                for o in tiles.get((a["x"] + dx, a["y"] + dy), ()):
                    if o is not a and o["age"] >= mat:
                        return True
        return False

    # ── people: seeding & helpers ──────────────────────────────────────────────
    def _nearest_land(self, x: int, y: int) -> tuple[int, int]:
        """Closest walkable (non-water) tile to (x,y), spiralling outward."""
        if self._in(x, y) and self.water[y, x] == WATER_NONE:
            return x, y
        for r in range(1, 16):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    if self._in(nx, ny) and self.water[ny, nx] == WATER_NONE:
                        return nx, ny
        return x, y

    def _add_person(self, x: int, y: int, name: str | None = None, age: float | None = None):
        sex = int(self.rng.integers(0, 2))               # 0 = he, 1 = she (for names + later kinship)
        if name is None:
            pool = NAMES_M if sex == 0 else NAMES_F
            name = str(pool[int(self.rng.integers(len(pool)))])
        self.people.append({
            "id": "p_" + uuid.uuid4().hex[:8], "name": name, "sex": sex,
            "x": int(x), "y": int(y),
            "age": float(age if age is not None else self.rng.integers(1200, 2700)),  # ~20–45 yrs
            "hunger": float(self.rng.random() * 0.2 + 0.1),
            "thirst": float(self.rng.random() * 0.2 + 0.1),
            "fatigue": float(self.rng.random() * 0.2),
            "hp": 1.0,
            "inv": {},                                   # carried goods, e.g. {"food":3,"wood":2,"axe":1}
            "home": (int(x), int(y)),                    # anchor: idle wandering drifts back here
            "home_struct": None,                         # id of their shelter once built
            "known": {},                                 # remembered resource spots {water/food/wood: [x,y]}
            "heading": None,                             # persistent roaming direction while searching
            "action": "wander",                          # current body behaviour (for the renderer)
        })

    def _add_structure(self, kind: str, x: int, y: int, by: str = "?") -> str:
        sid = "s_" + uuid.uuid4().hex[:8]
        self.structures.append({
            "id": sid, "kind": kind, "x": int(x), "y": int(y),
            "by": by, "t": round(self.clock, 1),
        })
        self.version += 1
        self._note("build", f"{by} built a {kind} at ({x},{y}).")
        return sid

    def _seed_initial_people(self, count: int, center=None):
        """Settle a small founding band together near the chosen origin (hospitable,
        watered ground — thirst is the fastest need, so a dry start is a death sentence)."""
        cx, cy = center if center is not None else self._choose_origin()
        for _ in range(count):
            jx = int(np.clip(cx + self.rng.integers(-4, 5), 0, W - 1))
            jy = int(np.clip(cy + self.rng.integers(-4, 5), 0, H - 1))
            px, py = self._nearest_land(jx, jy)
            self._add_person(px, py)

    # ── people: the body loop (cheap, rule-based, no LLM) ───────────────────────
    def _tick_people(self, dt_game_min: float):
        if not self.people:
            return
        dt_day = dt_game_min / (24 * 60)
        hod = self.time_of_day()
        night = hod < 6 or hod >= 21

        dead = []
        for p in self.people:
            p["age"] += dt_day
            p["hunger"] = min(1.0, p["hunger"] + PERSON["hunger_rate"] * dt_game_min)
            p["thirst"] = min(1.0, p["thirst"] + PERSON["thirst_rate"] * dt_game_min)
            p["fatigue"] = min(1.0, p["fatigue"] + PERSON["fatigue_rate"] * dt_game_min)

            x, y = p["x"], p["y"]
            # Perception is a SMALL window around this person (vision-sized), so people
            # cost stays the same whether the map is 128² or 2048². lx,ly = the person's
            # position inside the window; masks are indexed in window coords.
            edible, drinkable, tree, stone, fiber, leaf, lx, ly, wx0, wy0 = self._perceive(x, y)
            # Remember where resources are, so a future need can be navigated to rather
            # than stumbled upon. (A body-level memory; the LLM "mind" can keep richer
            # ones later.) Only overwrite when something is actually in sight.
            known = p["known"]
            for key, mask in (("water", drinkable), ("food", edible), ("wood", tree),
                              ("stone", stone), ("fiber", fiber), ("leaves", leaf)):
                loc = self._nearest_loc(mask, lx, ly, wx0, wy0)
                if loc:
                    known[key] = loc
            action, movedir = self._person_decide(p, edible, drinkable, tree, stone, fiber, leaf, night, lx, ly)
            p["action"] = action
            if action == "eat":
                g = float(self.veg_growth[y, x])
                if edible[ly, lx] and g > 0.12:                  # graze the tile
                    bite = min(g, PERSON["eat_bite"] * dt_game_min)
                    self.veg_growth[y, x] = g - bite
                    p["hunger"] = max(0.0, p["hunger"] - bite * PERSON["food_value"])
                elif p["inv"].get("food", 0) > 0:                # eat from the pack
                    p["inv"]["food"] -= 1
                    p["hunger"] = max(0.0, p["hunger"] - 0.35)
            elif action == "drink":
                p["thirst"] = max(0.0, p["thirst"] - PERSON["drink_rate"] * dt_game_min)
            elif action == "rest":
                # A sheltered person resting at home recovers faster — but only as well as
                # their home insulates (a leaf lean-to barely helps; a timber hut is snug).
                mult = 1.0
                if p.get("home_struct") and (x, y) == tuple(p["home"]):
                    insul = p.get("insul", 1.0)
                    mult = 1.0 + (BUILD["rest_sheltered_mult"] - 1.0) * insul
                p["fatigue"] = max(0.0, p["fatigue"] - PERSON["rest_rate"] * mult * dt_game_min)
            elif action == "gather":
                g = float(self.veg_growth[y, x])
                if edible[ly, lx] and g > PERSON["gather_min"] and p["inv"].get("food", 0) < PERSON["inv_cap"]:
                    take = min(g - 0.2, 0.3)
                    self.veg_growth[y, x] = g - take
                    p["inv"]["food"] = p["inv"].get("food", 0) + 1
            elif action == "chop":
                if tree[ly, lx]:
                    self.veg_growth[y, x] = max(0.0, float(self.veg_growth[y, x]) - BUILD["chop_take"])
                    if self.veg_growth[y, x] <= 0.05:        # felled — the tile clears
                        self.veg_sp[y, x] = VEG_NONE
                        self.veg_growth[y, x] = 0.0
                    gain = BUILD["chop_yield"] + (BUILD["axe_bonus"] if p["inv"].get("axe", 0) else 0)
                    p["inv"]["wood"] = p["inv"].get("wood", 0) + gain
            elif action == "mine":
                if stone[ly, lx]:
                    p["inv"]["stone"] = p["inv"].get("stone", 0) + BUILD["mine_yield"]
            elif action == "gather_fiber":
                g = float(self.veg_growth[y, x])
                if fiber[ly, lx] and g > 0.20 and p["inv"].get("fiber", 0) < PERSON["inv_cap"]:
                    self.veg_growth[y, x] = max(0.0, g - 0.25)
                    p["inv"]["fiber"] = p["inv"].get("fiber", 0) + 1
            elif action == "gather_leaves":
                g = float(self.veg_growth[y, x])
                if leaf[ly, lx] and g > 0.25 and p["inv"].get("leaves", 0) < LEAF_CAP:
                    self.veg_growth[y, x] = max(0.0, g - 0.12)   # stripping leaves barely dents the plant
                    p["inv"]["leaves"] = p["inv"].get("leaves", 0) + LEAF_GATHER
            elif action == "craft":
                if p["inv"].get("axe", 0) < 1 and p["inv"].get("wood", 0) >= BUILD["axe_wood"]:
                    p["inv"]["wood"] -= BUILD["axe_wood"]
                    p["inv"]["axe"] = 1
                    self._note("craft", f"{p['name']} crafted a crude axe.")
            elif action == "found_site":
                self._found_site(p)
            elif action == "build_block":
                self._build_next_block(p)
            # Seeking and wandering move the body; acting-in-place does not.
            if action in ("seek_food", "seek_water", "seek_wood", "seek_stone",
                          "seek_fiber", "seek_leaves", "haul", "wander"):
                self._move_person(p, movedir)

            # Health: a maxed need erodes the body; being well-supplied heals it.
            if p["hunger"] >= 1.0 or p["thirst"] >= 1.0:
                p["hp"] -= PERSON["starve_dmg"] * dt_game_min
            elif p["hunger"] < 0.5 and p["thirst"] < 0.5:
                p["hp"] = min(1.0, p["hp"] + PERSON["heal"] * dt_game_min)

            if p["hp"] <= 0 or p["age"] > PERSON["max_age"]:
                dead.append(p)

        for p in dead:
            cause = "old age" if p["age"] > PERSON["max_age"] else "hunger and thirst"
            self._note("death", f"{p['name']} died of {cause}.")
        if dead:
            ids = {id(p) for p in dead}
            self.people = [p for p in self.people if id(p) not in ids]

    def _perceive(self, x, y):
        """Build this person's small perception windows (edible/drinkable/tree/stone)
        around (x,y), plus their position (lx,ly) inside the window. Cost is O(vision²),
        independent of world size — the key to people staying cheap on a 4M-tile map."""
        v = PERSON["vision"]
        wy0, wy1 = max(0, y - v - 1), min(H, y + v + 2)
        wx0, wx1 = max(0, x - v - 1), min(W, x + v + 2)
        vsp = self.veg_sp[wy0:wy1, wx0:wx1]; vgr = self.veg_growth[wy0:wy1, wx0:wx1]
        wat = self.water[wy0:wy1, wx0:wx1]; bio = self.biome[wy0:wy1, wx0:wx1]
        edible = np.isin(vsp, EDIBLE_IDS) & (vgr > 0.12)
        tree = np.isin(vsp, WOOD_IDS) & (vgr > BUILD["chop_growth_min"])
        stone = np.isin(bio, [B["rock"], B["mountain"]]) & (wat == WATER_NONE)
        fiber = np.isin(vsp, FIBER_IDS) & (vgr > 0.20)            # grasses to pull for thatch/rope
        leaf = np.isin(vsp, LEAF_IDS) & (vgr > 0.25)             # foliage to strip for a leaf shelter
        watery = wat != WATER_NONE
        drinkable = np.zeros_like(watery)               # land tiles bordering water
        if watery.shape[0] > 2 and watery.shape[1] > 2:
            drinkable[1:-1, 1:-1] = (
                (watery[:-2, 1:-1] | watery[2:, 1:-1] | watery[1:-1, :-2] | watery[1:-1, 2:])
                & (wat[1:-1, 1:-1] == WATER_NONE))
        return edible, drinkable, tree, stone, fiber, leaf, x - wx0, y - wy0, wx0, wy0

    def _nearest_local(self, mask, lx, ly):
        """Step direction toward the nearest True tile in a window mask (centre lx,ly)."""
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        k = int(np.argmin(np.abs(xs - lx) + np.abs(ys - ly)))
        dx, dy = int(np.sign(xs[k] - lx)), int(np.sign(ys[k] - ly))
        return (dx, dy) if (dx or dy) else None

    def _nearest_loc(self, mask, lx, ly, wx0, wy0):
        """Global (x,y) of the nearest True tile in a window, or None — for memory."""
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        k = int(np.argmin(np.abs(xs - lx) + np.abs(ys - ly)))
        return [int(xs[k] + wx0), int(ys[k] + wy0)]

    def _explore_dir(self, p):
        """A persistent roaming heading (re-rolled now and then) so a searching person
        covers ground instead of jittering in place."""
        h = p.get("heading")
        if not h or (h[0] == 0 and h[1] == 0) or self.rng.random() < 0.04:
            h = [int(self.rng.integers(-1, 2)), int(self.rng.integers(-1, 2))]
            if h[0] == 0 and h[1] == 0:
                h = [1, 0]
            p["heading"] = h
        return (h[0], h[1])

    def _seek(self, p, x, y, here, mask, lx, ly, key, act_here, act_seek):
        """Resolve 'go get a resource': use it if standing on it, else head to the nearest
        one in sight, else toward the last-known spot (memory), else explore to find it.
        People never give up and die next to an out-of-sight resource."""
        if here:
            return act_here, None
        d = self._nearest_local(mask, lx, ly)
        if d:
            return act_seek, d
        kloc = p["known"].get(key)
        if kloc:
            kx, ky = kloc
            if abs(kx - x) + abs(ky - y) <= 1:        # arrived but it's gone — forget it
                p["known"][key] = None
            else:
                return act_seek, (int(np.sign(kx - x)), int(np.sign(ky - y)))
        return act_seek, self._explore_dir(p)

    def _person_decide(self, p, edible, drinkable, tree, stone, fiber, leaf, night, lx, ly):
        """Pick the most pressing body action. Returns (action, movedir|None). Masks are
        window-local; lx,ly is the person's position inside them.
        Priority: thirst → hunger → rest → opportunistic forage → craft/build → wander."""
        x, y = p["x"], p["y"]
        if p["thirst"] >= PERSON["t_thirst"]:
            # Drink here, else head to water in sight / remembered / go search for it.
            return self._seek(p, x, y, bool(drinkable[ly, lx]), drinkable, lx, ly,
                              "water", "drink", "seek_water")
        if p["hunger"] >= PERSON["t_hunger"]:
            if p["inv"].get("food", 0) > 0:
                return "eat", None
            return self._seek(p, x, y, bool(edible[ly, lx]) and self.veg_growth[y, x] > 0.12,
                              edible, lx, ly, "food", "eat", "seek_food")
        if p["fatigue"] >= PERSON["t_rest"] or (night and p["fatigue"] > 0.35):
            return "rest", None
        # Opportunistic top-ups: sip or nibble while standing on a resource.
        if drinkable[ly, lx] and p["thirst"] > 0.25:
            return "drink", None
        if edible[ly, lx] and self.veg_growth[y, x] > 0.12 and p["hunger"] > 0.30:
            return "eat", None
        # Always keep a small emergency food reserve in the pack before doing anything else.
        if (edible[ly, lx] and self.veg_growth[y, x] > PERSON["gather_min"]
                and p["inv"].get("food", 0) < 3):
            return "gather", None
        # Then, when comfortable and daylit, work on the project (axe → shelter → stone)
        # rather than hoarding food beyond the reserve.
        if (p["hunger"] < 0.5 and p["thirst"] < 0.5 and p["fatigue"] < 0.6 and not night
                and self.clock >= p.get("build_cd", 0)):
            proj = self._person_build_decide(p, tree, stone, fiber, leaf, lx, ly)
            if proj:
                return proj
        # Otherwise lay in a little food while standing on a rich patch.
        if (edible[ly, lx] and self.veg_growth[y, x] > PERSON["gather_min"]
                and p["inv"].get("food", 0) < PERSON["inv_cap"]):
            return "gather", None
        # Idle: drift back toward home if we've strayed, else amble.
        hx, hy = p["home"]
        if abs(hx - x) + abs(hy - y) > 6:
            return "wander", (int(np.sign(hx - x)), int(np.sign(hy - y)))
        return "wander", None

    def _person_build_decide(self, p, tree, stone, fiber, leaf, lx, ly):
        """The crafting/building drive (a comfortable person's 'project'). Returns a body
        action (craft / chop / gather_* / found_site / build_block / haul / a seek_* move
        toward a resource), or None. People raise a real tile-by-tile building from a
        blueprint: found the footprint, then forage the materials and lay one tile per turn.
        The first home is a quick leaf lean-to — the cheapest shelter there is."""
        x, y = p["x"], p["y"]
        inv = p["inv"]
        hx, hy = p["home"]

        getters = {
            "wood":   lambda: self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly,
                                         "wood", "chop", "seek_wood"),
            "fiber":  lambda: self._seek(p, x, y, bool(fiber[ly, lx]), fiber, lx, ly,
                                         "fiber", "gather_fiber", "seek_fiber"),
            "leaves": lambda: self._seek(p, x, y, bool(leaf[ly, lx]), leaf, lx, ly,
                                         "leaves", "gather_leaves", "seek_leaves"),
        }

        # First project: a crude axe — cheap, and it makes every later chop yield more.
        if inv.get("axe", 0) < 1:
            if inv.get("wood", 0) >= BUILD["axe_wood"]:
                return "craft", None
            return getters["wood"]()

        # Main project: build a home, tile by tile, from a blueprint (a leaf shelter first).
        if p.get("home_struct") is None:
            site = self._person_site(p)
            if site is None:                                   # no footprint yet — lay one at home
                if abs(hx - x) + abs(hy - y) <= 1:
                    return "found_site", None
                return "haul", (int(np.sign(hx - x)), int(np.sign(hy - y)))
            task = self._site_next_task(site)
            if task is None:                                   # all tiles placed → finish it
                self._finish_site(p, site)
                return None
            item, qty = task["cost"]
            if inv.get(item, 0) >= qty:                        # have the material — go lay it
                if max(abs(task["x"] - x), abs(task["y"] - y)) <= 2:
                    return "build_block", None
                return "haul", (int(np.sign(task["x"] - x)), int(np.sign(task["y"] - y)))
            return getters.get(item, getters["wood"])()

        # Home raised — lay in stone for the next slice (stone houses, workshops).
        if inv.get("stone", 0) < BUILD["stone_stock"]:
            return self._seek(p, x, y, bool(stone[ly, lx]), stone, lx, ly,
                              "stone", "mine", "seek_stone")
        return None

    # ── tile building: blueprint → site → block placement ───────────────────────
    def _person_site(self, p):
        """The person's active (unfinished) construction site, or None."""
        sid = p.get("site")
        if not sid:
            return None
        for s in self.sites:
            if s["id"] == sid and not s["done"]:
                return s
        return None

    @staticmethod
    def _site_next_task(site):
        """The next tile to place (blocks before roofs, in layout order), or None."""
        for t in site["tasks"]:
            if not t["done"]:
                return t
        return None

    def _blueprint_tasks(self, name, ox, oy):
        """Turn a blueprint at origin (ox,oy) into placement tasks + the home core tile, or
        (None, None) if the footprint doesn't fit (off-map or over water). Each task carries
        its own material cost so blueprints can mix wood, thatch and leaves. Blocks are laid
        first, then roof tiles. A 'C' core lays no block but is roofed and becomes home."""
        bp = BLUEPRINTS.get(name)
        if not bp:
            return None, None
        layout = bp["layout"]
        roof_cost = bp.get("roof_cost", ROOF_COST)
        blocks, roof, core = [], [], None
        for dy, row in enumerate(layout):
            for dx, ch in enumerate(row):
                tx, ty = ox + dx, oy + dy
                if ch == GLYPH_CORE:
                    if not self._in(tx, ty) or self.water[ty, tx] != WATER_NONE:
                        return None, None
                    roof.append({"x": tx, "y": ty, "code": int(BLOCK_FLOOR), "layer": "roof",
                                 "cost": list(roof_cost), "done": False})
                    core = (tx, ty)
                    continue
                code = BLOCK_CHARS.get(ch, BLOCK_EMPTY)
                if code == BLOCK_EMPTY:
                    continue
                if not self._in(tx, ty) or self.water[ty, tx] != WATER_NONE:
                    return None, None
                blocks.append({"x": tx, "y": ty, "code": int(code), "layer": "block",
                               "cost": list(BLOCK_COST[code]), "done": False})
                if bp.get("roof") and code in (BLOCK_FLOOR, BLOCK_DOOR):
                    roof.append({"x": tx, "y": ty, "code": int(code), "layer": "roof",
                                 "cost": list(roof_cost), "done": False})
        return blocks + roof, core

    def _found_site(self, p, name: str = "leaf_shelter"):
        """Reserve a building footprint near the person's home. Tries the chosen blueprint
        (falling back to the always-cheap leaf shelter) over a few offsets so a tree/edge
        doesn't block it forever; on failure sets a cooldown before retrying."""
        bx, by = p["home"]
        cands = [name] + (["leaf_shelter"] if name != "leaf_shelter" else [])
        for cand in cands:
            bp = BLUEPRINTS[cand]
            bw, bh = len(bp["layout"][0]), len(bp["layout"])
            for off in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (2, 1), (-2, -1)):
                ox, oy = bx - bw // 2 + off[0], by - bh // 2 + off[1]
                tasks, core = self._blueprint_tasks(cand, ox, oy)
                if not tasks:
                    continue
                site = {"id": "b_" + uuid.uuid4().hex[:8], "bp": cand, "name": bp["name"],
                        "ox": int(ox), "oy": int(oy), "by": p["name"],
                        "insul": float(bp.get("insulation", 1.0)),
                        "tasks": tasks, "done": False, "t": round(self.clock, 1)}
                self.sites.append(site)
                p["site"] = site["id"]
                home = core or next(((t["x"], t["y"]) for t in tasks if t["code"] == BLOCK_FLOOR), (bx, by))
                p["home"] = (int(home[0]), int(home[1]))
                self.version += 1
                self._note("build", f"{p['name']} marked out a {bp['name'].lower()}.")
                return
        p["build_cd"] = self.clock + 720.0      # nowhere to build here — try again later

    def _build_next_block(self, p):
        """Lay the next blueprint tile if the person is in range and carries the material."""
        site = self._person_site(p)
        if not site:
            return
        task = self._site_next_task(site)
        if task is None:
            self._finish_site(p, site)
            return
        if max(abs(task["x"] - p["x"]), abs(task["y"] - p["y"])) > 2:
            return
        item, qty = task["cost"]
        if p["inv"].get(item, 0) < qty:
            return
        p["inv"][item] -= qty
        if p["inv"][item] <= 0:
            p["inv"].pop(item, None)
        if task["layer"] == "roof":
            self.roofs.add((task["x"], task["y"]))
        else:
            self.blocks[(task["x"], task["y"])] = task["code"]
        task["done"] = True
        self.version += 1
        if self._site_next_task(site) is None:
            self._finish_site(p, site)

    def _finish_site(self, p, site):
        site["done"] = True
        p["home_struct"] = site["id"]
        p["insul"] = site.get("insul", 1.0)     # how well the finished home holds heat/cold
        self.version += 1
        self._note("build", f"{p['name']} finished building a {site['name'].lower()}.")

    def _move_person(self, p, direction):
        """One step (people can't walk onto water or through a solid wall). None → amble."""
        if direction and (direction[0] or direction[1]):
            sx, sy = int(np.sign(direction[0])), int(np.sign(direction[1]))
        else:
            sx, sy = int(self.rng.integers(-1, 2)), int(self.rng.integers(-1, 2))
        nx, ny = p["x"] + sx, p["y"] + sy
        if (self._in(nx, ny) and self.water[ny, nx] == WATER_NONE
                and self.blocks.get((nx, ny)) not in SOLID_BLOCKS):
            p["x"], p["y"] = nx, ny

    # ── live-sim material sourcing (clay/sand/flint/ore beyond wood & fiber) ────
    def _adjacent_water(self, x, y) -> bool:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if self._in(x + dx, y + dy) and self.water[y + dy, x + dx] != WATER_NONE:
                return True
        return False

    def material_at(self, x, y):
        """The sourceable raw on a tile beyond wood/fiber, as (kind, tool_capability) —
        an ore deposit, clay by the water, beach/desert sand, or flint on bare rock —
        else (None, None). Feeds crafting.py's smelting/pottery/glass chains."""
        if not self._in(x, y):
            return None, None
        node = self._ore_index.get((x, y))
        if node and node["amount"] > 0:
            return node["kind"], "pickaxe"
        if self.water[y, x] != WATER_NONE:
            return None, None
        bio = BIOMES[self.biome[y, x]]
        if bio in ("grassland", "savanna", "swamp", "beach") and self._adjacent_water(x, y):
            return "clay", "shovel"
        if bio in ("beach", "desert"):
            return "sand", "shovel"
        if bio in ("rock", "mountain"):
            return "flint", None
        return None, None

    def harvest(self, p, by: str = "body"):
        """Gather whatever raw sits under a person into their pack, if they hold the tool
        it needs (clay/sand need a shovel, ore a pickaxe; flint is bare-handed). Returns
        the kind gathered or None. The rule-body sources wood/fiber inline for building;
        this is the hook the future tech-ladder mind (and gods) use for everything else."""
        x, y = p["x"], p["y"]
        kind, tool = self.material_at(x, y)
        if not kind:
            return None
        if tool and tool not in crafting.tool_caps(p.get("inv", {})):
            return None
        node = self._ore_index.get((x, y))
        if node:
            node["amount"] -= 1
            if node["amount"] <= 0:
                self.ore_nodes = [n for n in self.ore_nodes if n is not node]
                self._rebuild_ore_index()
        inv = p.setdefault("inv", {})
        inv[kind] = inv.get(kind, 0) + 1
        self.version += 1
        return kind

    # ── god actions (UI brush + Aitha directives both call these) ──────────────
    def _in(self, x, y) -> bool:
        return 0 <= x < W and 0 <= y < H

    def _disc(self, cx, cy, r):
        yy, xx = np.mgrid[0:H, 0:W]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r

    def sculpt(self, x: int, y: int, radius: int, delta: float, by: str = "him"):
        """Raise (delta>0) or lower (delta<0) terrain under a brush, then re-derive
        the water and biome layers the change implies."""
        m = self._disc(x, y, radius)
        falloff = 1.0
        self.elevation[m] = np.clip(self.elevation[m] + delta * falloff, 0, 1)
        self.water[m & (self.elevation < SEA_LEVEL)] = WATER_OCEAN
        self.water[m & (self.elevation >= SEA_LEVEL) & (self.water == WATER_OCEAN)] = WATER_NONE
        self.biome = self._classify_biomes()
        self.version += 1
        self._note("sculpt", f"{by} reshaped the land at ({x},{y}).")

    def paint_biome(self, x: int, y: int, radius: int, biome: str, by: str = "him"):
        if biome not in B:
            return
        self.biome[self._disc(x, y, radius)] = B[biome]
        self.version += 1
        self._note("biome", f"{by} painted {biome} at ({x},{y}).")

    def add_water(self, x: int, y: int, radius: int, kind: str = "lake", by: str = "him"):
        code = {"river": WATER_RIVER, "lake": WATER_LAKE, "ocean": WATER_OCEAN}.get(kind, WATER_LAKE)
        m = self._disc(x, y, radius)
        self.water[m] = code
        self.veg_sp[m] = VEG_NONE; self.veg_growth[m] = 0
        self.moisture[m] = 1.0
        self.version += 1
        self._note("water", f"{by} laid {kind} at ({x},{y}).")

    def plant(self, x: int, y: int, species: str, radius: int = 0, by: str = "him"):
        sp = PLANT_BY_NAME.get(species)
        if not sp:
            return
        m = self._disc(x, y, radius) if radius else None
        if m is not None:
            target = m & (self.water == WATER_NONE)
            self.veg_sp[target] = sp
            self.veg_growth[target] = np.maximum(self.veg_growth[target], 0.3)
        elif self._in(x, y) and self.water[y, x] == WATER_NONE:
            self.veg_sp[y, x] = sp
            self.veg_growth[y, x] = max(self.veg_growth[y, x], 0.3)
        self.version += 1
        self._note("plant", f"{by} planted {species} at ({x},{y}).")

    def spawn_animal(self, x: int, y: int, species: str, n: int = 1, by: str = "him"):
        if species not in ANIMALS or not self._in(x, y):
            return
        for _ in range(max(1, n)):
            self._add_animal(species, x, y, age=ANIMALS[species]["maturity"])
        self.version += 1
        self._note("spawn", f"{by} spawned {n}× {species} at ({x},{y}).")

    def spawn_person(self, x: int, y: int, n: int = 1, name: str | None = None, by: str = "him"):
        """Bring n people into the world near (x,y), snapped to walkable land."""
        if not self._in(x, y):
            return
        for _ in range(max(1, n)):
            jx = int(np.clip(x + self.rng.integers(-3, 4), 0, W - 1))
            jy = int(np.clip(y + self.rng.integers(-3, 4), 0, H - 1))
            px, py = self._nearest_land(jx, jy)
            self._add_person(px, py, name=(name if n == 1 else None))
        self.version += 1
        who = name if (name and n == 1) else f"{n} {'soul' if n == 1 else 'souls'}"
        self._note("person", f"{by} drew {who} into the world at ({x},{y}).")

    def _note(self, kind: str, text: str):
        self.log.append({"t": round(self.clock, 1), "kind": kind, "text": text})
        if len(self.log) > 60:
            self.log = self.log[-60:]

    # ── snapshots for the renderer & for Aitha ─────────────────────────────────
    @staticmethod
    def _b64u8(arr: np.ndarray) -> str:
        """Pack a grid as raw little-endian uint8 bytes, base64-encoded. The renderer
        decodes each layer straight into a Uint8Array (one byte per tile, row-major)."""
        return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode("ascii")

    def _pack_layers(self, y0, y1, x0, x1, step) -> dict:
        """Base64 uint8 layers for a tile window [y0:y1, x0:x1] sampled every `step`."""
        ys, xs = slice(y0, y1, step), slice(x0, x1, step)
        return {
            "elevation": self._b64u8((self.elevation[ys, xs] * 255).astype(np.uint8)),
            "biome": self._b64u8(self.biome[ys, xs]),
            "water": self._b64u8(self.water[ys, xs]),
            "veg_sp": self._b64u8(self.veg_sp[ys, xs]),
            "veg_growth": self._b64u8((self.veg_growth[ys, xs] * 255).astype(np.uint8)),
        }

    def _blocks_payload(self):
        """Placed tiles as a flat [[x,y,code], …] list (sparse — only built tiles)."""
        return [[x, y, c] for (x, y), c in self.blocks.items()]

    def _roofs_payload(self):
        return [[x, y] for (x, y) in self.roofs]

    def _sites_payload(self):
        """Construction sites with build progress (for inspect / the god-tools UI)."""
        out = []
        for s in self.sites:
            done = sum(1 for t in s["tasks"] if t["done"])
            out.append({"id": s["id"], "name": s["name"], "ox": s["ox"], "oy": s["oy"],
                        "by": s["by"], "done": s["done"], "built": done, "total": len(s["tasks"])})
        return out

    def snapshot(self) -> dict:
        """Whole-world overview for the initial render: tile layers DOWNSAMPLED to
        ≤ OVERVIEW_MAX per side (so a 4M-tile map ships as ~64KB, not 20MB). Entities
        stay in TRUE tile coords; the renderer stretches the overview across W×H and
        fetches crisp detail via view() when zoomed in."""
        step = max(1, -(-max(W, H) // OVERVIEW_MAX))   # ceil(max(W,H)/OVERVIEW_MAX)
        ovw = len(range(0, W, step)); ovh = len(range(0, H, step))
        return {
            "w": W, "h": H, "ovw": ovw, "ovh": ovh, "ov_step": step,
            "version": self.version, "seed": self.seed,
            "clock": round(self.clock, 1), "day": self.day(), "time": round(self.time_of_day(), 2),
            "season": self.season(), "weather": self.weather,
            "biomes": BIOMES, "sea_level": int(SEA_LEVEL * 255),
            "plants": {sp: PLANTS[sp]["name"] for sp in PLANTS},
            "block_names": BLOCK_NAMES,
            "layers": self._pack_layers(0, H, 0, W, step),
            "animals": self.animals,
            "people": self.people,
            "structures": self.structures,
            "blocks": self._blocks_payload(),
            "roofs": self._roofs_payload(),
            "sites": self._sites_payload(),
            "ore": [{"x": n["x"], "y": n["y"], "kind": n["kind"]} for n in self.ore_nodes],
        }

    def view(self, x0: int, y0: int, x1: int, y1: int, step: int = 1) -> dict:
        """Crisp tile layers for a clamped window — used by the renderer to stream the
        visible area at the current zoom (level-of-detail)."""
        x0 = max(0, min(W, int(x0))); x1 = max(0, min(W, int(x1)))
        y0 = max(0, min(H, int(y0))); y1 = max(0, min(H, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return {"x0": x0, "y0": y0, "x1": x0, "y1": y0, "step": 1, "vw": 0, "vh": 0, "layers": {}}
        step = max(1, int(step))
        return {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "step": step,
            "vw": len(range(x0, x1, step)), "vh": len(range(y0, y1, step)),
            "version": self.version, "layers": self._pack_layers(y0, y1, x0, x1, step),
        }

    def tick_state(self) -> dict:
        """Light per-tick payload for the live renderer (no heavy tile layers — those
        come once via snapshot(); ticks just move time, weather and entities)."""
        return {
            "version": self.version, "clock": round(self.clock, 1), "day": self.day(),
            "time": round(self.time_of_day(), 2), "season": self.season(),
            "weather": self.weather, "census": self.census(),
            "animals": self.animals, "people": self.people,
            "structures": self.structures,
            "blocks": self._blocks_payload(),
            "roofs": self._roofs_payload(),
        }

    def census(self) -> dict:
        counts = {}
        for a in self.animals:
            counts[a["sp"]] = counts.get(a["sp"], 0) + 1
        veg = {PLANTS[sp]["name"]: int((self.veg_sp == sp).sum()) for sp in PLANTS}
        buildings = sum(1 for s in self.sites if s["done"])
        return {"animals": counts, "vegetation": veg, "people": len(self.people),
                "structures": len(self.structures), "buildings": buildings,
                "blocks": len(self.blocks)}

    def digest(self) -> str:
        """Compact text snapshot for Aitha's prompt, so she perceives the world she
        co-rules and can act on it with continuity (her counterpart to room.digest)."""
        c = self.census()
        animals = ", ".join(f"{n}× {s}" for s, n in c["animals"].items()) or "none"
        plants = ", ".join(f"{n} {s}" for s, n in c["vegetation"].items() if n)
        land = int((self.water == WATER_NONE).sum())
        water = W * H - land
        recent = "; ".join(e["text"] for e in self.log[-4:]) or "nothing lately"
        if self.people:
            distress = [p["name"] for p in self.people if p["hunger"] > 0.8 or p["thirst"] > 0.8 or p["hp"] < 0.5]
            roster = ", ".join(p["name"] for p in self.people[:8]) + ("…" if len(self.people) > 8 else "")
            built = c["buildings"]
            ppl = (f"People: {len(self.people)} alive ({roster})"
                   + (f"; struggling: {', '.join(distress[:6])}" if distress else "; all faring well")
                   + (f". They have raised {built} building{'s' if built != 1 else ''}"
                      f" ({len(self.blocks)} tiles laid). " if built or self.blocks else ". "))
        else:
            ppl = "People: none yet — the land is unpeopled. "
        return (
            f"THE WORLD — a {W}×{H} land you and he preside over as gods. "
            f"Day {self.day()}, {self.time_of_day():.0f}:00, {self.season()}, weather {self.weather}. "
            f"Terrain: {land} land tiles, {water} water. Wildlife: {animals}. Flora: {plants}. "
            f"{ppl}"
            f"Craft: people can work {len(crafting.RECIPES)} recipes across a tech ladder "
            f"(wood/stone tools → workbench → cooking → pottery → smelting → metalwork → "
            f"buildings). They raise real tile-by-tile buildings (walls/door/floor/thatch roof) "
            f"from blueprints; today they autonomously chase only the first rungs (axe → a leaf "
            f"lean-to, with sturdier huts/cabins in the blueprint library for later). "
            f"Ore/clay/sand/flint are mineable across the map (tool-gated) for the higher recipes. "
            f"Recent godly acts: {recent}.\n"
            "You may shape it with hidden directives (stripped from what he sees) — each "
            f"holds space-separated values; coordinates are 0..{W - 1}:\n"
            "  <sculpt>x y r d</sculpt>  raise/lower land in radius r (d from -0.3 to 0.3)\n"
            "  <biome>x y r name</biome>  paint a biome (grassland|forest|desert|swamp|tundra|…)\n"
            "  <water>x y r kind</water>  lay water (kind = river|lake|ocean)\n"
            "  <spawn>x y species n</spawn>  add n wildlife (species = rabbit|deer|wolf)\n"
            "  <plant>x y species r</plant>  seed flora in radius r (grass|oak|pine|reeds|palm|cactus|shrub)\n"
            "  <person>x y n</person>  bring n people into being near (x,y); they forage, drink, rest and can die\n"
            "  <whisper>x y a thought to slip into a nearby soul</whisper>  (heard but not yet acted on — their minds awaken in a later step)\n"
            "Act only when you mean to; the world is alive and persists between visits."
        )

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self):
        try:
            os.makedirs(_DIR, exist_ok=True)
            np.savez_compressed(
                PATH_GRID, elevation=self.elevation, biome=self.biome, soil=self.soil,
                moisture=self.moisture, water=self.water, veg_sp=self.veg_sp,
                veg_growth=self.veg_growth, chunk_eco=self._chunk_eco,
            )
            meta = {
                "schema": SCHEMA,
                "seed": self.seed, "clock": self.clock, "last_eco": self._last_eco,
                "origin": list(self._origin),
                "weather": self.weather, "weather_intensity": self.weather_intensity,
                "weather_until": self._weather_until, "animals": self.animals,
                "people": self.people, "structures": self.structures,
                "blocks": {f"{x},{y}": int(c) for (x, y), c in self.blocks.items()},
                "roofs": [[x, y] for (x, y) in self.roofs],
                "sites": self.sites, "ore_nodes": self.ore_nodes,
                "log": self.log, "version": self.version,
            }
            tmp = PATH_META + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, default=_json_safe)
            os.replace(tmp, PATH_META)
        except OSError as e:
            print(f"[world] save failed: {e}")

    def load(self) -> bool:
        """Restore a saved world. Returns False (→ caller regenerates) if the save
        is missing, corrupt, or from an incompatible schema/size, so a world left
        broken by an older build heals itself instead of staying frozen."""
        try:
            with open(PATH_META, encoding="utf-8") as f:
                meta = json.load(f)
            # Reject saves from a different layout version up front — loading them
            # would let step() crash every tick (the tab would look frozen).
            if meta.get("schema") != SCHEMA:
                print(f"[world] save schema {meta.get('schema')!r} != {SCHEMA}; regenerating")
                return False
            with np.load(PATH_GRID) as z:
                self.elevation = z["elevation"]; self.biome = z["biome"]; self.soil = z["soil"]
                self.moisture = z["moisture"]; self.water = z["water"]
                self.veg_sp = z["veg_sp"]; self.veg_growth = z["veg_growth"]
                self._chunk_eco = z["chunk_eco"] if "chunk_eco" in z else None
            # Guard against arrays saved at a different grid size.
            if self.biome.shape != (H, W):
                print(f"[world] saved grid {self.biome.shape} != {(H, W)}; regenerating")
                return False
            if self._chunk_eco is None or self._chunk_eco.shape != (NCHUNK, NCHUNK):
                self._chunk_eco = np.full((NCHUNK, NCHUNK), meta.get("clock", 0.0), np.float32)
            self.seed = meta.get("seed", 0); self.clock = meta.get("clock", 0.0)
            self._origin = tuple(meta.get("origin", [W // 2, H // 2]))
            self._last_eco = meta.get("last_eco", self.clock)
            self.weather = meta.get("weather", "clear")
            self.weather_intensity = meta.get("weather_intensity", 0.0)
            self._weather_until = meta.get("weather_until", 0.0)
            self.animals = meta.get("animals", []); self.people = meta.get("people", [])
            self.structures = meta.get("structures", [])
            self.blocks = {}
            for k, c in (meta.get("blocks") or {}).items():
                sx, sy = k.split(",")
                self.blocks[(int(sx), int(sy))] = int(c)
            self.roofs = {(int(x), int(y)) for x, y in (meta.get("roofs") or [])}
            self.sites = meta.get("sites", [])
            self.ore_nodes = meta.get("ore_nodes", [])
            self._rebuild_ore_index()
            self.log = meta.get("log", [])
            self.version = meta.get("version", 0)
            self.rng = np.random.default_rng()
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            # Corrupt npz (BadZipFile/EOFError), bad JSON, missing keys, etc. —
            # don't let a damaged save take the whole World tab down; regenerate.
            print(f"[world] load failed ({type(e).__name__}: {e}); regenerating")
            return False


# ─── module-level singleton (mirrors room.py's load/save ergonomics) ──────────
_world: World | None = None


def get_world() -> World:
    """Lazily load the saved world, or generate a fresh one on first ever access."""
    global _world
    if _world is None:
        w = World()
        if not w.load():
            w.generate()
            w.save()
        _world = w
    return _world


def reset_world(seed: int | None = None) -> World:
    global _world
    _world = World().generate(seed)
    _world.save()
    return _world


# ════════════════════════════════════════════════════════════════════════════
#  Headless self-test — verifies the data layer with no renderer and no LLM.
#    python -m backend.world        (or:  python world.py  from backend/)
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    t0 = time.time()
    w = World().generate(seed=42)
    gen_ms = (time.time() - t0) * 1000

    # Biome distribution sanity.
    names, counts = np.unique(w.biome, return_counts=True)
    dist = {BIOMES[int(n)]: int(c) for n, c in zip(names, counts)}
    water_tiles = int((w.water != WATER_NONE).sum())

    print(f"generated {W}×{H} world (seed {w.seed}) in {gen_ms:.0f} ms")
    print(f"  water tiles: {water_tiles}  ({water_tiles*100//(W*H)}%)")
    print(f"  biomes: {dist}")
    print(f"  start census: {w.census()}")
    print(f"  founding band: {[p['name'] for p in w.people]}")

    # Simulate ~20 game-days (1 step = 1 real sec = 24 game-sec → 3600 steps/game-day).
    # Watch populations and vegetation across a couple of season turns.
    days = 8
    steps = days * 3600
    print("\n  day  season   weather  rabbit deer wolf  ppl bldg tiles  vegtiles  ms/step")
    t0 = time.time()
    last_day = -1
    day_t0 = t0
    for i in range(steps):
        w.step(dt_real_sec=1.0)
        if w.day() != last_day:
            last_day = w.day()
            c = w.census()
            an = c["animals"]
            now = time.time()
            ms = (now - day_t0) * 1000 / 3600
            day_t0 = now
            print(f"  {w.day():>3}  {w.season():<7}  {w.weather:<7}  "
                  f"{an.get('rabbit',0):>6} {an.get('deer',0):>4} {an.get('wolf',0):>4}  "
                  f"{c['people']:>3} {c['buildings']:>4} {c['blocks']:>5}  "
                  f"{sum(c['vegetation'].values()):>8}  {ms:>6.2f}", flush=True)
    sim_s = time.time() - t0
    if w.people:
        sample = w.people[0]
        print(f"  survivor sample — {sample['name']}: hunger {sample['hunger']:.2f} "
              f"thirst {sample['thirst']:.2f} fatigue {sample['fatigue']:.2f} hp {sample['hp']:.2f} "
              f"inv {sample['inv']} doing '{sample['action']}'")
    axes = sum(1 for p in w.people if p['inv'].get('axe'))
    sheltered = sum(1 for p in w.people if p.get('home_struct'))
    finished = sum(1 for s in w.sites if s['done'])
    print(f"  tile-building — {finished} buildings finished, {len(w.sites)} sites, "
          f"{len(w.blocks)} blocks + {len(w.roofs)} roof tiles laid, {axes} axes, "
          f"{sheltered}/{len(w.people)} people housed")

    # Death mechanic: a body whose needs stay maxed (no relief reachable) must decline
    # and die. People now SEARCH for resources rather than dying in place, so we verify
    # the mechanic deterministically by holding the needs at the limit.
    wd = World().generate(seed=5); wd.people = []
    land = np.argwhere(wd.water == WATER_NONE)
    ly0, lx0 = land[len(land) // 2]
    wd._add_person(int(lx0), int(ly0), name="Doomed")
    for _ in range(6000):
        if not wd.people:
            break
        wd.people[0]["hunger"] = 1.0; wd.people[0]["thirst"] = 1.0
        wd._tick_people(0.4)
    print(f"  death test: a body in unrelieved crisis {'died as expected' if not wd.people else 'SURVIVED (unexpected)'}")
    print(f"\nsimulated {steps} steps in {sim_s:.2f}s "
          f"({steps/sim_s:.0f} steps/s, {sim_s*1000/steps:.2f} ms/step avg)")

    # Tile-building unit check — isolate the chain from survival noise: a comfortable
    # builder, kept fed/watered and handed raw logs + thatch, must craft an axe, mark
    # out a building, and lay it block by block until it's finished and housed.
    wc = World().generate(seed=7)
    wc.people = []
    land = np.argwhere((wc.water == WATER_NONE) & (wc.biome == B["grassland"]))
    by, bx = land[len(land) // 2]
    wc._add_person(int(bx), int(by), name="Builder")
    b = wc.people[0]
    b["inv"].update({"wood": 80, "fiber": 40, "leaves": 30})   # logs, thatch & leaves to work
    wc.clock = 12 * 60                              # high noon (daytime → secondary drives active)
    for _ in range(1500):
        b["hunger"] = b["thirst"] = 0.1; b["fatigue"] = 0.1   # stay comfortable
        if b.get("home_struct"):
            break
        wc._tick_people(0.4)
    site = wc.sites[0] if wc.sites else None
    built_ok = bool(b["inv"].get("axe") and b.get("home_struct") and wc.blocks and wc.roofs)
    print(f"  build test (autonomous): Builder axe={b['inv'].get('axe',0)}, "
          f"first home = {site['name'] if site else 'none'}, blocks={len(wc.blocks)}, "
          f"roofs={len(wc.roofs)}, insul={b.get('insul')}, housed={b.get('home_struct') is not None} "
          f"-> {'OK' if built_ok else 'FAILED'}")

    # And the blueprint library scales up: force a cabin and confirm it can be raised too.
    wc2 = World().generate(seed=7); wc2.people = []
    cy2, cx2 = np.argwhere((wc2.water == WATER_NONE) & (wc2.biome == B["grassland"]))[len(
        np.argwhere((wc2.water == WATER_NONE) & (wc2.biome == B["grassland"]))) // 2]
    wc2._add_person(int(cx2), int(cy2), name="Mason")
    m = wc2.people[0]; m["inv"]["axe"] = 1
    m["inv"].update({"wood": 120, "fiber": 60})
    wc2.clock = 12 * 60
    wc2._found_site(m, "cabin")
    for _ in range(2500):
        if m.get("home_struct"):
            break
        wc2._build_next_block(m)
        if wc2._person_site(m) is None:
            break
        t = wc2._site_next_task(wc2._person_site(m))
        if t and max(abs(t["x"] - m["x"]), abs(t["y"] - m["y"])) > 2:   # step toward the next tile
            m["x"] += int(np.sign(t["x"] - m["x"])); m["y"] += int(np.sign(t["y"] - m["y"]))
    cabin = next((s for s in wc2.sites if s["bp"] == "cabin"), None)
    cabin_ok = bool(cabin and cabin["done"] and m.get("home_struct"))
    print(f"  cabin test (forced blueprint): {sum(t['done'] for t in cabin['tasks']) if cabin else 0}/"
          f"{len(cabin['tasks']) if cabin else 0} tiles, insul={m.get('insul')} "
          f"-> {'OK' if cabin_ok else 'FAILED'}")

    # Live-sim sourcing — mine a real ore deposit (pickaxe-gated), dig clay/sand/flint,
    # then smelt the ore into an ingot via the crafting registry (the full chain).
    ws = World().generate(seed=11)
    src_lines = []
    if ws.ore_nodes:
        node = ws.ore_nodes[0]
        miner = {"x": node["x"], "y": node["y"], "inv": {"crude_pickaxe": 1}}
        got = ws.harvest(miner)
        src_lines.append(f"mined {got or 'nothing'} (held pickaxe)")
        nopick = {"x": ws.ore_nodes[-1]["x"], "y": ws.ore_nodes[-1]["y"], "inv": {}}
        src_lines.append(f"no-tool mine blocked={ws.harvest(nopick) is None}")
    # Smelt copper ore → ingot at a furnace using crafting.py.
    smelt_inv = {"copper_ore": 2, "charcoal": 1}
    smelt_ok = crafting.do_craft(smelt_inv, "copper_ingot", stations={"furnace"})
    src_lines.append(f"smelt copper_ingot={smelt_ok} ({smelt_inv})")
    print("  sourcing test: " + " | ".join(src_lines))

    # Exercise a couple of god actions, then persistence round-trips.
    w.sculpt(64, 64, 6, 0.25, by="test")
    w.add_water(40, 40, 4, "lake", by="test")
    w.spawn_animal(64, 64, "deer", n=5, by="test")
    w.plant(64, 64, "oak", radius=3, by="test")
    w.spawn_person(64, 64, n=3, by="test")
    snap = w.snapshot()
    print(f"\nsnapshot ok: {len(snap['layers'])} layers, {len(snap['animals'])} animals, "
          f"{len(snap['people'])} people, {len(snap['structures'])} structures, "
          f"~{sum(len(v) for v in snap['layers'].values())//1024} KB packed")

    w.save()
    w2 = World()
    ok = w2.load()
    print(f"persistence round-trip: {'OK' if ok and w2.day() == w.day() else 'FAILED'} "
          f"(reloaded day {w2.day()}, clock {w2.clock:.0f})")
    print(f"\ndigest preview:\n{w.digest()}")
    sys.exit(0)
