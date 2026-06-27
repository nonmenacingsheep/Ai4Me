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
import itertools
import json
import os
import shutil
import time
import uuid

import numpy as np

import crafting   # item/recipe registry (content for gods, UI & the mind)
import mind        # the people's inner life: memory, relationships, trade, goals, speech

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
SCHEMA = 7
CHUNK = 64                      # tiles per chunk → (W//CHUNK)² dormancy bookkeeping cells
NCHUNK = W // CHUNK
SEA_LEVEL = 0.36                # elevation below this is ocean
MOUNTAIN_LEVEL = 0.78           # elevation above this reads as mountain/rock

# Ecology only ever runs inside a box around the people, capped to this many tiles
# on a side so a scattered population can't blow the cost up; dormant chunks that
# re-enter the box fast-forward up to this many game-hours of growth at once.
ACTIVE_MAX = 768
ECO_CATCHUP_CAP = 48
ECO_TIME_BUDGET = 0.05          # max wall-seconds an ecology catch-up may spend in one tick
OVERVIEW_MAX = 256              # the whole-world snapshot is downsampled to ≤ this per side

# Time pace. This is a *thinking-first* world: the people deliberate about what their lives
# are for, and that wants a contemplative clock — so a day takes a few real hours, giving
# each mind room to choose, act, and be voiced rather than blurring past. (Lowering this
# only slows real-time pace; in-game balance is calibrated per game-minute and unchanged.)
GAME_SEC_PER_REAL_SEC = 8.0     # 1 real second == 8 game-seconds  →  1 game-day ≈ 3 real hours
# How often (game-minutes) a settled mind re-weighs its drives and may change its aim. The
# body actuates the standing intention every tick between these deliberations.
DELIBERATE_BEAT = 20.0
EXPLORE_LEASH = 40              # tiles from home a wanderer ranges before turning back
TINKER_BEAT = 200.0            # game-minutes between a mind's make-shift-craft experiments
TEACH_BEAT = 120.0            # game-minutes before a soul can be taught another craft
LEDGER_CAP = 240               # most entries kept in the Ledger of Making
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
SNAP_DIR = os.path.join(_DIR, "snapshots")      # named checkpoints the user can save & restore to
PATH_TEMPLATES = os.path.join(_DIR, "templates.json")  # the god's hand-authored building blueprints
                                                # (a library independent of any one world — survives resets)

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
WATER_NONE, WATER_RIVER, WATER_LAKE, WATER_OCEAN, WATER_SHALLOW = 0, 1, 2, 3, 4
# Water a person can WADE through (rivers & shingle shallows are fordable; lakes & ocean are
# barriers to be walked around). Fording leaves a soul WET for a spell (groundwork for a chill
# mechanic later). Deep water still bounds the world the band can reach on foot.
FORDABLE_WATER = (WATER_NONE, WATER_RIVER, WATER_SHALLOW)
WET_DURATION = 90.0                 # game-minutes a soul stays wet after wading
SEEN_FORGET = 720.0                 # game-minutes a remembered sighting of someone stays worth chasing
KNAP_CHOPS = 6                      # hand-chops of wood before a soul works out the axe (knaps a sharp edge)
GATHER_WORK = 2.5                   # game-minutes of hand-work per armful of leaves/fiber (so it's SEEN, not instant)
# Real labour takes real time: chopping a log, prying out stone and laying a building tile each
# take game-minutes of dwell (like gathering), so a soul is SEEN working rather than filling its
# pack or raising a wall in a blink. An axe roughly halves the felling work (see _chop_work).
CHOP_WORK  = 4.0                    # game-min to fell an armful of wood by hand (an axe cuts this down)
MINE_WORK  = 5.0                    # game-min to pry loose a measure of stone
BUILD_WORK = 4.0                    # game-min of labour to lay one building tile
# The eight step directions, used by the obstacle-aware walker to slide around barriers
# instead of freezing nose-to-the-water (the old greedy single-step just stopped dead).
_STEP_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

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
    # ── Two layers per need (game-DAYS; 1 day = 1440 game-min) ───────────────────────
    # Each need has a COMFORT signal (desire — what the mind weighs, rises EARLY) and a
    # physiological RESERVE (the true survival clock — hydration/satiety/stamina, 1=full
    # 0=failing). Comfort peaks long before the reserve runs out, so a soul actively seeks
    # relief well inside the safety margin: a healthy person feels very thirsty within half
    # a day but can endure ~3 days without water; very hungry within ~1.5 days but can last
    # ~3 weeks without food. Death/damage keys off the RESERVE, never off comfort.
    #
    # Comfort rise rates — desire saturates to 1.0 over its (short) comfort span:
    thirst_rate=0.00139, hunger_rate=0.00046, fatigue_rate=0.00069,   # spans ≈ 0.5d / 1.5d / 1.0d
    # Reserve drain rates — the slow physiological clock (drains to 0 over the survival span):
    hydration_drain=0.000231, satiety_drain=0.0000331, stamina_drain=0.000231,  # ≈ 3d / 21d / 3d
    t_hunger=0.50, t_thirst=0.45, t_rest=0.70,        # comfort thresholds to act on
    eat_bite=0.05, food_value=2.5,                     # graze speed / hunger-comfort restored per unit
    drink_rate=0.05, rest_rate=0.04,                   # thirst/fatigue comfort relieved per min
    # How much a relief action restores of the underlying reserve (per min while acting):
    hydrate_rate=0.02, feed_value=0.06, restore_rate=0.018,
    inv_cap=8, gather_min=0.30,                         # carry capacity / tile richness to gather
    # Health (hp) couples to the reserves: it erodes when any reserve falls into the danger
    # zone, and can only heal — and only up to a "vitality" ceiling — when the body is well
    # supplied. Vitality = min(satiety, stamina): chronic hunger or exhaustion drags the whole
    # body (and its hp ceiling) down, the malnutrition/sickness coupling.
    hp_danger=0.30, hp_safe=0.45,                      # reserve below danger erodes hp; above safe lets it heal
    starve_dmg=0.0026, heal=0.0009,                    # hp lost per unit of reserve-deficit / regained when sound
)
# Per-need wiring (comfort key, reserve key, comfort-rise rate, reserve-drain rate) so the
# body loop stays DRY across the three needs.
NEED_MODEL = (
    ("thirst",  "hydration", "thirst_rate",  "hydration_drain"),
    ("hunger",  "satiety",   "hunger_rate",  "satiety_drain"),
    ("fatigue", "stamina",   "fatigue_rate", "stamina_drain"),
)
# The home larder (P2 storage). A settled soul banks the surplus it carries above a travel
# reserve, then draws on that store when caught hungry/thirsty away from food with nothing in
# sight — turning a good forage into a buffer against a lean stretch. Kept ABOVE the barter
# surplus so banking never starves the gift/trade economy of spare food.
STORE_KEEP = {"food": 5, "safe_water": 3}   # carry up to this; bank only the excess (raw water stays on-person to boil)
# Communal granary (interdependence): once the band has roofs, working adults pour their HOME
# surplus (above a personal cushion) into a shared store that anyone can draw from — so children,
# the sick and non-foraging specialists lean on the pool the foragers fill. The seam the P2
# `store_access` stub was left for.
GRANARY_MIN_HOUSED = 3                        # the band raises a common granary once this many are housed
GRANARY_CUSHION = {"food": 8, "safe_water": 4}  # keep this much in your OWN larder before giving the rest
PROVISION_LOAD = 4          # gather this much above the travel reserve before hauling it home to bank
PROVISION_LEASH = 12        # stockpiling stays near home — never range far from water to lay in food

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

# ─── Stakes: exposure & predation ────────────────────────────────────────────
# A roof and the band are only worth something if the open is genuinely dangerous.
# EXPOSURE: caught out at night or in foul weather with no proper roof chills the body —
# it can't rest well and a bitter storm chips its health. A snug hut (insul 1.0) shields
# fully; a draughty leaf lean-to (0.12) barely; a portable sleeping mat not at all (it
# keeps no weather off). This is what gives a finished, well-insulated home survival worth.
EXPOSURE_FATIGUE = 0.0014     # extra fatigue/game-min when exposed, scaled by severity
EXPOSURE_HP      = 0.0011     # hp/game-min lost once exposure is SEVERE (cold storm, no roof)
EXPOSURE_SEVERE  = 0.5        # effective-exposure above which the cold starts to wound
# PREDATION: hungry wolves stalk people caught alone in the open. The band scares them off
# (others nearby) and a roof is safety — a lone soul far from shelter at night is in danger.
WOLF_MENACE_VISION = 7        # tiles a hungry wolf will close on a vulnerable person
WOLF_BAND_SAFETY   = 4        # others within this range of the target deter the wolf (the band protects)
WOLF_GUARDS_SAFE   = 2        # this many companions nearby and the wolf won't risk it
WOLF_BITE          = 0.17     # hp a wolf bite costs its victim
WOLF_BITE_GAIN     = 6.0      # energy a wolf gains from a person (well under a deer — people aren't easy prey)
FLEE_TRIGGER       = 0.55     # cached danger at/above which the BODY reflexively flees, overriding its aim
GUARD_RANGE        = 7        # a guardian shields band-mates (and faces down wolves) within this range
REST_HOMEWARD_MAX  = 8        # at rest, a soul walks home to sleep only if home is within this many tiles
                              # (farther off it sleeps where it stands — no fatal all-night march home)
# Cost-aware walking: each step is scored as (progress toward the goal × MOVE_PROGRESS_W) minus the
# tile's travel cost. Open ground is free; wading a river/shallow is slow; tiles near a prowling wolf
# carry a danger cost that falls off with distance — so a soul prefers dry, safe ground but still
# fords or braves danger when that's clearly the way to its goal.
MOVE_PROGRESS_W    = 3.0      # how strongly a step's headway toward the goal outweighs its cost
WADE_COST          = 2.0      # extra cost of stepping into a river/shallow (on top of half-speed wading)
LEAF_BRUSH_COST    = 3.5      # leaf panels are passable but "collide": a soul routes round them on dry
                              # ground when it can, brushing through only when that's the easy way — never
                              # BLOCKED (hard-solid leaf forced long detours into water → fatal disease)
PERSON_AVOID_COST  = 4.0      # souls don't walk INTO each other: a tile another soul stands on costs this
                              # much more, so they route round one another — but it's soft (never a hard
                              # block), so a crowd can't deadlock and no one is ever trapped by a neighbour
# Wandering with no errand: a soul picks a reachable ROAM waypoint a little way off and walks to it
# (pathfinding around walls), then picks a fresh one — so it actually crosses the ground instead of
# pacing back and forth against an obstacle the way a blind heading does.
ROAM_MIN = 5                  # nearest a roam waypoint is chosen
ROAM_MAX = 13                 # farthest a roam waypoint is chosen
ROAM_TIME = 150               # game-min before giving up on a waypoint (e.g. unreachable) and re-picking
# Perf: per-tick perception masks are precomputed over the box covering all people (big win for a
# CLUSTERED town); if the band is spread wider than this on either axis, fall back to per-soul windows.
PMASK_MAX_SPAN = 200
# Desire-line PATHS — feet wear trails into the ground along the routes the band actually uses (home↔
# water↔resources), so a cluster of huts reads as a real village. Purely COSMETIC (no movement effect).
FOOTFALL_CAP       = 60.0     # most a single tile's wear can build to
FOOTFALL_DECAY     = 0.78     # daily fade, so abandoned routes grass back over
FOOTFALL_PRUNE     = 1.5      # drop a tile from the map once its wear fades below this
FOOTFALL_PATH_MIN  = 6.0      # distinct-walker passes at which a tile reads as a shared worn path (sparser
                              # now that wear needs DIFFERENT souls, not one soul's repeated steps)
FOOTFALL_SEND_CAP  = 700      # most worn tiles streamed to the renderer (the busiest win)
PATH_PULL          = 0.4      # how much easier a worn path is to walk — souls FOLLOW the beaten track
                              # (a gentle cost cut: headway still dominates, so they only road-follow
                              # when it doesn't cost real ground — can't strand or force a detour)
# ROADS — a desire-path trodden hard enough HARDENS into a persistent road (the slime-mold result:
# near-optimal routes emerge from reinforce-on-use). Roads outlast the daily footfall fade, pull
# harder than a soft path, and grass over only if abandoned for good — the seed of a road NETWORK.
ROAD_HARDEN        = 24.0     # shared-wear at which a tile hardens into a road (a genuinely busy artery)
ROAD_PULL          = 0.65     # a road is easier going than a soft path (PATH_PULL) — souls follow it
ROAD_DECAY         = 0.08     # condition a road loses per day when no longer trodden
ROAD_PRUNE         = 0.15     # condition below which an abandoned road grasses back over
# SETTLEMENTS (M0) — the band's home becomes a first-class TOWN with a name, a centre and a roll of
# members; the foundation everything civic (zoning, a treasury, a planning authority, daughter
# colonies) will later hang on. Place-names are stitched from these.
SETTLEMENT_PREFIXES = ("Ash", "Stone", "River", "Green", "Oak", "Fair", "Hearth", "Elm",
                       "Bright", "Mill", "North", "Long", "Wind", "Fern")
SETTLEMENT_SUFFIXES = ("ford", "stead", "hollow", "haven", "wick", "bourne", "field", "vale",
                       "watch", "mere", "ridge", "barrow")
DANGER_AVOID_R     = 4        # tiles from a wolf at which danger starts to bend a path away
DANGER_AVOID_COST  = 1.6      # danger cost per tile closer than DANGER_AVOID_R to a wolf
# Site PLANNING: a soul weighs candidate spots and picks the best rather than the first that fits —
# near enough to water to drink, clear ground, clustered with the band but with room to breathe.
SITE_WATER_IDEAL   = 14       # water within this many tiles is a comfortable walk to drink (rewarded)
HOME_MIN_SPACING   = 3        # crowding an existing building closer than this is penalised (room for paths)
SITE_CANDIDATES_CAP = 60      # how many fitting spots to weigh before committing (bounds the cost)

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
SOLID_BLOCKS = {BLOCK_WALL}  # tiles people can't walk through. A leaf lean-to is flimsy and
# open-fronted, so it is NOT solid: people brush through its panels. (When leaf panels blocked
# movement, non-overlapping shelters packed along the narrow bank into a wall that boxed folk
# away from the water they'd settled beside — and they died of thirst at home.)

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
    # A communal Gathering Hall — not a home but a MONUMENT the band shares: a big, costly,
    # prestigious build an ambitious soul raises for everyone once their own roof is fine. It
    # is the first "status project" (Phase 2), and finishing it earns its builder lasting renown.
    "gathering": dict(name="Gathering Hall", roof=True, insulation=1.0, communal=True, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
    # A communal WORKSHOP — a specialized building (not a home): the band's shared bench. Crafting
    # near it goes faster, so a toolmaker who raises one makes gear & tools quicker for everyone.
    "workshop": dict(name="Workshop", roof=True, insulation=1.0, communal=True, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
    # A communal STOREHOUSE — a specialized building: while one stands, the band's STORED food
    # keeps better (slower spoilage) and is safer from vermin. A forager's answer to the hall.
    "storehouse": dict(name="Storehouse", roof=True, insulation=1.0, communal=True, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
    # A communal SMITHY — the metalworking shop (furnace + forge + anvil): standing beside it
    # unlocks smelting ore into ingots and forging metal tools/weapons — the deep crafting tree.
    "smithy": dict(name="Smithy", roof=True, insulation=1.0, communal=True, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
    # ── Community public works — the band's growing repertoire of buildings-with-a-PURPOSE,
    #    raised when a NEED arises (not pre-scripted). Each does something real once it stands. ──
    # A WELL: a stone shaft (its centre becomes a drinkable water source on completion) so the band
    # can settle away from the river and still drink — raised when home sits far from water.
    "well": dict(name="Well", roof=False, insulation=1.0, communal=True, layout=[
        "WFW",
        "FCF",
        "WFW",
    ]),
    # An INN: communal lodging — its roof shelters the unhoused & wanderers nearby, so a soul
    # without a home of its own isn't left to the cold. Raised when folk go unhoused.
    "inn": dict(name="Inn", roof=True, insulation=1.0, communal=True, layout=[
        "WWDWW",
        "WFFFW",
        "WFFFW",
        "WFFFW",
        "WWWWW",
    ]),
    # A WATCHTOWER: a high lookout that keeps wolves off the band within sight of it — raised
    # after the wolves have drawn blood.
    "watchtower": dict(name="Watchtower", roof=True, insulation=1.0, communal=True, layout=[
        "WWW",
        "WFW",
        "WDW",
    ]),
    # A MARKETPLACE — an open trading square (corner posts, an open floor) the band raises once it
    # trades in COIN: not a home or a workshop but the heart of its new economy, a civic sign that
    # a wooden village has become a trading people.
    "market": dict(name="Marketplace", roof=False, insulation=1.0, communal=True, layout=[
        "WFW",
        "FFF",
        "WFW",
    ]),
}

# The dwelling ladder — ascending comfort. Once a soul has any roof and is kitted out,
# their standing "life project" is to climb this: a draughty leaf lean-to → a snug timber
# hut → a roomy cabin. Each rung is built tile-by-tile from forageable wood & thatch (no
# station needed), so it's pure, visible material progress with the machinery that already
# works. (The deeper crafting tree — stations, metal, furniture — is a later rung that
# needs station-proximity; this is the foundation those layers build on.)
DWELLING_LADDER = ["leaf_shelter", "hut", "cabin"]
MONUMENT_BP = "gathering"          # the communal status build raised once a soul tops the ladder
WORKSHOP_BP = "workshop"           # a communal specialized building that speeds nearby crafting
WORKSHOP_RANGE = 6                 # tiles from a finished workshop within which crafting is faster
WORKSHOP_CRAFT_SPEED = 1.7        # craft-progress multiplier when working by the bench
WOOD_BUILD_RANGE = 22              # trees within this of home → a soul builds in TIMBER; else a leaf house
# Community public works — functional communal buildings the band raises when a NEED arises.
WELL_NEED_DIST    = 12            # settlement centroid farther than this from water → the band wants a well
INN_NEED_UNHOUSED = 2            # this many adults without a home → the band wants an inn
INN_RADIUS        = 5            # tiles from a finished inn within which an unhoused soul is sheltered
INN_SHELTER       = 0.55         # how well an inn shelters those nearby (vs an own roof's full insul)
WATCH_RADIUS      = 9            # tiles from a finished watchtower within which wolves are kept off
SMITHY_BP = "smithy"               # the communal metalworking shop (furnace+forge+anvil)
# A communal craft-building puts its STATIONS within reach of anyone working beside it — this is
# what opens the deep crafting tree to the band: a workshop is the bench/kiln/loom/tannery, a
# smithy is the furnace/forge/anvil, and a home hearth is the campfire.
CRAFT_BUILDING_STATIONS = {
    "workshop":  ("workbench", "kiln", "loom", "tannery"),
    "smithy":    ("furnace", "forge", "anvil"),
}
CRAFT_STATION_RANGE = 6            # tiles from a craft-building within which its stations are usable
# Electricity (modern era) — a generator/reactor powers tiles within POWER_RADIUS; a power pole
# within POWER_LINK of the fed grid relays it onward (a line of poles carries it across a village).
POWER_SOURCES = ("generator", "reactor")
POWER_RADIUS = 7                   # tiles a fed power node energizes around itself
POWER_LINK = 9                     # a pole within this of the fed grid joins it and relays onward
POWER_SHELTER_BONUS = 0.35         # how much an electrified (wired) home adds to its shelter
POWER_CRAFT_SPEED = 1.5            # electric tools: crafting in a powered area runs this much faster
# Reactor MELTDOWN (the modern era's stakes) — a reactor needs water within REACTOR_COOLING_RANGE to
# stay cooled. Sited far from any, heat builds until it melts down in fire: it's destroyed, the ground
# is scorched, and the band scatters in terror (the warned-of consequence made real). Only god-spawned
# reactors exist, so this is entirely inert until a god places one carelessly — never touches the band.
REACTOR_HEAT_RATE     = 0.02       # heat a dry (uncooled) reactor gains per game-minute
REACTOR_COOL_RATE     = 0.06       # heat a water-cooled reactor sheds per game-minute (cools faster than heats)
REACTOR_MELTDOWN_HEAT = 100.0      # heat at which an uncooled reactor melts down (~3.5 game-days dry)
MELTDOWN_SCORCH_R     = 3          # tiles of ground scorched black around the ruin
MELTDOWN_TERROR_R     = 9          # tiles within which souls are thrown into terror and flee
MELTDOWN_SINGE_R      = 2          # tiles within which a soul is singed (a capped fright, never slain)
MELTDOWN_SINGE        = 0.3        # hp a singed soul loses, floored so the blast can't kill outright
# Awe & wonder — a structure FAR beyond the band's craft (a generator/reactor) is perceived as the
# sublime: the curious approach to STUDY it, the cautious recoil. Study slowly yields INSIGHT, and
# enough insight lets a soul begin to puzzle out the first secret of the strangers' machines.
WONDER_KINDS = ("generator", "reactor")
WONDER_VISION = 18                 # tiles within which a wondrous structure is perceived
WONDER_INSIGHT_TO_LEARN = 24.0     # study-beats of insight before the first electricity craft is grasped
WONDER_RECIPE = "copper_coil"      # the first secret reverse-engineered from beholding the machine
WONDER_COOLING_INSIGHT = 10.0      # study-beats before a curious soul reasons WHY a reactor needs water
REACTOR_COOLING_RANGE = 6          # tiles a reactor wants water within to cool — beyond it, a soul is alarmed
STOREHOUSE_BP = "storehouse"       # a communal specialized building that protects the band's stored food
STORE_SPOIL_FACTOR = 0.5          # a storehouse halves how fast STORED food spoils
STORE_PEST_FACTOR = 0.4           # and cuts the chance of a vermin raid

# A building's FUNCTION — the real mechanical ROLE it fills. Every functional effect keys off this,
# not a fixed blueprint id, so the band's workshop/storehouse/hall works whether it's the built-in
# form OR one the band designed itself (Phase A.2: the LLM authors the FORM, the function is real).
BUILTIN_FUNCTION = {WORKSHOP_BP: "workshop", SMITHY_BP: "smithy",
                    STOREHOUSE_BP: "storehouse", MONUMENT_BP: "hall"}
AUTHORABLE_FUNCTIONS = ("home", "workshop", "smithy", "storehouse", "hall")  # what an LLM may design

# ─── Renown — social standing (Phase 2) ──────────────────────────────────────
# A soul's STANDING in the band: it grows from socially-visible achievements (raising a fine
# home, a communal monument, teaching a craft, giving freely) and fades slowly if not renewed,
# so status must be earned and maintained. Ambition turns standing into a pursued goal — the
# monument project — so settled, driven souls compete to leave a mark, not just nest.
RENOWN_GAIN = dict(dwelling=0.07, monument=0.55, teach=0.10, gift=0.05, help=0.06, contribute=0.05)
HELP_RANGE = 30               # how near a band-mate's unfinished build a housed soul will go to help
COMMISSION_FEE = 4            # food a soul pays the builder who raised its home — the seed of a labor economy
COMMISSION_COIN = 2          # …or, once money exists, this many coins instead
# A CONTRACTOR'S commission: a builder who raises a WHOLE home for another (not just a lent hand) is
# paid more — the deliberate "build a home for others for something in return". Bigger than the helper
# fee; coin once money exists, else food.
CONTRACTOR_FEE  = 12         # food a client pays the builder who raised their whole home
CONTRACTOR_COIN = 6          # …or, once money exists, this many coins instead
COMMISSION_RANGE = 16        # how near an unhoused soul must be for a settled builder to offer to build for them
MONEY_TRADE_XP = 6           # trades under a soul's belt before it could conceive of MONEY
COIN_MINT = 20               # coins the inventor of money strikes into being
# Self-authored projects ("aspirations") — a settled, content soul forms its OWN goal beyond mere
# survival (tidy the ground round its home, plant a garden by the door) and a PLAN of primitive
# skills carries it out. This is the open-vocabulary layer: the rule body seeds a few projects so
# it works offline; the LLM mind can later fill the same plan structure with richer goals.
ASPIRE_COOLDOWN = 2880.0      # game-min before a soul takes on another beautify-project (~2 days)
ASPIRE_RING = 2              # how far around home a tidy/garden project reaches
ASPIRE_MAX_STEPS = 10        # cap a plan's length so a project is bounded work
ASPIRE_KINDS = ("tidy", "garden", "art", "furnish")   # the executable project vocabulary the LLM can author into
DEFAULT_ASPIRE_GOAL = {"tidy": "tend the ground around my home",
                       "garden": "plant a garden by my door",
                       "art": "raise a grand work the whole band will marvel at",
                       "furnish": "make beds for my family"}
BED_REST_BONUS = 0.4         # how much resting in a proper BED speeds recovery (on top of the roof)
# Furniture is MADE from materials, not conjured: a soul must hold the makings to set a piece down
# (a bed is a pallet of leaves over fibre; tables/chairs/chests are timber). Flowers/cairns/art stay
# free (they're labour over found things). A soul short the makings skips that piece rather than
# magicking it into being.
FURNITURE_COST = {"bed": {"leaves": 4, "fiber": 2}, "table": {"wood": 3},
                  "chair": {"wood": 2}, "chest": {"wood": 3}}
# Governance — the band's FIRST NORM: pull your weight at the common granary. Judged once a day on
# a rolling window. Soft enforcement through reputation only (standing + how others regard you),
# never punishment — it nudges conduct without ever endangering a life.
GOV_MIN_BAND       = 3        # no commons-norm until the band is at least this many adults
GOV_CONTRIB        = 4.0      # granary units given (this window) to count as a steady contributor
GOV_FREERIDE       = 4.0      # granary units taken (this window) past which a non-giver is judged
GOV_FREERIDE_COST  = 0.05     # renown a chronic free-rider sheds per day
GOV_DISAPPROVAL    = 0.03     # how much each bandmate's regard for a free-rider dips per day
GOV_LEDGER_DECAY   = 0.6      # daily decay of the give/take ledger — recent conduct is what counts

# Phase B — INSTITUTIONS: a renown-recognised LEADER, facing a recurring problem, enacts a LAW the
# band then lives by. The LLM (the leader's judgement) picks WHICH wrong to name and frames it; the
# engine ENFORCES it deterministically — soft, through reputation only (a dip in standing & regard),
# never punishment that could endanger a life. Each law codifies one norm from an ENFORCEABLE
# vocabulary (the immune system: a law the engine can't actually judge is rejected). Inert with no
# model (only the default granary norm runs), so it can't move survival.
LAW_NORMS = ("hoarding", "labour", "peace")   # what an LLM may codify, beyond the always-on granary norm
LAW_MIN_RENOWN  = 2.5        # standing a soul needs to be heeded as the band's law-giver
LAW_PROBLEM_FRAC = 0.34      # a wrong is "recurring" once this share of able adults are in it
LAW_MAX = 3                  # most enacted laws the band carries (beyond the granary norm)
LAW_COOLDOWN = 4 * 24 * 60   # game-min between new laws (~4 days — lawmaking is rare and weighty)
LAW_SANCTION = 0.05          # renown an able violator sheds per day a law judges them
LAW_HOARD_CAP = 14           # personal food store above which (while the commons runs low) one hoards
LAW_GRANARY_LOW = 6          # commons food below which hoarding is a real wrong, not mere thrift
LAW_LABOUR_MIN = 1.0         # rolling "aided a build" credit below which an able soul shirks the work
LAW_PEACE_FOES = 3           # bandmates a soul holds in real enmity (sentiment < -0.3) to be a discord-sower
LAW_REPROACH = {"hoarding": "hoards food while the commons runs low",
                "labour": "never lends a hand when the band builds",
                "peace": "sows discord among us"}

# Phase C — CULTURE: a much-loved soul begins a TRADITION the band keeps each year. The LLM invents
# the festival (its name, its season, what it celebrates); the engine OBSERVES it — when that season
# turns, the band's own named feast deepens the seasonal gathering (bonds warm more than a generic
# one). Inert with no model (only the plain seasonal gathering runs), so it can't move survival.
CUSTOM_KINDS = ("feast",)    # what an LLM may found (festivals tied to a season; more kinds later)
CUSTOM_MIN_RENOWN = 1.5      # standing a soul needs for the band to take up the tradition they start
CUSTOM_MIN_POP = 8           # a band must have grown this big to have the leisure for traditions
CUSTOM_MAX = 4               # most traditions the band carries (≈ one per season)
CUSTOM_COOLDOWN = 3 * 24 * 60  # game-min between new traditions (~3 days)
CUSTOM_BOND_WARM = 0.10      # bond warmth at a band's OWN cherished feast (vs CUSTOM_BOND_PLAIN below)
CUSTOM_BOND_PLAIN = 0.06     # bond warmth at a plain turn-of-season gathering
RENOWN_DECAY = 0.012               # fraction of standing shed per game-day (≈ half-life ~8 weeks)
AMBITION_MONUMENT = 0.55           # a soul this ambitious will undertake a monument for the band
# Cooperative big-builds: a communal monument is heavy work — a tile is laid only when at least
# CO_OP_MIN builders are on hand (it takes several to raise a beam), so a lone soul can't grind it
# out alone and the band's ambitious converge into a crew (up to CO_OP_CREW_MAX) to raise it.
CO_OP_MIN = 2                      # builders that must be present at a communal site to lay a tile
CO_OP_CREW_MAX = 4                 # most builders that will join one communal build
CO_OP_RANGE = 6                    # how near the site a crew-mate counts as "on hand"

# Neighbourliness: a soul at loose ends who spots a band-mate raising their OWN home nearby walks
# over and OFFERS a hand; the builder accepts unless they dislike the helper. (Communal/commission
# builds already crew themselves — this is the unbidden good turn between neighbours.)
HELP_OFFER_RANGE = 9               # how near a neighbour's build must be to catch a passer-by's eye
HELP_OFFER_CD = 6 * 60            # game-min a soul waits before offering again (after a decline/a stint)
HELP_OK_SENTIMENT = -0.15         # a builder accepts help unless they regard the offerer below this

# Phase A — the DESIGN RATCHET: an LLM authors a new kind of building for a prospering band. Gated to
# stay RARE (one design buys days of game-life): a settled builder, a band grown past a threshold, a
# cooldown between designs, and a cap on the band's own library. Inert with no model (offline ceiling).
AUTHOR_MIN_POP = 10                # the band must have grown this big to spare imagination for new forms
AUTHOR_MAX = 6                     # cap on the band's self-authored design library
AUTHOR_COOLDOWN = 2 * 24 * 60      # game-minutes between authored buildings (~2 days)
PLY_WOOD_STOCK = 14                # a woodcutter plying their trade stocks timber up to this

# ─── Generations & lineage (Phase 4) ─────────────────────────────────────────
# A band that breeds true: bonded adults have children who inherit a blend of their parents'
# nature (but NOT their knowledge — culture must be taught afresh each generation, so it can
# grow or be lost), grow from dependent childhood to a calling of their own, and inherit home
# and a share of a parent's standing. This is where character becomes lineage and lineage,
# slowly, becomes culture. Ages are in game-days (DAYS_PER_YEAR == 60).
ADULT_AGE = 16 * DAYS_PER_YEAR     # childhood ends ~16 yrs: full capability + a vocation
CHILD_SKILL_GAIN = 0.00006         # skill a youngling earns per beat of practice (grows over a childhood)
ARROWS_PER_WHITTLE = 2             # arrows a child shapes from one length of wood
BREED_MIN_AGE = 18 * DAYS_PER_YEAR
BREED_MAX_AGE = 45 * DAYS_PER_YEAR
BREED_COOLDOWN_DAYS = 14.0         # game-days between a mother's children (births must outpace attrition)
BOND_WARMTH = 0.30                 # mutual sentiment at which two adults pair off
POP_CAP = int(os.getenv("AITHA_POP_CAP", "100"))   # ceiling on band size. Raised 36→100 so the band can
# grow into a TOWN (the civilization arc's whole point). The demographics already support it (a model on
# the real constants grows to the cap in decades, no extinction); the body tick scales linearly (~0.3ms/
# soul → ~30ms at 100, well under the 167ms budget); and the satiety BIRTH-GATE self-limits the band at
# its food carrying-capacity (births halt when a pair isn't well-fed), so a higher ceiling can't cause a
# Malthusian crash — it just lets the band find its natural, food-regulated size.
TRAIT_MUTATION = 0.08              # how far a child's nature drifts from the parental mean
RENOWN_LEGACY = 0.25               # share of a parent's standing that passes to each child

# ─── Illness (survival realism: waterborne disease) ──────────────────────────
# Raw water from a river or lake can carry sickness. A soul falls ill after an incubation,
# then suffers — fluid loss accelerates the drain on their reserves and a fever erodes
# health — and, if they pull through, recovers with a spell of immunity. They KNOW they are
# sick and feel where it hurts, but get only a vague HINT at the cause. The cure for now is
# not catching it (later: boiling water); a well-watered soul who keeps drinking can outlast
# the milder strains, but cholera's dehydration can kill if water isn't close to hand.
#   incub/dur in game-days; `drain` multiplies hydration/appetite loss while symptomatic;
#   `hp` is health eroded per game-minute of symptoms.
DISEASE = {
    "cholera":       dict(incub=1.0, dur=4.0,  drain=1.8, hp=0.00006,
                          hint="my guts run like a river — it was foul water, surely"),
    "dysentery":     dict(incub=2.0, dur=6.0,  drain=1.35, hp=0.00005,
                          hint="my belly gripes and bleeds — something I drank?"),
    "typhoid_fever": dict(incub=6.0, dur=12.0, drain=1.15, hp=0.00007,
                          hint="a slow fever burns in me — was it the water?"),
    # P3 — a fast, sharp food-poisoning from the wrong berries (NOT waterborne; learned by
    # which bush made you ill). Short and rarely fatal on its own, but a body already worn
    # thin by hunger/thirst can be tipped over — the cost of eating an unknown bush.
    "berry_sickness": dict(incub=0.25, dur=1.6, drain=1.5, hp=0.00004,
                           hint="my belly heaves and cramps — it was those berries"),
    # P4 — parasites/spoilage from eating RAW meat or fish. The lesson the band learns is to
    # COOK flesh over the hearth fire (cooked_meat/cooked_fish carry no such risk).
    "tainted_gut": dict(incub=1.5, dur=5.0, drain=1.4, hp=0.00005,
                        hint="a sickness churns in my gut — was the meat uncooked?"),
}
WATERBORNE = ("cholera", "dysentery", "typhoid_fever")   # only these come from raw water
WATER_INFECT_CHANCE = 0.012        # chance a single raw drink from a natural source infects
IMMUNITY_DAYS = 90.0               # how long recovery shields against that same disease
TEND_RECOVERY_BOOST = 0.6          # extra game-min shaved off an illness per game-min it's nursed
HEARTH_COST = {"wood": 3, "stone": 2, "leaves": 1}   # the campfire a soul raises to boil water by

# ─── Berry bushes (P3 gathering overhaul) ────────────────────────────────────
# Scattered bushes bear berries — a richer forage than grazing — but some are POISONOUS,
# and which is which is not written on them: a soul finds out by eating, then remembers
# that bush (lore) and shuns it. A picked bush re-ripens over a few days, so foraging is a
# renewable round rather than a one-off strip. Sparse node list (like ore) so the vast map
# costs nothing. Lore is the mitigation for the poison hazard, exactly as boiling is for water.
BERRY_BUSH_PER = 2500              # ~one bush per this many land tiles — common enough that a
                                   # settled band meets them often (so the poison/lore loop bites)
BERRY_POISON_FRAC = 0.28           # share of bushes that are poisonous
BERRY_REGROW_DAYS = 3.0            # a picked bush re-ripens over this
BERRY_POISON_CHANCE = 0.55         # chance eating an unknown poison bush actually sickens you
BERRY_HUNGER_RELIEF = 0.40         # comfort hunger a handful of berries slakes
BERRY_SATIETY = 0.10               # reserve nourishment a handful restores
BERRY_SEEK_RANGE = 18              # how far a hungry soul will divert to a worthwhile bush (kept tight
                                   # so chasing berries never strands a soul far from water)
BERRY_GOOD_BIAS = 3                # mild pull toward a known-good bush — enough to prefer a trusted one
                                   # nearby, but a distinctly CLOSER unknown bush still gets tried (and
                                   # so the poison gamble, and the lesson, stays a live part of foraging)
BERRY_BIOMES = {"grassland", "forest", "rainforest", "savanna", "swamp"}

# ─── Hunting & fishing (P4) ──────────────────────────────────────────────────
# A forager hunts game (rabbit/deer) and fishes the waters for MEAT/FISH — a far richer
# yield than grazing, but raw flesh must be COOKED over the home hearth or it sickens the gut
# (the same adapt-or-suffer arc as boiling water / learning poison bushes). A crude spear
# makes the kill far surer and the carcass yield bigger; bare hands rarely bring game down.
HUNT_VISION = 7                    # how far a hunter spots and pursues game
HUNT_KILL_SPEAR = 0.34             # kill chance per adjacent strike with a spear
HUNT_KILL_BARE = 0.12              # bare-handed per strike — poor per try, but a dogged pursuit of
                                   # many strikes still brings small game down (a spear makes it sure)
HUNT_MEAT_YIELD = {"rabbit": 2, "deer": 5}      # raw meat a carcass gives (a deer feeds the band)
FISH_CATCH_ROD = 0.05              # per game-min at the water with a rod
FISH_CATCH_BARE = 0.013           # tickling fish by hand — slow going
MEAT_STOCK = 6                     # a forager lays in raw flesh up to this, then cooks/stockpiles
# Hunting parties: big game (a deer) is too much for one — it takes a PARTY of HUNT_PARTY_MIN
# hunters on hand to bring down, and the carcass is shared among them. A lone hunter can only
# take small game (rabbits), so hunters depend on one another for the richest hauls.
HUNT_PARTY_MIN = 2                 # hunters near the quarry needed to down big game
HUNT_PARTY_RANGE = 4               # how near a band-mate counts as part of the hunting party
RAW_FLESH_SICKEN = 0.16            # chance a raw meat/fish meal brings on tainted_gut
COOKED_HUNGER_RELIEF = 0.55        # a cooked meal is the most filling thing there is
COOKED_SATIETY = 0.22
RAW_HUNGER_RELIEF = 0.35           # raw flesh fills you too — but it's a gamble
RAW_SATIETY = 0.12

# ─── Food spoilage, seasonal stockpiling & larder allure (P5) ────────────────
# Fresh food rots. Raw flesh turns in a day or two, cooked meals keep a while, gathered
# produce lasts longer, and only PRESERVED food (dried/smoked/pemmican) truly lasts — so a
# band that wants to outlast winter must dry its surplus at the hearth, not just pile up meat.
# Shelf lives are in game-DAYS; an item absent from this map never spoils (the preserved goods).
PERISHABLE = {"meat": 1.2, "fish": 1.0, "cooked_meat": 3.5, "cooked_fish": 3.0,
              "food": 8.0, "berry": 2.5}
# Stockpiling scales with the season: a band lays in heavily through AUTUMN against the lean
# white months, eases off in spring's plenty. Multiplies the provisioning/larder targets.
SEASON_STOCK_MULT = {"spring": 1.0, "summer": 1.15, "autumn": 2.1, "winter": 1.6}
PRESERVE_AT = 4                    # raw meat/fish at/above this is dried at the hearth for the long haul
# Allure: a fat larder of fresh food draws vermin. Stores above the threshold risk a raid that
# eats a chunk — the cost of hoarding perishables, and a nudge toward preserving instead.
PEST_STORE_THRESHOLD = 16          # stored food above this starts drawing pests
PEST_RAID_CHANCE = 0.16            # base per-day chance once over the threshold (scales with excess)
PEST_RAID_LOSS = 0.4               # fraction of the food store a raid carries off

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
WOOD_IDS = [sp for sp, info in PLANTS.items() if info["name"] in WOOD_PLANTS]
WATER_BUILD_BUFFER = 2      # tiles a soul keeps between a new building and water (flood-shy, but
                            # close enough that the walk to drink doesn't kill them — 3 was too far)
LEAF_GATHER = 3                          # leaves gained per pull (vs. 1 for fiber)
LEAF_CAP = 24                            # a person can carry a big bundle of leaves
ORE_KINDS = ("copper_ore", "tin_ore", "iron_ore", "gold_ore", "coal")
ORE_WEIGHTS = (0.30, 0.18, 0.30, 0.07, 0.15)
ORE_NODES_PER = 1_000_000               # ~1 deposit per this many tiles
STONE_PER_ROCK = 120                    # ~1 visible boulder per this many rock/mountain tiles


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
        self._last_spoil = 0.0           # game-minute of last food-spoilage pass (P5)
        self._last_pest = 0.0            # game-minute of last larder-pest check (P5)
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
        self.stone_nodes: list[dict] = []  # visible boulders on the highland rock (where stone is mined)
        self.berry_bushes: list[dict] = []  # scattered berry bushes (some poisonous), P3
        self._berry_index: dict[tuple[int, int], dict] = {}  # (x,y)->bush, rebuilt on load/seed
        self.granary: dict = {"store": {}, "x": None, "y": None}  # the band's shared common store
        self.decor: dict = {}            # (x,y) -> kind: flowers/art/furniture a soul sets in/around its home
        self.station_objs: dict = {}     # (x,y) -> kind: a crafted workbench/furnace/kiln, PLACED & visible
        self.footfall: dict = {}         # (x,y) -> wear: where feet fall most, worn into visible PATHS (cosmetic)
        self._foot_last: dict = {}       # (x,y) -> last walker id; a path wears only when DIFFERENT souls tread
        self.roads: dict = {}            # (x,y) -> condition: paths trodden hard enough HARDEN into roads
        self.settlements: list[dict] = []  # first-class SETTLEMENTS (the band's town(s)) — M0 foundation
        self.laws: list[dict] = []         # the band's enacted LAWS (LLM-authored, validated) — Phase B
        self.customs: list[dict] = []      # the band's TRADITIONS (LLM-authored festivals) — Phase C
        self.money_invented = False      # has a soul WORKED OUT money yet? (then coins circulate)
        self.money_inventor = None       # who gave the band the idea of money
        self.user_blueprints: list[dict] = []   # the god's hand-authored building templates (library)
        self.authored_blueprints: list[dict] = []  # the BAND's own designs (LLM-authored, validated) — Phase A
        self._load_templates()                    # pull the library off disk + register it for placement
        self.log: list[dict] = []        # recent god actions / notable events
        # Craft knowledge. Everyone is born knowing the basics (STARTER_RECIPES); the make-
        # shift survival crafts (water flask, etc.) must be DISCOVERED by an individual and
        # then SPREAD soul to soul by teaching. `known_recipes` is the band-wide union (for
        # the catalog/UI), but who-knows-what is now personal (p["recipes"]). Every first
        # making and every failed experiment is written into the Ledger of Making.
        self.known_recipes: set[str] = set(crafting.STARTER_RECIPES)
        self.ledger: list[dict] = []     # the Ledger of Making — discoveries + dead ends
        self.speed = 1.0                 # fast-forward multiplier (1×/2×/4×), set from the UI
        self.day_speed = 1.0             # extra fast-forward applied only during DAY hours (≥1)
        self.night_speed = 1.0           # extra fast-forward applied only during NIGHT hours (≥1)
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
        self._add_coastal_shelf()
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
        self._seed_stone_nodes()
        self._seed_berry_bushes()
        self._seed_initial_people(count=7, center=self._origin)

        self.clock = 8 * 60.0            # start at 08:00 on day 0
        self._last_eco = self._last_spoil = self._last_pest = self.clock
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

    def _seed_stone_nodes(self):
        """Scatter visible boulders across the highland rock — a clear on-map cue for where the
        band mines stone (rendered like ore). Mining stays biome-based; these just make the
        resource legible instead of an invisible property of grey terrain."""
        self.stone_nodes = []
        rock = np.argwhere(np.isin(self.biome, [B["mountain"], B["rock"]])
                           & (self.water == WATER_NONE))
        if not len(rock):
            return
        n = max(20, len(rock) // STONE_PER_ROCK)
        picks = self.rng.choice(len(rock), size=min(n, len(rock)), replace=False)
        for idx in picks:
            ry, rx = rock[idx]
            self.stone_nodes.append({"x": int(rx), "y": int(ry)})

    def _seed_berry_bushes(self):
        """Scatter berry bushes across the temperate/forest land, a fraction of them poisonous.
        Like ore, a small sparse list so the map stays cheap; the band discovers bushes (and
        which ones bite back) by ranging out and eating. Bushes start ripe (ripe_t=0)."""
        self.berry_bushes = []
        biome_ids = [B[name] for name in BERRY_BIOMES if name in B]
        land = np.argwhere(np.isin(self.biome, biome_ids) & (self.water == WATER_NONE))
        if not len(land):
            self._rebuild_berry_index()
            return
        n = max(12, (W * H) // BERRY_BUSH_PER)
        picks = self.rng.choice(len(land), size=min(n, len(land)), replace=False)
        for idx in picks:
            by, bx = land[idx]
            poison = bool(self.rng.random() < BERRY_POISON_FRAC)
            self.berry_bushes.append({"x": int(bx), "y": int(by), "poison": poison, "ripe_t": 0.0})
        self._rebuild_berry_index()

    def _rebuild_berry_index(self):
        self._berry_index = {(b["x"], b["y"]): b for b in self.berry_bushes}

    def _bush_ripe(self, b) -> bool:
        return self.clock >= b.get("ripe_t", 0.0)

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
        thresh = np.quantile(land_acc, 0.972) if land_acc.size else 1e9
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
        # River-mouth deltas (high sediment) get a much bigger budget so they read as
        # broad sandy beaches; steep/exposed shores stay thin.
        wf = np.clip(1.6 + sediment * 26.0 - slope * 22.0 - exposure * 2.2, 0.0, 12.0)
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
        for _ in range(12):                     # grow the beach inland, width ∝ budget
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

    def _add_coastal_shelf(self):
        """Mark a band of shallow water hugging the coast (rendered a lighter blue), which
        BULGES out where rivers meet the sea — the sediment a river dumps builds a shallow
        delta/fan offshore. Purely a water-layer distinction; the tiles are still ocean."""
        ocean = self.water == WATER_OCEAN
        if not ocean.any():
            return

        def grow(seed_mask, rings):
            m = seed_mask.copy()
            for _ in range(rings):
                acc = m.copy()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    acc |= np.roll(np.roll(m, dy, 0), dx, 1)
                m = acc
            return m

        land = self.water == WATER_NONE
        shelf = grow(land, 3) & ocean                       # a thin shelf all around the coast
        # Bulge: rivers carry sediment offshore, so the shelf reaches much farther out
        # opposite a river mouth (a delta fan).
        mouths = grow(self.water == WATER_RIVER, 1) & ocean
        if mouths.any():
            shelf |= grow(mouths, 12) & ocean
        self.water[shelf] = WATER_SHALLOW

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
        # Day/night compression: the clock can run faster during the day and/or night independently
        # (set from World Settings), so a god can keep days as they are but make nights flash by (or
        # vice-versa). The hours are still FULLY simulated — souls sleep, fatigue recovers, ecology
        # ticks by game-time — they just take less REAL time.
        hod = self.time_of_day()
        warp = getattr(self, "night_speed", 1.0) if (hod < 6 or hod >= 21) else getattr(self, "day_speed", 1.0)
        if warp > 1.0:
            dt_game_min *= warp
        self.clock += dt_game_min
        self._update_weather()
        self._tick_wildlife(dt_game_min)
        self._tick_people(dt_game_min)
        self._tick_reactors(dt_game_min)                           # a carelessly-sited reactor heats toward meltdown
        if self.clock - self._last_eco >= 60.0:                    # one game-hour elapsed
            self._tick_ecology_active()
            self._last_eco = self.clock
        if self.clock - self._last_spoil >= 60.0:                  # spoil perishables hourly (P5)
            self._tick_spoilage(min(1.0, (self.clock - self._last_spoil) / 1440.0))
            self._last_spoil = self.clock
        if self.clock - self._last_pest >= 1440.0:                 # vermin check once a game-day (P5)
            self._tick_pests()
            self._tick_governance()                                # judge the commons-norm once a day
            self._decay_footfall()                                 # worn paths fade where feet stop falling
            self._tick_settlements()                               # keep the band's town (name/centre/roll) current
            self._last_pest = self.clock
        season_now = self.season()
        if season_now != getattr(self, "_last_season", season_now):
            self._festival(season_now)                             # the band marks the turn of the year
        self._last_season = season_now
        self.version += 1

    def _festival(self, season_now):
        """A turn-of-the-season gathering: the band comes together, every bond warms a little, and
        the most-esteemed soul calls the festival — a shared rite that knits the group (#18). If the
        band has founded its OWN tradition for this season (Phase C), it is kept BY NAME and bonds
        deeper — a cherished feast means more than a generic gathering."""
        if len(self.people) < 2:
            return
        custom = next((c for c in self.customs if c.get("kind") == "feast"
                       and c.get("season") == season_now), None)
        warm = CUSTOM_BOND_WARM if custom else CUSTOM_BOND_PLAIN
        for i, a in enumerate(self.people):
            for b in self.people[i + 1:]:
                ra, rb = mind._rel(a, b, self.clock), mind._rel(b, a, self.clock)
                mind._adjust(ra, 0.02, warm); mind._adjust(rb, 0.02, warm)
            a["last_social_t"] = self.clock
        elder = max(self.people, key=lambda q: q.get("renown", 0.0))
        if custom:
            mind.speak(elder, f"Gather, all — it's {custom['name']}!", self.clock)
            self._note("culture", f"The band kept {custom['name']}"
                                  + (f" — {custom['value']}" if custom.get("value") else "") + ".")
        else:
            mind.speak(elder, f"Gather, all — we've come through to {season_now}!", self.clock)
            self._note("social", f"The band gathered to mark the turn to {season_now}.")

    def wants_new_custom(self, p) -> bool:
        """Rule trigger for the LLM culture-author: a much-loved, settled soul in a grown band with
        room for another tradition and lawmaking/custom rested (cooldown). Rare — a tradition is a
        lasting thing the whole band takes up."""
        if len(self.customs) >= CUSTOM_MAX or self.clock < getattr(self, "_custom_cd", 0.0):
            return False
        if p.get("home_struct") is None or p.get("renown", 0.0) < CUSTOM_MIN_RENOWN:
            return False
        return len(self.people) >= CUSTOM_MIN_POP

    def apply_authored_custom(self, data, by="the band") -> str | None:
        """Author→Validate→World for CULTURE (mirrors apply_authored_law): a soul founds a yearly
        FEAST. It is taken up only if its kind is one the band can actually keep (CUSTOM_KINDS), it's
        pinned to a real season, and the band hasn't a feast that season already. Returns the new
        tradition's name, or None."""
        if not isinstance(data, dict):
            return None
        kind = str(data.get("kind") or "feast").strip().lower()
        season = str(data.get("season") or "").strip().lower()
        name = (str(data.get("name") or "").strip())[:48]
        if kind not in CUSTOM_KINDS or season not in SEASONS or not name:
            return None
        if any(c.get("kind") == kind and c.get("season") == season for c in self.customs):
            return None
        value = (str(data.get("value") or "").strip())[:90]
        self.customs.append({"kind": kind, "season": season, "name": name, "value": value,
                             "by": by, "born": self.clock})
        self._custom_cd = self.clock + CUSTOM_COOLDOWN
        self.version += 1
        self._bump("custom_born")
        self._note("culture", f"{by} began a tradition — {name}, kept each {season}.")
        return name

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
        eco_min = float(self._chunk_eco[cy0:cy1 + 1, cx0:cx1 + 1].min())
        lag_hours = (self.clock - eco_min) / 60.0
        passes = int(min(max(lag_hours, 1), ECO_CATCHUP_CAP))
        # TIME-BOX the catch-up: a big lag over a large active region (e.g. when a god places
        # people into long-dormant terrain, or the band spreads toward the 768² cap) used to
        # run dozens of full-region passes in ONE tick — multi-second freezes that starved the
        # live loop. Cap the wall-time; advance the dormancy clock only by what we actually
        # simulated, so the rest catches up over the next few ticks instead of one stall.
        t0 = time.perf_counter()
        done = 0
        for _ in range(passes):
            self._tick_ecology(reg)
            done += 1
            if time.perf_counter() - t0 > ECO_TIME_BUDGET:
                break
        self._chunk_eco[cy0:cy1 + 1, cx0:cx1 + 1] = min(self.clock, eco_min + done * 60.0)

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

    def _wolf_threat_at(self, p, wolf_pos) -> float:
        """RAW wolf proximity to this soul (0..1), before any protection — 'is a wolf bearing
        down on me'. Zero under a roof. Drives the guardian's resolve (a ward to protect)."""
        if not wolf_pos or self._shelter_factor(p) > 0.5:
            return 0.0
        px, py = p["x"], p["y"]
        near = WOLF_MENACE_VISION + 1
        for (wx, wy) in wolf_pos:
            d = abs(wx - px) + abs(wy - py)
            if d < near:
                near = d
        if near > WOLF_MENACE_VISION:
            return 0.0
        return 1.0 - near / (WOLF_MENACE_VISION + 1)

    def _guardian_near(self, x, y, exclude=None) -> bool:
        """Is a living soul standing GUARD within range of (x,y)? Their watch lets others forage
        unafraid and warns wolves off — the protector's role the band comes to depend on."""
        for q in self.people:
            if q is exclude:
                continue
            if (q.get("intention") or {}).get("kind") == "guard" \
                    and abs(q["x"] - x) + abs(q["y"] - y) <= GUARD_RANGE:
                return True
        return False

    def _danger_at(self, p, wolf_pos) -> float:
        """OPERATIONAL danger that drives flight/forage-interruption: the raw wolf threat, eased
        by each companion within band range and ZEROED when a guardian stands watch nearby — so a
        protected forager keeps working instead of bolting. (Self-guardians don't shield self.)"""
        raw = self._wolf_threat_at(p, wolf_pos)
        if raw <= 0:
            return 0.0
        if self._guardian_near(p["x"], p["y"], exclude=p):
            return 0.0
        px, py = p["x"], p["y"]
        guards = sum(1 for q in self.people if q is not p
                     and abs(q["x"] - px) + abs(q["y"] - py) <= WOLF_BAND_SAFETY)
        return raw / (1 + guards)

    def _exposure_threat(self, night: bool) -> float:
        """How punishing it is to be OUTDOORS right now (0..1), from night cold + weather +
        season. A clear summer day is harmless; a winter night storm is brutal."""
        t = 0.0
        if night:
            t += 0.45
        w = self.weather
        t += {"rain": 0.30, "storm": 0.65, "snow": 0.70, "cloudy": 0.05}.get(w, 0.0)
        if self.season() == "winter":
            t += 0.25
        if w != "clear":                                 # foul weather bites harder the heavier it is
            t *= 0.7 + 0.3 * float(getattr(self, "weather_intensity", 1.0))
        return min(1.0, t)

    def _ensure_power(self):
        """(Re)build the set of ENERGIZED power nodes — generators/reactors and every pole the
        current keeps fed — whenever the structures change. A generator is a source; a power_pole
        within POWER_LINK of any energized node relays the current onward, so a line of poles
        carries power across the village. Cheap: there are only ever a handful of these."""
        if getattr(self, "_power_v", None) == self.version and hasattr(self, "_powered_nodes"):
            return
        sources = [(s["x"], s["y"]) for s in self.structures if s.get("kind") in POWER_SOURCES]
        poles = [(s["x"], s["y"]) for s in self.structures if s.get("kind") == "power_pole"]
        energized = set(sources)
        changed = True
        while changed:
            changed = False
            for pp in poles:
                if pp in energized:
                    continue
                if any(abs(pp[0] - e[0]) + abs(pp[1] - e[1]) <= POWER_LINK for e in energized):
                    energized.add(pp); changed = True
        self._powered_nodes = energized
        self._power_v = self.version

    def _powered(self, x, y) -> bool:
        """Is tile (x,y) within reach of the electrical grid (a generator or a fed pole)?"""
        self._ensure_power()
        return any(abs(x - e[0]) + abs(y - e[1]) <= POWER_RADIUS for e in self._powered_nodes)

    def _tick_reactors(self, dt_game_min: float):
        """A god-spawned REACTOR runs hot. Within reach of water it stays cooled; sited far from any,
        it has no cooling and HEAT builds until it MELTS DOWN. Entirely inert unless a god has placed
        a reactor (none exist in a natural band), so it never touches survival — it's the stakes the
        modern era brings when a machine is sited carelessly."""
        structs = getattr(self, "structures", None)
        if not structs:
            return
        melted = []
        for s in structs:
            if s.get("kind") != "reactor":
                continue
            if self._water_within(s["x"], s["y"], REACTOR_COOLING_RANGE):
                s["heat"] = max(0.0, s.get("heat", 0.0) - dt_game_min * REACTOR_COOL_RATE)
            else:
                s["heat"] = s.get("heat", 0.0) + dt_game_min * REACTOR_HEAT_RATE
                if s["heat"] >= REACTOR_MELTDOWN_HEAT:
                    melted.append(s)
        for s in melted:
            self._reactor_meltdown(s)

    def _reactor_meltdown(self, s):
        """The uncooled reactor bursts: it is destroyed (and falls off the power grid), the ground
        around it is scorched black, and every soul near it is thrown into terror — singed if very
        close (a capped fright, never slain), and left certain they were RIGHT to fear the thing."""
        rx, ry = s["x"], s["y"]
        self.structures = [q for q in self.structures if q is not s]
        self.version += 1                                    # drops it from the grid (_ensure_power rebuilds)
        self._note("disaster", f"the reactor ran too hot with no water to cool it — it has MELTED DOWN "
                               f"in fire and ruin at ({rx},{ry}).")
        for dy in range(-MELTDOWN_SCORCH_R, MELTDOWN_SCORCH_R + 1):
            for dx in range(-MELTDOWN_SCORCH_R, MELTDOWN_SCORCH_R + 1):
                tx, ty = rx + dx, ry + dy
                if self._in(tx, ty) and abs(dx) + abs(dy) <= MELTDOWN_SCORCH_R:
                    self.decor[(tx, ty)] = "scorch"          # blackened earth where it stood
        for p in self.people:
            d = abs(p["x"] - rx) + abs(p["y"] - ry)
            if d > MELTDOWN_TERROR_R:
                continue
            mind.remember(p, "the strangers' machine burst in fire — I was right to fear it",
                          0.95, "danger", self.clock)
            if d <= MELTDOWN_SINGE_R:
                p["hp"] = max(0.3, p.get("hp", 1.0) - MELTDOWN_SINGE)   # singed, not slain
            mind.speak(p, "The machine — it's burning! Run!", self.clock)
        self._bump("meltdown")

    def _nearest_wonder(self, p):
        """The nearest structure FAR beyond the band's craft (a generator/reactor) within sight,
        or None — the sublime thing a soul marvels at or recoils from. Returns (x, y, dist, kind)."""
        x, y = p["x"], p["y"]
        best, bd = None, WONDER_VISION + 1
        for s in self.structures:
            if s.get("kind") in WONDER_KINDS:
                d = abs(s["x"] - x) + abs(s["y"] - y)
                if d < bd:
                    best, bd = s, d
        return None if best is None else (best["x"], best["y"], bd, best["kind"])

    def _awe_react(self, p):
        """A soul's first sight of the sublime — wonder for the curious, dread for the cautious —
        burned as a vivid memory and spoken once. (Only the first beholding; later sights are quiet.)"""
        if p.get("_awed"):
            return
        p["_awed"] = True
        dread = mind._trait(p, "caution") > mind._trait(p, "curiosity")
        text = ("I have never seen the like — it fills me with dread." if dread
                else "I have never seen the like — what wonder is this?")
        mind.remember(p, text, 0.85, "awe", self.clock)
        mind.speak(p, text, self.clock)
        self._note("awe", f"{p['name']} beheld the strange machine in {'fear' if dread else 'wonder'}.")

    def _study_wonder(self, p, kind="generator", wx=None, wy=None):
        """Standing before the machine, a curious soul STUDIES it — slowly gathering insight until
        the first secret of the strangers' craft comes clear (reverse-engineering the modern tree).
        Along the way they reason about what the thing NEEDS to run — a reactor runs fearsomely hot
        and must be cooled — the seed of grasping a building's requirements & consequences."""
        p["insight"] = p.get("insight", 0.0) + 1.0
        if p["insight"] % 8 < 1:
            mind.remember(p, "I study the strangers' machine, trying to grasp how it is wrought",
                          0.5, "study", self.clock)
        # REASONING about the machine's REQUIREMENT — before grasping how it's built, a curious soul
        # works out that a reactor runs hot and needs water close by to keep from burning. The user's
        # own example ("they can't reason a reactor needs cooling") — now they can, and say so.
        if (kind == "reactor" and p["insight"] >= WONDER_COOLING_INSIGHT
                and not p.get("grasped_cooling")):
            p["grasped_cooling"] = True
            mind.remember(p, "that machine runs fearsomely hot — it must need water close by, or it would burn",
                          0.75, "study", self.clock)
            mind.speak(p, "It runs so hot — it must need water to stay cool, else it would burn itself up.",
                       self.clock)
            self._note("insight", f"{p['name']} reasoned the strange machine needs water to keep cool.")
        # …and from the REQUIREMENT to the CONSEQUENCE: a soul who has grasped the cooling, seeing the
        # reactor sited far from any water, raises the alarm — reasoning that THIS one is unsafe.
        if (kind == "reactor" and p.get("grasped_cooling") and not p.get("warned_reactor")
                and wx is not None and not self._water_within(wx, wy, REACTOR_COOLING_RANGE)):
            p["warned_reactor"] = True
            mind.remember(p, "the strangers' hot machine sits far from any water — it is not safe, it could burn",
                          0.8, "danger", self.clock)
            mind.speak(p, "This burning machine sits too far from water — it isn't safe here!", self.clock)
            self._note("alarm", f"{p['name']} warns the hot machine is sited dangerously far from water.")
        if p["insight"] >= WONDER_INSIGHT_TO_LEARN and not self._person_knows(p, WONDER_RECIPE):
            self._grant_recipe(p, WONDER_RECIPE, via="puzzled out from the strangers' machine",
                               rationale="studied the god's wondrous device until its first secret came clear")
            self._note("discovery", f"{p['name']} began to grasp how the strange machines are made.")

    def _marvel(self, p):
        """React to the sublime: the curious approach to STUDY it (and slowly learn), the cautious
        back away toward home. Either way they're awed. Returns a body action."""
        w = self._nearest_wonder(p)
        if w is None:
            return self._idle(p)
        wx, wy, dist, _kind = w
        self._awe_react(p)
        x, y = p["x"], p["y"]
        if mind._trait(p, "caution") > mind._trait(p, "curiosity") and dist < 7:
            hx, hy = p["home"]                                # recoil — keep clear of the thing
            return "wander", (hx - x, hy - y)
        if dist > 2:                                          # curious — go closer to see
            return "wander", (wx - x, wy - y)            # raw delta → pathfind round walls
        self._study_wonder(p, _kind, wx, wy)                 # at its foot — study it (and reason about it)
        return "rest", None

    def place_power(self, kind: str, x: int, y: int, by: str = "the god") -> str:
        """God tool: drop a generator, reactor or power pole into the world — the modern grid the
        wooden village can marvel at (and, in time, learn to build)."""
        if kind not in POWER_SOURCES and kind != "power_pole":
            return ""
        self._add_structure(kind, int(x), int(y), by=by)
        return f"{by} raised a {kind.replace('_', ' ')} at ({int(x)},{int(y)})"

    def _shelter_factor(self, p) -> float:
        """How well a soul is shielded from the open right now (0..1). Only a roof over one's
        OWN home tile shields — its strength is the home's insulation. A portable mat does
        not (it keeps no weather off). A home wired to the POWER grid shields better still — light
        and warmth against the night — so an electrified house is a real comfort and safety gain."""
        if p.get("home_struct") and (p["x"], p["y"]) == tuple(p["home"]):
            base = max(0.0, min(1.0, p.get("insul", 1.0)))
            if self._powered(p["home"][0], p["home"][1]):
                base = min(1.0, base + POWER_SHELTER_BONUS)
            return base
        # No roof of one's own — but a communal INN shelters the unhoused & wanderers who keep near it.
        if self._near_building("inn", INN_RADIUS, p["x"], p["y"]):
            return INN_SHELTER
        return 0.0

    def _wolves_menace_people(self, dt_game_min: float, night: bool) -> list:
        """Hungry wolves stalk lone people caught in the open: close in and, if they reach one
        who is unsheltered and unguarded, bite (real hp damage, occasionally lethal). The band
        deters them and a roof is safety, so danger is a reason to stay near others and to
        finish a home. Returns any people freshly killed (so the tick resolves their death)."""
        if not self.people:
            return []
        wolves = [a for a in self.animals
                  if a["sp"] == "wolf" and self.clock >= a.get("feed_next", 0)]
        if not wolves:
            return []
        killed = []
        bold = 1.3 if night else 0.7                     # wolves are bolder after dark
        for a in wolves:
            best, bd = None, WOLF_MENACE_VISION + 1
            for p in self.people:
                d = abs(p["x"] - a["x"]) + abs(p["y"] - a["y"])
                if d < bd:
                    best, bd = p, d
            if best is None:
                continue
            guards = sum(1 for q in self.people if q is not best
                         and abs(q["x"] - best["x"]) + abs(q["y"] - best["y"]) <= WOLF_BAND_SAFETY)
            if guards >= WOLF_GUARDS_SAFE or self._shelter_factor(best) > 0.5 \
                    or self._guardian_near(best["x"], best["y"], exclude=best) \
                    or self._powered(best["x"], best["y"]) \
                    or self._near_building("watchtower", WATCH_RADIUS, best["x"], best["y"]):
                continue                                 # band, roof, guardian, light, or WATCHTOWER — the wolf backs off
            if bd <= 1:                                  # in reach — strike
                if self.rng.random() < ANIMALS["wolf"]["kill_chance"] * bold / (1 + guards):
                    best["hp"] = max(0.0, best["hp"] - WOLF_BITE)
                    best["death_cause"] = "a wolf"
                    cap = ANIMALS["wolf"]["repro_at"] * ENERGY_CAP_MULT
                    a["energy"] = min(cap, a["energy"] + WOLF_BITE_GAIN)
                    a["feed_next"] = self.clock + ANIMALS["wolf"]["feed_cd"]
                    mind.remember(best, "a wolf set on me out in the open", 0.9, "danger", self.clock)
                    mind.speak(best, "Wolf! Get back!", self.clock)
                    self._note("danger", f"A wolf attacked {best['name']}.")
                    self._wolf_blooded = True             # the band now has cause to raise a watchtower
                    if best["hp"] <= 0:
                        killed.append(best)
            else:                                        # close the distance
                self._move_animal(a, (int(np.sign(best["x"] - a["x"])),
                                      int(np.sign(best["y"] - a["y"]))), ANIMALS["wolf"]["speed"])
        return killed

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
            if 0 <= nx < W and 0 <= ny < H and self.water[ny, nx] not in (WATER_OCEAN, WATER_SHALLOW):
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
            "hunger": float(self.rng.random() * 0.2 + 0.1),     # comfort/desire (drives behaviour)
            "thirst": float(self.rng.random() * 0.2 + 0.1),
            "fatigue": float(self.rng.random() * 0.2),
            "satiety": float(self.rng.random() * 0.15 + 0.85),   # reserve (true survival store, 1=full)
            "hydration": float(self.rng.random() * 0.15 + 0.85),
            "stamina": float(self.rng.random() * 0.15 + 0.85),
            "hp": 1.0,
            "inv": {},                                   # carried goods, e.g. {"food":3,"wood":2,"axe":1}
            "store": {},                                 # the larder at home: surplus food/water kept for later
            "store_access": {},                          # ids granted to draw from this store (→ household/lending)
            "home": (int(x), int(y)),                    # anchor: idle wandering drifts back here
            "home_struct": None,                         # id of their shelter once built
            "known": {},                                 # remembered resource spots {water/food/wood: [x,y]}
            "berry_lore": {},                            # learned bushes "x,y" -> "good"/"bad" (P3)
            "heading": None,                             # persistent roaming direction while searching
            "action": "wander",                          # current body behaviour (for the renderer)
        })
        mind.ensure_mind(self.people[-1], self.rng)      # attach mind + roll temperament
        return self.people[-1]

    def _add_structure(self, kind: str, x: int, y: int, by: str = "?") -> str:
        sid = "s_" + uuid.uuid4().hex[:8]
        self.structures.append({
            "id": sid, "kind": kind, "x": int(x), "y": int(y),
            "by": by, "t": round(self.clock, 1),
        })
        self.version += 1
        self._note("build", f"{by} built a {kind} at ({x},{y}).")
        return sid

    def _nearest_waterside(self, x: int, y: int, max_r: int = 60) -> tuple[int, int]:
        """Closest walkable tile that BORDERS water, spiralling out from (x,y). People settle
        here so their home — and the resting they do in it — is a step from a drink. A home
        far from water turns every thirst into a fatiguing trek and quietly kills a band."""
        for r in range(0, max_r):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:           # only the new ring each pass
                        continue
                    nx, ny = x + dx, y + dy
                    if (self._in(nx, ny) and self.water[ny, nx] == WATER_NONE
                            and self._adjacent_water(nx, ny)):
                        return nx, ny
        return self._nearest_land(x, y)

    def _seed_initial_people(self, count: int, center=None):
        """Settle a small founding band together by water near the chosen origin. Thirst is
        the fastest need, so a band that settles on the bank thrives; one that settles inland
        spends its days commuting to drink and dies of the round trip."""
        cx, cy = center if center is not None else self._choose_origin()
        bx, by = self._nearest_waterside(cx, cy)             # the band's shared waterside camp
        # Settle the band SPREAD OUT along the shore, each on their own bank tile a few tiles
        # from the next. Tight clustering used to make their shelters either grow inside one
        # another or (once that was forbidden) pack into a solid wall that boxed people away
        # from the very water they settled beside. Spacing at the waterside avoids both — and
        # unlike spacing homes after the fact, it never pushes anyone inland to die commuting.
        placed: list[tuple[int, int]] = []
        spread, min_gap, attempts = 11, 4, 0
        while len(placed) < count and attempts < count * 30:
            attempts += 1
            jx = int(np.clip(bx + self.rng.integers(-spread, spread + 1), 0, W - 1))
            jy = int(np.clip(by + self.rng.integers(-spread, spread + 1), 0, H - 1))
            px, py = self._nearest_waterside(jx, jy, max_r=14)
            if all(abs(px - qx) + abs(py - qy) >= min_gap for qx, qy in placed):
                placed.append((px, py))
        # If the shore was too cramped to space everyone, fill the rest however we can.
        while len(placed) < count:
            jx = int(np.clip(bx + self.rng.integers(-spread, spread + 1), 0, W - 1))
            jy = int(np.clip(by + self.rng.integers(-spread, spread + 1), 0, H - 1))
            placed.append(self._nearest_waterside(jx, jy, max_r=14))
        start = len(self.people)
        for px, py in placed:
            self._add_person(px, py)
        # Found some COUPLES so families — and a next generation — can begin from day one: pair
        # opposite-sex founders, settle each pair on a shared homesite, and start them bonded.
        founders = self.people[start:]
        males = [p for p in founders if p["sex"] == 0]
        females = [p for p in founders if p["sex"] == 1]
        for m, f in zip(males, females):
            if self.rng.random() < 0.75:
                m["partner"], f["partner"] = f["id"], m["id"]
                f["home"], f["x"], f["y"] = m["home"], m["x"], m["y"]   # she joins his homesite
                mind._rel(m, f, 0.0).update(trust=0.85, sentiment=0.65)
                mind._rel(f, m, 0.0).update(trust=0.85, sentiment=0.65)

    # ── people: the body loop (cheap, rule-based, no LLM) ───────────────────────
    def _tick_people(self, dt_game_min: float):
        if not self.people:
            return
        dt_day = dt_game_min / (24 * 60)
        hod = self.time_of_day()
        night = hod < 6 or hod >= 21
        wolf_pos = [(a["x"], a["y"]) for a in self.animals if a["sp"] == "wolf"]
        self._wolf_pos = wolf_pos                          # cached for the danger-aware walker
        self._person_pos = {(q["x"], q["y"]) for q in self.people}   # so souls route around one another
        self._build_perception_masks()                     # resource masks once/tick (vs np.isin per soul — perf)

        dead = []
        for p in self.people:
            mind.ensure_mind(p, self.rng)                # idempotent; covers any path that skipped it
            self._ensure_body(p)                          # default the physiological reserves (legacy saves)
            if self.clock >= p.get("cryst_cd", 0):       # identity forms slowly, from a life lived
                mind.crystallize_values(p)
                p["cryst_cd"] = self.clock + 1440.0      # once a game-day
            p["age"] += dt_day
            # Two layers move every tick: COMFORT (desire) rises on its short clock, the
            # physiological RESERVE drains on its long survival clock. The mind weighs comfort;
            # the body lives or dies by the reserve.
            ill_drain = self._illness_factor(p)          # >1 while a sickness wracks the body
            for ck, rk, crate, drate in NEED_MODEL:
                mult = ill_drain if rk in ("hydration", "satiety") else 1.0
                p[ck] = min(1.0, p[ck] + PERSON[crate] * dt_game_min)
                p[rk] = max(0.0, p[rk] - PERSON[drate] * mult * dt_game_min)
            self._tick_illness(p, dt_game_min)           # incubate, sicken, erode health, recover
            if p.get("renown", 0.0) > 0.0:                # standing fades slowly if not renewed
                p["renown"] = max(0.0, p["renown"] * (1.0 - RENOWN_DECAY * dt_day))

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
            # How much danger this soul is in from a prowling wolf right now (0..1), eased by
            # the band and nullified by a roof — cached so both the body's flee-reflex and the
            # mind's deliberation read the same read of the situation.
            p["_wolf_threat"] = self._wolf_threat_at(p, wolf_pos)   # raw peril (drives a guardian's resolve)
            p["_danger"] = self._danger_at(p, wolf_pos)
            action, movedir = self._person_decide(p, edible, drinkable, tree, stone, fiber, leaf, night, lx, ly)
            p["action"] = action
            if action == "eat":
                g = float(self.veg_growth[y, x])
                if self._eat_cooked(p):                          # a cooked meal is the finest fare — eat it first
                    pass
                elif edible[ly, lx] and g > 0.12:                # else graze the tile (free food underfoot)
                    bite = min(g, PERSON["eat_bite"] * dt_game_min)
                    self.veg_growth[y, x] = g - bite
                    p["hunger"] = max(0.0, p["hunger"] - bite * PERSON["food_value"])
                    self._refill(p, "satiety", bite * PERSON["food_value"] * 0.5)
                elif p["inv"].get("food", 0) > 0:                # gathered/berried food from the pack
                    p["inv"]["food"] -= 1
                    if p["inv"]["food"] <= 0:
                        p["inv"].pop("food", None)
                    p["hunger"] = max(0.0, p["hunger"] - 0.35)
                    self._refill(p, "satiety", PERSON["feed_value"] * 4)
                else:
                    self._eat_raw(p)                             # last resort — raw flesh, with its gamble
            elif action == "drink":
                p["thirst"] = max(0.0, p["thirst"] - PERSON["drink_rate"] * dt_game_min)
                self._refill(p, "hydration", PERSON["hydrate_rate"] * dt_game_min)
                self._fill_containers(p)                 # top up any flask while at the water
                self._maybe_infect(p)                    # raw water from the source can carry sickness
            elif action == "drink_pack":
                # Drink from a carried flask — the whole point of the water-bottle craft:
                # one stored drink fully slakes a good measure of thirst, anywhere.
                if p["inv"].get("water", 0) > 0:
                    p["inv"]["water"] -= 1
                    if p["inv"]["water"] <= 0:
                        del p["inv"]["water"]
                    p["thirst"] = max(0.0, p["thirst"] - 0.45)
                    self._refill(p, "hydration", 0.3)
                    self._maybe_infect(p)                # the flask holds raw water too — still a risk
            elif action == "drink_safe":
                # Boiled water from the pack — slakes thirst like any drink, but carries NO sickness.
                if p["inv"].get("safe_water", 0) > 0:
                    p["inv"]["safe_water"] -= 1
                    if p["inv"]["safe_water"] <= 0:
                        del p["inv"]["safe_water"]
                    p["thirst"] = max(0.0, p["thirst"] - 0.45)
                    self._refill(p, "hydration", 0.3)
            elif action == "rest":
                # A sheltered person resting at home recovers faster — but only as well as
                # their home insulates (a leaf lean-to barely helps; a timber hut is snug).
                mult = 1.0
                if p.get("home_struct") and (x, y) == tuple(p["home"]):
                    insul = p.get("insul", 1.0)
                    mult = 1.0 + (BUILD["rest_sheltered_mult"] - 1.0) * insul
                elif p["inv"].get("sleeping_mat", 0) > 0:    # a discovered mat helps anywhere
                    mult = 1.0 + (BUILD["rest_sheltered_mult"] - 1.0) * 0.5
                if self.decor.get((x, y)) == "bed":          # a proper BED — sleep deeper, mend faster
                    mult += BED_REST_BONUS
                p["fatigue"] = max(0.0, p["fatigue"] - PERSON["rest_rate"] * mult * dt_game_min)
                self._refill(p, "stamina", PERSON["restore_rate"] * mult * dt_game_min)
                if (x, y) == tuple(p["home"]):
                    if p.get("hearth"):
                        self._boil_at_home(p)            # tend the fire while resting: raw water → safe
                        self._cook_at_home(p)            # …and cook raw meat/fish into safe, hearty meals
                    self._deposit_home(p)                # bank the day's surplus into the larder
            elif action == "forage_berry":
                b = self._berry_index.get((x, y))
                if b is not None and self._bush_ripe(b):
                    self._forage_bush(p, b)
            elif action == "gather":
                g = float(self.veg_growth[y, x])
                food_cap = PERSON["inv_cap"] + (6 if p["inv"].get("forage_sack", 0) else 0)
                if edible[ly, lx] and g > PERSON["gather_min"] and p["inv"].get("food", 0) < food_cap:
                    take = min(g - 0.2, 0.3)
                    self.veg_growth[y, x] = g - take
                    p["inv"]["food"] = p["inv"].get("food", 0) + 1
            elif action == "hunt":
                pid = p.pop("_prey", None)               # set by _hunt only on an adjacent strike
                if pid:
                    prey = next((a for a in self.animals if a["id"] == pid), None)
                    if prey is not None and abs(prey["x"] - x) + abs(prey["y"] - y) <= 1:
                        self._resolve_hunt_strike(p, prey)
            elif action == "fish":
                if drinkable[ly, lx]:
                    rate = FISH_CATCH_ROD if p["inv"].get("fishing_rod", 0) else FISH_CATCH_BARE
                    if self.rng.random() < rate * dt_game_min:
                        p["inv"]["fish"] = p["inv"].get("fish", 0) + 1
                        self.version += 1
            elif action == "chop":
                # Felling wood is slow, dwelt-on labour; an axe roughly halves the work.
                has_axe = p["inv"].get("axe", 0) > 0
                chop_work = CHOP_WORK * (0.5 if has_axe else 1.0)
                if tree[ly, lx] and self._work_ready(p, "chop", dt_game_min, chop_work):
                    self.veg_growth[y, x] = max(0.0, float(self.veg_growth[y, x]) - BUILD["chop_take"])
                    if self.veg_growth[y, x] <= 0.05:        # felled — the tile clears
                        self.veg_sp[y, x] = VEG_NONE
                        self.veg_growth[y, x] = 0.0
                    gain = BUILD["chop_yield"] + (BUILD["axe_bonus"] if has_axe else 0)
                    p["inv"]["wood"] = p["inv"].get("wood", 0) + gain
                    # Hacking wood by hand is hard, slow work — and out of that struggle a soul
                    # WORKS OUT the axe (knaps a sharp edge). It then spreads by teaching, so not
                    # everyone must rediscover it. (No one is born knowing how to make a tool.)
                    if not has_axe and not self._person_knows(p, "crude_axe"):
                        p["knap"] = p.get("knap", 0) + 1
                        if p["knap"] >= KNAP_CHOPS and self._grant_recipe(
                                p, "crude_axe", via="worked out",
                                rationale="a sharpened stone bites far deeper than bare hands — an axe!"):
                            self._note("discovery", f"{p['name']} worked out how to make a crude axe.")
            elif action == "mine":
                if stone[ly, lx] and self._work_ready(p, "mine", dt_game_min, MINE_WORK):
                    p["inv"]["stone"] = p["inv"].get("stone", 0) + BUILD["mine_yield"]
            elif action == "gather_fiber":
                g = float(self.veg_growth[y, x])
                if (fiber[ly, lx] and g > 0.20 and p["inv"].get("fiber", 0) < PERSON["inv_cap"]
                        and self._gather_ready(p, "fiber", dt_game_min)):
                    self.veg_growth[y, x] = max(0.0, g - 0.25)
                    p["inv"]["fiber"] = p["inv"].get("fiber", 0) + 1
            elif action == "gather_leaves":
                g = float(self.veg_growth[y, x])
                if (leaf[ly, lx] and g > 0.25 and p["inv"].get("leaves", 0) < LEAF_CAP
                        and self._gather_ready(p, "leaves", dt_game_min)):
                    self.veg_growth[y, x] = 0.0                  # stripped bare — the foliage is gone until it regrows
                    p["inv"]["leaves"] = p["inv"].get("leaves", 0) + LEAF_GATHER
            elif action == "craft":
                # If nothing's underway yet, start the bootstrap axe (survival crafts start
                # their own item in the decide step). Then tick the active craft's timer;
                # the item is only granted when the work is finished.
                if (not p.get("craft") and p["inv"].get("axe", 0) < 1
                        and self._person_knows(p, "crude_axe") and self._can_make_tools(p)):
                    self._begin_craft(p, "crude_axe")
                self._advance_craft(p, dt_game_min)
            elif action == "found_site":
                cli = next((q for q in self.people if q["id"] == p.pop("next_client", None)), None)
                self._found_site(p, p.pop("next_bp", "leaf_shelter"),
                                 communal=p.pop("next_communal", False), rung=p.pop("next_rung", None),
                                 client=cli)
            elif action == "build_block":
                # Laying a tile is real labour — dwell on it. (The cabin self-test calls
                # _build_next_block directly, bypassing this timer, so it stays one-per-call.)
                if self._work_ready(p, "build", dt_game_min, BUILD_WORK):
                    self._build_next_block(p)
            # Seeking and wandering move the body; acting-in-place does not.
            if action in ("seek_food", "seek_water", "seek_wood", "seek_stone",
                          "seek_fiber", "seek_leaves", "seek_berry", "hunt", "fish",
                          "haul", "wander", "socialize", "flee", "guard"):
                self._move_person(p, movedir)

            # Health couples to the physiological RESERVES (never to comfort). Any reserve in
            # the danger zone erodes hp — the deeper, and the more reserves at once, the faster.
            # A sound body heals, but only up to its VITALITY ceiling (min of satiety & stamina),
            # so a chronically hungry or exhausted soul's hp is dragged down and slowly declines
            # even when it isn't outright starving — the malnutrition/exhaustion coupling.
            # Exposure — being caught out in the cold/wet without a roof taxes the body: it
            # tires faster and, when the weather turns severe, loses health. Cached so the mind
            # can weigh "I'm exposed — get under cover" when it deliberates.
            threat = self._exposure_threat(night)
            eff = threat * (1.0 - self._shelter_factor(p)) if threat > 0 else 0.0
            p["_exposed"] = round(eff, 3)
            if eff > 0:
                p["fatigue"] = min(1.0, p.get("fatigue", 0.0) + EXPOSURE_FATIGUE * eff * dt_game_min)
            if eff > EXPOSURE_SEVERE:
                p["hp"] = max(0.0, p["hp"] - EXPOSURE_HP * (eff - EXPOSURE_SEVERE) * dt_game_min)
                p["death_cause"] = "the cold"
            elif p.get("death_cause") == "the cold":         # no longer freezing — drop the stale cause
                p.pop("death_cause", None)

            res = (p["hydration"], p["satiety"], p["stamina"])
            deficit = sum(max(0.0, PERSON["hp_danger"] - r) for r in res)
            vitality = min(p["satiety"], p["stamina"])
            if deficit > 0:
                p["hp"] = max(0.0, p["hp"] - PERSON["starve_dmg"] * deficit * dt_game_min)
            elif min(res) > PERSON["hp_safe"]:
                p["hp"] = min(vitality, p["hp"] + PERSON["heal"] * dt_game_min)
                p.pop("death_cause", None)               # a recovering body sheds its near-miss

            if p["hp"] <= 0 or p["age"] > PERSON["max_age"]:
                dead.append(p)

        # Predation: hungry wolves harry anyone caught alone and unsheltered. A fatal bite
        # adds its victim to the dead (deduped — a soul already down won't be listed twice).
        seen_dead = {id(d) for d in dead}
        for victim in self._wolves_menace_people(dt_game_min, night):
            if id(victim) not in seen_dead:
                dead.append(victim); seen_dead.add(id(victim))

        # Social pass: people in sight of one another notice, gossip, trade — and adults pair
        # off. Then the band may bear children. This is where reputation, the barter economy
        # and now lineage all emerge.
        self._tick_minds_social()
        self._tick_reproduction()

        for p in dead:
            if p["age"] > PERSON["max_age"]:
                cause = "old age"
            elif p.get("death_cause"):                        # a wolf or the cold finished them
                cause = p["death_cause"]
            elif p.get("illness") and self.clock >= p["illness"]["onset_t"]:
                cause = p["illness"]["d"].replace("_", " ")   # the sickness took them
            else:                                            # name the reserve that gave out first
                cause = min((("thirst", p.get("hydration", 1.0)), ("hunger", p.get("satiety", 1.0)),
                             ("exhaustion", p.get("stamina", 1.0))), key=lambda kv: kv[1])[0]
            self._note("death", f"{p['name']} died of {cause}.")
            self._bequeath(p)                                # home + a share of standing pass to kin
            # Those who knew the dead carry it: a heavy, durable memory.
            mourner, grief = None, 0.0
            for q in self.people:
                if q is p:
                    continue
                if p["id"] in q.get("rel", {}) or mind._manhattan(p, q) <= PERSON["vision"]:
                    mind.remember(q, f"{p['name']} died of {cause}", 0.95, "death", self.clock)
                # The one who loved them most leads the mourning (a partner above all).
                bond = q.get("rel", {}).get(p["id"], {}).get("sentiment", 0.0)
                if q.get("partner") == p["id"]:
                    bond += 1.0
                if bond > grief:
                    mourner, grief = q, bond
            # Mourning rite: the closest soul grieves aloud — the band marks a loss, not just logs it.
            if mourner is not None and grief > 0.3:
                mind.speak(mourner, f"Rest now, {p['name']}. We'll remember you.", self.clock)
                self._note("death", f"{mourner['name']} mourns {p['name']}.")
        if dead:
            ids = {id(p) for p in dead}
            self.people = [p for p in self.people if id(p) not in ids]

    def _tick_minds_social(self):
        """Run encounters between every nearby pair of people (each pair once). Notable
        outcomes — trades, gossip — go to the world log so a god can watch culture form."""
        ppl = self.people
        for i in range(len(ppl)):
            for j in range(i + 1, len(ppl)):
                a, b = ppl[i], ppl[j]
                if mind._manhattan(a, b) > mind.SOCIAL_RADIUS:
                    continue
                for ev in mind.encounter(a, b, self.clock, self.rng):
                    self._note("social", ev)
                # Knowledge spreads soul to soul: a trusted neighbour passes on a craft.
                self._maybe_teach(a, b)
                # Two warm, unattached adults may pair off — the start of a family line.
                self._maybe_bond(a, b)
                # Generosity: whoever has resolved to *provide* gives, if they're beside
                # the other and carry a surplus — a one-way gift, the warmest social act.
                if mind._manhattan(a, b) <= 1:
                    for giver, taker in ((a, b), (b, a)):
                        inten = giver.get("intention") or {}
                        if (inten.get("kind") == "provide"
                                and giver.get("inv", {}).get("food", 0) > mind.TRADE_SURPLUS
                                and taker.get("inv", {}).get("food", 0) <= 1):
                            ev = mind.give(giver, taker, "food", self.clock)
                            if ev:
                                self._note("social", ev)
                        # Rescue: a caretaker nursing this sick soul shares food with them even
                        # off their last ration — a hungry, ailing body must be fed (#10).
                        if (inten.get("kind") == "tend" and inten.get("target") == taker["id"]
                                and giver.get("inv", {}).get("food", 0) > 0
                                and taker.get("illness") and taker.get("hunger", 0) > 0.4):
                            ev = mind.give(giver, taker, "food", self.clock)
                            if ev:
                                self._note("social", ev)
                        # A toolmaker (or anyone) beside a band-mate who lacks a piece of gear
                        # they carry a SPARE of hands it over — how crafted goods spread when
                        # they aren't barter goods. This is the toolmaker's social role.
                        for gid, _h, _r in self._GEAR:
                            if giver.get("inv", {}).get(gid, 0) >= 2 and taker.get("inv", {}).get(gid, 0) == 0:
                                ev = mind.give(giver, taker, gid, self.clock)
                                if ev:
                                    self._note("social", ev)
                                    self._record_trade(giver, taker)
                                break
                        # Help someone CRAFT: a band-mate who's worked out a piece of gear but is
                        # short the makings gets the missing material from someone who has it to
                        # spare — so the craft can go ahead. The other half of helping: not just
                        # lending labour on a build, but sharing the stuff a maker needs.
                        self._maybe_gift_craft_material(giver, taker)
                        # A toolmaker hands a spare TOOL to a band-mate who has none — how a
                        # builder gets their axe and a hunter their spear (tool-gating's payoff).
                        for _rid, key in self._TOOLS:
                            if giver.get("inv", {}).get(key, 0) >= 2 and taker.get("inv", {}).get(key, 0) == 0:
                                ev = mind.give(giver, taker, key, self.clock)
                                if ev:
                                    self._note("social", ev)
                                    # Maker-renown: a tool one fashioned, now serving another, builds
                                    # the maker's name as a craftsman the band relies on (#20).
                                    self._earn_renown(giver, RENOWN_GAIN["gift"],
                                                      f"made a {key} that now serves {taker['name']}")
                                    self._record_trade(giver, taker)
                                break
                    # The quiet bonds of daily life — kin breaking bread side by side (#1), and
                    # children at play (#5) — each warms a relationship a little. Gated to a small
                    # per-tick chance so standing together a while builds the bond gradually
                    # rather than spiking it, and to keep the pass cheap.
                    if self.rng.random() < 0.06:
                        _WORK = {"gather", "seek_food", "forage_berry", "seek_berry", "chop",
                                 "seek_wood", "mine", "hunt", "fish"}
                        both_eat = a.get("action") == "eat" and b.get("action") == "eat"
                        both_kids = a["age"] < ADULT_AGE and b["age"] < ADULT_AGE
                        both_work = a.get("action") in _WORK and b.get("action") in _WORK   # co-foraging
                        if both_eat or both_kids or both_work:
                            ra2, rb2 = mind._rel(a, b, self.clock), mind._rel(b, a, self.clock)
                            mind._adjust(ra2, 0.008, 0.03); mind._adjust(rb2, 0.008, 0.03)
                            if self.rng.random() < 0.15:
                                if both_kids:
                                    mind.speak(a, f"Tag — you're it, {b['name']}!", self.clock)
                                    self._note("social", f"{a['name']} and {b['name']} played together.")
                                elif both_eat:
                                    self._note("social", f"{a['name']} and {b['name']} shared a meal.")
                                else:
                                    self._note("social", f"{a['name']} and {b['name']} worked side by side.")

    # ── generations: bonding, birth, inheritance (Phase 4) ───────────────────────
    def _maybe_bond(self, a, b):
        """Two unattached adults of opposite sex who have grown warm and trusting pair off into
        a lifelong bond — the seed of a family. Bonds are mutual and exclusive."""
        if a.get("partner") or b.get("partner") or a["sex"] == b["sex"]:
            return
        if a["age"] < BREED_MIN_AGE or b["age"] < BREED_MIN_AGE:
            return
        if not (a.get("home_struct") and b.get("home_struct")):
            return
        ra, rb = a.get("rel", {}).get(b["id"], {}), b.get("rel", {}).get(a["id"], {})
        if (ra.get("sentiment", 0) >= BOND_WARMTH and rb.get("sentiment", 0) >= BOND_WARMTH
                and ra.get("trust", 0) >= 0.5 and rb.get("trust", 0) >= 0.5):
            a["partner"], b["partner"] = b["id"], a["id"]
            self._note("social", f"{a['name']} and {b['name']} became partners.")
            for one, two in ((a, b), (b, a)):
                mind.remember(one, f"{two['name']} and I became partners", 0.9, "social", self.clock)

    def _tick_reproduction(self):
        """Bonded, mature, well-fed and sheltered partners who are together may bear a child —
        gated by a mother's cooldown, the partners' nourishment (a resource check) and a band
        population ceiling, so the line grows slowly and only when the band can support it."""
        by_id = {p["id"]: p for p in self.people}
        # Cohabitation: a partner without a roof moves into the other's home, so a couple shares
        # one hearth (and is together when a child might come). Cheap, runs every social tick.
        for p in self.people:
            partner = by_id.get(p.get("partner"))
            if partner and not p.get("home_struct") and partner.get("home_struct"):
                p["home_struct"], p["home"] = partner["home_struct"], partner["home"]
                p["insul"] = partner.get("insul", 1.0)
        if len(self.people) >= POP_CAP:
            return
        for m in self.people:
            if m["sex"] != 1 or not (BREED_MIN_AGE <= m["age"] <= BREED_MAX_AGE):
                continue
            if self.clock < m.get("breed_cd", 0) or not m.get("home_struct"):
                continue
            f = by_id.get(m.get("partner"))
            if f is None or not (BREED_MIN_AGE <= f["age"] <= BREED_MAX_AGE):
                continue
            if mind._manhattan(m, f) > mind.SOCIAL_RADIUS:
                continue                                     # the partners must be together
            if m.get("satiety", 1) < 0.6 or f.get("satiety", 1) < 0.6:
                continue                                     # only a well-fed pair (resource gate)
            self._birth(m, f)
            if len(self.people) >= POP_CAP:
                break

    def _birth(self, mother, father):
        """A child is born into the family home. Its NATURE is a blend of its parents' (plus a
        little drift), but its KNOWLEDGE is blank beyond the universal starters — culture is not
        inherited, it must be taught afresh, so a band's crafts can grow or be lost across lives."""
        hx, hy = mother["home"]
        child = self._add_person(int(hx), int(hy), age=0.0)
        mt, ft = mother.get("traits", {}), father.get("traits", {})
        child["traits"] = {t: round(float(np.clip((mt.get(t, 0.5) + ft.get(t, 0.5)) / 2
                           + self.rng.normal(0.0, TRAIT_MUTATION), 0.1, 0.9)), 2) for t in mind.TRAITS}
        child["home_struct"] = mother.get("home_struct")     # sheltered in the family home
        child["insul"] = mother.get("insul", 1.0)
        child["parents"] = [mother["id"], father["id"]]
        child["lineage"] = mother.get("lineage") or father.get("lineage") or father["name"]
        for parent in (mother, father):
            parent.setdefault("children", []).append(child["id"])
            mind._rel(parent, child, self.clock).update(trust=0.95, sentiment=0.7)
            mind._rel(child, parent, self.clock).update(trust=0.95, sentiment=0.7)
        mother["breed_cd"] = self.clock + BREED_COOLDOWN_DAYS * 1440.0
        self.version += 1
        self._note("birth", f"{mother['name']} and {father['name']} had a child — {child['name']}.")
        mind.remember(mother, f"my child {child['name']} was born", 0.95, "birth", self.clock)
        mind.remember(father, f"my child {child['name']} was born", 0.9, "birth", self.clock)

    def _bequeath(self, dead):
        """When a soul dies, a share of their standing passes to each child (the renown of a
        line endures), and their home passes to an heir who needs one — a child still without a
        roof, else the surviving partner — so a raised home outlives its builder."""
        for q in self.people:
            if q is not dead and dead["id"] in q.get("parents", []):
                q["renown"] = q.get("renown", 0.0) + RENOWN_LEGACY * dead.get("renown", 0.0)
        home = dead.get("home_struct")
        if not home:
            return
        kids = [q for q in self.people if q is not dead and dead["id"] in q.get("parents", [])]
        heir = next((q for q in kids if q.get("home_struct") in (None, home)), None)
        if heir is None and dead.get("partner"):
            heir = next((q for q in self.people if q["id"] == dead["partner"]), None)
        if heir is not None and heir.get("home_struct") != home:
            heir["home_struct"], heir["home"], heir["insul"] = home, dead["home"], dead.get("insul", 1.0)
            hstore = heir.setdefault("store", {})        # the larder passes with the roof
            for k, v in dead.get("store", {}).items():
                hstore[k] = hstore.get(k, 0) + v
            self._note("birth", f"{heir['name']} inherited {dead['name']}'s home.")
            mind.remember(heir, f"I inherited {dead['name']}'s home", 0.8, "build", self.clock)

    @staticmethod
    def _resource_masks(vsp, vgr, wat, bio):
        """The resource boolean masks for a tile region (edible/tree/stone/fiber/leaf/drinkable) —
        the np.isin work, factored out so it can run ONCE over the whole people-box per tick instead
        of per soul (the np.isin hotspot at town scale)."""
        edible = np.isin(vsp, EDIBLE_IDS) & (vgr > 0.12)
        tree = np.isin(vsp, WOOD_IDS) & (vgr > BUILD["chop_growth_min"])
        stone = np.isin(bio, [B["rock"], B["mountain"]]) & (wat == WATER_NONE)
        fiber = np.isin(vsp, FIBER_IDS) & (vgr > 0.20)           # grasses to pull for thatch/rope
        leaf = np.isin(vsp, LEAF_IDS) & (vgr > 0.25)            # foliage to strip for a leaf shelter
        watery = wat != WATER_NONE
        drinkable = np.zeros_like(watery)                       # land tiles bordering water
        if watery.shape[0] > 2 and watery.shape[1] > 2:
            drinkable[1:-1, 1:-1] = (
                (watery[:-2, 1:-1] | watery[2:, 1:-1] | watery[1:-1, :-2] | watery[1:-1, 2:])
                & (wat[1:-1, 1:-1] == WATER_NONE))
        return edible, drinkable, tree, stone, fiber, leaf

    def _build_perception_masks(self):
        """Precompute the resource masks ONCE per tick over the box covering every soul's perception,
        so _perceive can SLICE them instead of running np.isin per soul — the big perf win at town
        scale. Survival-NEUTRAL: the masks are identical, just computed in bulk. None when no people."""
        if not self.people:
            self._pmask = None
            return
        v = PERSON["vision"] + 2                                # a tile of slack so every window sits inside
        xs = [p["x"] for p in self.people]; ys = [p["y"] for p in self.people]
        x0, x1 = max(0, min(xs) - v), min(W, max(xs) + v + 1)
        y0, y1 = max(0, min(ys) - v), min(H, max(ys) + v + 1)
        if (x1 - x0) > PMASK_MAX_SPAN or (y1 - y0) > PMASK_MAX_SPAN:
            self._pmask = None             # band too SPREAD to precompute cheaply (the win is for a clustered
            return                         # town); fall back to each soul computing its own small window
        ed, dr, tr, st, fi, lf = self._resource_masks(
            self.veg_sp[y0:y1, x0:x1], self.veg_growth[y0:y1, x0:x1],
            self.water[y0:y1, x0:x1], self.biome[y0:y1, x0:x1])
        self._pmask = {"edible": ed, "drinkable": dr, "tree": tr, "stone": st, "fiber": fi,
                       "leaf": lf, "x0": x0, "y0": y0, "x1": x1, "y1": y1}

    def _perceive(self, x, y):
        """Build this person's small perception windows (edible/drinkable/tree/stone)
        around (x,y), plus their position (lx,ly) inside the window. Cost is O(vision²),
        independent of world size — the key to people staying cheap on a 4M-tile map.
        Slices the per-tick precomputed masks when they cover the window (the common case), else
        falls back to computing the window directly."""
        v = PERSON["vision"]
        wy0, wy1 = max(0, y - v - 1), min(H, y + v + 2)
        wx0, wx1 = max(0, x - v - 1), min(W, x + v + 2)
        pm = getattr(self, "_pmask", None)
        if pm is not None and pm["x0"] <= wx0 and pm["y0"] <= wy0 and wx1 <= pm["x1"] and wy1 <= pm["y1"]:
            ry, rx = wy0 - pm["y0"], wx0 - pm["x0"]
            hy, hx = wy1 - wy0, wx1 - wx0
            sl = (slice(ry, ry + hy), slice(rx, rx + hx))
            return (pm["edible"][sl], pm["drinkable"][sl], pm["tree"][sl], pm["stone"][sl],
                    pm["fiber"][sl], pm["leaf"][sl], x - wx0, y - wy0, wx0, wy0)
        ed, dr, tr, st, fi, lf = self._resource_masks(
            self.veg_sp[wy0:wy1, wx0:wx1], self.veg_growth[wy0:wy1, wx0:wx1],
            self.water[wy0:wy1, wx0:wx1], self.biome[wy0:wy1, wx0:wx1])
        return ed, dr, tr, st, fi, lf, x - wx0, y - wy0, wx0, wy0

    def _nearest_local(self, mask, lx, ly):
        """The RAW delta toward the nearest True tile in a window mask (centre lx,ly) — so the mover
        PATHFINDS around walls to reach it (water, food, wood, …), instead of greedily oscillating
        against an obstacle. This is what stops a soul getting stuck on its OWN house wall on the way
        to water. None if the mask is empty or the nearest tile is the soul's own."""
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        k = int(np.argmin(np.abs(xs - lx) + np.abs(ys - ly)))
        dx, dy = int(xs[k] - lx), int(ys[k] - ly)
        return (dx, dy) if (dx or dy) else None

    def _nearest_loc(self, mask, lx, ly, wx0, wy0):
        """Global (x,y) of the nearest True tile in a window, or None — for memory."""
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        k = int(np.argmin(np.abs(xs - lx) + np.abs(ys - ly)))
        return [int(xs[k] + wx0), int(ys[k] + wy0)]

    def _seek_person(self, p):
        """Step direction toward the nearest other living person (for a social/trade goal),
        or None if alone or already beside someone. Stops adjacent so the social pass can
        run an encounter rather than walking onto them."""
        best, bd = None, PERSON["vision"] * 3 + 1       # only head toward company actually within reach
        for q in self.people:
            if q is p:
                continue
            d = abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])
            if d < bd:
                best, bd = q, d
        if best is None or bd <= 1:
            return None
        return (best["x"] - p["x"], best["y"] - p["y"])   # raw delta → pathfind to company round walls

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
                return act_seek, (kx - x, ky - y)        # raw delta to the remembered spot → pathfind
        return act_seek, self._explore_dir(p)

    def _mind_ctx(self, p, night) -> dict:
        """The slice of world a mind weighs when it deliberates: time, who's about, and the
        people it feels most warmly/coldly toward (and any nearby soul in want). Kept small
        and cheap — built once per deliberation, not per tick."""
        names = {q["id"]: q["name"] for q in self.people}
        fav_id = fav_name = foe_id = foe_name = None
        foe_mag = 0.0
        rels = p.get("rel", {})
        if rels:
            best = max(rels.items(), key=lambda kv: kv[1]["sentiment"])
            worst = min(rels.items(), key=lambda kv: kv[1]["sentiment"])
            if best[1]["sentiment"] > 0.15 and best[0] in names:
                fav_id, fav_name = best[0], best[1]["name"]
            if worst[1]["sentiment"] < -0.15 and worst[0] in names:
                foe_id, foe_name, foe_mag = worst[0], worst[1]["name"], -worst[1]["sentiment"]
        needy_id = needy_name = None
        nd = 999
        ail_id = ail_name = None
        ad = 999
        mentor_id = mentor_name = None
        mgap = 1
        owes = p.get("owes", {})
        my_recipes = p.get("recipes") or []
        healthy = p.get("hp", 1.0) > 0.6 and not (p.get("illness") and self.clock >= p["illness"]["onset_t"])
        for q in self.people:
            if q is p:
                continue
            d = abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])
            if q.get("inv", {}).get("food", 0) <= 1 and d <= PERSON["vision"] * 3:
                # Repay a past benefactor before a stranger — gratitude reweights need by debt.
                eff = d * (0.4 if q["id"] in owes else 1.0)
                if eff < nd:
                    needy_id, needy_name, nd = q["id"], q["name"], eff
            # A band-mate laid low by sickness (symptoms showing) within reach — only a soul who
            # is itself well goes to nurse them (#11 tend the sick / #10 rescue).
            if healthy and d < ad and d <= PERSON["vision"] * 3 \
                    and q.get("illness") and self.clock >= q["illness"]["onset_t"]:
                ail_id, ail_name, ad = q["id"], q["name"], d
            # A markedly more-skilled band-mate to apprentice oneself to — the soul seeks them out
            # to learn their crafts (apprenticeship; the teaching itself happens on contact).
            if d <= PERSON["vision"] * 3 and len(my_recipes) < 8:
                gap = sum(1 for r in (q.get("recipes") or []) if r not in my_recipes)
                if gap > mgap:
                    mentor_id, mentor_name, mgap = q["id"], q["name"], gap
        # Whereabouts memory: a soul KNOWS where someone is only by seeing them. Record every
        # person in sight now, so a later wish to find them can be navigated from a real last-known
        # spot rather than by magically tracking their position (feeds the non-omniscient seek).
        seen = p.setdefault("seen", {})
        vis = PERSON["vision"]
        for q in self.people:
            if q is p:
                continue
            if abs(q["x"] - p["x"]) + abs(q["y"] - p["y"]) <= vis:
                seen[q["id"]] = [q["x"], q["y"], self.clock]
        nearby = ", ".join(q["name"] for q in self.people
                           if q is not p and abs(q["x"] - p["x"]) + abs(q["y"] - p["y"]) <= PERSON["vision"] * 2)
        unsolved = self._person_unsolved(p)        # what THIS soul hasn't worked out yet
        gear = [r for r in ("leaf_flask", "forage_sack", "sleeping_mat") if self._person_knows(p, r)]
        needs_gear = any(p.get("inv", {}).get(r, 0) < 1 for r in gear)
        # Children don't build, take a calling, or pursue projects — they grow, forage, play and
        # learn (survival + belonging + curiosity) until they come of age.
        is_child = p["age"] < ADULT_AGE
        proj = None if is_child else self._project_for(p)   # standing life-project (dwelling ladder, monument)
        p["project"] = proj                  # stash so the body can pursue it (the mind only weighs it)
        # How far along the soul's current build is (0..1) — sunk-cost momentum so a half-raised
        # structure pulls harder and gets FINISHED rather than abandoned for some other whim.
        site = None if is_child else self._person_site(p)
        build_progress = (sum(1 for t in site["tasks"] if t["done"]) / len(site["tasks"])
                          if site and site.get("tasks") else 0.0)
        # The soul's calling — read from temperament but BENT toward the trade the band lacks, so the
        # division of labour stays balanced (not everyone the same calling). Hysteresis keeps it stable.
        voc = None if is_child else mind.vocation(p, self.people)
        p["vocation"] = voc
        needs_hearth = (not is_child and p.get("home_struct") is not None
                        and self._person_knows(p, "campfire") and not p.get("hearth"))
        # A housed soul with materials to spare looks for a band-mate's unfinished build nearby to
        # lend a hand on — the nearest site (not their own, not communal) whose next tile they can
        # actually pay for from their pack. This is what turns "I'm done" into "who needs a hand?".
        help_site = None
        if not is_child and p.get("home_struct") is not None:
            px, py = p["x"], p["y"]
            best_d = HELP_RANGE + 1
            for s in self.sites:
                if s.get("done") or s.get("communal") or s.get("owner") == p["id"]:
                    continue
                if self._site_next_task(s) is None:
                    continue
                d = abs(s["ox"] - px) + abs(s["oy"] - py)   # near enough to lend a hand (we'll gather the makings)
                if d < best_d:
                    best_d, help_site = d, s["id"]
        can_aspire = (not is_child and p.get("home_struct") is not None
                      and (p.get("plan") is not None or self.clock >= p.get("aspire_cd", 0.0)))
        return {
            "is_child": is_child,
            "needs_gear": needs_gear,
            "needs_hearth": needs_hearth,
            "help_site": help_site,
            "can_aspire": can_aspire,
            "aspire_kinds": list(ASPIRE_KINDS),
            "aspiring": p.get("plan") is not None,
            "aspire_why": (p.get("plan") or {}).get("goal") or "make my home a finer place to look on",
            "project": proj,
            "vocation": voc,
            "clock": self.clock, "night": night, "season": self.season(),
            "weather": self.weather, "time_str": f"{int(self.time_of_day()):02d}:00",
            "others_exist": len(self.people) > 1,
            "alive_ids": tuple(names.keys()), "_names": names,
            "fav_id": fav_id, "fav_name": fav_name,
            "foe_id": foe_id, "foe_name": foe_name, "foe_mag": foe_mag,
            "needy_id": needy_id, "needy_name": needy_name,
            "ail_id": ail_id, "ail_name": ail_name,
            "mentor_id": mentor_id, "mentor_name": mentor_name,
            "nearby": nearby or "no one",
            # What the band still hasn't figured out, and how hard-pressed this soul is —
            # so a curious, recently-thirsty person is the keenest to invent.
            "unsolved": unsolved,
            "unsolved_problems": [crafting.SURVIVAL_DISCOVERIES[r] for r in unsolved],
            "hardship": min(1.0, max(p.get("thirst", 0), p.get("fatigue", 0) * 0.6)),
            # Situational stakes the mind weighs: a prowling wolf nearby (danger) and being
            # caught out in foul weather without a roof (exposed). Both cached this tick.
            "danger": p.get("_danger", 0.0),
            # The keenest wolf-peril on any band-mate within guarding reach — a brave, settled
            # soul weighs this and stands watch over them instead of minding only itself.
            "ward_threat": max((q.get("_wolf_threat", 0.0) for q in self.people
                                if q is not p
                                and abs(q["x"] - p["x"]) + abs(q["y"] - p["y"]) <= GUARD_RANGE),
                               default=0.0),
            "exposed": p.get("_exposed", 0.0),
            "build_progress": build_progress,
            # The sublime: a machine far beyond the band's craft in sight. `novelty` fades as the
            # soul studies it, so awe gives way to understanding (then they go back to building).
            "wonder": (lambda w: {"dist": w[2], "kind": w[3],
                                  "novelty": max(0.1, 1.0 - p.get("insight", 0.0) / WONDER_INSIGHT_TO_LEARN)}
                       if w else None)(self._nearest_wonder(p)),
        }

    def _person_decide(self, p, edible, drinkable, tree, stone, fiber, leaf, night, lx, ly):
        """Actuate the mind's standing INTENTION (set by the drive arbiter / LLM) into one
        body action. Returns (action, movedir|None). The body no longer decides *what* to
        pursue — only *how* to take the next step toward what the mind has resolved on.

        Survival is not privileged here: it wins only because, when a need bites, its drive
        out-scores the rest and the arbiter picks drink/eat/rest. Two thin reflexes remain —
        a re-think when a need spikes mid-task, and a sip/bite when literally standing on
        relief — so a daydreaming wanderer never starves beside a stream."""
        x, y = p["x"], p["y"]
        # Fear reflex: a wolf right on top of an exposed soul overrides whatever they were doing —
        # bolt for home/the band now, don't wait for the next deliberation beat.
        if p.get("_danger", 0.0) >= FLEE_TRIGGER \
                and (p.get("intention") or {}).get("kind") != "guard":
            return self._flee(p)
        # (Re)deliberate when there's no aim, the beat has elapsed, or a need has spiked past
        # whatever the current intention is worth (an emergency interrupt).
        inten = p.get("intention")
        # Emergency interrupt: if a need spikes past whatever we're doing, re-deliberate —
        # UNLESS the current aim already relieves *that* need. (Crucially, a person resting
        # must still break off to drink when thirst turns dangerous — neglecting one need
        # while tending another is how the founding band quietly died.)
        relieves = {"drink": "thirst", "eat": "hunger", "rest": "fatigue"}
        worst, worst_u = max((("thirst", mind.need_urgency(p["thirst"], p.get("hydration", 1.0))),
                              ("hunger", mind.need_urgency(p["hunger"], p.get("satiety", 1.0))),
                              ("fatigue", mind.need_urgency(p["fatigue"], p.get("stamina", 1.0)))),
                             key=lambda kv: kv[1])
        spike = inten and relieves.get(inten.get("kind")) != worst \
            and worst_u > inten.get("u", 1.0) + mind.HYSTERESIS
        if (inten is None) or (self.clock >= p.get("delib_cd", 0)) or spike:
            ctx = self._mind_ctx(p, night)
            inten = mind.deliberate(p, ctx, self.rng)
            p["delib_cd"] = self.clock + DELIBERATE_BEAT * (0.7 + 0.6 * self.rng.random())

        kind = inten["kind"]; target = inten.get("target")

        # Reflex sips while passing relief (doesn't change the standing intention).
        if kind not in ("drink", "eat", "rest"):
            if p["inv"].get("safe_water", 0) > 0 and p["thirst"] > 0.3:
                return "drink_safe", None                    # boiled water is always the first choice
            # Only a soul who hasn't learned better gulps raw water on reflex; one who knows to
            # boil holds out for safe water (unless truly parched — handled by the drink intent).
            if drinkable[ly, lx] and p["thirst"] > 0.3 and not self._person_knows(p, "campfire"):
                return "drink", None
            if edible[ly, lx] and self.veg_growth[y, x] > 0.12 and p["hunger"] > 0.35:
                return "eat", None

        if kind == "drink":
            if p["inv"].get("safe_water", 0) > 0:   # boiled water in the pack — safe, drink it
                return "drink_safe", None
            if drinkable[ly, lx]:
                return "drink", None
            if p["inv"].get("water", 0) > 0:        # carried RAW water (a flask) — drink anywhere
                return "drink_pack", None
            if self._nearest_local(drinkable, lx, ly) is None and self._prefer_store(p, "water"):
                f = self._fetch_from_store(p, "water")   # no water in sight or nearer spring — try the larder
                if f:
                    return f
            return self._seek(p, x, y, False, drinkable, lx, ly, "water", "drink", "seek_water")
        if kind == "tinker":
            # Sit and puzzle out a make-shift craft — slow, gated, motivated by a felt
            # problem, and PERSONAL (it spreads to others later by teaching, not at once).
            # A settled soul does its puzzling back home, not alone out in the wild.
            home = self._homeward_if_away(p)
            if home:
                return home
            self._tinker(p, night)
            return "tinker", None
        if kind == "eat":
            inv = p["inv"]
            pack_food = (inv.get("food", 0) + inv.get("cooked_meat", 0) + inv.get("cooked_fish", 0)
                         + inv.get("meat", 0) + inv.get("fish", 0))
            if pack_food > 0 and p["hunger"] > 0.3:
                return "eat", None                      # eat from the pack (handler picks the best meal)
            berry = self._berry_seek(p)                 # a ripe bush in local reach beats grazing thin tiles
            if berry:
                return berry
            hunt = self._hunt(p, max_d=5)               # easy game right nearby is a meal worth taking
            if hunt:
                return hunt
            here_food = bool(edible[ly, lx]) and self.veg_growth[y, x] > 0.12
            if not here_food and self._nearest_local(edible, lx, ly) is None and self._prefer_store(p, "food"):
                f = self._fetch_from_store(p, "food")   # nothing growing in sight — fall back on the larder
                if f:
                    return f
            return self._seek(p, x, y, here_food, edible, lx, ly, "food", "eat", "seek_food")
        if kind == "rest":
            # Sleep under the roof if home is a short walk off (so the band beds down in its
            # dwellings, #2) — but only when not yet bone-tired and home is genuinely near;
            # a spent soul sleeps where it stands rather than marching all night into exhaustion.
            hx, hy = p["home"]
            hd = abs(hx - x) + abs(hy - y)
            if p.get("home_struct") and 1 < hd <= REST_HOMEWARD_MAX and p.get("fatigue", 0) < 0.8:
                return "wander", (hx - x, hy - y)
            return "rest", None
        if kind == "flee":
            return self._flee(p)
        if kind == "guard":
            return self._guard(p)
        if kind == "ply":
            # Ply one's trade: produce the surplus that division of labour and barter run on.
            return self._ply(p, edible, drinkable, tree, fiber, leaf, lx, ly)
        if kind == "provision":
            # Lay in a food reserve: gather a pack-load, carry it home, bank it in the larder.
            return self._provision(p, edible, lx, ly)
        if kind in ("build", "provide"):
            # Both lean on the build/forage machinery; provide also gathers a food surplus
            # to give away (the gift itself happens on contact in the social pass).
            proj = self._person_build_decide(p, tree, stone, fiber, leaf, lx, ly)
            if proj:
                return proj
            if kind == "provide" and target:
                move = self._seek_toward(p, target)
                if move is not None:
                    return "socialize", move
            return self._idle(p)
        if kind == "help":
            # Lend a hand on a band-mate's unfinished build — gather its makings and lay a tile.
            # The home still belongs to its owner (see _finish_site).
            return self._help_build(p, target, tree, stone, fiber, leaf, lx, ly)
        if kind == "forage":
            # A youngling gathers what little hands can, growing its foraging skill by doing.
            return self._child_forage(p, edible, lx, ly)
        if kind == "whittle":
            # A youngling whittles arrows and practises — growing a young crafter's skill.
            return self._child_whittle(p, tree, lx, ly)
        if kind in ("socialize", "befriend"):
            move = self._seek_toward(p, target) if target else self._seek_person(p)
            if move is not None:
                return "socialize", move
            return self._idle(p)          # already beside them — the social pass does the rest
        if kind == "explore":
            p["last_explore_t"] = self.clock
            lend = self._offer_help_maybe(p, tree, stone, fiber, leaf, lx, ly)
            if lend is not None:                          # a neighbour's raising their home nearby — pitch in
                return lend
            # Curiosity is leashed to home range: range out, but turn back before straying
            # past easy return to known water — wonder shouldn't be a death sentence. The leash
            # SHORTENS when the body is weak (hungry/thirsty/tired/hurt), so a soul never wanders
            # far from relief while running low — prudence reins in curiosity (#20).
            weak = max(p.get("thirst", 0), p.get("hunger", 0), p.get("fatigue", 0),
                       1.0 - p.get("hp", 1.0))
            leash = EXPLORE_LEASH * (1.0 - 0.6 * min(1.0, weak))
            hx, hy = p["home"]
            if abs(hx - x) + abs(hy - y) > leash:
                return "wander", (hx - x, hy - y)          # beyond the leash → pathfind home
            return "wander", self._roam_delta(p, hx, hy, leash)   # within leash → roam near home
        if kind == "tend" and target:
            # Go to the sick band-mate's side and nurse them (the recovery boost + any food
            # they're handed happen on contact in the illness tick / social pass).
            move = self._seek_toward(p, target)
            if move is not None:
                return "socialize", move
            return "tend", None
        if kind == "avoid" and target:
            t = next((q for q in self.people if q["id"] == target), None)
            if t is not None and (t["x"] != x or t["y"] != y):
                return "wander", (int(np.sign(x - t["x"])), int(np.sign(y - t["y"])))
        if kind == "marvel":
            return self._marvel(p)
        if kind == "aspire":
            # A self-authored project beyond survival — beautify one's own home (tidy / garden).
            return self._pursue_aspiration(p)
        lend = self._offer_help_maybe(p, tree, stone, fiber, leaf, lx, ly)
        if lend is not None:                              # at loose ends — lend a neighbour a hand if one's building
            return lend
        return self._idle(p)

    def _seek_toward(self, p, target_id):
        """Head toward a specific person WITHOUT omniscience. Straight to them if they're in
        sight; else to where we last saw them; else to their home (folk know where folk live);
        else give up. So a soul looks where it has reason to, finds them or doesn't, and moves on
        rather than tracking them like a homing missile."""
        t = next((q for q in self.people if q["id"] == target_id), None)
        if t is None:
            p.get("seen", {}).pop(target_id, None)
            return None
        x, y = p["x"], p["y"]
        seen = p.setdefault("seen", {})
        if abs(t["x"] - x) + abs(t["y"] - y) <= PERSON["vision"]:        # in sight — close in
            seen[target_id] = [t["x"], t["y"], self.clock]
            if abs(t["x"] - x) + abs(t["y"] - y) <= 1:
                return None
            return (t["x"] - x, t["y"] - y)                             # raw delta → pathfind round walls
        loc = seen.get(target_id)
        if loc and self.clock - loc[2] <= SEEN_FORGET:                  # go where we last saw them
            if abs(loc[0] - x) + abs(loc[1] - y) <= 1:                  # got there, they've moved on
                seen.pop(target_id, None)                               # the trail's cold — give up
                return None
            return (loc[0] - x, loc[1] - y)                             # raw delta → pathfind
        hx, hy = t.get("home", (x, y))                                  # try their home, then give up
        if abs(hx - x) + abs(hy - y) > 1:
            return (hx - x, hy - y)                                     # raw delta → pathfind to their home
        return None

    def _step_off_door(self, p):
        """If a soul is loitering ON a doorway, a step out to the most OPEN neighbouring tile — so
        idlers don't clog a building's single entrance (it just looks like they're blocking it,
        since people share tiles, but a clear doorway reads far better). None if not on a door."""
        x, y = p["x"], p["y"]
        if self.blocks.get((x, y)) != BLOCK_DOOR:
            return None
        best = None
        for dx, dy in _STEP_DIRS:
            nx, ny = x + dx, y + dy
            if not self._passable(nx, ny) or self.blocks.get((nx, ny)) == BLOCK_DOOR:
                continue
            score = 1 if self.blocks.get((nx, ny)) is None else 0   # open ground beats another tiled square
            if best is None or score > best[0]:
                best = (score, nx, ny)
        if best is not None:
            return "wander", (best[1] - x, best[2] - y)   # raw delta → pathfind round walls
        return None

    def _idle(self, p):
        """Drift home if strayed, else gather toward company — the resting state of a mind
        between aims. Idle souls drift toward a nearby band-mate rather than scattering, so the
        village reads as a gathering of folk rather than figures milling alone (#4)."""
        off = self._step_off_door(p)                       # never loiter in a doorway
        if off:
            return off
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) > 6:
            return "wander", (hx - p["x"], hy - p["y"])    # drift home → pathfind round walls
        # Drift toward the nearest neighbour within an easy stroll (not a cross-map trek), so
        # the idle band clusters; once beside someone, just amble (the social pass does the rest).
        best, bd = None, 11
        for q in self.people:
            if q is p:
                continue
            d = abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])
            if 1 < d < bd:
                best, bd = q, d
        if best is not None:
            return "wander", (best["x"] - p["x"], best["y"] - p["y"])   # drift to a neighbour → pathfind
        # No one to drift toward and home is near — SETTLE in place (an occasional amble keeps it
        # from looking frozen) instead of restlessly milling the village every single beat.
        if self.rng.random() < 0.15:
            return "wander", None
        return "rest", None

    # ── Self-authored projects: open-ended goals + a plan over composable skills ──────────────
    def _home_interior(self, p):
        """The floor tiles inside a soul's home — where furniture (beds for the family) goes."""
        s = next((q for q in self.sites if q["id"] == p.get("home_struct")), None)
        if not s:
            return []
        tiles = [(t["x"], t["y"]) for t in s.get("tasks", []) if t.get("code") == BLOCK_FLOOR]
        if not tiles and s.get("core"):                    # a leaf shelter has just its core
            tiles = [tuple(s["core"])]
        return tiles

    def _home_floor(self, p):
        """The real, walled-IN floor of a soul's PROPER home (a hut/cabin/…) — anchor tile first, so
        the first bed lands where the soul actually lies down to sleep. EMPTY for a leaf lean-to: it
        has no enclosed interior to furnish, which is why beds stopped appearing in the open."""
        s = next((q for q in self.sites if q["id"] == p.get("home_struct")), None)
        if not s:
            return []
        floor = [(t["x"], t["y"]) for t in s.get("tasks", []) if t.get("code") == BLOCK_FLOOR]
        home = tuple(p.get("home", ()))
        if home in floor:                                  # sleep where you rest: the bed goes on the anchor
            floor.remove(home); floor.insert(0, home)
        return floor

    def _form_aspiration(self, p):
        """A settled, content soul DREAMS UP its own project — tidy the overgrown ground around its
        home, or plant a flower garden by the door — and lays a PLAN of primitive skill-steps to
        carry it out. Returns the plan dict, or None if there's nothing worth doing. (The rule body
        seeds these so it works with no model; the LLM mind can author richer ones into this shape.)"""
        hx, hy = p["home"]
        cur, amb = mind._trait(p, "curiosity"), mind._trait(p, "ambition")
        occ = self._occupied_tiles()
        ring = []
        for dy in range(-ASPIRE_RING, ASPIRE_RING + 1):
            for dx in range(-ASPIRE_RING, ASPIRE_RING + 1):
                tx, ty = hx + dx, hy + dy
                if (dx == 0 and dy == 0) or not self._in(tx, ty) \
                        or self.water[ty, tx] != WATER_NONE or (tx, ty) in occ:
                    continue
                ring.append((tx, ty))
        if not ring:
            return None
        overgrown = [(tx, ty) for tx, ty in ring if float(self.veg_growth[ty, tx]) > 0.3]
        bare = [(tx, ty) for tx, ty in ring
                if (tx, ty) not in self.decor and float(self.veg_growth[ty, tx]) <= 0.3]
        corners = [(hx - ASPIRE_RING, hy - ASPIRE_RING), (hx + ASPIRE_RING, hy - ASPIRE_RING),
                   (hx - ASPIRE_RING, hy + ASPIRE_RING), (hx + ASPIRE_RING, hy + ASPIRE_RING)]
        art_tiles = [(tx, ty) for tx, ty in corners
                     if (tx, ty) in bare and (tx, ty) not in self.decor]

        def build(kind, goal):
            """Ground a project KIND into an executable plan over the skill library — or None if
            it can't be done here (no bare ground for a garden, etc.). This is what lets the LLM
            author a goal in words: it picks the kind, the body composes the skills."""
            goal = (goal or "").strip() or DEFAULT_ASPIRE_GOAL.get(kind, "make my home finer")
            if kind == "art" and (art_tiles or bare):
                # A GRAND WORK, not a lone stone: a centrepiece chosen by temperament (the ambitious
                # raise a soaring OBELISK, the curious carve a TOTEM, others a STATUE) ringed by
                # standing stones — a little monument by the home for the band to marvel at.
                spots = art_tiles or bare
                cx0, cy0 = spots[0]
                # the centrepiece reflects temperament, with room for variety: the ambitious raise
                # monumental works (obelisk/arch/statue/brazier), the curious whimsical ones
                # (totem/fountain/shrine/banner), others something solid and proud.
                if amb >= 0.55:
                    palette = ["obelisk", "arch", "statue", "brazier"]
                elif cur >= 0.55:
                    palette = ["totem", "fountain", "shrine", "banner"]
                else:
                    palette = ["statue", "brazier", "banner", "totem"]
                center = palette[int(self.rng.integers(len(palette)))]
                steps = [["place", cx0, cy0, center]]
                ring = [t for t in bare if t != (cx0, cy0)][:4]
                steps += [["place", rx, ry, "cairn"] for rx, ry in ring]
                return {"goal": goal, "kind": "art", "i": 0, "steps": steps}
            if kind == "garden" and bare:
                return {"goal": goal, "kind": "garden", "i": 0,
                        "steps": [["place", tx, ty, "flower"] for tx, ty in bare[:6]]}
            if kind == "tidy" and overgrown:
                return {"goal": goal, "kind": "tidy", "i": 0,
                        "steps": [["clear", tx, ty] for tx, ty in overgrown[:ASPIRE_MAX_STEPS]]}
            if kind == "furnish":                            # furnish the home: beds first, then a hearth-table, chairs, a chest
                empty = [t for t in self._home_floor(p)      # PROPER home only (a leaf lean-to has no interior)
                         if t not in self.decor and t not in self.station_objs]
                if empty:
                    want = (["bed"] * max(1, self._household_size(p))  # a bed per soul, then the comforts of a settled home
                            + ["table", "chest", "chair", "chair"])
                    steps = [["place", tx, ty, k] for (tx, ty), k in zip(empty, want)]
                    if steps:
                        return {"goal": goal, "kind": "furnish", "i": 0, "steps": steps}
            return None

        # An LLM-AUTHORED project (the mind's own reasoned goal — see mind.apply_deliberation) takes
        # precedence when the model has set one and it can actually be carried out here.
        lp = p.pop("llm_project", None)
        if isinstance(lp, dict):
            plan = build(str(lp.get("kind", "")).strip().lower(), lp.get("goal"))
            if plan:
                return plan
        # OFFLINE / fallback — choose by TASTE: ambitious raise standing stones, curious plant a
        # garden, the orderly tidy the thicket. Each weighed by temperament, the best-fit chosen.
        weights = {"furnish": 0.34 + 0.10 * self._household_size(p),   # a soul with family wants beds most
                   "art": 0.30 + 0.55 * amb, "garden": 0.30 + 0.55 * cur, "tidy": 0.30 + 0.35 * (1.0 - cur)}
        for kind in sorted(weights, key=lambda k: -weights[k]):
            plan = build(kind, None)
            if plan:
                return plan
        return None

    def _pursue_aspiration(self, p):
        """Actuate a self-authored project: form one if none, then run its next plan-step."""
        plan = p.get("plan")
        if not plan or plan.get("done"):
            plan = self._form_aspiration(p)
            if not plan:
                p["aspire_cd"] = self.clock + ASPIRE_COOLDOWN
                p.pop("plan", None)
                return self._idle(p)
            p["plan"] = plan
            mind.remember(p, f"I've a mind to {plan['goal']}", 0.4, "intent", self.clock)
        steps, i = plan["steps"], plan.get("i", 0)
        if i >= len(steps):
            self._complete_aspiration(p, plan)
            return self._idle(p)
        step = steps[i]
        skill, tx, ty = step[0], step[1], step[2]
        if max(abs(tx - p["x"]), abs(ty - p["y"])) > 1:      # walk to the spot
            return "wander", (tx - p["x"], ty - p["y"])   # raw delta → pathfind to the plan spot
        if skill == "clear":                                 # primitive skills the body composes
            self._clear_ground(tx, ty, 0)
        elif skill == "place":                               # plant a flower / raise a cairn / set furniture
            kind = step[3] if len(step) > 3 else "flower"
            cost = FURNITURE_COST.get(kind)                  # furniture is MADE from materials, not conjured
            if cost and not all(p["inv"].get(m, 0) >= n for m, n in cost.items()):
                plan["i"] = i + 1                            # short the makings — skip this piece, don't fake it
                return "tend", None
            if cost:
                for m, n in cost.items():
                    p["inv"][m] = p["inv"].get(m, 0) - n
                    if p["inv"][m] <= 0:
                        del p["inv"][m]
            self.decor[(tx, ty)] = kind
            self.version += 1
        plan["i"] = i + 1
        self._bump("aspire_step")
        return "tend", None

    def _complete_aspiration(self, p, plan):
        """A finished project — a small pride, a finer home, and a cooldown before the next."""
        p.pop("plan", None)
        p["aspire_cd"] = self.clock + ASPIRE_COOLDOWN
        self._bump("aspire_done")
        done = {"garden": "planted a garden by their door",
                "art": "raised a grand work — a wonder of their own making",
                "tidy": "tidied the ground around their home",
                "furnish": "made beds for their family"}.get(plan.get("kind"), "made their home finer")
        mind.remember(p, f"I {plan['goal']} — my home is finer for it", 0.6, "pride", self.clock)
        grand = plan.get("kind") == "art"
        mind.speak(p, "Let them come and marvel at what I've raised!" if grand
                   else "There — a finer place to call my own.", self.clock)
        if grand:                                            # a wonder lifts standing more than tidying does
            self._earn_renown(p, RENOWN_GAIN.get("build", 0.08), "raised a wonder the band admires")
        self._note("life", f"{p['name']} {done}.")
        self._earn_renown(p, RENOWN_GAIN.get("gift", 0.05) * 0.5, "made a finer home — a quiet pride")

    def _homeward_if_away(self, p, reach: int = 4):
        """A step toward home if a SETTLED soul has strayed from it — so making and puzzling-out
        (craft / research) happen back in the safe settlement, not alone in the dangerous wild.
        None when the soul is homeless (it works where it is) or already home."""
        if not p.get("home_struct"):
            return None
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) <= reach:
            return None
        return "wander", (hx - p["x"], hy - p["y"])       # raw delta → pathfind home

    def _flee(self, p):
        """Bolt for safety: home is shelter and the band is protection, so run there. Once on
        the home tile, hunker down (rest) — a roof is the safest place to be."""
        hx, hy = p["home"]
        x, y = p["x"], p["y"]
        if abs(hx - x) + abs(hy - y) > 0:
            return "flee", (hx - x, hy - y)
        return "rest", None

    def _guard(self, p):
        """Stand watch over the band-mate most in a wolf's sights: stride to their side and
        hold there (the body's _guardian_near check then shields them and warns the wolf off).
        With no one left to ward, fall back to idle — the threat has passed."""
        x, y = p["x"], p["y"]
        ward, wd = None, GUARD_RANGE + 1
        for q in self.people:
            if q is p:
                continue
            t = q.get("_wolf_threat", 0.0)
            if t <= 0:
                continue
            d = abs(q["x"] - x) + abs(q["y"] - y)
            if d <= GUARD_RANGE and t > 0 and d < wd:
                ward, wd = q, d
        if ward is None:
            return self._idle(p)
        if wd > 1:                                   # close in to put myself between ward and wolf
            return "guard", (ward["x"] - x, ward["y"] - y)   # raw delta → pathfind to the ward
        return "guard", None                         # at their side — hold the watch

    def _person_build_decide(self, p, tree, stone, fiber, leaf, lx, ly):
        """The crafting/building drive (a comfortable person's 'project'). Returns a body
        action (craft / chop / gather_* / found_site / build_block / haul / a seek_* move
        toward a resource), or None. People raise a real tile-by-tile building from a
        blueprint: found the footprint, then forage the materials and lay one tile per turn.
        The first home is a quick leaf lean-to — the cheapest shelter there is."""
        x, y = p["x"], p["y"]
        inv = p["inv"]
        hx, hy = p["home"]

        # Already mid-craft? Keep at it (the item takes in-world time to finish). The drive
        # arbiter still owns this person — if a need bites, survival wins the next deliberation
        # and they break off here; the half-done craft waits, its inputs already reserved.
        if p.get("craft"):
            return "craft", None

        getters = {
            "wood":   lambda: self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly,
                                         "wood", "chop", "seek_wood"),
            "fiber":  lambda: self._seek(p, x, y, bool(fiber[ly, lx]), fiber, lx, ly,
                                         "fiber", "gather_fiber", "seek_fiber"),
            "leaves": lambda: self._seek(p, x, y, bool(leaf[ly, lx]), leaf, lx, ly,
                                         "leaves", "gather_leaves", "seek_leaves"),
            "stone":  lambda: self._seek(p, x, y, bool(stone[ly, lx]), stone, lx, ly,
                                         "stone", "mine", "seek_stone"),
        }

        # Once a TOOLMAKER has worked out the axe (by chopping wood by hand — see the chop
        # handler), they fashion one; it makes every later chop yield more. A non-toolmaker can't
        # make their own — they build and gather bare-handed until a toolmaker gifts them one.
        if inv.get("axe", 0) < 1 and self._person_knows(p, "crude_axe") and self._can_make_tools(p):
            if inv.get("wood", 0) >= BUILD["axe_wood"]:
                return "craft", None
            return getters["wood"]()

        # First home: build a quick leaf lean-to, tile by tile, from its blueprint.
        if p.get("home_struct") is None:
            return self._pursue_building(p, "leaf_shelter", getters)

        # Home raised — now make any DISCOVERED make-shift survival gear they still lack
        # (the water flask first; it's what frees them from the riverbank).
        gear = self._survival_craft_decide(p, fiber, leaf, getters)
        if gear:
            return gear

        # A soul who has learned that raw water sickens raises a HEARTH at home to boil it safe.
        hearth = self._hearth_decide(p, getters)
        if hearth:
            return hearth

        # The life-project: climb the dwelling ladder to a snugger home (hut, then cabin).
        # This is what fills the once-empty hours after survival is met — a real, visible goal.
        proj = p.get("project")
        if proj and proj.get("kind") == "dwelling":
            act = self._pursue_building(p, proj["bp"], getters, rung=proj.get("rung"))
            if act:
                return act
        # The status project: an ambitious soul whose own home is fine raises a communal
        # monument for the band — a visible bid for lasting standing (Phase 2). If a half-built
        # hall was orphaned by its raiser's death, adopt and finish it rather than starting anew.
        if proj and proj.get("kind") == "monument":
            if self._person_site(p) is None:
                # Join an in-progress communal build (the crew), or adopt an orphaned one, before
                # founding a fresh hall — so the band raises ONE monument together, not many alone.
                join = self._communal_build_to_join(p) or self._orphaned_monument()
                if join is not None:
                    p["site"] = join["id"]
            act = self._pursue_building(p, proj["bp"], getters, communal=True)
            if act:
                return act
        # A COMMISSION: the builder raises a whole home FOR a client (owner = the client), built in
        # the builder's own settlement; the client moves in and pays the builder when it's done.
        if proj and proj.get("kind") == "commission":
            client = next((q for q in self.people if q["id"] == proj.get("client")), None)
            # Client gone, or already housed properly some other way → abandon the job and release any
            # half-built site, so the builder is never stuck forever on a commission it can't finish.
            if client is None or self._current_dwelling_bp(client) in ("hut", "cabin"):
                p.pop("project", None)
                site = self._person_site(p)
                if site is not None and site.get("commission"):
                    p.pop("site", None)
            else:
                act = self._pursue_building(p, proj["bp"], getters, client=client)
                if act:
                    return act

        # With home, gear, hearth and dwelling all handled, a settled soul beside a station turns
        # to the DEEP craft tree — planks, tools, cooking, pottery, and (with ore to hand) metal.
        # The whole tree is now open to the band, gated by stations/tools/materials, not knowledge.
        tech = self._tech_craft_decide(p, getters)
        if tech:
            return tech

        # Otherwise lay in stone for the next slice (stone houses, workshops).
        if inv.get("stone", 0) < BUILD["stone_stock"]:
            return self._seek(p, x, y, bool(stone[ly, lx]), stone, lx, ly,
                              "stone", "mine", "seek_stone")
        return None

    def _hearth_decide(self, p, getters):
        """A soul who has learned raw water sickens raises a HEARTH (campfire) at home to boil it
        clean: gather the materials, then lay it. Returns a body action, or None if not needed."""
        if not self._person_knows(p, "campfire") or p.get("hearth"):
            return None
        inv = p["inv"]
        for mat, need in HEARTH_COST.items():
            if inv.get(mat, 0) < need:
                return getters[mat]()                         # short on a material — go get it
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) > 1:           # have it all — bring it home and build
            return "haul", (hx - p["x"], hy - p["y"])    # raw delta → the mover pathfinds around walls
        for mat, need in HEARTH_COST.items():
            inv[mat] = inv.get(mat, 0) - need
            if inv[mat] <= 0:
                inv.pop(mat, None)
        p["hearth"] = True
        self._add_structure("campfire", hx, hy, by=p["name"])
        mind.remember(p, "raised a hearth — now I can boil my water clean", 0.8, "build", self.clock)
        return "tend", None

    def _boil_at_home(self, p):
        """Tend the home fire: convert a measure of raw flask water into SAFE (boiled) water, up
        to the capacity of the vessels carried. Cheap, runs each rest-tick beside the hearth."""
        inv = p["inv"]
        if inv.get("water", 0) <= 0:
            return
        cap = sum(crafting.CONTAINER_WATER.get(c, 0) * inv.get(c, 0) for c in crafting.CONTAINER_WATER)
        if cap <= 0 or inv.get("safe_water", 0) >= cap:        # need a vessel to hold boiled water
            return
        inv["water"] -= 1
        if inv["water"] <= 0:
            inv.pop("water", None)
        inv["safe_water"] = inv.get("safe_water", 0) + 1

    # ── the home larder (P2 storage + fetch) ───────────────────────────────────
    def _can_access_store(self, fetcher, owner) -> bool:
        """Whether `fetcher` may draw from `owner`'s home store. For now a soul owns its own
        larder outright; the grant map (`store_access`, keyed by id) is the seam the coming
        household-sharing and temporary-lending phases hang on — a partner or a soul handed a
        standing errand will read as accessible here without the call-sites changing."""
        if fetcher is owner or fetcher["id"] == owner["id"]:
            return True
        grant = owner.get("store_access", {}).get(fetcher["id"])
        if grant is None:
            return False
        return grant in (True, "always") or (isinstance(grant, (int, float)) and self.clock < grant)

    def _granary_store(self):
        """The band's shared common store, or None until it exists. It comes into being — at the
        centre of the settled homes — once GRANARY_MIN_HOUSED souls have roofs, the moment a band
        is established enough to keep a commons. Returns the store dict to deposit into / draw from."""
        g = self.granary
        if g.get("x") is not None:
            self._relocate_granary_if_stranded()         # self-heal a stranded/under-water granary
        if g.get("x") is None:
            housed = [q for q in self.people if q.get("home_struct")]
            if len(housed) >= GRANARY_MIN_HOUSED:
                cx = int(sum(q["home"][0] for q in housed) / len(housed))
                cy = int(sum(q["home"][1] for q in housed) / len(housed))
                gx, gy = self._dry_spot_near(cx, cy)     # the centroid can fall in a lake — nudge to dry land
                g["x"], g["y"] = gx, gy
                self._add_structure("granary", gx, gy, by="the band")
                self._note("build", "The band raised a common granary — a shared store against lean days.")
            else:
                return None
        return g["store"]

    def _dry_spot_near(self, cx, cy):
        """The nearest buildable tile to (cx,cy): on land, set back from water (flood-shy) and
        not already built on. Spirals outward; falls back to merely-dry, then to the point itself,
        so a granary/structure never lands in the water even when the home centroid does."""
        occ = self._occupied_tiles()

        def ok(x, y, buffered):
            if not self._in(x, y) or self.water[y, x] != WATER_NONE or (x, y) in occ:
                return False
            return not (buffered and self._water_within(x, y, WATER_BUILD_BUFFER))

        for buffered in (True, False):                   # prefer a flood-shy spot, else any dry land
            for r in range(0, 24):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if max(abs(dx), abs(dy)) != r:
                            continue
                        if ok(cx + dx, cy + dy, buffered):
                            return cx + dx, cy + dy
        return cx, cy

    def _deposit_home(self, p):
        """Resting at home, a soul banks the surplus food/water it is carrying above its travel
        reserve into the larder — so a good forage outlives the day. Only survival consumables
        are stored (building stock and gear stay on-person where the build logic expects them).
        A well-stocked larder then OVERFLOWS into the band's common granary, so a forager's
        bounty feeds the children, the sick and the specialists who cannot lay in their own."""
        inv, store = p["inv"], p.setdefault("store", {})
        for key, keep in STORE_KEEP.items():
            spare = inv.get(key, 0) - keep
            if spare > 0:
                store[key] = store.get(key, 0) + spare
                inv[key] = keep
                if inv[key] <= 0:
                    inv.pop(key, None)
        gst = self._granary_store()
        if gst is not None:
            for key, cushion in GRANARY_CUSHION.items():       # overflow above a personal cushion → the commons
                spare = store.get(key, 0) - cushion
                if spare > 0:
                    gst[key] = gst.get(key, 0) + spare
                    store[key] = cushion
                    p["gran_given"] = p.get("gran_given", 0.0) + spare   # pulling their weight (norm)

    def _prefer_store(self, p, want):
        """Decide whether to fall back on the larder rather than forage. Only when the store
        actually holds it, home is within ranging distance, AND home is no farther than any
        remembered wild spot — so a needy soul is never marched PAST nearer water/food to the
        larder (that detour, overriding a closer known spring, quietly cost thirst deaths)."""
        store = p.get("store", {})
        gst = self.granary.get("store") if self.granary.get("x") is not None else {}
        if want == "food":
            stocked = store.get("food", 0) > 0 or gst.get("food", 0) > 0
        else:
            stocked = (store.get("safe_water", 0) + store.get("water", 0)
                       + gst.get("safe_water", 0)) > 0
        if not stocked:
            return False
        hx, hy = p["home"]
        dh = abs(hx - p["x"]) + abs(hy - p["y"])
        if dh > EXPLORE_LEASH:                       # too far to march a needy soul home
            return False
        kloc = p.get("known", {}).get("food" if want == "food" else "water")
        if kloc:
            dk = abs(kloc[0] - p["x"]) + abs(kloc[1] - p["y"])
            if dk < dh:                              # a remembered wild spot is nearer — use it
                return False
        return True

    def _fetch_from_store(self, p, want):
        """Caught hungry/thirsty away from home with nothing in sight, head for the larder and
        draw a unit. `want` is 'food' or 'water'. Returns a body action (haul toward home, or the
        consume action once home and the unit is in the pack), or None when the store can't help."""
        store = p.get("store", {})
        gst = self.granary.get("store") if self.granary.get("x") is not None else None
        keys = ["food"] if want == "food" else ["safe_water", "water"]   # boiled water first
        has_personal = any(store.get(k, 0) > 0 for k in keys)
        has_common = gst is not None and any(gst.get(k, 0) > 0 for k in (keys if want == "food" else ["safe_water"]))
        if not has_personal and not has_common:
            return None
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) > 1:                      # still on the way home
            return "haul", (hx - p["x"], hy - p["y"])    # raw delta → the mover pathfinds around walls
        for k in keys:                                                   # at the larder — withdraw one
            if store.get(k, 0) > 0:
                store[k] -= 1
                if store[k] <= 0:
                    store.pop(k, None)
                p["inv"][k] = p["inv"].get(k, 0) + 1
                return ("eat" if want == "food" else ("drink_safe" if k == "safe_water" else "drink_pack")), None
        # Own larder is bare — lean on the common granary the band keeps.
        for k in (keys if want == "food" else ["safe_water"]):
            if gst and gst.get(k, 0) > 0:
                gst[k] -= 1
                if gst[k] <= 0:
                    gst.pop(k, None)
                p["inv"][k] = p["inv"].get(k, 0) + 1
                p["gran_taken"] = p.get("gran_taken", 0.0) + 1          # leaning on the commons (norm)
                mind.remember(p, "drew from the common store when my own ran bare", 0.5, "social", self.clock)
                return ("eat" if want == "food" else "drink_safe"), None
        return None

    def _pursue_building(self, p, bp_name, getters, communal: bool = False, rung=None, client=None):
        """Raise a building from a blueprint tile by tile: found the footprint at home, then
        forage each tile's material and lay it. Returns a body action, or None when there's
        nothing to do this beat (between steps, or just finished). Shared by the first
        lean-to, every dwelling-ladder upgrade, the communal monument, and a COMMISSION (a home
        built for a `client`, who becomes its owner)."""
        x, y = p["x"], p["y"]
        inv = p["inv"]
        hx, hy = p["home"]
        site = self._person_site(p)
        if site is None:                                       # no footprint yet — lay one at home
            if p.get("build_cd", 0) > self.clock:              # nowhere to build last time; bide
                return None
            if abs(hx - x) + abs(hy - y) <= 1:
                p["next_bp"] = bp_name                          # the handler founds this blueprint
                p["next_communal"] = communal
                p["next_rung"] = rung
                p["next_client"] = client["id"] if client else None   # owner-to-be, for a commission
                return "found_site", None
            return "haul", (hx - x, hy - y)              # raw delta → the mover pathfinds around walls
        task = self._site_next_task(site)
        if task is None:                                       # all tiles placed → finish it
            self._finish_site(p, site)
            return None
        item, qty = task["cost"]
        if inv.get(item, 0) >= qty:                            # have the material — go lay it
            if max(abs(task["x"] - x), abs(task["y"] - y)) <= 2:
                return "build_block", None
            return "haul", (task["x"] - x, task["y"] - y)   # raw delta → pathfind around walls
        return getters.get(item, getters["wood"])()

    def _current_dwelling_bp(self, p):
        """The blueprint name of the soul's current finished home, or None if they've no roof
        yet. Looked up from the site `home_struct` points at (finished sites are kept)."""
        sid = p.get("home_struct")
        if not sid:
            return None
        for s in self.sites:
            if s["id"] == sid:
                return s.get("rung") or s.get("bp")     # the ladder rung (a designed home reports its rung)
        return None

    def _orphaned_monument(self):
        """An in-progress communal build no living soul is still raising — a monument whose
        raiser died, or a building the GOD marked out (site-mode template placement) — free for
        an ambitious builder to adopt and finish, so a half-built hall is never abandoned forever."""
        live_sites = {q.get("site") for q in self.people}
        for s in self.sites:
            if s["done"] or s["id"] in live_sites:
                continue
            if s["bp"] == MONUMENT_BP or s.get("orphan"):
                return s
        return None

    def _communal_in_progress(self) -> bool:
        """Is any communal monument currently being raised (unfinished)?"""
        return any((s["bp"] == MONUMENT_BP or s.get("communal")) and not s["done"] for s in self.sites)

    def _site_crew(self, site_id) -> list:
        """The living builders currently assigned to a site."""
        return [q for q in self.people if q.get("site") == site_id]

    def _communal_build_to_join(self, p):
        """An in-progress communal build whose crew isn't yet full and is nearest to `p` — so an
        ambitious soul joins the group effort rather than each starting their own hall."""
        best, bd = None, 1e9
        for s in self.sites:
            if s["done"] or not (s["bp"] == MONUMENT_BP or s.get("communal")):
                continue
            if len(self._site_crew(s["id"])) >= CO_OP_CREW_MAX:
                continue
            d = abs(s["ox"] - p["x"]) + abs(s["oy"] - p["y"])
            if d < bd:
                best, bd = s, d
        return best

    def _site_crew_present(self, site) -> int:
        """How many of the site's crew are ON HAND (near the footprint) — the bodies available to
        lay a heavy communal tile this beat."""
        ox, oy = site["ox"], site["oy"]
        return sum(1 for q in self._site_crew(site["id"])
                   if abs(q["x"] - ox) + abs(q["y"] - oy) <= CO_OP_RANGE)

    def _household_size(self, p) -> int:
        """How many mouths share this soul's home: themself, a partner, and any still-young
        children — what a home needs to be sized for."""
        n = 1 + (1 if p.get("partner") else 0)
        kids = p.get("children", [])
        if kids:
            ages = {q["id"]: q["age"] for q in self.people}
            n += sum(1 for c in kids if ages.get(c, ADULT_AGE) < ADULT_AGE)
        return n

    def _design_leaf_home(self, p, fam: int) -> str:
        """An all-LEAF home — leaf-panel walls round a roofed core, no wood at all. What a soul
        raises when timber is out of reach (or simply preferred): humble and draughtier than a
        timber house, but a real, coherent home built entirely from leaves."""
        iw = max(1, min(3, 1 + fam // 2))
        ih = max(1, min(3, 1 + fam // 2))
        Wt, Ht = iw + 2, ih + 2
        rows = [["L" if (rx in (0, Wt - 1) or ry in (0, Ht - 1)) else "."
                 for rx in range(Wt)] for ry in range(Ht)]
        rows[Ht // 2][Wt // 2] = "C"                           # the roofed home core
        rows[0][Wt // 2] = "."                                 # an open doorway in the leaf wall
        layout = ["".join(r) for r in rows]
        bid = f"home_{p['id']}"
        BLUEPRINTS[bid] = dict(name=f"{p['name']}'s Leaf House", roof=True, insulation=0.45,
                               roof_cost=("leaves", 1), layout=layout)
        return bid

    def _home_template_for(self, p):
        """A god-authored home template to raise as a PRIOR — so the band builds the GOD's designs,
        not just the parametric default. Picks the smallest non-communal template roomy enough for
        the household; None when there's none suitable (or it's a timber design and no wood is to
        hand). This is the long-promised 'templates as priors': the god's blueprints become the
        forms a soul reaches for first. The band's OWN (LLM-authored) designs count too — Phase A —
        so a soul reaches for a home its people designed, not only the god's or the parametric one."""
        priors = self.user_blueprints + self.authored_blueprints   # the god's designs AND the band's own
        if not priors:
            return None
        fam = self._household_size(p)
        hx, hy = p["home"]
        wood_ok = p["inv"].get("wood", 0) >= 4 or self._wood_within(hx, hy, WOOD_BUILD_RANGE)
        best, best_floor = None, None
        roomy_cap = fam * 3 + 6                                 # a home shouldn't be MUCH bigger than the
        # household needs — a family of two doesn't raise a 100-tile hall just because the god drew
        # one. Templates wildly oversized for this soul are skipped; the parametric designer (which
        # sizes to the household) then builds an appropriately modest home.
        for ub in priors:
            if ub.get("communal"):
                continue
            layout = ub.get("layout") or []
            floor = sum(row.count("F") + row.count(GLYPH_CORE) for row in layout)
            if floor < max(1, fam) or floor > roomy_cap:       # too small OR absurdly oversized
                continue
            timber = any(("W" in r or "F" in r or "D" in r) for r in layout)
            if timber and not wood_ok:                         # a timber template but no wood to raise it
                continue
            if best is None or floor < best_floor:
                best, best_floor = ub["id"], floor
        return best

    def _validate_blueprint(self, layout) -> tuple:
        """Lint a building layout: it must enclose at least one floor/core tile, every such tile
        must be reachable FROM OUTSIDE (through doors or open/leaf panels — no walled-off room a
        soul could never enter), and the footprint must be a sane size. The guard that keeps a
        generated or god-drawn building actually usable. Returns (ok: bool, reason: str)."""
        if not layout:
            return False, "empty layout"
        Hh = len(layout)
        Ww = max(len(r) for r in layout)
        grid = [r.ljust(Ww, ".") for r in layout]
        solid = {"W", "O"}                                # walls & windows block; leaf is soft-passable
        floors = [(x, y) for y in range(Hh) for x in range(Ww) if grid[y][x] in ("F", "C")]
        if not floors:
            return False, "no floor"
        if Hh * Ww > 144 or len(floors) > 60:
            return False, "oversized footprint"
        # Flood from OUTSIDE the box inward through every non-solid tile; each floor must be touched.
        seen = {(-1, -1)}
        q = [(-1, -1)]
        qi = 0
        while qi < len(q):
            x, y = q[qi]
            qi += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in seen or not (-1 <= nx <= Ww and -1 <= ny <= Hh):
                    continue
                if 0 <= nx < Ww and 0 <= ny < Hh and grid[ny][nx] in solid:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
        if not all(f in seen for f in floors):
            return False, "a room is walled off from outside"
        return True, "ok"

    def _partition_rooms(self, rows, x0, y0, x1, y1, rooms_left, rng) -> int:
        """Recursively split an interior rect (glyph coords, inclusive) into rooms — a BSP partition.
        Each split lays an inner WALL across the longer axis with a single DOORWAY, so the rooms form
        a tree joined by doors and every room stays reachable. Returns the count of leaf rooms made."""
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if rooms_left <= 1 or (w < 4 and h < 4):           # enough rooms, or too small to halve
            return 1
        if w >= h and w >= 4:                              # a vertical wall (home is wider here)
            cut = x0 + int(rng.integers(2, w - 1))         # ≥1 floor column survives each side
            for yy in range(y0, y1 + 1):
                rows[yy][cut] = "W"
            rows[y0 + (y1 - y0) // 2][cut] = "D"           # a doorway through the partition
            a = self._partition_rooms(rows, x0, y0, cut - 1, y1, (rooms_left + 1) // 2, rng)
            b = self._partition_rooms(rows, cut + 1, y0, x1, y1, rooms_left // 2, rng)
            return a + b
        if h >= 4:                                         # a horizontal wall
            cut = y0 + int(rng.integers(2, h - 1))
            for xx in range(x0, x1 + 1):
                rows[cut][xx] = "W"
            rows[cut][x0 + (x1 - x0) // 2] = "D"
            a = self._partition_rooms(rows, x0, y0, x1, cut - 1, (rooms_left + 1) // 2, rng)
            b = self._partition_rooms(rows, x0, cut + 1, x1, y1, rooms_left // 2, rng)
            return a + b
        return 1

    def _design_dwelling(self, p, rung: str, client=None) -> str:
        """A soul DESIGNS a home. For its OWN, a fitting god template is used if there is one;
        otherwise it generates one parametrically — a timber room sized to the household and shaped
        by the BUILDER's skill, or an all-leaf house when wood is scarce. With `client` set the home
        is designed for THAT soul instead (a commission): sized to the client's household, keyed to
        them, and always timber (a proper home is a real upgrade — the builder gathers the wood)."""
        if client is None:
            prior = self._home_template_for(p)
            if prior:
                self._bump("template_prior")
                return prior
        who = client if client is not None else p              # the home's future OWNER (sizes the household)
        fam = self._household_size(who)
        skill = (p.get("skills") or {}).get("building", 0.0)   # but the BUILDER's hands shape it
        # Material is chosen for the WHOLE house, never mixed: build in timber only when wood is
        # actually to hand (carried, or trees near home); otherwise raise a coherent all-LEAF house
        # rather than a half-finished timber one waiting on wood that isn't coming.
        hx, hy = p["home"]
        timber = p["inv"].get("wood", 0) >= 4 or self._wood_within(hx, hy, WOOD_BUILD_RANGE)
        if not timber and client is None:
            return self._design_leaf_home(p, fam)
        if rung == "hut":                                       # a modest first proper home
            iw = max(1, min(3, 1 + fam // 2))
            ih = max(1, min(3, 1 + fam // 2))
        else:                                                   # a cabin (or finer) — larger, roomier
            iw = max(2, min(4, 2 + fam // 2 + int(skill * 1.5)))
            ih = max(2, min(5, 2 + fam // 2 + int(skill * 2.0)))
        Wt, Ht = iw + 2, ih + 2                                 # +2 for the wall ring
        rows = [["W" if (rx in (0, Wt - 1) or ry in (0, Ht - 1)) else "F"
                 for rx in range(Wt)] for ry in range(Ht)]
        # A roomier household earns more ROOMS: a skilled builder partitions the interior by BSP
        # (a hearth room, sleeping rooms), each joined to the next by an inner doorway so the home
        # stays walkable. Small homes stay a single open room. Scales 2→4 rooms with the family.
        rooms_target = 1
        if fam >= 4 and skill > 0.25 and (ih >= 4 or iw >= 4):
            rooms_target = max(2, min(4, (fam + 2) // 2))
        rooms = self._partition_rooms(rows, 1, 1, Wt - 2, Ht - 2, rooms_target, self.rng) \
            if rooms_target > 1 else 1
        # A front door in the top wall, set where it opens onto FLOOR (never into an inner wall).
        cols = [cx for cx in range(1, Wt - 1) if rows[1][cx] == "F"]
        dcol = min(cols, key=lambda cx: abs(cx - Wt // 2)) if cols else Wt // 2
        rows[0][dcol] = "D"
        if skill > 0.3 and Ht >= 4:                            # a skilled hand adds windows to the long walls
            rows[Ht // 2][0] = "O"
            rows[Ht // 2][Wt - 1] = "O"
        layout = ["".join(r) for r in rows]
        kind = "Hut" if rung == "hut" else "Cabin"
        nm = f"{who['name']}'s {kind}" + (f" ({rooms} rooms)" if rooms > 1 else "")
        bid = f"home_{who['id']}"                               # keyed to the OWNER (a commission won't clobber the builder's own)
        BLUEPRINTS[bid] = dict(name=nm, roof=True, insulation=1.0, layout=layout)
        return bid

    def _commission_client(self, builder):
        """The nearest leaf-sheltered adult a settled BUILDER could offer to raise a proper home
        for — or None if there's no one to build for (or someone's already on it). Targets only
        souls who've already got a humble shelter (never the homeless-survival path), so a
        commission is a pure UPGRADE for a fee and can never leave a soul worse off."""
        bx, by = builder["x"], builder["y"]
        claimed = {s.get("owner") for s in self.sites if s.get("commission") and not s.get("done")}
        best, best_d = None, COMMISSION_RANGE + 1
        for q in self.people:
            if q is builder or q["id"] in claimed or q.get("age", 0) < ADULT_AGE:
                continue
            if self._current_dwelling_bp(q) != "leaf_shelter":   # only upgrade a humble leaf shelter
                continue
            d = abs(q["x"] - bx) + abs(q["y"] - by)
            if d < best_d:
                best, best_d = q, d
        return best

    def _project_for(self, p):
        """The soul's standing life-project once survival & first shelter are met: climb the
        dwelling ladder to a snugger home. Returns a project dict {kind, bp, why} or None
        (no home yet, or already at the top rung). The mind weighs this against company,
        wandering and rest; an unambitious, content soul may simply linger in their hut."""
        if p.get("home_struct") is None:
            return None                                        # first shelter is the survival path's job
        cur = self._current_dwelling_bp(p)
        try:
            i = DWELLING_LADDER.index(cur)
        except ValueError:
            i = len(DWELLING_LADDER) - 1                       # unknown/custom home — treat as topped out
        # The status project: once a soul has a real home (a hut or better), an AMBITIOUS one
        # may turn to raising the band's communal monument — choosing lasting standing over a
        # finer house of their own. It takes precedence over further nesting; less driven souls
        # keep climbing the dwelling ladder instead.
        mon = {"kind": "monument", "bp": self._authored_for("hall") or MONUMENT_BP,
               "why": "raise a gathering hall — a place for us all, and a name that lasts"}
        has_real_home = i >= DWELLING_LADDER.index("hut")
        # Already mid communal build (hall OR workshop)? Keep at it rather than switching projects.
        my_site = next((s for s in self.sites if s["id"] == p.get("site") and not s.get("done")), None)
        if my_site is not None and my_site.get("communal"):
            return {"kind": "monument", "bp": my_site["bp"], "why": "finish what we're raising together"}
        # Already raising a home FOR ANOTHER (a commission)? See it through.
        if my_site is not None and my_site.get("commission"):
            return {"kind": "commission", "bp": my_site["bp"], "client": my_site.get("owner"),
                    "why": "finish the home I'm raising for them"}
        # A TOOLMAKER's specialty project: raise the communal WORKSHOP (the band's bench) if there
        # isn't one yet — gear, tools and everyone's crafting go faster beside it. It's their
        # answer to the ambitious soul's gathering hall.
        voc = p.get("vocation") or mind.vocation(p)
        band_has_shop = self._has_function("workshop")
        if (voc == "toolmaker" and has_real_home and not band_has_shop
                and (self._communal_build_to_join(p) is not None or not self._communal_in_progress())):
            return {"kind": "monument", "bp": self._authored_for("workshop") or WORKSHOP_BP,
                    "why": "raise a workshop — a bench for us all, and faster hands"}
        # Once the band has a workshop, a toolmaker raises the SMITHY — the metalworking shop that
        # opens smelting and metal tools (the deep tree). Their crowning specialty build.
        if (voc == "toolmaker" and has_real_home and band_has_shop and not self._has_function("smithy")
                and (self._communal_build_to_join(p) is not None or not self._communal_in_progress())):
            return {"kind": "monument", "bp": self._authored_for("smithy") or SMITHY_BP,
                    "why": "raise a smithy — to smelt ore and forge true metal tools"}
        # A FORAGER's specialty: raise the communal STOREHOUSE so the band's food keeps against
        # lean days (slower spoilage, fewer vermin) — their answer to the bench and the hall.
        if (voc == "forager" and has_real_home and not self._has_function("storehouse")
                and (self._communal_build_to_join(p) is not None or not self._communal_in_progress())):
            return {"kind": "monument", "bp": self._authored_for("storehouse") or STOREHOUSE_BP,
                    "why": "raise a storehouse — to keep our food against lean days"}
        # A BUILDER's calling: OFFER to raise a proper home for a leaf-sheltered neighbour, paid a
        # fee on completion — the deliberate labour market ("build a home for others for something in
        # return"). Their answer to the toolmaker's bench and the forager's storehouse, and only when
        # not tied up in a communal raise. Designed to the CLIENT's household, in the builder's skill.
        if voc == "builder" and has_real_home and not self._communal_in_progress():
            client = self._commission_client(p)
            if client is not None:
                bp_id = self._design_dwelling(p, "hut", client=client)
                return {"kind": "commission", "bp": bp_id, "client": client["id"],
                        "why": f"raise {client['name']} a proper home — for a fair price"}
        # PUBLIC WORKS — a settled soul turns its hand to whatever the band most needs built (a well,
        # an inn, a watchtower). Same co-op gating as any monument: a real home first, and a crew.
        if has_real_home and (self._communal_build_to_join(p) is not None or not self._communal_in_progress()):
            wants = self._community_wants(p)
            if wants:
                return wants[0]
        band_has_hall = self._has_function("hall")
        # A communal build is a GROUP effort: an ambitious soul undertakes one, OR joins an
        # in-progress one whose crew isn't full — so the band's builders converge to raise it
        # together (a lone soul can't lay its heavy tiles; see _build_next_block). Only begun when
        # the band actually has a crew's worth of ambitious builders, so no one commits to a hall
        # they could never raise alone.
        ambitious_housed = sum(1 for q in self.people if q.get("home_struct")
                               and mind._trait(q, "ambition") >= AMBITION_MONUMENT)
        if (has_real_home and not band_has_hall and ambitious_housed >= CO_OP_MIN
                and mind._trait(p, "ambition") >= AMBITION_MONUMENT
                and (self._communal_build_to_join(p) is not None or not self._communal_in_progress())):
            return mon
        # Otherwise keep climbing the dwelling ladder to a snugger home — but the soul now DESIGNS
        # that home itself, sized to its household and shaped by its building skill, rather than
        # raising a one-size-fits-all blueprint.
        if i + 1 < len(DWELLING_LADDER):
            nxt = DWELLING_LADDER[i + 1]
            bp_id = self._design_dwelling(p, nxt)
            return {"kind": "dwelling", "bp": bp_id, "rung": nxt,
                    "why": f"design and raise a {BLUEPRINTS[bp_id]['name'].lower()} to fit my own"}
        return None

    # Survival gear the band has discovered, in priority order, with what each does and the
    # raw it ultimately comes from (rope is made from fiber on the spot).
    _GEAR = (("leaf_flask", "water", "leaves"), ("forage_sack", "sack", "fiber"),
             ("sleeping_mat", "mat", "fiber"))
    # Tools only a TOOLMAKER fashions (recipe id, inv key) — so builders & hunters depend on the
    # toolmaker for their gear, who keeps spares to hand round (the heart of tool-gating).
    _TOOLS = (("crude_axe", "axe"), ("crude_spear", "crude_spear"))

    def _can_make_tools(self, p) -> bool:
        """Only a toolmaker has the craft to fashion proper tools (axe/spear/rod). Everyone else
        depends on them — a builder chops bare-handed (slower) until gifted an axe, a hunter needs
        a spear from the maker's bench. This is what makes the toolmaker's calling matter."""
        return (p.get("vocation") or mind.vocation(p)) == "toolmaker"

    MATERIAL_GIFT_SPARE = 3      # raw a giver keeps for itself before handing the surplus to a maker
    GIFT_COOLDOWN = 480.0        # game-min a giver waits between material gifts (a kindness, not a faucet)

    def _maybe_gift_craft_material(self, giver, taker):
        """If `taker` knows a piece of gear but lacks it AND is short a raw material that `giver`
        has to spare, hand a unit over so the maker can get on with it. Rate-limited per giver so
        it's an occasional kindness, not a faucet that drains the giver every adjacent tick."""
        if self.clock < giver.get("gift_cd", 0.0):
            return
        tinv = taker.get("inv", {})
        ginv = giver.get("inv", {})
        for rid, _have_key, _raw in self._GEAR:
            if not self._person_knows(taker, rid):
                continue
            if rid == "leaf_flask" and tinv.get("leaf_flask", 0) >= 1:
                continue
            if rid in ("forage_sack", "sleeping_mat") and tinv.get(rid, 0) >= 1:
                continue
            need = crafting.missing(tinv, rid)
            for mat, qn in need.items():
                if mat == "rope":                          # an intermediate — the maker spins it themselves
                    continue
                if ginv.get(mat, 0) >= qn + self.MATERIAL_GIFT_SPARE:
                    ev = mind.give(giver, taker, mat, self.clock)
                    if ev:
                        self._note("social", ev)
                        self._bump("gift_material")
                        giver["gift_cd"] = self.clock + self.GIFT_COOLDOWN
                        self._earn_renown(giver, RENOWN_GAIN["gift"],
                                          f"shared {mat} so {taker['name']} could make a {rid.replace('_', ' ')}")
                        self._record_trade(giver, taker)   # trade XP — the road to inventing money
                    return
        return

    def _survival_craft_decide(self, p, fiber, leaf, getters):
        """If the band knows a make-shift craft this person lacks, gather its materials and
        assemble it. Rope (fiber→rope) is spun as an intermediate when a recipe calls for it.
        Returns a body action, or None when they're fully kitted out."""
        inv = p["inv"]
        for rid, have_key, _raw in self._GEAR:
            if not self._person_knows(p, rid):
                continue
            if rid == "leaf_flask" and inv.get("leaf_flask", 0) >= 1:
                continue
            if rid in ("forage_sack", "sleeping_mat") and inv.get(rid, 0) >= 1:
                continue
            if crafting.can_craft(inv, rid, stations=(), tools=None):
                home = self._homeward_if_away(p)               # assemble it back at the settlement
                if home:
                    return home
                self._begin_craft(p, rid)                      # have everything — start it (takes time)
                return "craft", None
            need = crafting.missing(inv, rid)                  # what's short — go get it
            if "rope" in need and inv.get("rope", 0) < need["rope"]:
                if inv.get("fiber", 0) >= 3:
                    home = self._homeward_if_away(p)           # spin it at home, not in the wild
                    if home:
                        return home
                    self._begin_craft(p, "rope")               # spin fiber into rope (also takes time)
                    return "craft", None
                return getters["fiber"]()
            for mat in ("leaves", "fiber"):
                if need.get(mat, 0) > 0 and mat in getters:
                    return getters[mat]()
            break
        return None

    def _provision(self, p, edible, lx, ly):
        """Lay in food against lean days: gather until the pack holds a load over the travel
        reserve, then carry it home and bank the surplus in the larder. Comfort-gated upstream,
        so a soul only does this when its own needs are quiet — survival always comes first."""
        x, y = p["x"], p["y"]
        inv = p["inv"]
        hx, hy = p["home"]
        # Stockpiling stays a near-home chore: never let laying-in food range a soul far from its
        # own water. If we've strayed past the leash, head back rather than chase another bush.
        if abs(hx - x) + abs(hy - y) > PROVISION_LEASH:
            return "haul", (hx - x, hy - y)              # raw delta → the mover pathfinds around walls
        cap = PERSON["inv_cap"] + (6 if inv.get("forage_sack", 0) else 0)
        load = min(STORE_KEEP["food"] + PROVISION_LOAD, cap)   # carry the reserve plus a load to bank
        if inv.get("food", 0) < load:
            berry = self._berry_seek(p)                # a ripe bush near home is the best larder-filler
            if berry:
                return berry
            here = bool(edible[ly, lx]) and self.veg_growth[y, x] > PERSON["gather_min"]
            return self._seek(p, x, y, here, edible, lx, ly, "food", "gather", "seek_food")
        if abs(hx - x) + abs(hy - y) > 1:                      # loaded — bring it home to the larder
            return "haul", (hx - x, hy - y)              # raw delta → the mover pathfinds around walls
        self._deposit_home(p)                                  # bank the surplus above the travel reserve
        return "tend", None

    def _ply(self, p, edible, drinkable, tree, fiber, leaf, lx, ly):
        """A settled specialist plies its trade in idle hours, building the surplus division of
        labour runs on: a FORAGER fills the larder (game, fish, berries, grass), a BUILDER stocks
        timber, a TOOLMAKER crafts spare gear. The differing surpluses are what barter moves."""
        x, y = p["x"], p["y"]
        inv = p["inv"]
        if p.get("craft"):
            return "craft", None
        voc = p.get("vocation", "forager")
        if voc == "forager":
            cap = PERSON["inv_cap"] + (6 if inv.get("forage_sack", 0) else 0)
            meat_target = int(MEAT_STOCK * SEASON_STOCK_MULT.get(self.season(), 1.0))   # hunt harder pre-winter
            # Game first — a carcass is the richest haul. Then cast for fish if water's at hand.
            if inv.get("meat", 0) < meat_target:
                hunt = self._hunt(p)
                if hunt:
                    return hunt
            if inv.get("fish", 0) < meat_target and (drinkable[ly, lx] or self._nearest_local(drinkable, lx, ly)):
                fish = self._fish(p, drinkable, lx, ly)
                if fish:
                    return fish
            if inv.get("food", 0) < cap:
                berry = self._berry_seek(p)             # a forager works the bushes too — richer pickings
                if berry:
                    return berry
                here = bool(edible[ly, lx]) and self.veg_growth[y, x] > PERSON["gather_min"]
                return self._seek(p, x, y, here, edible, lx, ly, "food", "gather", "seek_food")
            return self._idle(p)
        if voc == "builder":
            if inv.get("wood", 0) < PLY_WOOD_STOCK:
                return self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly, "wood", "chop", "seek_wood")
            return self._idle(p)
        if voc == "toolmaker":
            getters = {
                "wood":   lambda: self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly,
                                             "wood", "chop", "seek_wood"),
                "fiber":  lambda: self._seek(p, x, y, bool(fiber[ly, lx]), fiber, lx, ly,
                                             "fiber", "gather_fiber", "seek_fiber"),
                "leaves": lambda: self._seek(p, x, y, bool(leaf[ly, lx]), leaf, lx, ly,
                                             "leaves", "gather_leaves", "seek_leaves"),
            }
            act = self._toolmaker_ply(p, getters)
            if act:
                return act
        return self._idle(p)

    def _toolmaker_ply(self, p, getters):
        """Make a SPARE of each gear the band has worked out (one to keep, one to pass on), the
        toolmaker's contribution: when met, a spare flows to a band-mate who has none (social
        pass). Returns a body action or None when fully stocked."""
        inv = p["inv"]
        for rid, _have, _raw in self._GEAR:
            if not self._person_knows(p, rid):
                continue
            if inv.get(rid, 0) >= 2:                           # one to use, one spare to give
                continue
            if crafting.can_craft(inv, rid, stations=(), tools=None):
                self._begin_craft(p, rid)
                return "craft", None
            need = crafting.missing(inv, rid)
            if "rope" in need and inv.get("rope", 0) < need["rope"]:
                if inv.get("fiber", 0) >= 3:
                    self._begin_craft(p, "rope")
                    return "craft", None
                return getters["fiber"]()
            for mat in ("leaves", "fiber"):
                if need.get(mat, 0) > 0 and mat in getters:
                    return getters[mat]()
            break
        # Then TOOLS — the toolmaker's defining work. Keep a couple of axes (one to use, one to
        # hand a builder) and a spare spear for a hunter, so the band's gear flows from one bench.
        for rid, key in self._TOOLS:
            if not self._person_knows(p, rid) or inv.get(key, 0) >= 2:
                continue
            if rid == "crude_axe":
                if inv.get("wood", 0) >= BUILD["axe_wood"]:
                    self._begin_craft(p, rid)
                    return "craft", None
                return getters["wood"]()
            # Spears/rods need flint or a workbench — make them only when the materials are already
            # at hand (no forced flint-mining yet); otherwise leave it for a later tech rung.
            if crafting.can_craft(inv, rid, stations=(), tools=None):
                self._begin_craft(p, rid)
                return "craft", None
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

    def _water_within(self, x, y, r: int) -> bool:
        """Is there any water within r tiles of (x,y)? Used to keep buildings back from the
        shore (flood-shy siting)."""
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        return bool(np.any(self.water[y0:y1, x0:x1] != WATER_NONE))

    def _wood_within(self, x, y, r: int) -> bool:
        """Is there standing timber (a grown tree) within r tiles of (x,y)? Decides whether a soul
        can realistically build in TIMBER or must fall back to a leaf house."""
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        return bool(np.any(np.isin(self.veg_sp[y0:y1, x0:x1], WOOD_IDS)
                           & (self.veg_growth[y0:y1, x0:x1] > 0.2)))

    def _has_building(self, bp: str) -> bool:
        """Does a finished building of this blueprint stand anywhere (e.g. a communal storehouse)?"""
        return any(s.get("done") and s.get("bp") == bp for s in self.sites)

    def _near_building(self, bp: str, r: int, x: int, y: int) -> bool:
        """Is (x,y) within r of a FINISHED building of this blueprint (an inn, a watchtower, …)?"""
        for s in self.sites:
            if s.get("done") and s.get("bp") == bp and abs(s["ox"] - x) + abs(s["oy"] - y) <= r:
                return True
        return False

    # ── building FUNCTION — effects key off the role, so an LLM-authored form works too (A.2) ──
    def _bp_function(self, bp_id) -> str:
        """The mechanical role a blueprint fills: a built-in workshop/smithy/storehouse/hall, or
        whatever an authored design declared. '' for a plain home/monument with no special effect."""
        if bp_id in BUILTIN_FUNCTION:
            return BUILTIN_FUNCTION[bp_id]
        return (BLUEPRINTS.get(bp_id) or {}).get("function") or ""

    def _has_function(self, func: str) -> bool:
        """Has the band finished ANY building (built-in OR self-designed) that fills this role?"""
        return any(s.get("done") and self._bp_function(s.get("bp")) == func for s in self.sites)

    def _near_function(self, p, func: str, r: int) -> bool:
        """Is the soul within r of a FINISHED building filling this role (whatever its design)?"""
        x, y = p["x"], p["y"]
        for s in self.sites:
            if (s.get("done") and self._bp_function(s.get("bp")) == func
                    and abs(s["ox"] - x) + abs(s["oy"] - y) <= r):
                return True
        return False

    def _authored_for(self, func: str):
        """The id of an LLM-authored design for this role, if the band has dreamt one up — so it
        raises ITS OWN workshop/storehouse/hall instead of the built-in form. Newest design wins."""
        for ab in reversed(self.authored_blueprints):
            if ab.get("function") == func:
                return ab["id"]
        return None

    def _community_wants(self, p):
        """The PUBLIC WORKS the band needs but hasn't raised — its growing repertoire of functional
        buildings, each triggered by a real shortfall (not a script). Returns communal-build
        projects, best-need first; the co-op machinery raises them like any monument."""
        out = []
        homes = [(int(q["home"][0]), int(q["home"][1])) for q in self.people if q.get("home_struct")]
        if homes and not self._has_building("well"):
            cx = sum(h[0] for h in homes) // len(homes)
            cy = sum(h[1] for h in homes) // len(homes)
            if not self._water_within(cx, cy, WELL_NEED_DIST):          # settled far from water
                out.append(("well", "dig a well — water close to home at last"))
        unhoused = sum(1 for q in self.people if q["age"] >= ADULT_AGE and not q.get("home_struct"))
        if unhoused >= INN_NEED_UNHOUSED and not self._has_building("inn"):
            out.append(("inn", "raise an inn — a roof for those who have none"))
        if getattr(self, "_wolf_blooded", False) and not self._has_building("watchtower"):
            out.append(("watchtower", "raise a watchtower — to keep the wolves off us"))
        # Once the band trades in COIN, it raises a MARKETPLACE — a civic heart for its new economy.
        if getattr(self, "money_invented", False) and not self._has_building("market"):
            out.append(("market", "raise a marketplace — a place to trade now that we have coin"))
        return [{"kind": "monument", "bp": b, "why": w} for b, w in out]

    def _tile_unbuildable(self, tx, ty, occupied, avoid: int):
        """Why a tile can't take a building tile (a short, human reason), or None if it can — so
        the same check both rejects a footprint AND lets a soul SAY why it won't build there."""
        if not self._in(tx, ty):
            return "off the edge of the world"
        if self.water[ty, tx] != WATER_NONE:
            return "in the water"
        if (tx, ty) in occupied:
            return "right where something already stands"
        if avoid:
            if self._water_within(tx, ty, avoid):
                return "too close to the water"
            # A soul PREFERS a natural clearing — open ground is a better homesite than the deep
            # woods (proven: clearing-sited bands survive markedly better). It's only a preference:
            # the relaxed fallback (avoid=0) still lets them build on wooded ground when no clearing
            # is near, and whatever veg is under the footprint is CLEARED tile-by-tile as it's built.
            if self.veg_sp[ty, tx] in WOOD_IDS and self.veg_growth[ty, tx] > 0.2:
                return "in among the trees"
        return None

    def _build_reason(self, name, ox, oy, avoid: int, occupied=None):
        """The reason a blueprint won't fit at (ox,oy) — the first tile that fails — or None if
        it fits. Lets a soul reason aloud about a spot instead of silently giving up."""
        bp = BLUEPRINTS.get(name)
        if not bp:
            return "a building I don't know how to raise"
        occupied = occupied if occupied is not None else self._occupied_tiles()
        for dy, row in enumerate(bp["layout"]):
            for dx, ch in enumerate(row):
                if ch != GLYPH_CORE and BLOCK_CHARS.get(ch, BLOCK_EMPTY) == BLOCK_EMPTY:
                    continue
                why = self._tile_unbuildable(ox + dx, oy + dy, occupied, avoid)
                if why:
                    return why
        return None

    def _tile_waterlocked(self, x, y) -> bool:
        """Is a land tile cut off by water on every side — somewhere the band can't actually walk
        to? (A structure stranded like this needs moving.)"""
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self._in(nx, ny) and self.water[ny, nx] == WATER_NONE:
                    return False
        return True

    def _relocate_granary_if_stranded(self):
        """Self-heal a granary that ended up unreachable — standing IN the water, or on a scrap of
        land walled off by it. The band reasons 'we can't get to it' and hauls it to dry ground
        near the homes. Fixes worlds saved before flood-shy siting, and keeps it honest after."""
        g = self.granary
        if g.get("x") is None:
            return
        x, y = int(g["x"]), int(g["y"])
        if self._in(x, y) and self.water[y, x] == WATER_NONE and not self._tile_waterlocked(x, y):
            return                                       # reachable dry ground — all is well
        housed = [q for q in self.people if q.get("home_struct")]
        if housed:
            cx = int(sum(q["home"][0] for q in housed) / len(housed))
            cy = int(sum(q["home"][1] for q in housed) / len(housed))
        else:
            cx, cy = x, y
        nx, ny = self._dry_spot_near(cx, cy)
        if (nx, ny) == (x, y):
            return
        g["x"], g["y"] = nx, ny
        for s in self.structures:
            if s.get("kind") == "granary":
                s["x"], s["y"] = nx, ny
                break
        self.version += 1
        self._note("build", "The granary stood where no one could reach it — the band hauled it to dry ground.")

    @staticmethod
    def _rotate_layout(layout, k: int):
        """Rotate a glyph layout 90°×k clockwise (k in 0..3). Glyphs are orientation-free, so this
        just carries the door (and windows) round to a different wall — the basis for ORIENTING a
        building so its door faces open ground rather than always pointing the same way."""
        rows = [list(r) for r in layout]
        for _ in range(k % 4):
            rows = [list(col) for col in zip(*rows[::-1])]
        return ["".join(r) for r in rows]

    def _blueprint_tasks(self, name, ox, oy, occupied=None, avoid: int = 0, layout=None):
        """Turn a blueprint at origin (ox,oy) into placement tasks + the home core tile, or
        (None, None) if the footprint doesn't fit (off-map, over water, or OVERLAPPING an
        existing building/site — which is why shelters used to grow inside one another). Each
        task carries its own material cost so blueprints can mix wood, thatch and leaves.
        Blocks are laid first, then roof tiles. A 'C' core lays no block but is roofed/home.
        When `avoid` > 0 (autonomous siting) the footprint must also keep that many tiles from
        water (flood-shy) and never sit on a standing tree — so the band stops building on the
        shore and over the woods. (God placement passes avoid=0 — the god may build anywhere.)
        A `layout` override lets a soul try the building in a chosen ROTATION (see _found_site)."""
        bp = BLUEPRINTS.get(name)
        if not bp:
            return None, None
        occupied = occupied if occupied is not None else self._occupied_tiles()

        def bad(tx, ty) -> bool:
            return self._tile_unbuildable(tx, ty, occupied, avoid) is not None

        layout = layout if layout is not None else bp["layout"]
        roof_cost = bp.get("roof_cost", ROOF_COST)
        blocks, roof, core = [], [], None
        for dy, row in enumerate(layout):
            for dx, ch in enumerate(row):
                tx, ty = ox + dx, oy + dy
                if ch == GLYPH_CORE:
                    if bad(tx, ty):
                        return None, None
                    roof.append({"x": tx, "y": ty, "code": int(BLOCK_FLOOR), "layer": "roof",
                                 "cost": list(roof_cost), "done": False})
                    core = (tx, ty)
                    continue
                code = BLOCK_CHARS.get(ch, BLOCK_EMPTY)
                if code == BLOCK_EMPTY:
                    continue
                if bad(tx, ty):
                    return None, None
                blocks.append({"x": tx, "y": ty, "code": int(code), "layer": "block",
                               "cost": list(BLOCK_COST[code]), "done": False})
                if bp.get("roof") and code in (BLOCK_FLOOR, BLOCK_DOOR):
                    roof.append({"x": tx, "y": ty, "code": int(code), "layer": "roof",
                                 "cost": list(roof_cost), "done": False})
        return blocks + roof, core

    def _occupied_tiles(self) -> set:
        """Every tile already taken by a placed block, a roof, or a pending construction
        site — so a new footprint can be rejected before it's laid over someone's home."""
        occ = set(self.blocks.keys())
        occ |= set(self.roofs)
        for s in self.sites:
            if s.get("done"):
                continue
            for t in s["tasks"]:
                occ.add((t["x"], t["y"]))
        return occ

    def _door_approaches(self) -> set:
        """Every tile that must stay CLEAR in front of a DOOR — the open step a soul takes to come
        and go — so a new building can never wall a neighbour into their home by building across the
        doorway (the spatial-awareness gap: siting used to forbid only OVERLAP, not blocking). Covers
        doors already placed (in self.blocks) and doors still planned (in pending sites)."""
        reserved = set()
        blocks = self.blocks
        for (x, y), code in blocks.items():
            if code != BLOCK_DOOR:
                continue
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + ddx, y + ddy
                if self._in(nx, ny) and (nx, ny) not in blocks and self.water[ny, nx] == WATER_NONE:
                    reserved.add((nx, ny))             # the open approach just outside the door
        for s in self.sites:
            if s.get("done"):
                continue
            btiles = {(t["x"], t["y"]) for t in s["tasks"] if t.get("layer") == "block"}
            for t in s["tasks"]:
                if t.get("code") != BLOCK_DOOR:
                    continue
                x, y = t["x"], t["y"]
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + ddx, y + ddy
                    if self._in(nx, ny) and (nx, ny) not in btiles and self.water[ny, nx] == WATER_NONE:
                        reserved.add((nx, ny))
        return reserved

    def _door_openness(self, tasks, occupied, anchors) -> float:
        """Score an ORIENTATION: reward a door that opens onto CLEAR GROUND and faces OUT (away from
        the settlement's heart) instead of into a neighbour's wall — the spatial sense to point a
        doorway somewhere a soul can actually come and go. Added to the site score in _found_site."""
        btiles = {(t["x"], t["y"]) for t in tasks if t.get("layer") == "block"}
        doors = [(t["x"], t["y"]) for t in tasks if t.get("code") == BLOCK_DOOR]
        if not doors:
            return 0.0
        cx = cy = None
        if anchors:
            cx = sum(a[0] for a in anchors) / len(anchors)
            cy = sum(a[1] for a in anchors) / len(anchors)
        score = 0.0
        for (x, y) in doors:
            outdir = None
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + ddx, y + ddy) not in btiles:           # the open side outside the door
                    outdir = (ddx, ddy)
                    break
            if outdir is None:
                continue
            ddx, ddy = outdir
            for step in (1, 2):                                # clear ground for two tiles out the door
                tx, ty = x + ddx * step, y + ddy * step
                if (self._in(tx, ty) and (tx, ty) not in occupied and (tx, ty) not in btiles
                        and self.water[ty, tx] == WATER_NONE):
                    score += 0.8
            if cx is not None and abs(x + ddx - cx) + abs(y + ddy - cy) > abs(x - cx) + abs(y - cy):
                score += 0.5                                   # the door faces OUT of the settlement
        return score

    @staticmethod
    def _site_offsets():
        """Footprint origins to try, spiralling outward from home so a site lands as close
        as it can but steps away ring by ring when the near ground is taken."""
        offs = [(0, 0)]
        for r in range(1, 10):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r:
                        offs.append((dx, dy))
        return offs

    def _settlement_anchors(self):
        """The fixed points a soul plans around: every home, plus the common granary — the body
        of the settlement a new building should sit WITH (clustered) but not ON (spaced)."""
        anchors = [(int(q["home"][0]), int(q["home"][1])) for q in self.people if q.get("home_struct")]
        g = self.granary
        if g.get("x") is not None:
            anchors.append((int(g["x"]), int(g["y"])))
        return anchors

    def _settlement_name(self) -> str:
        """Stitch a place-name for the band's home (Ashford, Stonehaven, …)."""
        rng = self.rng
        return (SETTLEMENT_PREFIXES[int(rng.integers(len(SETTLEMENT_PREFIXES)))]
                + SETTLEMENT_SUFFIXES[int(rng.integers(len(SETTLEMENT_SUFFIXES)))])

    def _tick_settlements(self):
        """Maintain the band's first-class SETTLEMENT — the town that emerges from where folk have
        made their homes. For now the housed band is ONE settlement (daughter colonies come later):
        its CENTRE tracks the homes' centroid, its members are the housed souls, its size their
        count. The civic foundation (M0) that zoning, a treasury and a planning authority hang on."""
        homed = [q for q in self.people if q.get("home_struct")]
        if not homed:
            return
        cx = int(sum(q["home"][0] for q in homed) / len(homed))
        cy = int(sum(q["home"][1] for q in homed) / len(homed))
        if not self.settlements:
            self.settlements = [{"id": "set_" + uuid.uuid4().hex[:8], "name": self._settlement_name(),
                                 "cx": cx, "cy": cy, "members": [], "pop": 0,
                                 "founded_t": round(self.clock, 1)}]
            self._note("culture", f"the band's home is a place now — they call it {self.settlements[0]['name']}.")
        s = self.settlements[0]
        s["members"] = [q["id"] for q in homed]
        s["pop"] = len(homed)
        s["cx"], s["cy"] = cx, cy

    def _nearest_resource_dist(self, cx, cy):
        """Manhattan distance to the nearest stone boulder or ore node, or None if the map has
        none — what a smithy/workshop wants close: the stuff it works."""
        best = None
        for n in self.stone_nodes:
            d = abs(n["x"] - cx) + abs(n["y"] - cy)
            if best is None or d < best:
                best = d
        for n in self.ore_nodes:
            d = abs(n["x"] - cx) + abs(n["y"] - cy)
            if best is None or d < best:
                best = d
        return best

    def _site_purpose_bonus(self, bp_name, cx, cy) -> float:
        """A communal building belongs where its PURPOSE is best served, not just clustered: a
        SMITHY or WORKSHOP wants the stone and ore it works close to hand. A modest nudge layered on
        the site score, so the band raises its craft buildings WITH a reason — by the rock — while
        the water/cluster terms still keep them near home. (Homes pass bp_name=None: no change.)"""
        if bp_name in (WORKSHOP_BP, SMITHY_BP):
            d = self._nearest_resource_dist(cx, cy)
            if d is not None:
                return max(0.0, 2.5 - d * 0.10)              # the closer to stone/ore, the better
        return 0.0

    def _score_site(self, cx, cy, anchors, communal: bool, origin=None, bp_name=None) -> float:
        """How GOOD a building site (cx,cy) is — the heart of planning. Rewards a comfortable walk to
        water and clear ground; clusters near the band but penalises crowding a neighbour; a
        communal building is pulled hard toward the centre so it sits among everyone. Crucially it
        also favours building CLOSE to where the soul already is — a roof raised soon beats a finer
        plot reached after a long, exposed trek (which was getting people killed by the cold). A
        purpose bonus then nudges special buildings toward what they need (a smithy by the stone)."""
        score = 0.0
        score += 3.0 if self._water_within(cx, cy, SITE_WATER_IDEAL) else -2.5   # don't strand from drink
        if not (self.veg_sp[cy, cx] in WOOD_IDS and self.veg_growth[cy, cx] > 0.2):
            score += 0.6                                                          # open ground reads better
        if origin is not None:
            score -= (abs(origin[0] - cx) + abs(origin[1] - cy)) * 0.35          # build near — don't trek far
        if anchors:
            nd = min(abs(ax - cx) + abs(ay - cy) for ax, ay in anchors)
            if nd < HOME_MIN_SPACING:
                score -= (HOME_MIN_SPACING - nd) * 2.0                            # too cramped — leave room
            ccx = sum(a[0] for a in anchors) / len(anchors)
            ccy = sum(a[1] for a in anchors) / len(anchors)
            cdist = abs(ccx - cx) + abs(ccy - cy)
            score -= cdist * (0.16 if communal else 0.05)                         # cluster (hard for communal)
        score += self._site_purpose_bonus(bp_name, cx, cy)                       # built WITH a reason
        return score

    def _reaches_water(self, start, planned_solid=(), budget=2500) -> bool:
        """Bounded BFS from `start` over passable ground — treating this building's about-to-be-laid
        WALLS (`planned_solid`) as already solid — to a land tile bordering water (a drink spot).
        The siting guard: a home must never seal its own folk away from drink. Bounded, so it stays
        cheap and never sweeps the whole 2048² map; returns True the instant it touches a shore."""
        sx, sy = int(start[0]), int(start[1])
        if not self._in(sx, sy):
            return False
        solid = set(planned_solid)
        seen = {(sx, sy)}
        q = [(sx, sy)]
        qi = 0
        while qi < len(q) and qi < budget:
            x, y = q[qi]
            qi += 1
            if self.water[y, x] == WATER_NONE:               # standing on land — is water one step off?
                for dx, dy in _STEP_DIRS:
                    ax, ay = x + dx, y + dy
                    if self._in(ax, ay) and self.water[ay, ax] != WATER_NONE:
                        return True
            for dx, dy in _STEP_DIRS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in seen or (nx, ny) in solid:
                    continue
                if self._passable(nx, ny):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _choose_reachable_site(self, found, communal: bool, top_k: int = 8):
        """From the score-sorted candidates, pick the best whose occupants could actually REACH
        WATER (a home must not seal its folk from drink — even soft leaf walls aside, real walls and
        a tight cluster can box a doorway in). A communal build has no occupants, so the top score
        wins outright. Falls back to the top score if none of the top-K prove reachable — putting a
        roof somewhere always beats leaving a soul homeless (siting must never strand anyone)."""
        if communal:
            return found[0]
        for c in found[:top_k]:
            _, ox, oy, tasks, core, cxy = c
            wall = {(t["x"], t["y"]) for t in tasks
                    if t.get("layer") == "block" and t.get("code") == BLOCK_WALL}
            if self._reaches_water(core or cxy, planned_solid=wall):
                return c
        return found[0]

    def _found_site(self, p, name: str = "leaf_shelter", communal: bool = False, rung=None, client=None):
        """PLAN a building footprint: rather than grabbing the first tile that fits, a soul weighs
        the candidate spots around its anchor and picks the BEST — near enough to water, on clear
        ground, clustered with the band but with room to breathe (`_score_site`). Falls back to the
        cheap leaf shelter, then to any dry ground, so it never ends up homeless. A `communal` site
        (a monument) is NOT the builder's home, so it doesn't move their home anchor; a `client`
        commission is OWNED by the client (it becomes their home) and likewise leaves the builder's."""
        bx, by = p["home"]
        occupied = self._occupied_tiles() | self._door_approaches()   # never build across a doorway
        anchors = self._settlement_anchors()
        cands = [name] + (["leaf_shelter"] if name != "leaf_shelter" else [])
        for avoid in (WATER_BUILD_BUFFER, 0):            # prefer a flood-shy spot, else any dry ground
            for cand in cands:
                bp = BLUEPRINTS[cand]
                # Try the building in each DISTINCT rotation, so its door can be ORIENTED toward open
                # ground instead of always facing the same way (a symmetric layout dedups to fewer).
                rot_layouts, seen_l = [], set()
                for k in (0, 1, 2, 3):
                    lr = self._rotate_layout(bp["layout"], k)
                    key = tuple(lr)
                    if key not in seen_l:
                        seen_l.add(key)
                        rot_layouts.append(lr)
                per_rot = max(8, SITE_CANDIDATES_CAP // len(rot_layouts))
                found = []                               # [(score, ox, oy, tasks, core, (cx,cy)), …]
                for lr in rot_layouts:
                    bw, bh = len(lr[0]), len(lr)
                    rfits = 0
                    for off in self._site_offsets():
                        ox, oy = bx - bw // 2 + off[0], by - bh // 2 + off[1]
                        tasks, core = self._blueprint_tasks(cand, ox, oy, occupied, avoid=avoid, layout=lr)
                        if not tasks:
                            continue
                        cx, cy = core if core else (ox + bw // 2, oy + bh // 2)
                        s = (self._score_site(cx, cy, anchors, communal, origin=(bx, by), bp_name=cand)
                             + self._door_openness(tasks, occupied, anchors))   # orient the door to open ground
                        found.append((s, ox, oy, tasks, core, (cx, cy)))
                        rfits += 1
                        if rfits >= per_rot:             # weighed enough spots in THIS orientation
                            break
                if not found:
                    continue                             # this blueprint doesn't fit anywhere — try the next
                found.sort(key=lambda c: -c[0])
                chosen = self._choose_reachable_site(found, communal)
                _, ox, oy, tasks, core = chosen[:5]
                site = {"id": "b_" + uuid.uuid4().hex[:8], "bp": cand, "name": bp["name"],
                        "ox": int(ox), "oy": int(oy), "by": p["name"],
                        "owner": (client["id"] if client else p["id"]),   # a commission belongs to the CLIENT
                        "rung": rung if (rung and cand == name) else cand,   # ladder rung (custom homes still track; leaf fallback stays leaf)
                        "core": [int(core[0]), int(core[1])] if core else None,  # the heart tile (a well's shaft, a home's hearth)
                        "insul": float(bp.get("insulation", 1.0)),
                        "tasks": tasks, "done": False, "t": round(self.clock, 1)}
                site["communal"] = bool(communal)
                if client is not None:
                    site["commission"] = True
                    site["builder"] = p["id"]            # the contractor — paid by the client on completion
                self.sites.append(site)
                p["site"] = site["id"]
                if not communal and client is None:      # only the builder's OWN home moves its anchor
                    home = core or next(((t["x"], t["y"]) for t in tasks if t["code"] == BLOCK_FLOOR), (bx, by))
                    p["home"] = (int(home[0]), int(home[1]))
                self.version += 1
                self._bump("plan_found")
                if client is not None:                   # the ASK: a builder offers to raise a home for another
                    self._bump("commission_started")
                    self._note("build", f"{p['name']} took on raising a home for {client['name']}.")
                    mind.speak(p, f"{client['name']} — I'll raise you a proper home, for a fair price.", self.clock)
                    mind.remember(client, f"{p['name']} offered to build me a proper home", 0.7, "social", self.clock)
                    return
                verb = "began a" if communal else "marked out a"
                self._note("build", f"{p['name']} {verb} {bp['name'].lower()}.")
                # Say WHAT they're raising, so a watcher can tell at a glance (a house? an inn? a well?).
                what = bp["name"].split("'s")[-1].strip().lower() if "'s" in bp["name"] else bp["name"].lower()
                mind.speak(p, f"I'll raise {'an' if what[:1] in 'aeiou' else 'a'} {what} here.", self.clock)
                return
        # Nowhere fit the footprint — reason about WHY (it'd be in the water / among the trees /
        # no room) so the soul thinks it through rather than silently giving up, then waits.
        bw0, bh0 = len(BLUEPRINTS[name]["layout"][0]), len(BLUEPRINTS[name]["layout"])
        why = self._build_reason(name, bx - bw0 // 2, by - bh0 // 2, WATER_BUILD_BUFFER, occupied) \
            or "there's just no room here"
        mind.remember(p, f"I can't raise it here — it'd be {why}.", 0.4, "build", self.clock)
        mind.speak(p, f"Not here — {why}.", self.clock)
        p["build_cd"] = self.clock + 720.0      # nowhere to build here — try again later

    def _grow_skill(self, p, name, amt=CHILD_SKILL_GAIN):
        """Nudge a soul's hands-on skill upward (capped). How a youngling who keeps at a thing
        slowly gets good at it — the seed of competent adulthood."""
        sk = p.setdefault("skills", {})
        sk[name] = min(1.0, sk.get(name, 0.0) + amt)

    def _child_forage(self, p, edible, lx, ly):
        """A youngling forages food — useful work small hands CAN do, and practice that grows
        their foraging skill a little each beat. Surplus banks at home like anyone's (kids help
        feed the band too), but they never range as a provisioner would."""
        self._grow_skill(p, "foraging")
        x, y = p["x"], p["y"]
        return self._seek(p, x, y, bool(edible[ly, lx]), edible, lx, ly, "food", "eat", "seek_food")

    def _child_whittle(self, p, tree, lx, ly):
        """A youngling whittles arrows from a length of wood — safe handiwork that grows a young
        crafter's skill. With no wood to hand they go gather some first."""
        inv = p["inv"]
        if inv.get("wood", 0) >= 1:
            inv["wood"] -= 1
            if inv["wood"] <= 0:
                inv.pop("wood", None)
            inv["arrows"] = inv.get("arrows", 0) + ARROWS_PER_WHITTLE
            self._grow_skill(p, "crafting", CHILD_SKILL_GAIN * 1.5)
            self._bump("child_whittle")
            return "whittle", None
        x, y = p["x"], p["y"]
        return self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly, "wood", "chop", "seek_wood")

    def _bump(self, key):
        """Tiny behaviour counter so a canary can confirm a new behaviour actually FIRES live
        (a recurring blind spot — a behaviour can be correct yet never trigger)."""
        d = self.__dict__.setdefault("_beh", {})
        d[key] = d.get(key, 0) + 1

    def _helpable_build_near(self, p):
        """A band-mate's OWN home-in-progress within eyeshot that this soul could pitch in on — not a
        communal/commission raise (those crew themselves), not their own, and still needing work.
        Nearest first; None if there's nothing to lend a hand with."""
        x, y = p["x"], p["y"]
        best, bd = None, HELP_OFFER_RANGE + 1
        for s in self.sites:
            if s.get("done") or s.get("communal") or s.get("commission") or s.get("owner") == p["id"]:
                continue
            ox, oy = s.get("ox", x), s.get("oy", y)
            d = abs(ox - x) + abs(oy - y)
            if d >= bd:
                continue
            if all(t.get("done") for t in s.get("tasks", [])):
                continue                                  # nothing left to lay
            best, bd = s, d
        return best

    def _offer_help_maybe(self, p, tree, stone, fiber, leaf, lx, ly):
        """While at loose ends, a hale soul who SEES a neighbour raising their own home walks over,
        OFFERS a hand, and — if the builder is willing — pitches in. Returns a body action while
        offering/helping, or None to carry on wandering. The good turn that makes a village feel
        like neighbours, not strangers sharing ground."""
        # Already committed to a build? See it through, then rest the offer.
        hid = p.get("helping")
        if hid:
            site = next((s for s in self.sites if s["id"] == hid and not s.get("done")), None)
            if site is not None and not all(t.get("done") for t in site.get("tasks", [])):
                return self._help_build(p, hid, tree, stone, fiber, leaf, lx, ly)
            p.pop("helping", None)
            p["help_offer_cd"] = self.clock + HELP_OFFER_CD
            return None
        # Only a settled, hale, unhurried soul offers — never while houseless, sick, or running low.
        if (p.get("home_struct") is None or p.get("illness")
                or max(p.get("hunger", 0), p.get("thirst", 0), p.get("fatigue", 0)) > 0.5
                or self.clock < p.get("help_offer_cd", 0.0)):
            return None
        site = self._helpable_build_near(p)
        if site is None:
            return None
        owner = next((q for q in self.people if q["id"] == site.get("owner")), None)
        if owner is None:
            return None
        x, y = p["x"], p["y"]
        if abs(owner["x"] - x) + abs(owner["y"] - y) > 2:        # walk over to ask (pathfinds round walls)
            return "socialize", (owner["x"] - x, owner["y"] - y)
        # Beside them — ASK. The builder accepts unless they regard the offerer poorly.
        sentiment = p.get("rel", {}).get(owner["id"], {}).get("sentiment", 0.0)
        if sentiment < HELP_OK_SENTIMENT:
            p["help_offer_cd"] = self.clock + HELP_OFFER_CD
            mind.speak(owner, "I've got it — but thank you.", self.clock)
            return None
        p["helping"] = site["id"]                                # accepted — fall in and help
        mind.speak(p, f"Want a hand with that, {owner['name']}?", self.clock)
        mind.speak(owner, "Aye — much obliged!", self.clock)
        return self._help_build(p, site["id"], tree, stone, fiber, leaf, lx, ly)

    def _help_build(self, p, site_id, tree, stone, fiber, leaf, lx, ly):
        """Lend a hand on a band-mate's unfinished build: FORAGE the next tile's material if short
        of it (just as the owner would), then walk over and lay the tile. The finished home still
        goes to its owner. Yields to idle if the site is gone/finished or its next tile needs a
        material this helper can't fetch."""
        site = next((s for s in self.sites if s["id"] == site_id and not s.get("done")), None)
        if site is None:
            return self._idle(p)
        task = self._site_next_task(site)
        if task is None:
            return self._idle(p)
        item, qty = task["cost"]
        x, y = p["x"], p["y"]
        if p["inv"].get(item, 0) < qty:                  # short the makings — go gather them
            getters = {
                "wood":   lambda: self._seek(p, x, y, bool(tree[ly, lx]), tree, lx, ly, "wood", "chop", "seek_wood"),
                "fiber":  lambda: self._seek(p, x, y, bool(fiber[ly, lx]), fiber, lx, ly, "fiber", "gather_fiber", "seek_fiber"),
                "leaves": lambda: self._seek(p, x, y, bool(leaf[ly, lx]), leaf, lx, ly, "leaves", "gather_leaves", "seek_leaves"),
                "stone":  lambda: self._seek(p, x, y, bool(stone[ly, lx]), stone, lx, ly, "stone", "mine", "seek_stone"),
            }
            g = getters.get(item)
            return g() if g else self._idle(p)
        tx, ty = task["x"], task["y"]
        if max(abs(tx - x), abs(ty - y)) > 1:
            return "wander", (tx - x, ty - y)            # walk to the build tile → pathfind round walls
        before = sum(1 for t in site["tasks"] if t["done"])
        self._build_next_block(p, site=site)            # lay it (credits the owner on completion)
        if sum(1 for t in site["tasks"] if t["done"]) > before:
            self._bump("help_block")
            p["aided"] = p.get("aided", 0.0) + 1.0       # rolling "lends a hand" credit (the labour norm)
            if not site.get("done"):
                self._earn_renown(p, RENOWN_GAIN.get("help", 0.3) * 0.4, "lent a hand on a neighbour's build")
        return "build", None

    def _build_next_block(self, p, site=None):
        """Lay the next blueprint tile if the person is in range and carries the material. With an
        explicit `site` a soul lays a tile on SOMEONE ELSE's build (helping) — the finished home
        still goes to its owner, not to whoever happened to place the last tile."""
        site = site if site is not None else self._person_site(p)
        if not site:
            return
        task = self._site_next_task(site)
        if task is None:
            self._finish_site(p, site)
            return
        if max(abs(task["x"] - p["x"]), abs(task["y"] - p["y"])) > 2:
            return
        # Cooperative big-builds: a communal monument's tiles are too heavy for one — they're laid
        # only when a crew of CO_OP_MIN is on hand. A lone builder waits at the site (the join logic
        # draws others in), so raising the hall is a genuine group effort, not a solo grind.
        if site.get("communal") and self._site_crew_present(site) < CO_OP_MIN:
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
            self._clear_ground(task["x"], task["y"])     # trample the growth under & around the tile
        task["done"] = True
        self._grow_skill(p, "building", CHILD_SKILL_GAIN)   # hands learn the trade by laying tile
        if site.get("owner") and p.get("id") != site["owner"]:   # remember who built it FOR the owner
            hs = site.setdefault("helpers", [])
            if p["id"] not in hs:
                hs.append(p["id"])
        self.version += 1
        if self._site_next_task(site) is None:
            self._finish_site(p, site)

    def _clear_ground(self, x, y, radius: int = 1):
        """Strip the vegetation under a placed tile and the ring around it — a building stands on
        cleared, trodden ground, not in a thicket. Clears BOTH the growth and the species, so a
        tree or shrub the tile sat on actually vanishes (not just shrinks). Cheap box edit."""
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        self.veg_growth[y0:y1, x0:x1] = 0.0
        self.veg_sp[y0:y1, x0:x1] = VEG_NONE

    def _earn_renown(self, p, amount: float, why: str) -> None:
        """Raise a soul's social standing for a visible deed, and lodge it as a proud memory."""
        p["renown"] = p.get("renown", 0.0) + amount
        mind.remember(p, why, min(0.95, 0.5 + amount), "renown", self.clock)

    def _record_trade(self, a, b):
        """Both parties gain a little TRADE EXPERIENCE — and out of enough of it, a clever soul may
        one day conceive of money. This is the ground the currency idea grows from."""
        a["trade_n"] = a.get("trade_n", 0) + 1
        b["trade_n"] = b.get("trade_n", 0) + 1
        self._maybe_invent_money(a)
        self._maybe_invent_money(b)

    def _maybe_invent_money(self, p):
        """A trade-worn, curious soul has the INSIGHT that a durable token can stand for any debt or
        trade — and so MONEY comes into the world, once, named for its inventor. No system decreed
        it; it emerged from a band that had bartered enough to feel the need. The inventor strikes
        the first coins; they then circulate through the labour market (commissions paid in coin)."""
        if getattr(self, "money_invented", False) or p.get("age", 0) < ADULT_AGE:
            return
        if p.get("trade_n", 0) < MONEY_TRADE_XP or mind._trait(p, "curiosity") < 0.5:
            return
        if self.rng.random() > 0.25:                       # an insight, not a certainty
            return
        self.money_invented = True
        self.money_inventor = p["name"]
        p["inv"]["coin"] = p["inv"].get("coin", 0) + COIN_MINT
        self._bump("money_invented")
        self._note("culture", f"{p['name']} reasoned that a marked token could stand for any trade or "
                              "debt — and so MONEY was born. The band would never barter blind again.")
        mind.remember(p, "I saw it — a token can stand for anything we owe one another. I have given the band MONEY.",
                      0.95, "discovery", self.clock)
        mind.speak(p, "A token — let it stand for what we owe each other! This changes everything.", self.clock)
        self._earn_renown(p, 0.6, "gave the band the very idea of money — a name to outlast the ages")

    def _spend_food(self, q, n: int) -> bool:
        """Pay `n` food from a soul's pack, then its home larder — for a commission/trade. Returns
        False (no change) if it can't afford the whole sum (you can't pay what you don't have)."""
        inv, store = q.setdefault("inv", {}), q.setdefault("store", {})
        if inv.get("food", 0) + store.get("food", 0) < n + 1:        # keep a bite for itself
            return False
        from_inv = min(inv.get("food", 0), n)
        inv["food"] = inv.get("food", 0) - from_inv
        if inv.get("food", 0) <= 0:
            inv.pop("food", None)
        rem = n - from_inv
        if rem > 0:
            store["food"] = store.get("food", 0) - rem
            if store.get("food", 0) <= 0:
                store.pop("food", None)
        return True

    def _dismantle_site(self, site_id):
        """Tear down a finished building: pull its blocks and roof off the map and drop the site —
        used when a soul upgrades, so the old leaf shelter doesn't linger as an abandoned husk."""
        s = next((q for q in self.sites if q["id"] == site_id), None)
        if not s:
            return None
        for t in s["tasks"]:
            self.blocks.pop((t["x"], t["y"]), None)
            self.roofs.discard((t["x"], t["y"]))
        self.sites = [q for q in self.sites if q is not s]
        self.version += 1
        return s

    def _finish_site(self, p, site):
        site["done"] = True
        self.version += 1
        if BLUEPRINTS.get(site["bp"], {}).get("communal"):
            # A WELL's shaft fills with water on completion — a real drinking source the band can
            # gather at, so it can live away from the river (the building's PURPOSE made real).
            if site["bp"] == "well" and site.get("core"):
                cx, cy = site["core"]
                if self._in(cx, cy):
                    self.water[cy, cx] = WATER_RIVER
            # A MARKETPLACE finished is a civilizational milestone — the band has become a trading
            # people, with a civic place to deal. Marked as culture, not just another build.
            if site["bp"] == "market":
                self._note("culture", f"{p['name']} raised the band's first MARKETPLACE — a wooden "
                                      "village has become a trading people.")
                mind.speak(p, "Here we'll trade — bring your goods and your coin!", self.clock)
            # A monument/public work: it doesn't house the builder, but it crowns them — the band
            # gains a shared landmark with a purpose, and its raiser wins lasting renown.
            self._note("build", f"{p['name']} finished a {site['name'].lower()} for the band.")
            self._earn_renown(p, RENOWN_GAIN["monument"],
                              f"raised a {site['name'].lower()} for us all — a name that will last")
            return
        # The HOME belongs to whoever raised it (the owner), not whoever happened to lay the last
        # tile — so a band-mate who lent a hand finishing the walls doesn't end up claiming it. For a
        # COMMISSION the owner is the CLIENT it was built for; they move in (their anchor shifts here).
        owner = next((q for q in self.people if q["id"] == site.get("owner")), None) or p
        commission = bool(site.get("commission"))
        if commission:
            core = site.get("core")
            hxy = tuple(core) if core else next(((t["x"], t["y"]) for t in site["tasks"]
                                                 if t["code"] == BLOCK_FLOOR), None)
            if hxy:
                owner["home"] = (int(hxy[0]), int(hxy[1]))
        # Moving up the dwelling ladder: pull down the owner's old, lesser home now that a finer
        # one stands, so the settlement isn't littered with the husks of outgrown shelters.
        old = owner.get("home_struct")
        if old and old != site["id"]:
            gone = self._dismantle_site(old)
            if gone:
                self._note("build", f"{owner['name']} pulled down their old {gone['name'].lower()}.")
        owner["home_struct"] = site["id"]
        owner["insul"] = site.get("insul", 1.0)     # how well the finished home holds heat/cold
        if commission:                              # the builder raised it FOR the client
            builder = next((q for q in self.people if q["id"] == site.get("builder")), None)
            bname = builder["name"] if builder else site.get("by", "a builder")
            self._note("build", f"{bname} finished {owner['name']}'s new {site['name'].lower()}.")
            mind.remember(owner, f"{bname} built me a fine {site['name'].lower()} — a proper home at last",
                          0.85, "build", self.clock)
            if builder is not None:
                self._earn_renown(builder, RENOWN_GAIN["dwelling"],
                                  f"built {owner['name']} a fine home — good paid work, well done")
        else:
            self._note("build", f"{owner['name']} finished building a {site['name'].lower()}.")
            mind.remember(owner, f"raised my own {site['name'].lower()} — a home at last", 0.85,
                          "build", self.clock)
            self._earn_renown(owner, RENOWN_GAIN["dwelling"], f"raised a fine {site['name'].lower()} of my own")
            if p is not owner:                      # a band-mate laid the final hand — honour it
                self._earn_renown(p, RENOWN_GAIN.get("help", 0.3),
                                  f"lent a hand raising {owner['name']}'s home")
                mind.remember(p, f"helped {owner['name']} finish their home", 0.6, "renown", self.clock)
        # COMMISSION — the owner PAYS everyone who built their home for them, as much as it can
        # spare: the first stir of a LABOUR ECONOMY (work traded for goods), emerging from folk
        # building for folk. No one decreed it; it arose from a builder helping and a soul grateful.
        for hid in site.get("helpers", []):
            if hid == owner["id"]:
                continue
            helper = next((q for q in self.people if q["id"] == hid), None)
            if helper is None:
                continue
            # The CONTRACTOR who raised a whole commissioned home earns a bigger fee than a soul who
            # merely lent a hand. Pay in COIN once the band has money and the owner holds some; else food.
            is_contractor = commission and hid == site.get("builder")
            fee_food = CONTRACTOR_FEE if is_contractor else COMMISSION_FEE
            fee_coin = CONTRACTOR_COIN if is_contractor else COMMISSION_COIN
            paid = None
            if getattr(self, "money_invented", False) and owner["inv"].get("coin", 0) >= fee_coin:
                owner["inv"]["coin"] -= fee_coin
                helper["inv"]["coin"] = helper["inv"].get("coin", 0) + fee_coin
                paid = f"{fee_coin} coin"
            elif self._spend_food(owner, fee_food):
                helper["inv"]["food"] = helper["inv"].get("food", 0) + fee_food
                paid = f"{fee_food} food"
            if not paid:
                continue
            self._bump("contractor_paid" if is_contractor else "commission")
            verb = "for building their home" if is_contractor else "for helping raise their home"
            self._note("trade", f"{owner['name']} paid {helper['name']} {paid} {verb}.")
            mind.remember(helper, f"was paid {paid} for raising {owner['name']}'s home — honest work, honest pay",
                          0.6, "trade", self.clock)
            mind.remember(owner, f"paid {helper['name']} {paid} to help raise my home", 0.5, "trade", self.clock)
            rel = (helper.get("rel") or {}).get(owner["id"])
            if rel:
                rel["trades"] = rel.get("trades", 0) + 1
            self._record_trade(helper, owner)              # trade XP — the road to inventing money

    # ── god-authored building templates (the Templates god-tool) ────────────────
    # The god designs buildings in a blank-grid editor and saves them as blueprints in the
    # SAME glyph format the band's own buildings use, so a hand-drawn smithy or longhouse can
    # be placed on the map (instantly, or as a site for souls to raise) — and is stored ready
    # for the AI to draw on as a "prior" in a later phase (it ignores them for now).
    def _load_templates(self) -> None:
        try:
            with open(PATH_TEMPLATES, encoding="utf-8") as f:
                self.user_blueprints = json.load(f) or []
        except (FileNotFoundError, ValueError, OSError):
            self.user_blueprints = []
        self._register_user_blueprints()

    def _save_templates(self) -> None:
        try:
            os.makedirs(_DIR, exist_ok=True)
            tmp = PATH_TEMPLATES + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.user_blueprints, f, default=_json_safe)
            os.replace(tmp, PATH_TEMPLATES)
        except OSError as e:
            print(f"[world] template save failed: {e}")

    def _register_user_blueprints(self) -> None:
        """Inject the library into the shared BLUEPRINTS registry (keyed by id) so all the
        existing footprint-fit + placement machinery treats them like any built-in blueprint."""
        for ub in self.user_blueprints:
            BLUEPRINTS[ub["id"]] = {
                "name": ub.get("name", "Building"),
                "roof": bool(ub.get("roof", True)),
                "insulation": float(ub.get("insulation", 1.0)),
                "layout": list(ub.get("layout", [])),
                "communal": bool(ub.get("communal", False)),
                "capacity": int(ub.get("capacity", 0)),
                "user": True,
            }

    def _register_authored(self) -> None:
        """Inject the BAND's own (LLM-authored, validated) designs into BLUEPRINTS, so the build
        machinery raises them like any built-in. Mirrors _register_user_blueprints for god templates."""
        for ab in self.authored_blueprints:
            BLUEPRINTS[ab["id"]] = {
                "name": ab.get("name", "Building"), "roof": bool(ab.get("roof", True)),
                "insulation": float(ab.get("insulation", 1.0)), "layout": list(ab.get("layout", [])),
                "communal": bool(ab.get("communal", False)), "authored": True,
                "function": ab.get("function", "home"),     # the real ROLE its effect keys off (A.2)
            }

    def apply_authored_building(self, data, by="the band") -> str | None:
        """The Author→Validate→World loop for ARCHITECTURE (mirrors apply_llm_discovery for crafts):
        take an LLM-authored building DESIGN (name + purpose + glyph layout) and, only if it survives
        the validators — known glyphs & sane dims (_valid_layout) AND a usable building (floor
        enclosed, every room reachable, sane size — _validate_blueprint) — register it as a new
        blueprint the band can raise. The validators are the IMMUNE SYSTEM: a hallucinated 113-tile
        mansion or a walled-off room is set aside, never built. Returns the new blueprint id, or None.
        The mind PROPOSES; the deterministic engine DISPOSES."""
        if not isinstance(data, dict):
            return None
        name = (str(data.get("name") or "").strip())[:40]
        layout = data.get("layout")
        if isinstance(layout, list):
            layout = [str(r) for r in layout]
        if not name or not self._valid_layout(layout):       # known glyphs (. F W D O # L C), 1–16 sq
            return None
        ok, why = self._validate_blueprint(layout)           # floor enclosed, reachable from outside, sane size
        if not ok:
            self._note("design", f"a plan for a {name.lower()} was set aside — {why}.")
            self._bump("design_rejected")
            return None
        func = str(data.get("function") or "home").strip().lower()
        if func not in AUTHORABLE_FUNCTIONS:                  # the band only designs roles it understands
            func = "home"
        communal = (func != "home")                          # a workshop/store/hall is a PUBLIC work, not a home
        bid = "auth_" + uuid.uuid4().hex[:8]
        bp = {"id": bid, "name": name, "roof": True, "insulation": 1.0, "layout": layout,
              "communal": communal, "authored": True, "function": func,
              "purpose": (str(data.get("purpose") or "").strip())[:80]}
        self.authored_blueprints.append(bp)
        self._register_authored()
        self._authored_cd = self.clock + AUTHOR_COOLDOWN   # rest the design ratchet — one buys days
        self.version += 1
        self._bump("authored_building")
        role = "" if func == "home" else f" — a {func} for the band"
        self._note("design", f"{by} worked out a new kind of building — a {name.lower()}{role}"
                             + (f" ({bp['purpose']})" if bp["purpose"] else "") + ".")
        return bid

    def wants_new_building(self, p) -> bool:
        """Rule trigger for the LLM design ratchet: a SETTLED BUILDER in a band prosperous enough to
        spare the imagination — grown past AUTHOR_MIN_POP — when the band has room left in its design
        library and hasn't authored one lately (cooldown). Rare by construction: the LLM is the rare,
        expensive AUTHOR; this cheap rule decides WHEN it's worth waking. Inert until a band thrives."""
        if len(self.authored_blueprints) >= AUTHOR_MAX:
            return False
        if p.get("vocation") != "builder" or p.get("home_struct") is None:
            return False                                   # the band's settled architect, not a struggler
        if self.clock < getattr(self, "_authored_cd", 0.0):
            return False                                   # still resting from the last design
        return len(self.people) >= AUTHOR_MIN_POP          # only a grown, prospering band

    @staticmethod
    def _valid_layout(layout) -> bool:
        """A layout is a non-empty list of equal-length rows of known glyphs, ≤ 16×16."""
        if not isinstance(layout, list) or not layout or len(layout) > 16:
            return False
        w = len(layout[0])
        if not (1 <= w <= 16):
            return False
        ok = set(BLOCK_CHARS) | {GLYPH_CORE}
        return all(isinstance(r, str) and len(r) == w and all(c in ok for c in r) for r in layout)

    def list_templates(self) -> list:
        """The library for the dropdown — each with its layout so the UI can draw a thumbnail."""
        return [dict(ub) for ub in self.user_blueprints]

    def save_template(self, data: dict) -> dict:
        """Create or update a hand-authored blueprint. Returns {ok, id|error}."""
        if not isinstance(data, dict):
            return {"ok": False, "error": "bad payload"}
        layout = data.get("layout")
        if not self._valid_layout(layout):
            return {"ok": False, "error": "layout must be 1–16 rows of equal length using . F W D O # L C"}
        name = (str(data.get("name") or "").strip() or "Building")[:40]
        ub = {
            "id": data.get("id") or ("ub_" + uuid.uuid4().hex[:8]),
            "name": name,
            "layout": [str(r) for r in layout],
            "roof": bool(data.get("roof", True)),
            "insulation": max(0.0, min(1.0, float(data.get("insulation", 1.0)))),
            "communal": bool(data.get("communal", False)),
            "capacity": max(0, int(data.get("capacity", 0))),
            "t": round(self.clock, 1),
        }
        self.user_blueprints = [b for b in self.user_blueprints if b["id"] != ub["id"]]
        self.user_blueprints.append(ub)
        self._register_user_blueprints()
        self._save_templates()
        return {"ok": True, "id": ub["id"]}

    def delete_template(self, bp_id: str) -> dict:
        before = len(self.user_blueprints)
        self.user_blueprints = [b for b in self.user_blueprints if b["id"] != bp_id]
        BLUEPRINTS.pop(bp_id, None)
        self._save_templates()
        return {"ok": len(self.user_blueprints) < before}

    def place_template(self, bp_id: str, x: int, y: int, instant: bool = True, by: str = "him") -> dict:
        """Place a hand-authored building on the map at (x,y) as the footprint CENTRE. `instant`
        stamps the finished building in one stroke (god world-building); otherwise it drops an
        adoptable construction SITE the band will raise tile by tile. Returns {ok, result}, and
        on a bad spot reports WHY (off-map / over water / overlapping) — the dry-run fit check."""
        if bp_id not in BLUEPRINTS:
            return {"ok": False, "result": "no such template"}
        bp = BLUEPRINTS[bp_id]
        bw, bh = len(bp["layout"][0]), len(bp["layout"])
        ox, oy = int(x) - bw // 2, int(y) - bh // 2
        tasks, core = self._blueprint_tasks(bp_id, ox, oy)
        if not tasks:
            return {"ok": False, "result": "won't fit there — off the map, over water, or overlapping a building"}
        if not instant:
            site = {"id": "b_" + uuid.uuid4().hex[:8], "bp": bp_id, "name": bp["name"],
                    "ox": int(ox), "oy": int(oy), "by": by, "insul": float(bp.get("insulation", 1.0)),
                    "tasks": tasks, "done": False, "t": round(self.clock, 1),
                    "communal": True, "orphan": True}      # orphan+communal → a free builder adopts it
            self.sites.append(site)
            self.version += 1
            self._note("build", f"{by} marked out a {bp['name'].lower()} for the band to raise.")
            return {"ok": True, "result": f"marked out a {bp['name']} at ({x},{y}) for the band to build"}
        for t in tasks:                                    # instant: stamp every tile, finished
            if t["layer"] == "roof":
                self.roofs.add((t["x"], t["y"]))
            else:
                self.blocks[(t["x"], t["y"])] = t["code"]
        sid = self._add_structure(bp_id, ox, oy, by=by)
        self.version += 1
        self._note("build", f"{by} placed a {bp['name']} at ({x},{y}).")
        return {"ok": True, "result": f"placed a {bp['name']} at ({x},{y})", "struct": sid}

    def _work_ready(self, p, key, dt, work: float) -> bool:
        """Spread a piece of manual labour over visible game-time: returns True (and resets the
        timer) only once `work` game-minutes have accrued at this task. Keyed by `key`, so
        switching to a different job banks no stale progress (one timer slot per soul, since a
        soul does one thing at a time). Powers timed gathering, chopping, mining and building —
        so a soul is SEEN working rather than completing in a blink."""
        gw = p.get("gwork")
        if not gw or gw.get("k") != key:
            gw = {"k": key, "t": 0.0}
            p["gwork"] = gw
        gw["t"] += dt
        if gw["t"] >= work:
            gw["t"] = 0.0
            return True
        return False

    def _gather_ready(self, p, key, dt) -> bool:
        """Thin wrapper: an armful of leaves/fiber takes GATHER_WORK game-min of hand-work."""
        return self._work_ready(p, key, dt, GATHER_WORK)
        return False

    def _passable(self, nx, ny) -> bool:
        """A tile a person may step onto: in bounds, not deep water (rivers/shallows are fordable),
        and not a solid wall. Leaf panels stay PASSABLE (a soul is never trapped behind flimsy
        leaves) but the mover charges a brush cost so they're routed-around — soft collision."""
        return (self._in(nx, ny) and self.water[ny, nx] in FORDABLE_WATER
                and self.blocks.get((nx, ny)) not in SOLID_BLOCKS)

    def _step_to(self, p, nx, ny):
        """Commit a step. Wading water wets the soul AND slows them: stepping into a river/shallow
        sets a flag that costs them their next move beat (half-speed through water). Each footfall
        on dry land wears the ground a little — the band treads its routes into visible paths."""
        if self.water[ny, nx] in (WATER_RIVER, WATER_SHALLOW):
            p["wet_until"] = self.clock + WET_DURATION
            p["_wade"] = True
        else:                                                  # dry ground remembers where SHARED feet fall:
            last = self._foot_last                             # wear builds only when a DIFFERENT soul treads,
            if last.get((nx, ny)) != p["id"]:                  # so one soul pacing (or a lone household route)
                ff = self.footfall                             # never wears a path — only a route the band SHARES
                w = ff.get((nx, ny), 0.0) + 1.0
                ff[(nx, ny)] = w if w < FOOTFALL_CAP else FOOTFALL_CAP
                last[(nx, ny)] = p["id"]
        p["x"], p["y"] = nx, ny

    def _compute_path(self, x, y, gx, gy, cap=700):
        """A bounded, terrain-aware shortest path from (x,y) toward (gx,gy) — A* (Dijkstra + a
        goal heuristic) over passable tiles, capped at `cap` expansions so it never scans the whole
        2048² map. Step costs mirror the greedy mover (water/leaf cost more, roads/paths less), so
        the path prefers dry ground and the road network. If the goal can't be reached within the
        cap it heads for the nearest tile it COULD reach — so a soul always makes progress. Returns
        the list of tiles to walk (first step first), or None if it's already there / boxed in."""
        start, goal = (x, y), (gx, gy)
        if start == goal:
            return None
        pq = [(abs(gx - x) + abs(gy - y), 0.0, start)]
        came = {start: None}
        cost = {start: 0.0}
        best_t, best_h = start, abs(gx - x) + abs(gy - y)
        seen = 0
        while pq and seen < cap:
            _, c, cur = heapq.heappop(pq)
            seen += 1
            if cur == goal:
                best_t = goal
                break
            cx, cy = cur
            h = abs(gx - cx) + abs(gy - cy)
            if h < best_h:                                 # remember the closest tile reached so far
                best_h, best_t = h, cur
            for dx, dy in _STEP_DIRS:
                nx, ny = cx + dx, cy + dy
                if not self._passable(nx, ny):
                    continue
                sc = 1.0
                if self.water[ny, nx] in (WATER_RIVER, WATER_SHALLOW):
                    sc += WADE_COST                        # keep paths on dry ground (wading risks chill/illness)
                elif self.blocks.get((nx, ny)) == BLOCK_LEAF:
                    sc += LEAF_BRUSH_COST
                elif (nx, ny) in self.roads:
                    sc = max(0.1, 1.0 - ROAD_PULL)         # follow the road network where it goes our way
                elif self.footfall.get((nx, ny), 0.0) >= FOOTFALL_PATH_MIN:
                    sc = max(0.1, 1.0 - PATH_PULL)
                nc = c + sc
                if nc < cost.get((nx, ny), 1e18):
                    cost[(nx, ny)] = nc
                    came[(nx, ny)] = cur
                    heapq.heappush(pq, (nc + (abs(gx - nx) + abs(gy - ny)), nc, (nx, ny)))
        if best_t == start:
            return None
        node, path = best_t, []
        while node is not None and node != start:
            path.append(node)
            node = came.get(node)
        path.reverse()
        return path or None

    def _path_step(self, p, gx, gy):
        """The next tile toward (gx,gy) along a path that routes AROUND walls — cached on the soul so
        the search runs only when needed (path exhausted, goal moved, or the next tile got blocked),
        not every beat. Returns a neighbour tile, or None (caller falls back to the greedy step)."""
        x, y = p["x"], p["y"]
        cache = p.get("_path")
        if cache and cache["goal"] == (gx, gy) and cache["steps"]:
            nxt = cache["steps"][0]
            if abs(nxt[0] - x) + abs(nxt[1] - y) == 1 and self._passable(*nxt):
                cache["steps"].pop(0)
                return nxt
        path = self._compute_path(x, y, gx, gy)
        if not path:
            p.pop("_path", None)
            return None
        p["_path"] = {"goal": (gx, gy), "steps": path[1:]}
        return path[0]

    def _pick_roam(self, p, ax, ay, radius):
        """A reachable WANDER waypoint within `radius` of an anchor (ax,ay) — kept near HOME and inside
        the soul's leash so a wanderer never strays toward unsafe water, on dry open ground. None if
        no spot found in a few tries."""
        radius = max(float(ROAM_MIN), min(float(radius), float(ROAM_MAX)))
        for _ in range(8):
            ang = self.rng.random() * 6.283185
            dist = ROAM_MIN + self.rng.random() * (radius - ROAM_MIN)
            gx = int(ax + np.cos(ang) * dist)
            gy = int(ay + np.sin(ang) * dist)
            if (self._in(gx, gy) and self.water[gy, gx] == WATER_NONE
                    and self.blocks.get((gx, gy)) not in SOLID_BLOCKS):
                return (gx, gy)
        return None

    def _roam_delta(self, p, ax, ay, radius):
        """Wandering with no errand: the RAW delta to a self-chosen roam waypoint near the anchor
        (the mover then PATHFINDS there), picking a fresh one on arrival or when it's gone stale — so
        a wanderer crosses the ground instead of pacing, while staying within its leash of home.
        Falls back to a blind heading if no waypoint is reachable."""
        x, y = p["x"], p["y"]
        roam = p.get("roam")
        if (not roam or (abs(roam[0] - x) + abs(roam[1] - y) <= 2)
                or self.clock >= p.get("roam_until", 0.0)
                or abs(roam[0] - ax) + abs(roam[1] - ay) > radius + 3):   # waypoint drifted past the leash
            roam = self._pick_roam(p, ax, ay, radius)
            if roam is None:
                return self._explore_dir(p)
            p["roam"], p["roam_until"] = roam, self.clock + ROAM_TIME
        return (roam[0] - x, roam[1] - y)

    def _move_person(self, p, direction):
        """One cost-aware step. Each open neighbour is scored by how much HEADWAY it makes toward
        the goal (weighted) minus its travel cost (slow water, nearby danger). The best wins — so
        a soul rounds lakes and walls instead of freezing against them, skirts wolves, and prefers
        dry ground, while still fording or braving danger when that's plainly the way. With no
        heading it ROAMS along a persistent wander direction (re-rolled now and then) rather than
        jittering back and forth between two tiles — so an idle soul actually covers ground.

        For a DIRECTED move toward a real, multi-tile goal (a haul home, a remembered spot, a
        band-mate — passed as the RAW delta, |dx|+|dy| > 2) it uses the bounded PATHFINDER so a soul
        routes AROUND buildings instead of oscillating against a wall. With NO goal (a wander) it
        walks to a self-chosen ROAM waypoint, also pathfound — so it crosses the ground instead of
        pacing. Flee/repulsion and short hops pass sign-vectors (≤ 2) and keep the cheap greedy step."""
        x, y = p["x"], p["y"]
        if p.pop("_wade", False):                          # slogging through water — lose this beat
            return
        mag = (abs(int(direction[0])) + abs(int(direction[1]))) if direction else 0
        if mag > 2:                                        # DIRECTED toward a real goal → pathfind round walls
            gx, gy = x + int(direction[0]), y + int(direction[1])
            step = self._path_step(p, gx, gy)
            if step is not None:
                self._step_to(p, step[0], step[1])
                return
            # nothing reachable found → fall through to the greedy step (using the sign direction)
        else:
            p.pop("_path", None)                           # a short hop / blind amble → greedy
        if not direction or (direction[0] == 0 and direction[1] == 0):
            direction = self._explore_dir(p)               # a steady wander heading, not a dither
        sx = int(np.sign(direction[0])) if direction else 0
        sy = int(np.sign(direction[1])) if direction else 0
        # Danger only matters when a wolf is actually near THIS soul — resolve the short list ONCE
        # (the common case is none), so the per-neighbour scoring stays cheap in the hot path.
        near_wolves = [(wx, wy) for wx, wy in getattr(self, "_wolf_pos", ())
                       if abs(wx - x) + abs(wy - y) <= DANGER_AVOID_R + 1]
        water, rng = self.water, self.rng
        best, best_score = None, -1e9
        for dx, dy in _STEP_DIRS:
            nx, ny = x + dx, y + dy
            if not self._passable(nx, ny):
                continue
            cost = 1.0
            if water[ny, nx] in (WATER_RIVER, WATER_SHALLOW):
                cost += WADE_COST
            elif self.blocks.get((nx, ny)) == BLOCK_LEAF:
                cost += LEAF_BRUSH_COST                     # soft collision: prefer going ROUND a leaf wall on dry
                                                            # ground; brush through only when that's the easy way
            elif (nx, ny) in self.roads:
                cost -= ROAD_PULL                           # a real ROAD is the easiest going — souls follow it
            elif self.footfall.get((nx, ny), 0.0) >= FOOTFALL_PATH_MIN:
                cost -= PATH_PULL                           # a worn path is easy going — feet follow the beaten track
            if (nx, ny) in getattr(self, "_person_pos", ()):
                cost += PERSON_AVOID_COST                   # don't walk into another soul — route around them
            for wx, wy in near_wolves:                      # usually empty → skipped
                d = abs(wx - nx) + abs(wy - ny)
                if d < DANGER_AVOID_R:
                    cost += (DANGER_AVOID_R - d) * DANGER_AVOID_COST
            progress = dx * sx + dy * sy                    # -2..2 (0 when ambling — cost decides)
            score = progress * MOVE_PROGRESS_W - cost + float(rng.random()) * 0.5
            if score > best_score:
                best, best_score = (nx, ny), score
        if best is not None:
            self._step_to(p, best[0], best[1])

    def _decay_footfall(self):
        """Once a day: worn paths fade and grass back over where feet stop falling, while the most
        beaten tracks HARDEN into persistent ROADS. A road still trodden stays in good repair; an
        abandoned one slowly grasses over — so the road network tracks where the band really goes."""
        ff = self.footfall
        # Harden heavily-trodden ground into roads, and keep trodden roads in repair.
        for t, w in ff.items():
            if w >= ROAD_HARDEN:
                self.roads[t] = 1.0
        if self.roads:
            kept = {}
            for t, cond in self.roads.items():
                nc = min(1.0, cond + 0.3) if ff.get(t, 0.0) >= FOOTFALL_PATH_MIN else cond - ROAD_DECAY
                if nc >= ROAD_PRUNE:
                    kept[t] = nc
            self.roads = kept
        if ff:
            self.footfall = {t: w * FOOTFALL_DECAY for t, w in ff.items() if w * FOOTFALL_DECAY >= FOOTFALL_PRUNE}
            self._foot_last = {t: w for t, w in self._foot_last.items() if t in self.footfall}  # drop stale walkers

    def _paths_payload(self):
        """The worn tiles to draw as paths — those above FOOTFALL_PATH_MIN, busiest first, capped,
        each with a 0..1 intensity so the renderer can fade a faint trail into a beaten track."""
        worn = [(t, w) for t, w in self.footfall.items() if w >= FOOTFALL_PATH_MIN]
        worn.sort(key=lambda kw: -kw[1])
        return [[t[0], t[1], round(min(1.0, w / FOOTFALL_CAP), 2)] for t, w in worn[:FOOTFALL_SEND_CAP]]

    # ── physiological reserves (the body layer beneath comfort) ──────────────────
    def _ensure_body(self, p):
        """Default the physiological reserves for any person dict that predates them (legacy
        saves, or a god-spawned soul that skipped _add_person). Seed each reserve from its
        comfort signal so a thirsty old-save soul starts plausibly low, not brimming full."""
        if "hydration" in p:
            return
        p["hydration"] = max(0.15, 1.0 - 0.5 * p.get("thirst", 0.2))
        p["satiety"] = max(0.15, 1.0 - 0.5 * p.get("hunger", 0.2))
        p["stamina"] = max(0.15, 1.0 - 0.5 * p.get("fatigue", 0.2))
        p.setdefault("hp", 1.0)

    @staticmethod
    def _refill(p, key, amount):
        """Restore a reserve by `amount`, but never above its VITALITY ceiling: an exhausted
        body can't fully rehydrate or refeed (stamina caps hydration/satiety) and a malnourished
        one can't fully rest off its fatigue (satiety caps stamina). The ceiling only limits the
        top-up — it never yanks an already-higher reserve down."""
        # Cap floor stays ABOVE the danger line (DANGER_RESERVE) so the coupling can sap a
        # depleted body's recovery without trapping a reserve in permanent danger (a spiral).
        if key == "stamina":
            cap = 0.55 + 0.45 * p.get("satiety", 1.0)
        else:                                                # hydration / satiety
            cap = 0.55 + 0.45 * p.get("stamina", 1.0)
        target = max(p.get(key, 0.0), cap)                   # don't pull a higher reserve down
        p[key] = min(p.get(key, 0.0) + amount, target)

    # ── illness: waterborne disease (survival realism) ───────────────────────────
    def _maybe_infect(self, p):
        """A raw drink may carry sickness. No effect if already ill or still immune to every
        strain. On infection the disease incubates silently before symptoms strike."""
        if p.get("illness") or self.rng.random() >= WATER_INFECT_CHANCE:
            return
        imm = p.get("immune", {})
        choices = [d for d in WATERBORNE if self.clock >= imm.get(d, 0.0)]
        if not choices:
            return
        d = str(self.rng.choice(choices))
        spec = DISEASE[d]
        p["illness"] = {"d": d, "infected_t": self.clock,
                        "onset_t": self.clock + spec["incub"] * 1440.0,
                        "end_t": self.clock + (spec["incub"] + spec["dur"]) * 1440.0,
                        "known": False}

    def _maybe_poison(self, p):
        """Eating a poisonous bush may bring on a sharp berry-sickness (unless already ill or
        still immune from a past bout). Reuses the same disease machinery as foul water."""
        if p.get("illness") or self.rng.random() >= BERRY_POISON_CHANCE:
            return False
        if self.clock < p.get("immune", {}).get("berry_sickness", 0.0):
            return False
        spec = DISEASE["berry_sickness"]
        p["illness"] = {"d": "berry_sickness", "infected_t": self.clock,
                        "onset_t": self.clock + spec["incub"] * 1440.0,
                        "end_t": self.clock + (spec["incub"] + spec["dur"]) * 1440.0,
                        "known": False}
        return True

    # ── berry foraging & bush lore (P3) ──────────────────────────────────────────
    def _learn_bush(self, p, b, bad: bool):
        """Record what a bush turned out to be, so the soul shuns a bad one and returns to a
        good one. A bush that ever sickened them is remembered as bad for good."""
        lore = p.setdefault("berry_lore", {})
        key = f"{b['x']},{b['y']}"
        if bad or lore.get(key) != "bad":                 # 'bad' is sticky — never downgraded to good
            lore[key] = "bad" if bad else "good"

    def _forage_bush(self, p, b):
        """Pick a ripe bush: slake hunger, pocket a handful, set it re-ripening, and learn what
        it was — a safe bush becomes trusted, a poisonous one that bites becomes a bush to shun."""
        b["ripe_t"] = self.clock + BERRY_REGROW_DAYS * 1440.0   # picked — re-ripens over days
        p["hunger"] = max(0.0, p["hunger"] - BERRY_HUNGER_RELIEF)
        self._refill(p, "satiety", BERRY_SATIETY)
        p["inv"]["food"] = p["inv"].get("food", 0) + 1          # a handful for the pack/larder too
        if b["poison"]:
            if self._maybe_poison(p):                          # unlucky — it bites back
                self._learn_bush(p, b, bad=True)               # now a bush they'll shun
                mind.remember(p, "those berries turned my stomach — I'll shun that bush",
                              0.7, "illness", self.clock)
        else:
            self._learn_bush(p, b, bad=False)                  # safe — a bush worth returning to
        self.version += 1

    # ── hunting, fishing & cooking (P4) ──────────────────────────────────────────
    def _maybe_taint(self, p):
        """A raw meat/fish meal may bring on tainted_gut (parasites/spoilage). No effect if
        already ill or still immune. The lesson: cook flesh over the hearth."""
        if p.get("illness") or self.rng.random() >= RAW_FLESH_SICKEN:
            return False
        if self.clock < p.get("immune", {}).get("tainted_gut", 0.0):
            return False
        spec = DISEASE["tainted_gut"]
        p["illness"] = {"d": "tainted_gut", "infected_t": self.clock,
                        "onset_t": self.clock + spec["incub"] * 1440.0,
                        "end_t": self.clock + (spec["incub"] + spec["dur"]) * 1440.0,
                        "known": False}
        return True

    def _party_near(self, x, y, exclude=None) -> int:
        """How many people are within hunting-party range of (x,y) — the band-mates on hand to
        join a hunt (or, at the quarry, the party available to bring big game down)."""
        return sum(1 for q in self.people if q is not exclude
                   and abs(q["x"] - x) + abs(q["y"] - y) <= HUNT_PARTY_RANGE)

    def _nearest_prey(self, p, max_d=HUNT_VISION, big_ok=True):
        """The nearest huntable animal within `max_d`, or None. With `big_ok` False (a lone
        hunter), deer are skipped — only small game (rabbits) is worth a solo chase."""
        x, y = p["x"], p["y"]
        best = None; best_d = max_d + 1
        for a in self.animals:
            if a["sp"] not in ("rabbit", "deer"):
                continue
            if a["sp"] == "deer" and not big_ok:
                continue
            d = abs(a["x"] - x) + abs(a["y"] - y)
            if d < best_d:
                best, best_d = a, d
        return best

    def _resolve_hunt_strike(self, p, prey) -> bool:
        """One strike at adjacent quarry. Big game (deer) needs a PARTY of HUNT_PARTY_MIN on hand
        or the strike just scatters it; a felled deer is SHARED among the party, while small game
        is the lone hunter's own. Returns True on a kill. A spear makes the strike far surer."""
        has_spear = p["inv"].get("crude_spear", 0) > 0
        big = prey["sp"] == "deer"
        party = ([q for q in self.people
                  if abs(q["x"] - prey["x"]) + abs(q["y"] - prey["y"]) <= HUNT_PARTY_RANGE]
                 if big else [p])
        if big and len(party) < HUNT_PARTY_MIN:
            return False                                       # too few hands — the deer breaks away
        if self.rng.random() >= (HUNT_KILL_SPEAR if has_spear else HUNT_KILL_BARE):
            return False
        self.animals = [a for a in self.animals if a["id"] != prey["id"]]
        yld = HUNT_MEAT_YIELD.get(prey["sp"], 2) + (1 if has_spear else 0)
        if big and len(party) > 1:                             # share the carcass among the party
            share = max(1, yld // len(party))
            for q in party:
                q["inv"]["meat"] = q["inv"].get("meat", 0) + share
            self._note("hunt", f"{p['name']}'s party brought down a {prey['sp']}.")
        else:
            p["inv"]["meat"] = p["inv"].get("meat", 0) + yld
            self._note("hunt", f"{p['name']} brought down a {prey['sp']}.")
        self.version += 1
        return True

    def _hunt(self, p, max_d=HUNT_VISION):
        """Pursue the nearest game within `max_d`; a body action toward it, or a strike when
        adjacent (the kill is resolved in the action handler). None when there's nothing to chase.
        Big game is only pursued when a band-mate is near enough to make a party."""
        big_ok = self._party_near(p["x"], p["y"], exclude=p) >= (HUNT_PARTY_MIN - 1)
        prey = self._nearest_prey(p, max_d, big_ok=big_ok)
        if prey is None:
            return None
        dx, dy = prey["x"] - p["x"], prey["y"] - p["y"]
        if abs(dx) + abs(dy) <= 1:
            p["_prey"] = prey["id"]                 # stash the quarry for the strike handler
            return "hunt", None
        return "hunt", (dx, dy)

    def _fish(self, p, drinkable, lx, ly):
        """Fish the water's edge: cast in place when beside water, else step toward the nearest
        water in sight. Returns a body action, or None if no water is reachable in sight."""
        if drinkable[ly, lx]:
            return "fish", None
        d = self._nearest_local(drinkable, lx, ly)
        if d:
            return "fish", d
        return None

    def _cook_at_home(self, p):
        """Tend the hearth fire: turn raw meat/fish into food. A big raw haul is DRIED into
        preserved stores (dried_meat/dried_fish — they never spoil, the larder for winter); a
        smaller catch is simply cooked for eating now. Runs each rest-tick beside the hearth."""
        inv = p["inv"]
        for raw, cooked, dried, cost in (("meat", "cooked_meat", "dried_meat", 2),
                                         ("fish", "cooked_fish", "dried_fish", 2)):
            n = inv.get(raw, 0)
            if n >= max(PRESERVE_AT, cost):              # surplus — preserve it against the lean months
                inv[raw] = n - cost
                if inv[raw] <= 0:
                    inv.pop(raw, None)
                inv[dried] = inv.get(dried, 0) + 1
            elif n > 0:                                  # a little — cook a meal
                inv[raw] = n - 1
                if inv[raw] <= 0:
                    inv.pop(raw, None)
                inv[cooked] = inv.get(cooked, 0) + 1

    def _tick_spoilage(self, dt_days):
        """Fresh food rots over game-time (raw flesh fastest, then cooked, then produce); only
        preserved stores keep. Decays each perishable stack — in the pack AND the home larder —
        proportionally to elapsed time over its shelf life. Cheap: a few people × a few items."""
        if dt_days <= 0:
            return
        has_store = self._has_function("storehouse")              # the band's stores keep better with one
        for p in self.people:
            for hold in (p.get("inv"), p.get("store")):
                if not hold:
                    continue
                stored = hold is p.get("store")
                for item, shelf in PERISHABLE.items():
                    n = hold.get(item, 0)
                    if n <= 0:
                        continue
                    rate = dt_days / shelf
                    if stored and has_store:                       # a proper storehouse slows spoilage
                        rate *= STORE_SPOIL_FACTOR
                    lost = int(n * rate)
                    if self.rng.random() < (n * rate - lost):
                        lost += 1
                    if lost > 0:
                        hold[item] = n - lost
                        if hold[item] <= 0:
                            hold.pop(item, None)

    # ── Phase B: LAWS — the LLM-as-leader codifies a norm; the engine enforces it (reputation only) ──
    def _band_leader(self):
        """The soul the band would heed as its law-giver: the highest-renown able adult, if their
        standing clears LAW_MIN_RENOWN. None if no one yet carries that authority."""
        best, br = None, LAW_MIN_RENOWN
        for q in self.people:
            if q["age"] < ADULT_AGE or q.get("illness"):
                continue
            r = q.get("renown", 0.0)
            if r >= br:
                best, br = q, r
        return best

    def _law_violates(self, p, norm) -> bool:
        """Does this soul stand in breach of `norm` right now? Only able adults are ever judged — a
        child, the sick or the badly hurt are never in the wrong for failing the band."""
        if p["age"] < ADULT_AGE or p.get("illness") or p.get("hp", 1.0) <= 0.5:
            return False
        if norm == "hoarding":
            return (p.get("store", {}).get("food", 0) > LAW_HOARD_CAP
                    and self.granary.get("store", {}).get("food", 0) < LAW_GRANARY_LOW)
        if norm == "labour":
            return p.get("aided", 0.0) < LAW_LABOUR_MIN and any(not s.get("done") for s in self.sites)
        if norm == "peace":
            return sum(1 for r in (p.get("rel") or {}).values()
                       if r.get("sentiment", 0.0) < -0.3) >= LAW_PEACE_FOES
        return False

    def _sanction(self, p, norm) -> None:
        """Soft enforcement: a law-breaker sheds a little standing and a little of the band's regard
        — disapproval, never punishment that could endanger a life (mirrors the granary norm)."""
        p["renown"] = max(0.0, p.get("renown", 0.0) - LAW_SANCTION)
        self._note("norm", f"the band marks that {p['name']} {LAW_REPROACH.get(norm, 'breaks our law')}.")
        for q in self.people:
            if q is p:
                continue
            rel = (q.get("rel") or {}).get(p["id"])
            if rel:
                rel["sentiment"] = max(-1.0, rel["sentiment"] - GOV_DISAPPROVAL)
        self._bump("law_enforced")

    def _law_problem(self):
        """A recurring WRONG the band has no law against yet — the norm the most able adults are in
        breach of, if that share clears LAW_PROBLEM_FRAC. Returns the norm id, or None. This is the
        cheap rule that tells the leader there's something worth legislating."""
        adults = [q for q in self.people if q["age"] >= ADULT_AGE and not q.get("illness")
                  and q.get("hp", 1.0) > 0.5]
        if len(adults) < GOV_MIN_BAND:
            return None
        enacted = {law["norm"] for law in self.laws}
        worst, worst_frac = None, LAW_PROBLEM_FRAC
        for norm in LAW_NORMS:
            if norm in enacted:
                continue
            frac = sum(1 for q in adults if self._law_violates(q, norm)) / len(adults)
            if frac >= worst_frac:
                worst, worst_frac = norm, frac
        return worst

    def wants_new_law(self, p) -> bool:
        """Rule trigger for the LLM legislator: p is the band's recognised LEADER, a recurring wrong
        goes un-named, the band has room for another law, and lawmaking has rested (cooldown). Rare
        by construction — one law is weighty and stands for days."""
        if len(self.laws) >= LAW_MAX or self.clock < getattr(self, "_law_cd", 0.0):
            return False
        leader = self._band_leader()
        if leader is None or leader["id"] != p["id"]:
            return False
        return self._law_problem() is not None

    def apply_authored_law(self, data, by="the band") -> str | None:
        """Author→Validate→World for LAWS (mirrors apply_authored_building): the leader's design names
        a wrong to forbid; it is ENACTED only if the wrong is one the engine can actually judge (norm
        in LAW_NORMS) and isn't already law. The validator is the immune system — an unenforceable or
        duplicate law is set aside. Returns the norm id enacted, or None."""
        if not isinstance(data, dict):
            return None
        norm = str(data.get("norm") or "").strip().lower()
        if norm not in LAW_NORMS or any(law["norm"] == norm for law in self.laws):
            return None
        name = (str(data.get("name") or "").strip())[:48] or f"the law against {norm}"
        value = (str(data.get("value") or "").strip())[:90]
        self.laws.append({"norm": norm, "name": name, "value": value, "by": by, "enacted": self.clock})
        self._law_cd = self.clock + LAW_COOLDOWN
        self.version += 1
        self._bump("law_enacted")
        self._note("law", f"{by} gave the band a law — {name} ({LAW_REPROACH.get(norm, norm)}).")
        return norm

    def _tick_governance(self):
        """The band's first NORM, judged once a game-day: pull your weight at the common granary.
        Steady contributors earn standing; an ABLE soul who keeps drawing from the commons without
        ever filling it draws quiet disapproval — a dip in their renown and in how bandmates regard
        them. Soft enforcement through reputation alone (never punishment that could endanger a
        life), on a rolling window so it's recent conduct that's weighed, not a lifetime ledger."""
        if self.granary.get("x") is None:
            return
        adults = [q for q in self.people if q["age"] >= ADULT_AGE]
        if len(adults) < GOV_MIN_BAND:
            return
        for p in self.people:
            given, taken = p.get("gran_given", 0.0), p.get("gran_taken", 0.0)
            able = p["age"] >= ADULT_AGE and not p.get("illness") and p.get("hp", 1.0) > 0.5
            if given >= GOV_CONTRIB and given >= taken:
                self._earn_renown(p, RENOWN_GAIN["contribute"],
                                  "I give more to the commons than I take — the band leans on me")
            elif able and taken >= GOV_FREERIDE and given <= taken * 0.25:
                p["renown"] = max(0.0, p.get("renown", 0.0) - GOV_FREERIDE_COST)
                self._note("norm", f"the band notes {p['name']} leans on the commons but rarely fills it.")
                for q in self.people:
                    if q is p:
                        continue
                    rel = (q.get("rel") or {}).get(p["id"])
                    if rel:                                    # only those who actually know them judge
                        rel["sentiment"] = max(-1.0, rel["sentiment"] - GOV_DISAPPROVAL)
                self._bump("gov_judged")
            for law in self.laws:                          # any ENACTED law judges its wrong (soft, reputation)
                if self._law_violates(p, law["norm"]):
                    self._sanction(p, law["norm"])
            p["gran_given"], p["gran_taken"] = given * GOV_LEDGER_DECAY, taken * GOV_LEDGER_DECAY
            p["aided"] = p.get("aided", 0.0) * GOV_LEDGER_DECAY   # the labour-credit window rolls too

    def _tick_pests(self):
        """A fat larder of fresh food draws vermin. Each game-day, a store past the threshold
        risks a raid that carries off a chunk — the price of hoarding perishables in the open,
        and a nudge toward DRYING the surplus (preserved goods don't tempt them)."""
        has_store = self._has_function("storehouse")             # a storehouse keeps vermin off the stores
        for p in self.people:
            store = p.get("store")
            if not store:
                continue
            food = store.get("food", 0)
            if food <= PEST_STORE_THRESHOLD:
                continue
            chance = min(0.85, PEST_RAID_CHANCE * (food / PEST_STORE_THRESHOLD))
            if has_store:
                chance *= STORE_PEST_FACTOR
            if self.rng.random() < chance:
                taken = max(1, int(food * PEST_RAID_LOSS))
                store["food"] = food - taken
                if store["food"] <= 0:
                    store.pop("food", None)
                self._note("pest", f"vermin raided {p['name']}'s stores, carrying off {taken} food.")
                mind.remember(p, "vermin have been at my stores — I must dry and stow food better",
                              0.6, "loss", self.clock)

    def _eat_cooked(self, p):
        """Eat a cooked meal from the pack — the most filling, and wholly safe."""
        inv = p["inv"]
        for k in ("cooked_meat", "cooked_fish"):
            if inv.get(k, 0) > 0:
                inv[k] -= 1
                if inv[k] <= 0:
                    inv.pop(k, None)
                p["hunger"] = max(0.0, p["hunger"] - COOKED_HUNGER_RELIEF)
                self._refill(p, "satiety", COOKED_SATIETY)
                return True
        return False

    def _eat_raw(self, p):
        """Eat raw meat/fish — filling, but it may bring on tainted_gut. A last resort."""
        inv = p["inv"]
        for k in ("meat", "fish"):
            if inv.get(k, 0) > 0:
                inv[k] -= 1
                if inv[k] <= 0:
                    inv.pop(k, None)
                p["hunger"] = max(0.0, p["hunger"] - RAW_HUNGER_RELIEF)
                self._refill(p, "satiety", RAW_SATIETY)
                if self._maybe_taint(p):
                    mind.remember(p, "the raw meat sat ill in me — I should cook it next time",
                                  0.7, "illness", self.clock)
                return True
        return False

    def _ensure_bush_index(self):
        """A spatial bucket of berry bushes by cell, so _nearest_bush checks only the cells near a
        soul instead of every bush on the map (perf at town scale). Bushes are static, so it rebuilds
        only when the list changes. Survival-NEUTRAL — same bushes considered, just found faster."""
        if getattr(self, "_bush_idx_n", -1) == len(self.berry_bushes):
            return
        c = max(1, BERRY_SEEK_RANGE)
        idx = {}
        for b in self.berry_bushes:
            idx.setdefault((b["x"] // c, b["y"] // c), []).append(b)
        self._bush_idx, self._bush_idx_n = idx, len(self.berry_bushes)

    def _nearest_bush(self, p):
        """The best ripe bush worth foraging within reach: nearest one this soul doesn't KNOW to
        be poisonous (a known-good bush is preferred, an unknown one is a gamble worth taking).
        Returns the bush dict or None."""
        self._ensure_bush_index()
        lore = p.get("berry_lore", {})
        x, y = p["x"], p["y"]
        c = max(1, BERRY_SEEK_RANGE)
        cx, cy = x // c, y // c
        best = None; best_score = None
        for gx in (cx - 1, cx, cx + 1):                   # cell == seek range, so 3×3 covers the reach
            for gy in (cy - 1, cy, cy + 1):
                for b in self._bush_idx.get((gx, gy), ()):
                    d = abs(b["x"] - x) + abs(b["y"] - y)
                    if d > BERRY_SEEK_RANGE or not self._bush_ripe(b):
                        continue
                    tag = lore.get(f"{b['x']},{b['y']}")
                    if tag == "bad":
                        continue                          # known poison — shun it
                    score = d - (BERRY_GOOD_BIAS if tag == "good" else 0)   # mild bias toward a trusted bush
                    if best_score is None or score < best_score:
                        best, best_score = b, score
        return best

    def _berry_seek(self, p):
        """If a worthwhile berry bush is in reach, a body action toward it (or foraging it when
        underfoot); else None so the soul falls back on grazing/seeking as before."""
        b = self._nearest_bush(p)
        if b is None:
            return None
        dx, dy = b["x"] - p["x"], b["y"] - p["y"]
        if dx == 0 and dy == 0:
            return "forage_berry", None
        return "seek_berry", (dx, dy)

    def _tended(self, p) -> bool:
        """Is a well band-mate keeping vigil at this sick soul's side — standing within a tile
        with the resolve to tend them? Their nursing eases the illness."""
        for q in self.people:
            if q is p:
                continue
            if (q.get("intention") or {}).get("kind") == "tend" \
                    and (q["intention"].get("target") == p["id"]) \
                    and abs(q["x"] - p["x"]) + abs(q["y"] - p["y"]) <= 1:
                return True
        return False

    def _illness_factor(self, p):
        """How much a current sickness accelerates fluid/appetite loss (1.0 when healthy or
        still incubating; >1 once symptoms set in)."""
        ill = p.get("illness")
        if ill and self.clock >= ill["onset_t"]:
            return DISEASE[ill["d"]]["drain"]
        return 1.0

    def _tick_illness(self, p, dt_game_min):
        """Advance any sickness: surface it to the soul at onset (they KNOW they're ill and get
        a vague hint at the cause), erode health while symptomatic, and recover — with a spell
        of immunity — once it has run its course."""
        ill = p.get("illness")
        if not ill:
            return
        now = self.clock
        if now < ill["onset_t"]:
            return                                           # still incubating, no symptoms yet
        if not ill.get("known"):
            ill["known"] = True
            self._note("illness", f"{p['name']} has fallen ill.")
            mind.remember(p, f"I've fallen ill — {DISEASE[ill['d']]['hint']}", 0.9, "illness", now)
        # A caretaker keeping vigil at the sick soul's side eases the illness: the body loses
        # less health and weathers it sooner — so nursing visibly matters (#11).
        tended = self._tended(p)
        hp_rate = DISEASE[ill["d"]]["hp"] * (0.5 if tended else 1.0)
        p["hp"] = max(0.0, p["hp"] - hp_rate * dt_game_min)
        if tended:
            ill["end_t"] -= TEND_RECOVERY_BOOST * dt_game_min   # the sickness runs its course faster
        if now >= ill["end_t"]:                              # weathered it
            p.setdefault("immune", {})[ill["d"]] = now + IMMUNITY_DAYS * 1440.0
            p["illness"] = None
            mind.remember(p, "the sickness has passed — I feel myself again", 0.7, "illness", now)
            if not p.get("knows_boil"):
                # The hard-won lesson: bad water made me ill — fire must clean it. (Spreads to
                # others by teaching, like any craft; grants the campfire recipe to build a hearth.)
                p["knows_boil"] = True
                if "campfire" not in p.setdefault("recipes", []):
                    p["recipes"].append("campfire")
                mind.remember(p, "I will boil my water over a fire from now on — raw water sickens", 0.85, "build", now)

    # ── live-sim material sourcing (clay/sand/flint/ore beyond wood & fiber) ────
    def _fill_containers(self, p):
        """Top a person's carried water up to the capacity of the flasks/skins they hold."""
        inv = p["inv"]
        cap = sum(crafting.CONTAINER_WATER.get(c, 0) * inv.get(c, 0) for c in crafting.CONTAINER_WATER)
        if cap > 0 and inv.get("water", 0) < cap:
            inv["water"] = cap

    # ── Crafting takes time ──────────────────────────────────────────────────────
    # A craft is no longer instant: beginning one pays its inputs up front and starts a
    # game-minute timer (crafting.craft_minutes). The worker holds station with a ⚙ over
    # their head until it finishes, when the item is granted. State lives on the person as
    # p["craft"] = {rid, out, qty, left, total}; None when idle.
    _CRAFT_DONE_NOTE = {
        "crude_axe": "shaped my first axe from wood",
        "rope": "twisted fiber into a length of rope",
        "leaf_flask": "made a leaf flask to carry water",
        "forage_sack": "wove a sack to forage more",
        "sleeping_mat": "wove a mat to rest well anywhere",
    }

    def _begin_craft(self, p, rid: str, stations=()) -> bool:
        """Start crafting `rid` if nothing's already underway: pay the inputs now and set a
        timer. Returns True if a craft is active afterwards (so the caller holds position).
        `stations` are the crafting stations in reach (from a nearby workshop/smithy/hearth) —
        what unlocks the deeper tree beyond bare-handed work."""
        if p.get("craft"):
            return True
        inv = p["inv"]
        if rid == "crude_axe":                                  # the first tool — must be WORKED OUT first
            if not self._person_knows(p, "crude_axe") or inv.get("wood", 0) < BUILD["axe_wood"]:
                return False
            inv["wood"] -= BUILD["axe_wood"]
            out, qty = "axe", 1
        else:
            if rid in crafting.SURVIVAL_DISCOVERIES and not self._person_knows(p, rid):
                return False                                     # can't make what they haven't worked out
            if not crafting.can_craft(inv, rid, stations=stations, tools=None):
                return False
            r = crafting.RECIPES[rid]
            for k, n in r["inp"].items():                       # reserve inputs up front
                inv[k] = inv.get(k, 0) - n
                if inv[k] <= 0:
                    del inv[k]
            out, qty = r["out"], r["qty"]
        mins = crafting.craft_minutes(rid)
        p["craft"] = {"rid": rid, "out": out, "qty": qty, "left": mins, "total": mins}
        self.version += 1
        return True

    def _near_workshop(self, p) -> bool:
        """Is the soul working within reach of a finished workshop (the band's bench) — the built-in
        form OR one the band designed itself (A.2)?"""
        return self._near_function(p, "workshop", WORKSHOP_RANGE)

    def _place_station_obj(self, p, kind: str) -> None:
        """Set a freshly-crafted personal station down as a VISIBLE object in the soul's home — so
        a workbench/furnace/kiln actually appears in the dwelling instead of living as an abstract
        list. Prefers a clear interior floor tile; failing that, just outside the door."""
        taken = set(self.station_objs)
        interior = [t for t in self._home_interior(p)
                    if t not in taken and self.decor.get(t) != "bed"]
        spot = interior[0] if interior else None
        if spot is None:
            hx, hy = p.get("home", (p["x"], p["y"]))
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                t = (hx + dx, hy + dy)
                if self._in(t[0], t[1]) and t not in taken and self.water[t[1], t[0]] == WATER_NONE:
                    spot = t
                    break
        if spot is not None:
            self.station_objs[spot] = kind
            self.version += 1

    def _stations_for(self, p) -> set:
        """The crafting STATIONS a soul can use right now: the campfire of its own hearth, plus
        any communal craft-building (workshop → bench/kiln/loom/tannery, smithy → furnace/forge/
        anvil) it's standing near. This is the gate that opens the deep crafting tree."""
        st = set()
        if p.get("hearth"):
            st.add("campfire")
        x, y = p["x"], p["y"]
        # PERSONAL stations a soul has built for itself (a workbench, then a furnace, …) — usable
        # when it's home. This is how a soul climbs the tree WITHOUT waiting on a communal workshop:
        # it makes its own bench first, and each station it builds unlocks the next.
        if p.get("stations") and p.get("home") \
                and abs(p["home"][0] - x) + abs(p["home"][1] - y) <= CRAFT_STATION_RANGE:
            st.update(p["stations"])
        for s in self.sites:
            if not s.get("done"):
                continue
            kinds = CRAFT_BUILDING_STATIONS.get(s.get("bp"))
            if kinds and abs(s["ox"] - x) + abs(s["oy"] - y) <= CRAFT_STATION_RANGE:
                st.update(kinds)
        return st

    def _tech_craft_decide(self, p, getters):
        """Climb the DEEP crafting tree when settled and beside a station. The whole 130-recipe
        tree is now ACCESSIBLE — gated only by the stations in reach (from the workshop/smithy/
        hearth), the tools held, and the materials to hand, not by secret knowledge. Walks the
        ordered TECH_LADDER and makes the first thing it usefully can; gathers a missing common
        raw (wood/stone/fiber/leaves) to get there. Metal waits on ore the soul actually has — no
        far expeditions that would strand it. Returns a body action, or None if nothing to do here."""
        if p.get("craft"):
            return "craft", None
        # Climb the tree only from a position of REAL safety — well-watered, well-fed, well-rested.
        # The tree is a luxury, never a need, so a fetch errand for raws can't strand a soul into
        # thirst far from the river. (Survival lesson: an uncapped errand will kill — see the
        # homeward-walk cap. This keeps deep-crafting strictly above the survival floor.)
        if p.get("thirst", 0) > 0.25 or p.get("hunger", 0) > 0.25 or p.get("fatigue", 0) > 0.55:
            return None
        stations = self._stations_for(p)
        built = set(p.get("stations", []))
        inv = p["inv"]
        tool_rids = {rid for rid, _ in self._TOOLS}
        SAFE_RAWS = {"wood", "stone", "fiber", "leaves"}
        for rid in crafting.TECH_LADDER:
            r = crafting.recipe(rid)
            if not r or r["out"] in crafting.STRUCTURE_KINDS:  # don't auto-raise big structures/homes
                continue
            if rid in tool_rids and not self._can_make_tools(p):
                continue                                       # tools are the toolmaker's craft (tool-gating)
            is_station = r["out"] in crafting.STATION_KINDS
            if is_station:
                if r["out"] in built or inv.get(rid, 0) >= 1:  # one of each station is enough — keep climbing
                    continue
            else:
                cap = 3 if r["tier"] <= 1 else 1               # keep a few staples; one of the finer goods
                if inv.get(rid, 0) >= cap:
                    continue
            if r["station"] and r["station"] not in stations:
                continue
            if r["tool"] and r["tool"] not in crafting.tool_caps(inv):
                continue
            if crafting.can_craft(inv, rid, stations=stations):
                # CRAFTING A STATION is how the climb begins: the first workbench needs no station,
                # and once built it unlocks the workbench tier; then a furnace, a kiln, and so on —
                # a soul builds its own way up the tree, no communal workshop required.
                if self._begin_craft(p, rid, stations=stations):
                    self._bump("tech_craft")
                    return "craft", None
            # Can't yet — fetch ONE missing COMMON raw to move toward it; anything needing ore/
            # clay/sand or an intermediate we lack is left for an earlier ladder rung (or a god).
            for mat in crafting.missing(inv, rid):
                if mat in SAFE_RAWS and mat in getters:
                    return getters[mat]()
        return None

    def _advance_craft(self, p, dt_game_min: float) -> bool:
        """Tick an in-progress craft; grant the item and clear the state when it finishes.
        Returns True on the tick it completes. Working by the communal workshop speeds it up."""
        c = p.get("craft")
        if not c:
            return False
        speed = WORKSHOP_CRAFT_SPEED if self._near_workshop(p) else 1.0
        if self._powered(p["x"], p["y"]):                # electric tools speed every craft
            speed *= POWER_CRAFT_SPEED
        c["left"] = max(0.0, c["left"] - dt_game_min * speed)
        if c["left"] > 0:
            return False
        p["inv"][c["out"]] = p["inv"].get(c["out"], 0) + c["qty"]
        # A crafted STATION (workbench/furnace/kiln/forge/loom/tannery/anvil) is PLACED at the
        # soul's home as a VISIBLE object and thereafter opens its tier of the tree there — the
        # soul's own climb. The kind is also indexed on the soul for the fast station-in-reach gate.
        if c["out"] in crafting.STATION_KINDS:
            stations = p.setdefault("stations", [])
            if c["out"] not in stations:
                stations.append(c["out"])
                self._place_station_obj(p, c["out"])
        self.version += 1
        rid = c["rid"]
        p["craft"] = None
        nice = rid.replace("_", " ")
        self._note("craft", f"{p['name']} finished a {nice}.")
        mind.remember(p, self._CRAFT_DONE_NOTE.get(rid, f"crafted a {nice} with my own hands"),
                      0.6, "craft", self.clock)
        return True

    def _craft_known(self, p, rid: str) -> bool:
        """Craft a recipe the band has discovered, consuming inputs from the person's pack.
        Tier-0 survival crafts need no station; returns True on success. (Instant — used by
        tests and any path that wants an immediate result; the live body uses _begin_craft
        so crafting takes time.)"""
        if rid not in self.known_recipes:
            return False
        ok = crafting.do_craft(p["inv"], rid, stations=(), tools=None)
        if ok:
            self.version += 1
        return ok

    # ── Recipe knowledge: personal, discovered, taught, logged ───────────────────
    # The felt problem that prompts each experiment — invention is MOTIVATED, so a soul
    # works out the water flask when thirst has been biting, not by idle luck.
    _DISCOVERY_TRIGGER = {
        "leaf_flask":   lambda p, night: p.get("thirst", 0) > 0.28,
        "forage_sack":  lambda p, night: p.get("hunger", 0) > 0.28,
        "sleeping_mat": lambda p, night: p.get("fatigue", 0) > 0.32,
        "campfire":     lambda p, night: night or p.get("fatigue", 0) > 0.45,
    }
    _DISCOVERY_RATIONALE = {
        "leaf_flask":   "If I fold broad leaves and bind them with cord, they might hold water to carry.",
        "forage_sack":  "A pouch woven of fibre and cord could carry far more than my two hands.",
        "sleeping_mat": "Leaves layered over fibre would make a mat to rest on, away from home.",
        "campfire":     "Stack wood on stone and work it hard enough — maybe I can keep the fire.",
    }

    def _person_knows(self, p, rid: str) -> bool:
        return rid in crafting.STARTER_RECIPES or rid in p.get("recipes", [])

    def _person_unsolved(self, p) -> list:
        known = p.get("recipes", [])
        return [rid for rid in crafting.SURVIVAL_DISCOVERIES if rid not in known]

    def _ledger_add(self, entry: dict) -> None:
        self.ledger.append(entry)
        if len(self.ledger) > LEDGER_CAP:
            del self.ledger[0]

    def _grant_recipe(self, p, rid: str, via: str, rationale: str = "") -> bool:
        """Teach/record a survival craft to ONE soul: add it to their knowledge, burn a proud
        (or grateful) memory, and write it into the Ledger of Making. Returns True if new to
        them. `via` ∈ worked out | reasoned out | taught by <name>."""
        if not rid or self._person_knows(p, rid):
            return False
        p.setdefault("recipes", []).append(rid)
        self.known_recipes.add(rid)
        name = rid.replace("_", " ")
        problem = crafting.SURVIVAL_DISCOVERIES.get(rid, "")
        first = not any(e.get("rid") == rid for e in self.ledger if e.get("kind") == "made")
        mind.remember(p, f"I {via} how to make a {name}" + (f" — for {problem}" if problem else ""),
                      0.9, "discovery", self.clock)
        if via in ("worked out", "reasoned out"):
            mind.speak(p, f"I've worked out how to make a {name}!", self.clock)
            vals = p.setdefault("values", {t: 0.0 for t in mind.TRAITS})
            vals["curiosity"] = round(min(mind.VALUE_CAP, vals.get("curiosity", 0.0) + 0.05), 3)
        self._ledger_add({"kind": "made", "rid": rid, "name": name, "who": p["name"],
                          "who_id": p["id"], "via": via, "rationale": rationale, "first": first,
                          "day": self.day(), "clock": round(self.clock, 1),
                          "time": f"{int(self.time_of_day()):02d}:00"})
        return True

    def _record_failed(self, p, guess) -> None:
        """Log a make-shift experiment that came to nothing — the Ledger's failed-inventions
        column. De-duplicated per person so one tinkerer can't flood it."""
        if not guess:
            return
        combo = ", ".join(sorted(guess))
        for e in reversed(self.ledger[-12:]):
            if e.get("kind") == "failed" and e.get("who_id") == p["id"] and e.get("combo") == combo:
                return
        self._ledger_add({"kind": "failed", "who": p["name"], "who_id": p["id"], "combo": combo,
                          "day": self.day(), "clock": round(self.clock, 1),
                          "time": f"{int(self.time_of_day()):02d}:00"})

    def _tinker(self, p, night: bool) -> None:
        """Gated offline invention: only now and then (TINKER_BEAT), and only toward a problem
        the person is actually feeling, do they try a fresh combination of materials. A hit
        becomes a real discovery; a miss is logged as a dead end. The slow, earned offline
        path — the LLM mind can leap straight to the answer separately."""
        if self.clock < p.get("tinker_cd", 0):
            return
        p["tinker_cd"] = self.clock + TINKER_BEAT * (0.7 + 0.6 * self.rng.random())
        unsolved = self._person_unsolved(p)
        targets = [rid for rid in unsolved
                   if self._DISCOVERY_TRIGGER.get(rid, lambda *_: True)(p, night)]
        if not targets:
            return
        tried = {tuple(t) for t in p.get("tried", [])}
        combos = [c for k in (2, 3) for c in itertools.combinations(mind.CRAFT_VOCAB, k)
                  if c not in tried]
        if not combos:
            return
        guess = list(combos[int(self.rng.integers(len(combos)))])
        rid = crafting.identify(set(guess), targets)            # solves a problem they FEEL?
        if rid:
            p.setdefault("tried", []).append(guess)             # don't re-derive it
            self._grant_recipe(p, rid, via="worked out",
                               rationale=self._DISCOVERY_RATIONALE.get(rid, ""))
            self._note("discovery", f"{p['name']} worked out how to make a {rid.replace('_', ' ')}.")
        else:
            # Burn it as a dead end ONLY if it matches no survival craft at all; a combo that
            # would solve a problem they don't yet feel is left for when they do.
            if crafting.identify(set(guess), self._person_unsolved(p)) is None:
                p.setdefault("tried", []).append(guess)
            self._record_failed(p, guess)

    def apply_llm_discovery(self, p, data: dict) -> str | None:
        """Route an LLM craft-hypothesis through the same grant/log path as offline invention:
        canonicalize the guessed materials, and if they match a craft this soul hasn't worked
        out, they reason it into being. Returns the recipe id, or None. Used by the mind loop."""
        if not isinstance(data, dict):
            return None
        raw = data.get("ingredients") or []
        guess = {m for m in (mind.canon_material(str(x)) for x in raw) if m}
        say = str(data.get("say", "") or "")
        if say:
            mind.speak(p, say, self.clock)
        rid = crafting.identify(guess, self._person_unsolved(p))
        if rid:
            self._grant_recipe(p, rid, via="reasoned out", rationale=say or
                               self._DISCOVERY_RATIONALE.get(rid, ""))
            return rid
        if guess:
            self._record_failed(p, guess)
        return None

    def _maybe_teach(self, a, b) -> None:
        """When two souls are together, the one who knows a survival craft the other lacks may
        pass it on — gated by trust and a per-learner cooldown, so knowledge spreads over days
        rather than all at once. Records a 'taught by' line in the Ledger."""
        for teacher, learner in ((a, b), (b, a)):
            if self.clock < learner.get("learn_cd", 0):
                continue
            gap = [rid for rid in (teacher.get("recipes") or []) if not self._person_knows(learner, rid)]
            if not gap:
                continue
            rel = learner.get("rel", {}).get(teacher["id"]) or {}
            trust = rel.get("trust", 0.0)
            # A wide skill gap means an eager apprentice and much to impart — teaching takes hold
            # more readily the more the learner has to learn (apprenticeship).
            chance = 0.5 + 0.4 * min(1.0, len(gap) / 4.0)
            if trust < 0.4 or self.rng.random() > chance:
                continue
            rid = gap[int(self.rng.integers(len(gap)))]
            if self._grant_recipe(learner, rid, via=f"taught by {teacher['name']}",
                                  rationale=f"{teacher['name']} showed me how"):
                learner["learn_cd"] = self.clock + TEACH_BEAT * (0.7 + 0.6 * self.rng.random())
                self._note("discovery",
                           f"{teacher['name']} taught {learner['name']} to make a {rid.replace('_', ' ')}.")
                self._earn_renown(teacher, RENOWN_GAIN["teach"],
                                  f"taught {learner['name']} to make a {rid.replace('_', ' ')}")
                return

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

    def whisper(self, x: int, y: int, text: str, by: str = "him"):
        """A god slips a thought into the nearest soul. It lands as a vivid memory the
        mind will weigh when it next decides and reflects — divine inspiration, not a
        command (the body still obeys its needs first)."""
        text = (text or "").strip()
        if not text or not self.people:
            return
        target = min(self.people, key=lambda p: abs(p["x"] - x) + abs(p["y"] - y))
        mind.ensure_mind(target)
        mind.remember(target, f"a thought came to me, as if from nowhere: {text}", 0.9,
                      "whisper", self.clock)
        self.version += 1
        self._note("whisper", f"{by} whispered to {target['name']}.")

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

    def _stockpiles_payload(self):
        """Visible stores the band keeps: the common granary and every home larder that actually
        holds something — so a god can SEE the food/wood/stone piling up, not just trust it's
        there. Each entry carries its location, contents, and total count for a ground pile."""
        out = []
        g = self.granary
        if g.get("x") is not None and g.get("store"):
            items = {k: int(v) for k, v in g["store"].items() if v > 0}
            if items:
                out.append({"x": g["x"], "y": g["y"], "items": items,
                            "total": sum(items.values()), "communal": True})
        for p in self.people:
            if not p.get("home_struct"):
                continue
            store = {k: int(v) for k, v in (p.get("store") or {}).items() if v > 0}
            if store:
                hx, hy = p["home"]
                out.append({"x": int(hx), "y": int(hy), "items": store,
                            "total": sum(store.values()), "communal": False})
        return out

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
            "season": self.season(), "weather": self.weather, "speed": self.speed,
            "day_speed": self.day_speed, "night_speed": self.night_speed,
            "biomes": BIOMES, "sea_level": int(SEA_LEVEL * 255),
            "plants": {sp: PLANTS[sp]["name"] for sp in PLANTS},
            "block_names": BLOCK_NAMES,
            "layers": self._pack_layers(0, H, 0, W, step),
            "animals": self.animals,
            "people": self._people_light(),
            "structures": self.structures,
            "blocks": self._blocks_payload(),
            "roofs": self._roofs_payload(),
            "sites": self._sites_payload(),
            "ore": [{"x": n["x"], "y": n["y"], "kind": n["kind"]} for n in self.ore_nodes],
            "stone": [[n["x"], n["y"]] for n in self.stone_nodes],
            "berries": [{"x": b["x"], "y": b["y"], "ripe": self._bush_ripe(b)} for b in self.berry_bushes],
            "stockpiles": self._stockpiles_payload(),
            "decor": [[x, y, k] for (x, y), k in self.decor.items()],
            "stations": [[x, y, k] for (x, y), k in self.station_objs.items()],
            "paths": self._paths_payload(),
            "roads": [[x, y, round(c, 2)] for (x, y), c in self.roads.items()],
            "settlements": self.settlements,
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

    # Per-person keys too heavy to ship every tick (the mind's memory stream + distilled
    # beliefs grow over a life). The live renderer doesn't need them; the inspector panel
    # pulls the full mind on demand via person_detail(). Stripping them keeps the ~6/s
    # broadcast small so the world stays smooth instead of stuttering.
    _HEAVY_PERSON_KEYS = ("memory", "reflections", "craft")

    def _people_light(self) -> list:
        """People minus their heavy mind state, with a compact crafting summary added so
        the renderer can float a ⚙ (and progress) over a worker's head."""
        out = []
        for p in self.people:
            q = {k: v for k, v in p.items() if k not in self._HEAVY_PERSON_KEYS}
            c = p.get("craft")
            if c and c.get("total"):
                q["crafting"] = {"rid": c["rid"], "out": c["out"],
                                 "pct": round(max(0.0, 1.0 - c["left"] / c["total"]), 3)}
            out.append(q)
        return out

    def tick_state(self) -> dict:
        """Light per-tick payload for the live renderer (no heavy tile layers — those
        come once via snapshot(); ticks just move time, weather and entities)."""
        return {
            "version": self.version, "clock": round(self.clock, 1), "day": self.day(),
            "time": round(self.time_of_day(), 2), "season": self.season(),
            "weather": self.weather, "census": self.census(),
            "animals": self.animals, "people": self._people_light(),
            "structures": self.structures,
            "blocks": self._blocks_payload(),
            "roofs": self._roofs_payload(),
        }

    def person_detail(self, pid) -> dict | None:
        """The full mind of one person — temperament, lived values, the memory stream,
        distilled beliefs, relationships and current intention — for the inspector panel
        opened by double-clicking them. Returns None if no such living person."""
        for p in self.people:
            if str(p.get("id")) == str(pid):
                c = p.get("craft")
                d = dict(p)
                if c and c.get("total"):
                    d["crafting"] = {"rid": c["rid"], "out": c["out"],
                                     "pct": round(max(0.0, 1.0 - c["left"] / c["total"]), 3),
                                     "left_min": round(c["left"], 1)}
                # Resolve kin to names for the inspector's family panel, and the life stage.
                nm = {q["id"]: q["name"] for q in self.people}
                d["stage"] = "child" if p["age"] < ADULT_AGE else ("elder" if p["age"] > 0.8 * PERSON["max_age"] else "adult")
                d["age_years"] = round(p["age"] / DAYS_PER_YEAR, 1)
                d["kin"] = {
                    "partner": nm.get(p.get("partner")),
                    "parents": [nm[i] for i in p.get("parents", []) if i in nm],
                    "children": [nm[i] for i in p.get("children", []) if i in nm],
                    "lineage": p.get("lineage"),
                }
                return d
        return None

    # Vegetation tallies scan the whole (up to 4M-tile) map, so they're cached and only
    # recomputed every so often in game-time — the counts drift slowly and needn't be exact
    # every tick. This is the difference between a smooth world and a stuttering one.
    CENSUS_VEG_EVERY = 60.0      # game-minutes between full vegetation recounts

    def census(self) -> dict:
        counts = {}
        for a in self.animals:
            counts[a["sp"]] = counts.get(a["sp"], 0) + 1
        veg = getattr(self, "_census_veg", None)
        if veg is None or (self.clock - getattr(self, "_census_veg_t", -1e9)) > self.CENSUS_VEG_EVERY:
            veg = {PLANTS[sp]["name"]: int((self.veg_sp == sp).sum()) for sp in PLANTS}
            self._census_veg = veg
            self._census_veg_t = self.clock
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
            distress = [p["name"] for p in self.people                       # genuine danger = the body failing, not mere discomfort
                        if p.get("hydration", 1) < 0.35 or p.get("satiety", 1) < 0.35
                        or p.get("stamina", 1) < 0.35 or p["hp"] < 0.6
                        or (p.get("illness") and p["illness"].get("known"))]
            roster = ", ".join(p["name"] for p in self.people[:8]) + ("…" if len(self.people) > 8 else "")
            built = c["buildings"]
            ppl = (f"People: {len(self.people)} alive ({roster})"
                   + (f"; struggling: {', '.join(distress[:6])}" if distress else "; all faring well")
                   + (f". They have raised {built} building{'s' if built != 1 else ''}"
                      f" ({len(self.blocks)} tiles laid). " if built or self.blocks else ". "))
            kids = sum(1 for q in self.people if q["age"] < ADULT_AGE)
            if kids:
                ppl += f"{kids} of them are children. "
            vocs = {}
            for q in self.people:
                if q.get("home_struct") and q.get("vocation"):
                    vocs[q["vocation"]] = vocs.get(q["vocation"], 0) + 1
            if sum(vocs.values()) >= 3:
                ppl += ("Their trades: "
                        + ", ".join(f"{n} {v}{'s' if n != 1 else ''}" for v, n in sorted(vocs.items())) + ". ")
            famous = max(self.people, key=lambda q: q.get("renown", 0.0), default=None)
            if famous and famous.get("renown", 0.0) > 0.15:
                halls = sum(1 for s in self.sites if s.get("communal") and s["done"])
                ppl += (f"Most esteemed among them is {famous['name']} (renown "
                        f"{famous['renown']:.2f}). " + (f"The band has raised {halls} gathering "
                        f"hall{'s' if halls != 1 else ''}. " if halls else ""))
            inner = mind.digest(self.people, self.clock)
            if inner:
                ppl += "Their minds: " + inner + " "
            invented = [r for r in crafting.SURVIVAL_DISCOVERIES if r in self.known_recipes]
            if invented:
                ppl += ("They have worked out: "
                        + ", ".join(r.replace("_", " ") for r in invented) + ". ")
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
            f"from blueprints, and once survival is met they autonomously climb the dwelling "
            f"ladder (axe → a leaf lean-to → a snug hut → a roomy cabin), so settled souls keep "
            f"raising ever-finer homes rather than standing idle. "
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
            "  <whisper>x y a thought to slip into a nearby soul</whisper>  (lands as a vivid memory they weigh when next they decide — inspiration, not a command)\n"
            "Act only when you mean to; the world is alive and persists between visits."
        )

    # ── persistence ────────────────────────────────────────────────────────────
    def save(self, grids: bool = True):
        """Persist the world. The MUTABLE state (people, sites, blocks, the economy, …) is a small
        JSON written every time. The 2048² terrain GRIDS are ~130MB to compress, so they're written
        only when `grids` is True (the loop does this occasionally) or when none are on disk yet —
        the rest of the time we skip them, which is what stops the periodic save FREEZE. The grids
        drift slowly (only vegetation/ecology change at all), so a few minutes' staleness is fine.

        Every write is ATOMIC (staged to a .tmp, then os.replace): an interrupted or partial write
        — app closed mid-save, a crash — can never leave a truncated/corrupt file. That truncation,
        on the non-atomic 1.6s grid write, is what used to corrupt world.npz and silently reset the
        world on next launch. On a full (grid) save the previous good pair is also rotated to .bak,
        so load() can fall back to it rather than regenerate."""
        try:
            os.makedirs(_DIR, exist_ok=True)
            full = grids or not os.path.exists(PATH_GRID)
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
                "stone_nodes": self.stone_nodes,
                "berry_bushes": self.berry_bushes, "granary": self.granary,
                "decor": [[x, y, k] for (x, y), k in self.decor.items()],
                "stations": [[x, y, k] for (x, y), k in self.station_objs.items()],
                "footfall": [[x, y, round(w, 1)] for (x, y), w in self.footfall.items()],
                "roads": [[x, y, round(c, 2)] for (x, y), c in self.roads.items()],
                "foot_v": 1,                                   # paths/roads now wear only on SHARED use (see load)
                "settlements": self.settlements,
                "authored_blueprints": self.authored_blueprints,   # the band's own designs (Phase A)
                "laws": self.laws,                                 # the band's enacted laws (Phase B)
                "customs": self.customs,                           # the band's traditions (Phase C)
                "log": self.log, "version": self.version,
                "known_recipes": sorted(self.known_recipes),
                "ledger": self.ledger,
                "speed": self.speed, "day_speed": self.day_speed, "night_speed": self.night_speed,
                "money_invented": self.money_invented, "money_inventor": self.money_inventor,
            }
            mtmp = PATH_META + ".tmp"
            with open(mtmp, "w", encoding="utf-8") as f:
                json.dump(meta, f, default=_json_safe)
            if full:
                # Stage the new grid fully to a temp BEFORE touching anything live, so a failed
                # compress leaves the existing save intact. Then back up the current good pair to
                # .bak and swap both new files in. Worlds can no longer be lost to a partial write.
                gtmp = PATH_GRID + ".tmp"
                with open(gtmp, "wb") as gf:           # a file object → numpy won't append ".npz"
                    np.savez_compressed(
                        gf, elevation=self.elevation, biome=self.biome, soil=self.soil,
                        moisture=self.moisture, water=self.water, veg_sp=self.veg_sp,
                        veg_growth=self.veg_growth, chunk_eco=self._chunk_eco,
                    )
                for path in (PATH_GRID, PATH_META):
                    if os.path.exists(path):
                        try:
                            os.replace(path, path + ".bak")
                        except OSError:
                            pass
                os.replace(gtmp, PATH_GRID)
            os.replace(mtmp, PATH_META)                # atomic; on a meta-only save this is the whole write
        except OSError as e:
            print(f"[world] save failed: {e}")

    def load(self) -> bool:
        """Restore a saved world, falling back to the .bak pair before ever giving up — so a single
        corrupt/partial file recovers from the last good save instead of resetting the world. Returns
        False (→ caller regenerates) only when neither the primary nor the backup can be read."""
        if self._load_files(PATH_META, PATH_GRID):
            return True
        if (os.path.exists(PATH_META + ".bak") and os.path.exists(PATH_GRID + ".bak")
                and self._load_files(PATH_META + ".bak", PATH_GRID + ".bak")):
            print("[world] primary save unreadable — recovered from backup (.bak)")
            return True
        return False

    def _load_files(self, meta_path: str, grid_path: str) -> bool:
        """Restore a saved world from a specific meta/grid pair. Returns False (→ caller tries the
        backup, then regenerates) if the save is missing, corrupt, or from an incompatible
        schema/size, so a world left broken by an older build heals itself instead of staying frozen."""
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            # Reject saves from a different layout version up front — loading them
            # would let step() crash every tick (the tab would look frozen).
            if meta.get("schema") != SCHEMA:
                print(f"[world] save schema {meta.get('schema')!r} != {SCHEMA}; regenerating")
                return False
            with np.load(grid_path) as z:
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
            self._last_spoil = self._last_pest = self.clock     # P5 timers — start fresh from the saved clock
            self.weather = meta.get("weather", "clear")
            self.weather_intensity = meta.get("weather_intensity", 0.0)
            self._weather_until = meta.get("weather_until", 0.0)
            self.animals = meta.get("animals", []); self.people = meta.get("people", [])
            for _p in self.people:                       # heal legacy saves that predate the mind
                mind.ensure_mind(_p, self.rng)
            self.structures = meta.get("structures", [])
            self.blocks = {}
            for k, c in (meta.get("blocks") or {}).items():
                sx, sy = k.split(",")
                self.blocks[(int(sx), int(sy))] = int(c)
            self.roofs = {(int(x), int(y)) for x, y in (meta.get("roofs") or [])}
            self.sites = meta.get("sites", [])
            self.ore_nodes = meta.get("ore_nodes", [])
            self.stone_nodes = meta.get("stone_nodes", [])
            if not self.stone_nodes:                      # older saves predate boulders — scatter them now
                self._seed_stone_nodes()
            self.granary = meta.get("granary") or {"store": {}, "x": None, "y": None}
            self.decor = {(int(d[0]), int(d[1])): d[2] for d in (meta.get("decor") or [])}
            self.station_objs = {(int(d[0]), int(d[1])): d[2] for d in (meta.get("stations") or [])}
            self.footfall = {(int(d[0]), int(d[1])): float(d[2]) for d in (meta.get("footfall") or [])}
            self.roads = {(int(d[0]), int(d[1])): float(d[2]) for d in (meta.get("roads") or [])}
            if meta.get("foot_v", 0) < 1:                      # one-time scrub: the old per-step paths/roads
                self.footfall = {}; self.roads = {}; self._foot_last = {}   # wore everywhere — start clean
            self.settlements = meta.get("settlements", []) or []   # M0; a pre-M0 save self-heals on the daily tick
            self.authored_blueprints = meta.get("authored_blueprints", []) or []   # the band's own designs
            self.laws = meta.get("laws", []) or []                                  # enacted laws (Phase B)
            self.customs = meta.get("customs", []) or []                            # traditions (Phase C)
            self._register_authored()                     # re-inject them into BLUEPRINTS for the build machinery
            self._relocate_granary_if_stranded()          # heal a granary saved in the water
            self.berry_bushes = meta.get("berry_bushes", [])
            if self.berry_bushes:
                self._rebuild_berry_index()
            else:
                self._seed_berry_bushes()      # a save predating berries — populate the map now
            self._rebuild_ore_index()
            self.log = meta.get("log", [])
            self.known_recipes = set(meta.get("known_recipes") or crafting.STARTER_RECIPES)
            self.ledger = meta.get("ledger") or []
            self.speed = float(meta.get("speed") or 1.0)
            self.day_speed = float(meta.get("day_speed") or 1.0)
            self.night_speed = float(meta.get("night_speed") or 1.0)
            self.money_invented = bool(meta.get("money_invented"))
            self.money_inventor = meta.get("money_inventor")
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

    # ─── named snapshots — explicit, user-controlled saves (settings → World saves) ──────
    @staticmethod
    def _snap_id(name: str) -> str:
        """A filesystem-safe, unique folder id from a display name (+ a timestamp so two saves can
        share a name)."""
        base = "".join(c if (c.isalnum() or c in "-_") else "-" for c in (name or "").strip())
        base = "-".join(part for part in base.split("-") if part)[:28] or "save"
        return f"{base}-{int(time.time())}"

    def save_snapshot(self, name: str = "") -> dict:
        """Save a NAMED checkpoint the user can return to. Forces a full, current save first, then
        copies the save pair into snapshots/<id>/. Returns the snapshot's info record."""
        self.save(grids=True)                              # make the on-disk save current + complete
        disp = (name or "").strip()[:40] or time.strftime("%b %d, %H:%M")
        sid = self._snap_id(disp)
        dest = os.path.join(SNAP_DIR, sid)
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(PATH_META, os.path.join(dest, "world.json"))
        shutil.copy2(PATH_GRID, os.path.join(dest, "world.npz"))
        info = {"id": sid, "name": disp, "created": time.time(),
                "day": self.day(), "pop": len(self.people)}
        with open(os.path.join(dest, "snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(info, f)
        return info

    def list_snapshots(self) -> list:
        """All saved snapshots, newest first — each {id, name, created, day, pop} for the list."""
        out = []
        if os.path.isdir(SNAP_DIR):
            for d in os.listdir(SNAP_DIR):
                info_path = os.path.join(SNAP_DIR, d, "snapshot.json")
                if os.path.exists(info_path):
                    try:
                        with open(info_path, encoding="utf-8") as f:
                            out.append(json.load(f))
                    except (OSError, ValueError):
                        pass
        out.sort(key=lambda s: s.get("created", 0), reverse=True)
        return out

    def _stage_restore(self, sid: str) -> bool:
        """Snapshot the current world (so a restore is undoable), then copy a named snapshot's files
        over the live save. The actual reload is an atomic singleton swap in restore_world() — never
        an in-place reload that could race the engine tick. Returns True if the snapshot staged."""
        sid = os.path.basename((sid or "").strip())        # guard against path traversal
        src = os.path.join(SNAP_DIR, sid)
        smeta, sgrid = os.path.join(src, "world.json"), os.path.join(src, "world.npz")
        if not (sid and os.path.exists(smeta) and os.path.exists(sgrid)):
            return False
        self.save_snapshot("before restore")               # keep the current world recoverable
        try:
            shutil.copy2(smeta, PATH_META)
            shutil.copy2(sgrid, PATH_GRID)
        except OSError:
            return False
        return True

    def delete_snapshot(self, sid: str) -> bool:
        """Delete a named snapshot. Returns True if it existed and was removed."""
        sid = os.path.basename((sid or "").strip())
        dest = os.path.join(SNAP_DIR, sid)
        if sid and os.path.isdir(dest):
            try:
                shutil.rmtree(dest)
                return True
            except OSError:
                pass
        return False


# ─── module-level singleton (mirrors room.py's load/save ergonomics) ──────────
_world: World | None = None


def get_world() -> World:
    """Lazily load the saved world, or generate a fresh one on first ever access."""
    global _world
    if _world is None:
        w = World()
        if not w.load():
            # A save EXISTS but neither it nor its backup could be read — do NOT silently overwrite
            # it (that's what used to turn a transient corruption into a permanent reset). Set it
            # aside, recoverable, then start fresh.
            if os.path.exists(PATH_META) or os.path.exists(PATH_GRID):
                _quarantine_save()
            w.generate()
            w.save()
        _world = w
    return _world


def _quarantine_save() -> None:
    """Move an unreadable save out of the way (instead of overwriting it) so a damaged world can
    still be recovered by hand, into ~/.ai4me/corrupt-<timestamp>/."""
    dest = os.path.join(_DIR, "corrupt-" + time.strftime("%Y%m%d-%H%M%S"))
    try:
        os.makedirs(dest, exist_ok=True)
        for path in (PATH_META, PATH_GRID, PATH_META + ".bak", PATH_GRID + ".bak"):
            if os.path.exists(path):
                shutil.move(path, os.path.join(dest, os.path.basename(path)))
        print(f"[world] unreadable save quarantined to {dest} (recoverable; world regenerated)")
    except OSError as e:
        print(f"[world] could not quarantine the unreadable save: {e}")


def reset_world(seed: int | None = None) -> World:
    global _world
    _world = World().generate(seed)
    _world.save()
    return _world


def restore_world(sid: str) -> "World | None":
    """Restore a named snapshot by swapping in a freshly-loaded world — atomic like reset_world, so
    the running tick finishes on the old world and the next tick gets the restored one (no in-place
    reload racing the engine). The current world is snapshotted first, so a restore is undoable.
    Returns the restored world, or None if the snapshot was missing/unreadable."""
    global _world
    cur = get_world()
    if not cur._stage_restore(sid):                # validates + snapshots current + copies files over
        return None
    w = World()
    if not w.load():                              # load the just-restored save into a fresh world
        return None
    _world = w
    return w


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

    # Simulate ~8 game-days. We advance a fixed 24 game-seconds per step (independent of the
    # real-time pace constant), so this test keeps the same granularity and runtime whatever
    # GAME_SEC_PER_REAL_SEC is set to. Watch populations across season turns.
    days = 8
    step_dt = 24.0 / GAME_SEC_PER_REAL_SEC          # real-sec per step → 24 game-sec/step
    steps = int(days * 86400 / 24.0)
    print("\n  day  season   weather  rabbit deer wolf  ppl bldg tiles  vegtiles  ms/step")
    t0 = time.time()
    last_day = -1
    day_t0 = t0
    for i in range(steps):
        w.step(dt_real_sec=step_dt)
        if w.day() != last_day:
            last_day = w.day()
            c = w.census()
            an = c["animals"]
            now = time.time()
            ms = (now - day_t0) * 1000 / (steps / days)
            day_t0 = now
            print(f"  {w.day():>3}  {w.season():<7}  {w.weather:<7}  "
                  f"{an.get('rabbit',0):>6} {an.get('deer',0):>4} {an.get('wolf',0):>4}  "
                  f"{c['people']:>3} {c['buildings']:>4} {c['blocks']:>5}  "
                  f"{sum(c['vegetation'].values()):>8}  {ms:>6.2f}", flush=True)
    sim_s = time.time() - t0
    deaths = [e["text"] for e in w.log if e["kind"] == "death"]
    print(f"  survival: {len(w.people)}/7 of the founding band alive; "
          f"deaths logged: {deaths if deaths else 'none'}")
    beh = getattr(w, "_beh", {})
    print(f"  behaviours fired: planned-foundings={beh.get('plan_found', 0)}, "
          f"help-blocks-laid={beh.get('help_block', 0)}, craft-materials-gifted={beh.get('gift_material', 0)}, "
          f"home-commissions-paid={beh.get('commission', 0)}")
    kids = [q for q in w.people if q["age"] < ADULT_AGE]
    ksk = {k: round(v, 2) for k, v in (kids[0].get("skills", {}) if kids else {}).items()}
    print(f"  younglings: {len(kids)} children; arrows-whittled={beh.get('child_whittle', 0)}; "
          f"sample child skills={ksk or 'n/a'}")
    cbuildings = {s["bp"] for s in w.sites if s.get("done") and s.get("communal")}
    deep = sorted({r["out"] for q in w.people for it in q.get("inv", {})
                   for r in [crafting.recipe(it)] if r and r["tier"] >= 1})
    print(f"  craft tree: deep-crafts fired={beh.get('tech_craft', 0)}; communal builds={sorted(cbuildings) or 'none'}; "
          f"tier1+ goods on hand={deep[:8] or 'none'}")
    print(f"  self-authored projects: aspirations finished={beh.get('aspire_done', 0)}, "
          f"plan-steps done={beh.get('aspire_step', 0)}, flowers/decor placed={len(w.decor)}")
    lore = sum(len(p.get("berry_lore", {})) for p in w.people)
    bad_lore = sum(1 for p in w.people for v in p.get("berry_lore", {}).values() if v == "bad")
    berry_ill = sum(1 for p in w.people if (p.get("illness") or {}).get("d") == "berry_sickness"
                    or "berry_sickness" in p.get("immune", {}))
    print(f"  berries: {len(w.berry_bushes)} bushes on the map; "
          f"band learned {lore} bush(es) ({bad_lore} poison); souls who weathered berry-sickness: {berry_ill}")
    hunts = sum(1 for e in w.log if e["kind"] == "hunt")
    flesh = sum(p["inv"].get(k, 0) for p in w.people for k in ("meat", "fish", "cooked_meat", "cooked_fish"))
    tainted = sum(1 for p in w.people if (p.get("illness") or {}).get("d") == "tainted_gut"
                  or "tainted_gut" in p.get("immune", {}))
    print(f"  hunt/fish: {hunts} kills logged; flesh now held (raw+cooked): {flesh}; "
          f"souls who weathered raw-flesh sickness: {tainted}")
    pests = sum(1 for e in w.log if e["kind"] == "pest")
    preserved = sum(p["inv"].get(k, 0) + p.get("store", {}).get(k, 0)
                    for p in w.people for k in ("dried_meat", "dried_fish"))
    print(f"  spoilage/seasons: season now {w.season()}; pest raids logged: {pests}; "
          f"preserved (dried) stores held: {preserved}")
    if w.people:
        sample = w.people[0]
        print(f"  survivor sample — {sample['name']}: comfort[h/t/f] "
              f"{sample['hunger']:.2f}/{sample['thirst']:.2f}/{sample['fatigue']:.2f} "
              f"reserve[sat/hyd/stm] {sample.get('satiety',1):.2f}/{sample.get('hydration',1):.2f}/"
              f"{sample.get('stamina',1):.2f} hp {sample['hp']:.2f} "
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
        # Death keys off the physiological RESERVES, so hold those at empty (comfort alone
        # no longer kills — that's the whole point of the body/comfort split).
        wd.people[0]["hydration"] = 0.0; wd.people[0]["satiety"] = 0.0; wd.people[0]["stamina"] = 0.0
        wd._tick_people(0.4)
    print(f"  death test: a body in unrelieved crisis {'died as expected' if not wd.people else 'SURVIVED (unexpected)'}")
    print(f"\nsimulated {steps} steps in {sim_s:.2f}s "
          f"({steps/sim_s:.0f} steps/s, {sim_s*1000/steps:.2f} ms/step avg)")

    # Tile-building unit check — isolate the chain from survival noise: a comfortable builder,
    # kept fed/watered and handed raw logs + thatch + an axe (tools are the toolmaker's job now —
    # gating tested separately), must mark out a building and lay it block by block until housed.
    wc = World().generate(seed=7)
    wc.people = []
    land = np.argwhere((wc.water == WATER_NONE) & (wc.biome == B["grassland"]))
    by, bx = land[len(land) // 2]
    wc._add_person(int(bx), int(by), name="Builder")
    b = wc.people[0]
    b["inv"].update({"wood": 80, "fiber": 40, "leaves": 30, "axe": 1})   # logs, thatch, leaves & an axe
    b["recipes"] = ["crude_axe"]
    wc.clock = 12 * 60                              # high noon (daytime → secondary drives active)
    for _ in range(1500):
        b["hunger"] = b["thirst"] = 0.1; b["fatigue"] = 0.1   # stay comfortable
        if b.get("home_struct"):
            break
        wc._tick_people(0.4)
    site = wc.sites[0] if wc.sites else None
    built_ok = bool(b.get("home_struct") and wc.blocks and wc.roofs)
    print(f"  build test (autonomous): Builder axe={b['inv'].get('axe',0)}, "
          f"first home = {site['name'] if site else 'none'}, blocks={len(wc.blocks)}, "
          f"roofs={len(wc.roofs)}, insul={b.get('insul')}, housed={b.get('home_struct') is not None} "
          f"-> {'OK' if built_ok else 'FAILED'}")

    # BLUEPRINT LINT — every built-in design AND every parametrically-generated home must be a
    # usable building: floor enclosed, every room reachable from outside (no walled-off space), and
    # sanely sized. Sweeps the home designer across household sizes & skills to catch a bad layout
    # (an orphan room, a door into a wall) before a soul ever tries to live in one.
    wl = World().generate(seed=11)
    lint_bad = []
    for bpn, bp in list(BLUEPRINTS.items()):
        ok, why = wl._validate_blueprint(bp["layout"])
        if not ok:
            lint_bad.append((bpn, why))
    designed = 0
    for fam in range(1, 9):
        for sk in (0.0, 0.4, 0.8):
            wl.people = []; wl._add_person(50, 50, name="L"); lp = wl.people[0]
            lp["age"] = ADULT_AGE + 5; lp["home"] = (50, 50)
            lp["inv"] = {"wood": 99}; lp["skills"] = {"building": sk}; lp["children"] = []
            for ci in range(max(0, fam - 1)):
                wl._add_person(50, 52, name=f"c{ci}"); kk = wl.people[-1]; kk["age"] = 2
                lp["children"].append(kk["id"])
            if fam >= 2 and lp["children"]:
                lp["partner"] = lp["children"].pop(0)
            for rung in ("hut", "cabin"):
                bid = wl._design_dwelling(lp, rung)
                ok, why = wl._validate_blueprint(BLUEPRINTS[bid]["layout"])
                designed += 1
                if not ok:
                    lint_bad.append((f"{rung}/fam{fam}/sk{sk}", why))
    lint_ok = not lint_bad
    print(f"  blueprint-lint test: builtins + {designed} designed homes all valid={lint_ok}"
          + (f" BAD={lint_bad[:5]}" if lint_bad else "") + f" -> {'OK' if lint_ok else 'FAILED'}")

    # ARCHITECTURE AUTHORING (Phase A) — the LLM design ratchet. An authored building is REGISTERED
    # only if it survives the validators (the immune system); the rule trigger wakes the (rare, costly)
    # LLM only for a SETTLED BUILDER in a prospering band. Inert offline — with no model the band
    # builds exactly as before, so this can't move survival.
    wa = World().generate(seed=5)
    # A.2: an authored WORKSHOP carries a FUNCTION → built communally, and the band would raise ITS
    # design (not the built-in) and crafting near it is faster. Built-ins still resolve to their role.
    a_good = wa.apply_authored_building({"name": "Tinkers' Shed", "function": "workshop",
                                         "purpose": "to craft", "layout": ["WWDWW", "WFCFW", "WFFFW", "WWWWW"]})
    a_walled = wa.apply_authored_building({"name": "Tomb", "layout": ["WWWWW", "WFFFW", "WWWWW"]})  # no door
    a_huge = wa.apply_authored_building({"name": "Mega", "layout": ["F" * 20] * 20})                # oversized
    func_ok = (wa._bp_function(WORKSHOP_BP) == "workshop" and wa._bp_function(a_good) == "workshop"
               and BLUEPRINTS[a_good]["communal"] and wa._authored_for("workshop") == a_good)
    a_land = np.argwhere(wa.water == WATER_NONE)
    wa.people = []
    for i in range(AUTHOR_MIN_POP):
        aly, alx = (int(v) for v in a_land[i * 37]); wa._add_person(alx, aly, name=f"A{i}")
    a_arch = wa.people[0]; a_arch["vocation"] = "builder"; a_arch["home_struct"] = "h"
    wa._authored_cd = 0.0                                           # clear the cooldown left by a_good
    a_trig_b = wa.wants_new_building(a_arch)                        # settled builder, grown band → wants
    a_forg = wa.people[1]; a_forg["vocation"] = "forager"; a_forg["home_struct"] = "h"
    a_trig_f = wa.wants_new_building(a_forg)                        # a forager is not the architect → no
    auth_ok = (bool(a_good) and a_good in BLUEPRINTS and a_walled is None and a_huge is None
               and func_ok and a_trig_b and not a_trig_f)
    print(f"  authoring test (Phase A/A.2): registered={bool(a_good)} func_routes={func_ok} "
          f"rejects-walled={a_walled is None} rejects-huge={a_huge is None} "
          f"trigger builder={a_trig_b}/forager={a_trig_f} -> {'OK' if auth_ok else 'FAILED'}")

    # LAWS (Phase B) — a renown-recognised LEADER, facing a recurring wrong, enacts a law the engine
    # then judges by REPUTATION only. An unenforceable/duplicate law is set aside; the unable are
    # never judged. Inert offline (only the default granary norm runs), so it can't move survival.
    wl = World().generate(seed=5); wl.people = []
    wl.granary = {"store": {"food": 2}, "x": 50, "y": 50}                  # commons LOW
    for i in range(5):
        wl._add_person(50 + i, 50, name=f"L{i}"); _p = wl.people[i]
        _p["age"] = ADULT_AGE + 5; _p["renown"] = 0.5; _p["hp"] = 1.0; _p["store"] = {}
    wl_lead = wl.people[0]; wl_lead["renown"] = 3.0
    for i in (1, 2):
        wl.people[i]["store"] = {"food": LAW_HOARD_CAP + 5}               # 2/5 hoard → a recurring wrong
    l_problem = wl._law_problem() == "hoarding"
    wl._law_cd = 0.0
    l_leader = wl.wants_new_law(wl_lead) and not wl.wants_new_law(wl.people[1])   # only the leader legislates
    l_enact = wl.apply_authored_law({"norm": "hoarding", "name": "The Sharing Law"}, by="L0") == "hoarding"
    l_reject = (wl.apply_authored_law({"norm": "hoarding"}) is None                # already enacted
                and wl.apply_authored_law({"norm": "teleport"}) is None)           # unenforceable
    _rb = [wl.people[i]["renown"] for i in (1, 2)]
    wl._tick_governance()
    l_enforce = all(wl.people[i]["renown"] < _rb[k] for k, i in enumerate((1, 2))) and wl_lead["renown"] == 3.0
    law_ok = l_problem and l_leader and l_enact and l_reject and l_enforce
    print(f"  law test (Phase B): problem={l_problem} leader-only={l_leader} enacts={l_enact} "
          f"rejects-dup/bad={l_reject} enforces-soft={l_enforce} -> {'OK' if law_ok else 'FAILED'}")

    # CUSTOMS (Phase C) — a much-loved soul founds a yearly FEAST; the engine keeps it, deepening
    # that season's gathering. A bad kind/season or a second feast for a taken season is set aside.
    # Inert offline (only the plain seasonal gathering runs), so it can't move survival.
    wcu = World().generate(seed=5); wcu.people = []
    for i in range(CUSTOM_MIN_POP):
        wcu._add_person(50 + i, 50, name=f"K{i}"); _p = wcu.people[i]
        _p["age"] = ADULT_AGE + 5; _p["renown"] = 0.3; _p["hp"] = 1.0; _p["home_struct"] = "h"
    wcu.people[0]["renown"] = 2.0; wcu._custom_cd = 0.0
    c_wants = wcu.wants_new_custom(wcu.people[0]) and not wcu.wants_new_custom(wcu.people[1])
    c_enact = wcu.apply_authored_custom({"kind": "feast", "season": "autumn", "name": "Harvest Home",
                                         "value": "the gathering-in"}, by="K0") == "Harvest Home"
    c_reject = (wcu.apply_authored_custom({"kind": "feast", "season": "autumn", "name": "x"}) is None
                and wcu.apply_authored_custom({"kind": "rave", "season": "autumn", "name": "x"}) is None
                and wcu.apply_authored_custom({"kind": "feast", "season": "monsoon", "name": "x"}) is None)
    _cb = mind._rel(wcu.people[0], wcu.people[1], wcu.clock)["sentiment"]
    wcu._festival("autumn")
    c_observe = (mind._rel(wcu.people[0], wcu.people[1], wcu.clock)["sentiment"] > _cb
                 and any(e["kind"] == "culture" and "Harvest Home" in e.get("text", "") for e in wcu.log))
    custom_ok = c_wants and c_enact and c_reject and c_observe
    print(f"  custom test (Phase C): respected-only={c_wants} founds={c_enact} "
          f"rejects-dup/badkind/badseason={c_reject} feast-kept+bonds={c_observe} "
          f"-> {'OK' if custom_ok else 'FAILED'}")

    # PERF — the per-tick precomputed perception masks (the np.isin hotspot at town scale) must be
    # byte-IDENTICAL to computing each window directly, or survival quietly shifts. The guard for the
    # optimization that took ~52ms/step at 83 souls down to ~13ms.
    wpf = World().generate(seed=5); wpf.people = []
    for (px, py) in [(50, 50), (62, 55), (45, 48), (70, 60)]:
        wpf._add_person(px, py, name="pf")
    wpf._build_perception_masks()
    mask_ok = True
    for (px, py) in [(50, 50), (55, 52), (66, 58), (45, 48), (70, 60)]:
        sliced = wpf._perceive(px, py)
        _sv = wpf._pmask; wpf._pmask = None; direct = wpf._perceive(px, py); wpf._pmask = _sv
        if not all(np.array_equal(s, d) if isinstance(s, np.ndarray) else s == d
                   for s, d in zip(sliced, direct)):
            mask_ok = False
    print(f"  perf test: perception-masks-identical={mask_ok} -> {'OK' if mask_ok else 'FAILED'}")

    # SETTLEMENT (M0) — the housed band becomes a first-class town: named, centred on its homes,
    # with a roll of members; a pre-M0 (empty) save self-heals on the daily tick; none when homeless.
    ws = World().generate(seed=11); ws.people = []
    for i, (sx, sy) in enumerate([(50, 50), (54, 50), (52, 54)]):
        ws._add_person(sx, sy, name=f"S{i}"); sp = ws.people[-1]; sp["age"] = ADULT_AGE + 3
        sp["home"] = (sx, sy); sp["home_struct"] = f"h{i}"
    ws._tick_settlements()
    st = ws.settlements[0] if ws.settlements else {}
    ws.settlements = []; ws._tick_settlements()                    # self-heal from a pre-M0 save
    healed = len(ws.settlements) == 1
    set_ok = (bool(st.get("name")) and st.get("cx") == 52 and st.get("pop") == 3
              and len(st.get("members", [])) == 3 and healed)
    print(f"  settlement test: name={st.get('name')} centre=({st.get('cx')},{st.get('cy')}) "
          f"pop={st.get('pop')} self-heal={healed} -> {'OK' if set_ok else 'FAILED'}")

    # Axe is EARNED, not innate (P-competence): no one is born knowing how to make a tool; a soul
    # WORKS IT OUT after chopping wood by hand KNAP_CHOPS times, and it then spreads by teaching.
    wk = World().generate(seed=7); wk.people = []
    ky, kx = np.argwhere((wk.water == WATER_NONE) & (wk.biome == B["grassland"]))[0]
    wk._add_person(int(kx), int(ky), name="Knapper")
    kn = wk.people[0]
    born_knowing = wk._person_knows(kn, "crude_axe")            # should be False — not a starter anymore
    for _ in range(KNAP_CHOPS):                                 # simulate hand-chops working it out
        kn["knap"] = kn.get("knap", 0) + 1
        if kn["knap"] >= KNAP_CHOPS:
            wk._grant_recipe(kn, "crude_axe", via="worked out")
    knapped = wk._person_knows(kn, "crude_axe")
    axe_ok = (not born_knowing) and knapped
    print(f"  axe-discovery test: born-knowing-axe={born_knowing} learns-by-knapping={knapped} "
          f"-> {'OK' if axe_ok else 'FAILED'}")

    # TOOL-GATING — only a toolmaker can fashion an axe; a builder who knows the recipe can't,
    # and gets one handed over when a toolmaker carrying a spare stands beside them.
    wtg = World().generate(seed=7); wtg.people = []
    tgy, tgx = np.argwhere((wtg.water == WATER_NONE) & (wtg.biome == B["grassland"]))[0]
    wtg._add_person(int(tgx), int(tgy), name="Smith")
    tmk = wtg.people[0]; tmk["traits"] = {"sociability": 0.5, "ambition": 0.3, "curiosity": 0.8, "caution": 0.3}
    tmk["vocation"] = mind.vocation(tmk); tmk["recipes"] = ["crude_axe"]; tmk["inv"] = {"wood": 10, "home": 1}
    wtg._add_person(int(tgx) + 1, int(tgy), name="Hauler")
    bld = wtg.people[1]; bld["traits"] = {"sociability": 0.5, "ambition": 0.8, "curiosity": 0.3, "caution": 0.3}
    bld["vocation"] = mind.vocation(bld); bld["recipes"] = ["crude_axe"]; bld["inv"] = {"wood": 10}
    getters = {"wood": lambda: ("seek_wood", None), "fiber": lambda: ("seek_fiber", None), "leaves": lambda: ("seek_leaves", None)}
    bld["home_struct"] = "s"      # housed builder, knows the recipe, has wood — must NOT make an axe
    bld_makes = wtg._person_build_decide(bld, np.zeros((1, 1), bool), np.zeros((1, 1), bool),
                                         np.zeros((1, 1), bool), np.zeros((1, 1), bool), 0, 0)
    craft_rid = (bld.get("craft") or {}).get("rid")
    builder_blocked = craft_rid not in ("crude_axe", "crude_spear")   # never fashioned a TOOL
    # (a non-toolmaker may still craft sticks/a workbench to climb the tree — that's intended;
    #  tool-gating only forbids the toolmaker's own tools)
    tmk_can = wtg._can_make_tools(tmk) and not wtg._can_make_tools(bld)
    # The toolmaker carries two spare axes; standing beside the axe-less builder, one flows over.
    tmk["inv"]["axe"] = 2
    for _ in range(3):
        wtg._tick_minds_social()
    gifted_ok = bld["inv"].get("axe", 0) >= 1
    tool_gate_ok = tmk_can and builder_blocked and gifted_ok
    print(f"  tool-gating test: toolmaker-only={tmk_can} builder-can't-make={builder_blocked} "
          f"axe-gifted-to-builder={gifted_ok} -> {'OK' if tool_gate_ok else 'FAILED'}")

    # COOPERATIVE BIG-BUILDS — a communal monument's tiles need a crew (CO_OP_MIN) on hand: a
    # lone builder can't lay them; a second crew-mate present unblocks the work.
    wcb = World().generate(seed=7); wcb.people = []
    cby, cbx = np.argwhere((wcb.water == WATER_NONE) & (wcb.biome == B["grassland"]))[
        len(np.argwhere((wcb.water == WATER_NONE) & (wcb.biome == B["grassland"]))) // 2]
    wcb._add_person(int(cbx), int(cby), name="Mason1"); ma = wcb.people[0]
    ma["home"] = (int(cbx), int(cby)); ma["inv"] = {"wood": 50, "fiber": 20}
    wcb._found_site(ma, MONUMENT_BP, communal=True)
    csite = wcb._person_site(ma)
    coop_ok = False
    if csite:
        tsk = wcb._site_next_task(csite); ma["x"], ma["y"] = tsk["x"], tsk["y"]
        n0 = len(wcb.blocks)
        wcb._build_next_block(ma)                          # alone — crew 1 < CO_OP_MIN → blocked
        solo_blocked = len(wcb.blocks) == n0
        wcb._add_person(int(tsk["x"]), int(tsk["y"]), name="Mason2"); mb = wcb.people[1]
        mb["site"] = csite["id"]; mb["inv"] = {"wood": 50}
        wcb._build_next_block(ma)                          # crew 2 ≥ CO_OP_MIN → tile lays
        crew_builds = any(t["done"] for t in csite["tasks"])
        coop_ok = solo_blocked and crew_builds
    print(f"  co-op build test: lone-builder-blocked={solo_blocked if csite else '?'} "
          f"crew-of-two-builds={crew_builds if csite else '?'} -> {'OK' if coop_ok else 'FAILED'}")

    # OFFER A HAND — a hale, settled soul at loose ends who spots a NEIGHBOUR raising their own home
    # walks over, asks, and pitches in; a disliked builder declines; a hungry passer-by doesn't offer.
    wh = World().generate(seed=5); wh.people = []
    hl = np.argwhere(wh.water == WATER_NONE)[len(np.argwhere(wh.water == WATER_NONE)) // 2]
    hby, hbx = int(hl[0]), int(hl[1])
    def _hsite():
        return {"id": "hs", "owner": "OWN", "bp": "hut", "ox": hbx, "oy": hby, "done": False,
                "communal": False, "tasks": [{"x": hbx, "y": hby, "code": 1, "cost": ["wood", 1], "done": False}]}
    def _hsetup(hx, hy):
        wh.people = []
        wh._add_person(hbx, hby, name="Own"); _o = wh.people[0]; _o["id"] = "OWN"; _o["home"] = (hbx, hby)
        wh._add_person(hx, hy, name="Help"); _h = wh.people[1]; _h["id"] = "HLP"; _h["home"] = (hbx + 6, hby)
        _h["home_struct"] = "hh"; _h["age"] = ADULT_AGE + 5; _h["inv"] = {}
        _h["hunger"] = _h["thirst"] = _h["fatigue"] = 0.1; _h["hp"] = 1.0; _h["help_offer_cd"] = 0.0
        return _o, _h
    def _hgrids(_h):
        e, d, t, s, f, l, glx, gly, _, _ = wh._perceive(_h["x"], _h["y"]); return (t, s, f, l, glx, gly)
    _o, _h = _hsetup(hbx + 1, hby); wh.sites = [_hsite()]
    h_accept = (wh._offer_help_maybe(_h, *_hgrids(_h)) is not None and _h.get("helping") == "hs"
                and bool(_h.get("say")))
    _o, _h = _hsetup(hbx + 1, hby); wh.sites = [_hsite()]; _h["rel"] = {"OWN": {"sentiment": -0.5, "name": "Own"}}
    h_decline = wh._offer_help_maybe(_h, *_hgrids(_h)) is None and _h.get("helping") is None
    _o, _h = _hsetup(hbx + 1, hby); _h["hunger"] = 0.8; wh.sites = [_hsite()]
    h_gated = wh._offer_help_maybe(_h, *_hgrids(_h)) is None
    help_ok = h_accept and h_decline and h_gated
    print(f"  offer-a-hand test: accepts={h_accept} disliked-declines={h_decline} hungry-skips={h_gated} "
          f"-> {'OK' if help_ok else 'FAILED'}")

    # HUNTING PARTIES — big game (a deer) can't be taken by one: a lone hunter's strikes never
    # fell it; a party of HUNT_PARTY_MIN on hand brings it down and shares the meat.
    whp = World().generate(seed=7); whp.people = []; whp.animals = []
    py, px = np.argwhere((whp.water == WATER_NONE) & (whp.biome == B["grassland"]))[0]
    whp._add_person(int(px), int(py), name="Hunter1"); h1 = whp.people[0]; h1["inv"] = {"crude_spear": 1}
    whp._add_animal("deer", int(px) + 1, int(py)); deer = whp.animals[-1]
    solo_failed = all(not whp._resolve_hunt_strike(h1, deer) for _ in range(200))   # never fells it alone
    # lone hunter targets small game only (deer skipped without a party near)
    whp._add_animal("rabbit", int(px) - 1, int(py))
    targets_small = whp._nearest_prey(h1, big_ok=False)["sp"] == "rabbit"
    whp._add_person(int(px) + 1, int(py) - 1, name="Hunter2"); h2 = whp.people[1]; h2["inv"] = {"crude_spear": 1}
    party_killed = any(whp._resolve_hunt_strike(h1, deer) for _ in range(400))      # a party fells it
    shared = h2["inv"].get("meat", 0) > 0                                           # and shares the meat
    party_ok = solo_failed and targets_small and party_killed and shared
    print(f"  hunting-party test: lone-can't-fell-deer={solo_failed} solo-targets-small={targets_small} "
          f"party-fells-it={party_killed} meat-shared={shared} -> {'OK' if party_ok else 'FAILED'}")

    # STAKES — exposure & predation give a roof and the band real survival worth.
    wx = World().generate(seed=7); wx.people = []
    xy, xx = np.argwhere((wx.water == WATER_NONE) & (wx.biome == B["grassland"]))[0]
    wx._add_person(int(xx), int(xy), name="Exposed")
    xp = wx.people[0]; xp["home_struct"] = None
    wx.weather = "storm"; wx.weather_intensity = 1.0
    threat = wx._exposure_threat(night=True)
    open_eff = threat * (1.0 - wx._shelter_factor(xp))             # roofless in a night storm
    xp["home_struct"] = "s_h"; xp["home"] = (xp["x"], xp["y"]); xp["insul"] = 1.0
    snug_eff = threat * (1.0 - wx._shelter_factor(xp))             # a snug hut shields fully
    exposure_ok = open_eff > EXPOSURE_SEVERE and snug_eff < 0.05
    print(f"  exposure test: night-storm exposure open={open_eff:.2f} vs snug-hut={snug_eff:.2f} "
          f"-> {'OK' if exposure_ok else 'FAILED'}")

    # A hungry wolf beside a lone, unsheltered soul draws blood; the band (companions near) deters it.
    xp["home_struct"] = None; wx.animals = []
    wx._add_animal("wolf", int(xp["x"]) + 1, int(xp["y"]))
    wolf = wx.animals[0]
    bit = False
    for _ in range(200):
        xp["hp"] = max(xp["hp"], 1.0); wolf["feed_next"] = 0.0
        wolf["x"], wolf["y"] = int(xp["x"]) + 1, int(xp["y"])     # keep it on the doorstep & hungry
        wx._wolves_menace_people(0.4, night=True)
        if xp["hp"] < 1.0:
            bit = True; break
    xp["hp"] = 1.0
    for k in range(WOLF_GUARDS_SAFE):                              # now ring the victim with companions
        wx._add_person(int(xp["x"]), int(xp["y"]), name=f"Guard{k}")
    for _ in range(80):
        wolf["feed_next"] = 0.0; wolf["x"], wolf["y"] = int(xp["x"]) + 1, int(xp["y"])
        wx._wolves_menace_people(0.4, night=True)
    guarded_ok = xp["hp"] == 1.0
    print(f"  predation test: lone soul bitten={bit}, guarded soul unharmed={guarded_ok} "
          f"-> {'OK' if (bit and guarded_ok) else 'FAILED'}")

    # GUARDIAN ROLE — a single soul STANDING WATCH (not just any companion) wards off the wolf
    # and zeroes the ward's operational danger, and a bold soul chooses to guard while a timid
    # one flees. Tests the deterrence, the danger-nulling, and the drive in one pass.
    wgd = World().generate(seed=11); wgd.people = []
    wyy, wxx = np.argwhere((wgd.water == WATER_NONE) & (wgd.biome == B["grassland"]))[0]
    wgd._add_person(int(wxx), int(wyy), name="Ward")
    ward = wgd.people[0]; ward["home_struct"] = None; ward["hp"] = 1.0
    wgd.animals = []; wgd._add_animal("wolf", int(ward["x"]) + 1, int(ward["y"]))
    wolfg = wgd.animals[0]
    # A lone protector who is NOT guarding (no intention) doesn't shield — the wolf still bites.
    prot = wgd._add_person(int(ward["x"]) + 2, int(ward["y"]), name="Protector")
    prot = wgd.people[-1]; prot["intention"] = {"kind": "rest"}
    wpos = [(wolfg["x"], wolfg["y"])]
    danger_idle = wgd._danger_at(ward, wpos)
    prot["intention"] = {"kind": "guard"}                          # now they stand watch
    danger_guarded = wgd._danger_at(ward, wpos)
    bit2 = False
    for _ in range(120):
        ward["hp"] = 1.0; wolfg["feed_next"] = 0.0
        wolfg["x"], wolfg["y"] = int(ward["x"]) + 1, int(ward["y"])
        prot["x"], prot["y"] = int(ward["x"]) + 2, int(ward["y"])  # keep the guardian in range
        wgd._wolves_menace_people(0.4, night=True)
        if ward["hp"] < 1.0:
            bit2 = True; break
    deter_ok = danger_idle > 0.05 and danger_guarded == 0.0 and not bit2
    # The drive: a bold soul (low caution) with a roof and a threatened ward picks "guard";
    # a timid one with the same ctx flees instead.
    bold = wgd._add_person(int(wxx), int(wyy), name="Bold"); bold = wgd.people[-1]
    bold["home_struct"] = "s_h"; bold["traits"]["caution"] = 0.05; bold["values"] = {}
    timid = wgd._add_person(int(wxx), int(wyy), name="Timid"); timid = wgd.people[-1]
    timid["home_struct"] = "s_h"; timid["traits"]["caution"] = 0.95; timid["values"] = {}
    gctx = {"ward_threat": 0.6, "danger": 0.5, "clock": 0.0, "night": False, "others_exist": True}
    bold_kind = mind.deliberate(bold, gctx, wgd.rng)["kind"]
    timid_kind = mind.deliberate(timid, gctx, wgd.rng)["kind"]
    drive_ok = bold_kind == "guard" and timid_kind == "flee"
    guardian_ok = deter_ok and drive_ok
    print(f"  guardian test: danger idle={danger_idle:.2f}→guarded={danger_guarded:.2f}, ward-bitten={bit2}, "
          f"bold→{bold_kind}/timid→{timid_kind} -> {'OK' if guardian_ok else 'FAILED'}")

    # COMMUNAL GRANARY — a forager's home surplus overflows into a shared store, and a soul
    # whose own larder is bare draws from the commons (the interdependence the pool creates).
    wg = World().generate(seed=7); wg.people = []
    gyy, gxx = np.argwhere((wg.water == WATER_NONE) & (wg.biome == B["grassland"]))[0]
    for i in range(GRANARY_MIN_HOUSED):
        wg._add_person(int(gxx) + i, int(gyy), name=f"Forager{i}")
    for q in wg.people:
        q["home_struct"] = "s_h"; q["home"] = (int(q["x"]), int(q["y"]))
    donor = wg.people[0]; donor["inv"]["food"] = 30          # a big haul to bank
    wg._deposit_home(donor)                                  # → personal larder, then overflow to commons
    granary_ok = wg.granary["x"] is not None and wg.granary["store"].get("food", 0) > 0
    wg._add_person(int(gxx), int(gyy) + 1, name="Hungry")
    needy = wg.people[-1]
    needy["home_struct"] = "s_h"; needy["home"] = (int(needy["x"]), int(needy["y"]))
    needy["store"] = {}; needy["inv"] = {}
    stock_before = wg.granary["store"].get("food", 0)
    wg._fetch_from_store(needy, "food")
    drew_ok = needy["inv"].get("food", 0) > 0 and wg.granary["store"].get("food", 0) < stock_before
    print(f"  granary test: commons stocked={granary_ok} ({wg.granary['store'].get('food',0)} food), "
          f"needy soul drew from it={drew_ok} -> {'OK' if (granary_ok and drew_ok) else 'FAILED'}")

    # TEMPLATES — the god designs a building and places it: instant stamps it finished, a bad
    # spot is refused with a reason, and site-mode drops an adoptable orphan site. Cleans up after.
    wt = World().generate(seed=7); wt.people = []
    save_res = wt.save_template({"name": "Smithy", "layout": ["WDW", "WFW", "WWW"], "insulation": 1.0})
    tid = save_res.get("id")
    listed_ok = any(b["id"] == tid for b in wt.list_templates())
    land_t = np.argwhere((wt.water == WATER_NONE) & (wt.biome == B["grassland"]))
    tyy, txx = land_t[len(land_t) // 2]
    n_blocks0 = len(wt.blocks)
    place = wt.place_template(tid, int(txx), int(tyy), instant=True)
    placed_ok = place["ok"] and len(wt.blocks) > n_blocks0
    wyy, wxx = np.argwhere(wt.water != WATER_NONE)[0]               # a spot in the water — must refuse
    bad = wt.place_template(tid, int(wxx), int(wyy), instant=True)
    refused_ok = (not bad["ok"]) and ("fit" in bad["result"])
    site_res = wt.place_template(tid, int(txx) + 8, int(tyy) + 8, instant=False)
    site_ok = site_res["ok"] and any(s.get("orphan") for s in wt.sites)
    wt.delete_template(tid)                                         # don't pollute the real library
    templates_ok = save_res["ok"] and listed_ok and placed_ok and refused_ok and site_ok
    print(f"  templates test: saved+listed={listed_ok} placed-instant={placed_ok} "
          f"bad-spot-refused={refused_ok} site-mode={site_ok} -> {'OK' if templates_ok else 'FAILED'}")

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

    # Minds: two comfortable people with complementary surpluses, set side by side, should
    # meet, build a relationship, and trade through the body loop — and a god's whisper
    # should land as a memory. (All rule-based; no model needed.)
    wm = World().generate(seed=5); wm.people = []
    gy, gx = np.argwhere((wm.water == WATER_NONE) & (wm.biome == B["grassland"]))[0]
    wm._add_person(int(gx), int(gy), name="Bram")
    wm._add_person(int(gx) + 1, int(gy), name="Cael")
    bram, cael = wm.people
    bram["home_struct"] = cael["home_struct"] = "s_test"        # comfortable enough to deal
    bram["inv"] = {"food": 8, "wood": 0}; cael["inv"] = {"food": 0, "wood": 8}
    wm.clock = 12 * 60
    for _ in range(40):
        bram["hunger"] = bram["thirst"] = bram["fatigue"] = 0.1
        cael["hunger"] = cael["thirst"] = cael["fatigue"] = 0.1
        wm._tick_minds_social()
        wm.clock += 30
    traded = bram["inv"].get("wood", 0) >= 1 and cael["inv"].get("food", 0) >= 1
    knows = cael["id"] in bram.get("rel", {})
    wm.whisper(int(gx), int(gy), "the high ground holds good stone", by="test")
    whispered = any(m["kind"] == "whisper" for m in bram.get("memory", []) + cael.get("memory", []))
    minds_ok = traded and knows and whispered
    print(f"  minds test: traded={traded} knows={knows} whisper_landed={whispered} "
          f"trust={bram['rel'][cael['id']]['trust']:.2f} -> {'OK' if minds_ok else 'FAILED'}")

    # Thinking-first: the body actuates the mind's INTENTION. A parched soul deliberates to
    # drink; a sated, housed loner reaches for company — survival is just one drive winning
    # when it bites, not a hardcoded priority. Driven entirely through the real body loop.
    wt = World().generate(seed=5); wt.people = []
    ty, tx = np.argwhere((wt.water == WATER_NONE) & (wt.biome == B["grassland"]))[0]
    wt._add_person(int(tx), int(ty), name="Thirsty")
    th = wt.people[0]; th["thirst"] = 0.9
    edible, drinkable, tree, stone, fiber, leaf, lx, ly, _, _ = wt._perceive(th["x"], th["y"])
    wt._person_decide(th, edible, drinkable, tree, stone, fiber, leaf, False, lx, ly)
    survive_first = th["intention"]["kind"] == "drink"
    # Now a comfortable, housed, lonely soul beside a friend should choose company, not calories.
    wt._add_person(int(tx) + 2, int(ty), name="Mae")
    loner = wt.people[1]
    loner["home_struct"] = "s"; loner["thirst"] = loner["hunger"] = loner["fatigue"] = 0.1
    loner["traits"]["sociability"] = 0.9; loner["last_social_t"] = -3000
    loner["delib_cd"] = 0; wt.clock = 5000
    e2, d2, t2, s2, f2, l2, lx2, ly2, _, _ = wt._perceive(loner["x"], loner["y"])
    wt._person_decide(loner, e2, d2, t2, s2, f2, l2, False, lx2, ly2)
    meaning_when_safe = loner["intention"]["kind"] in ("socialize", "befriend", "explore")
    think_ok = survive_first and meaning_when_safe
    print(f"  thinking-first test: thirsty→{th['intention']['kind']}, "
          f"safe-loner→{loner['intention']['kind']} -> {'OK' if think_ok else 'FAILED'}")

    # CIRCADIAN SLEEP — a soul keeps daylight hours: mild daytime drowsiness does NOT send it
    # to bed (it pushes through and does something), but the same tiredness AT NIGHT does, and
    # genuine daytime exhaustion (a spent stamina reserve) still earns rest. Fixes "they sleep
    # whenever". Driven through the real drive arbiter.
    ws = World().generate(seed=5); ws.people = []
    syy, sxx = np.argwhere((ws.water == WATER_NONE) & (ws.biome == B["grassland"]))[0]
    ws._add_person(int(sxx), int(syy), name="Drowsy")
    dz = ws.people[0]; dz["home_struct"] = "s_h"; dz["home"] = (int(dz["x"]), int(dz["y"]))
    dz["thirst"] = dz["hunger"] = 0.1; dz["traits"]["sociability"] = 0.8; dz["last_social_t"] = -3000
    dz["fatigue"] = 0.55; dz["stamina"] = 0.7            # tired-ish, but reserve sound
    day_ctx = ws._mind_ctx(dz, night=False)
    day_kind = max(mind.drives(dz, day_ctx), key=lambda d: d[2])[0]   # what wins by day
    night_ctx = ws._mind_ctx(dz, night=True)
    night_kind = max(mind.drives(dz, night_ctx), key=lambda d: d[2])[0]
    dz["fatigue"] = 0.95; dz["stamina"] = 0.2           # now truly spent — should rest even by day
    spent_ctx = ws._mind_ctx(dz, night=False)
    spent_kind = max(mind.drives(dz, spent_ctx), key=lambda d: d[2])[0]
    sleep_ok = day_kind != "rest" and night_kind == "rest" and spent_kind == "rest"
    print(f"  circadian test: day-drowsy→{day_kind}, night→{night_kind}, day-exhausted→{spent_kind} "
          f"-> {'OK' if sleep_ok else 'FAILED'}")

    # PRUDENCE HABITS — a hurt soul lies low to mend (#9), a forager with a full larder eases off
    # hauling more food (#14), and a weak soul's wander-leash shortens so it won't stray from
    # relief (#20). All driven through the real arbiter / actuation.
    wp = World().generate(seed=5); wp.people = []
    pyy, pxx = np.argwhere((wp.water == WATER_NONE) & (wp.biome == B["grassland"]))[0]
    wp._add_person(int(pxx), int(pyy), name="Hurt")
    hz = wp.people[0]; hz["home_struct"] = "s_h"; hz["home"] = (int(hz["x"]), int(hz["y"]))
    hz["thirst"] = hz["hunger"] = 0.1; hz["fatigue"] = 0.4; hz["stamina"] = 0.8
    hz["hp"] = 0.45                                         # wounded — should choose to rest and mend
    hurt_kind = max(mind.drives(hz, wp._mind_ctx(hz, night=False)), key=lambda d: d[2])[0]
    hz["hp"] = 1.0
    inj_ok = hurt_kind == "rest"
    # Forager ply eases off once the larder is brimming.
    fctx = {"vocation": "forager", "home_struct": "s_h", "clock": 5000.0, "night": False,
            "season": "spring", "others_exist": True}
    hz["home_struct"] = "s_h"; hz["fatigue"] = 0.1
    def _ply_u(store):
        hz["store"] = store
        return next((d[2] for d in mind.drives(hz, fctx) if d[0] == "ply"), 0.0)
    empty_ply, full_ply = _ply_u({"food": 0}), _ply_u({"food": 99})
    hz["store"] = {}
    enough_ok = full_ply < empty_ply * 0.5
    # Weak soul's leash shortens (the actuation math: a near-starving wanderer far out turns home).
    hz["hunger"] = 0.1; hz["fatigue"] = 0.9; hz["x"] = int(hz["home"][0]) + 20; hz["y"] = int(hz["home"][1])
    e4, d4, t4, s4, f4, l4, lx4, ly4, _, _ = wp._perceive(hz["x"], hz["y"])
    hz["intention"] = {"kind": "explore", "u": 9.9}; hz["delib_cd"] = wp.clock + 9e9   # hold explore
    wa, wmove = wp._person_decide(hz, e4, d4, t4, s4, f4, l4, False, lx4, ly4)
    # directed moves now carry the RAW delta (the mover pathfinds); a weak soul still turns HOMEward
    homeward = (wmove is not None and np.sign(wmove[0]) == np.sign(hz["home"][0] - hz["x"]) and wmove[1] == 0)
    leash_ok = homeward
    prudence_ok = inj_ok and enough_ok and leash_ok
    print(f"  prudence test: hurt→{hurt_kind}, full-ply {full_ply:.2f}<empty {empty_ply:.2f}={enough_ok}, "
          f"weak-leashes-home={leash_ok} -> {'OK' if prudence_ok else 'FAILED'}")

    # SOCIAL TEXTURE — two souls meeting GREET each other with a spoken line (#3, conversation
    # bubbles), and idle souls DRIFT toward company instead of scattering (#4 clustering).
    wc = World().generate(seed=5); wc.people = []
    cyy, cxx = np.argwhere((wc.water == WATER_NONE) & (wc.biome == B["grassland"]))[0]
    wc._add_person(int(cxx), int(cyy), name="Ada")
    wc._add_person(int(cxx) + 1, int(cyy), name="Ben")
    a, b = wc.people; wc.clock = 5000.0
    for q in (a, b):
        q.pop("say", None)
    mind.encounter(a, b, wc.clock, wc.rng)
    greeted = bool(a.get("say")) and bool(b.get("say"))
    # Idle clustering: a soul with a neighbour a short stroll off ambles toward them.
    a["home"] = (int(a["x"]), int(a["y"])); b["x"], b["y"] = int(a["x"]) + 5, int(a["y"])
    _act, move = wc._idle(a)
    clusters = move is not None and np.sign(move[0]) == 1 and move[1] == 0   # raw delta toward the neighbour
    social_ok = greeted and clusters
    print(f"  social-texture test: greeted={greeted} (\"{a.get('say','')}\"), idle-clusters={clusters} "
          f"-> {'OK' if social_ok else 'FAILED'}")

    # CARE — a well, housed soul resolves to TEND a sick band-mate nearby (#11), and a sick soul
    # kept company by a caretaker loses less health (nursing matters / #10 rescue feeds them).
    wn = World().generate(seed=5); wn.people = []
    nyy, nxx = np.argwhere((wn.water == WATER_NONE) & (wn.biome == B["grassland"]))[0]
    wn._add_person(int(nxx), int(nyy), name="Healer")
    wn._add_person(int(nxx) + 1, int(nyy), name="Sick")
    heal, sick = wn.people; wn.clock = 5000.0
    heal["home_struct"] = "s_h"; heal["home"] = (int(heal["x"]), int(heal["y"]))
    heal["thirst"] = heal["hunger"] = heal["fatigue"] = 0.1; heal["hp"] = 1.0
    heal["traits"]["sociability"] = 0.9; heal["store"] = {"food": 99}   # no provision pull
    heal["last_social_t"] = heal["last_explore_t"] = wn.clock           # not starved for company/novelty
    dz_d = next(iter(DISEASE))
    sick["illness"] = {"d": dz_d, "infected_t": 0.0, "onset_t": 0.0, "end_t": 1e9, "known": True}
    heal["delib_cd"] = 0
    care_kind = max(mind.drives(heal, wn._mind_ctx(heal, night=False)), key=lambda d: d[2])
    tend_picks = care_kind[0] == "tend" and care_kind[1] == sick["id"]
    # Recovery: same illness, same span — nursed vs alone — the nursed body loses less health.
    heal["intention"] = {"kind": "tend", "target": sick["id"]}
    tended_flag = wn._tended(sick)
    sick["hp"] = 1.0; wn._tick_illness(sick, 10.0); hp_nursed = sick["hp"]
    heal["intention"] = {"kind": "rest"}
    sick["hp"] = 1.0; sick["illness"]["end_t"] = 1e9; wn._tick_illness(sick, 10.0); hp_alone = sick["hp"]
    care_ok = tend_picks and tended_flag and (1.0 - hp_nursed) < (1.0 - hp_alone)
    print(f"  care test: well-soul→{care_kind[0]} (sick={tend_picks}), nursed-hploss {1-hp_nursed:.3f}"
          f"<alone {1-hp_alone:.3f} -> {'OK' if care_ok else 'FAILED'}")

    # GRATITUDE & APPRENTICESHIP — a soul repays a past benefactor before a stranger even when the
    # stranger is nearer (#gratitude), and a low-skill soul seeks out a far more-skilled band-mate
    # to learn from (#apprenticeship). Both surfaced through the real ctx + drive arbiter.
    wk = World().generate(seed=5); wk.people = []
    kyy, kxx = np.argwhere((wk.water == WATER_NONE) & (wk.biome == B["grassland"]))[0]
    wk._add_person(int(kxx), int(kyy), name="Giver")
    wk._add_person(int(kxx) + 3, int(kyy), name="Stranger")    # nearer, but no debt owed
    wk._add_person(int(kxx) + 6, int(kyy), name="Patron")      # farther, but a past benefactor
    giver, stranger, patron = wk.people; wk.clock = 5000.0
    giver["home_struct"] = "s_h"; giver["home"] = (int(giver["x"]), int(giver["y"]))
    giver["inv"]["food"] = 9
    for q in (stranger, patron):
        q["inv"]["food"] = 0
    giver["owes"] = {patron["id"]: 100.0}                      # patron once gave to me
    gctx2 = wk._mind_ctx(giver, night=False)
    gratitude_ok = gctx2.get("needy_id") == patron["id"]       # repay the patron first
    # Apprenticeship: a near-unskilled soul beside a master should see them as a mentor.
    learner = giver
    learner["recipes"] = ["rope"]; stranger["recipes"] = ["rope"]
    patron["recipes"] = ["rope", "crude_axe", "leaf_flask", "forage_sack", "campfire", "sleeping_mat"]
    learner.pop("owes", None)
    actx = wk._mind_ctx(learner, night=False)
    appr_ok = actx.get("mentor_id") == patron["id"] \
        and any(d[0] == "befriend" and d[1] == patron["id"] for d in mind.drives(learner, actx))
    print(f"  gratitude/apprentice test: repay-patron={gratitude_ok}, mentor-found={appr_ok} "
          f"-> {'OK' if (gratitude_ok and appr_ok) else 'FAILED'}")

    # CULTURE — a soul of standing passes a vivid TALE into another's memory (#16, lore spreads),
    # and a maker who hands over a tool that now serves another earns renown as a craftsman (#20).
    wl = World().generate(seed=5); wl.people = []
    lyy, lxx = np.argwhere((wl.water == WATER_NONE) & (wl.biome == B["grassland"]))[0]
    wl._add_person(int(lxx), int(lyy), name="Teller")
    wl._add_person(int(lxx) + 1, int(lyy), name="Listener")
    teller, listener = wl.people; wl.clock = 5000.0
    teller["renown"] = 0.6
    mind.remember(teller, "a wolf set on the band and we drove it off", 0.9, "danger", 0.0)
    got_tale = False
    for _ in range(80):
        if mind._tell_tale(teller, listener, wl.clock, wl.rng):
            got_tale = True; break
    story_ok = got_tale and any(m["kind"] == "tale" for m in listener.get("memory", []))
    # Maker-renown: gifting a spare tool to a band-mate who has none lifts the maker's standing.
    wl.clock = 6000.0
    teller["inv"] = {"axe": 2}; listener["inv"] = {}
    r0 = teller.get("renown", 0.0)
    for q in (teller, listener):                              # trust enough that they'll interact
        q.setdefault("rel", {})
    wl._tick_minds_social()
    maker_ok = teller.get("renown", 0.0) > r0 and listener["inv"].get("axe", 0) == 1
    print(f"  culture test: tale-spread={story_ok}, maker-renown {r0:.2f}->{teller.get('renown',0):.2f}={maker_ok} "
          f"-> {'OK' if (story_ok and maker_ok) else 'FAILED'}")

    # FESTIVAL & DEFERENCE — a turn-of-season gathering warms every bond and the elder calls it
    # (#18), and a renowned voice sways opinion further than a nobody's (#15).
    wv = World().generate(seed=5); wv.people = []
    vyy, vxx = np.argwhere((wv.water == WATER_NONE) & (wv.biome == B["grassland"]))[0]
    for i in range(3):
        wv._add_person(int(vxx) + i, int(vyy), name=f"Folk{i}")
    wv.clock = 5000.0
    elder = wv.people[0]; elder["renown"] = 0.9
    before = mind._rel(wv.people[1], wv.people[2], wv.clock)["sentiment"]
    wv._festival("summer")
    after = mind._rel(wv.people[1], wv.people[2], wv.clock)["sentiment"]
    festival_ok = after > before and bool(elder.get("say"))
    # Deference: same opinion, louder when the speaker has standing.
    third = "p_ghost"
    famous = {"id": "p_fame", "name": "Famous", "renown": 1.0,
              "rel": {third: {"name": "X", "sentiment": 0.8, "trust": 0.7, "met": 0.0, "trades": 0, "last": 0.0}}}
    humble = {"id": "p_humb", "name": "Humble", "renown": 0.0,
              "rel": {third: {"name": "X", "sentiment": 0.8, "trust": 0.7, "met": 0.0, "trades": 0, "last": 0.0}}}
    lf = {"id": "p_lf", "name": "Lf", "rel": {}}
    lh = {"id": "p_lh", "name": "Lh", "rel": {}}
    mind._gossip(famous, lf, wv.clock); mind._gossip(humble, lh, wv.clock)
    deference_ok = lf["rel"][third]["sentiment"] > lh["rel"][third]["sentiment"]
    print(f"  festival/deference test: bond {before:.2f}->{after:.2f} elder-calls={bool(elder.get('say'))}, "
          f"famous {lf['rel'][third]['sentiment']:.2f}>humble {lh['rel'][third]['sentiment']:.2f}={deference_ok} "
          f"-> {'OK' if (festival_ok and deference_ok) else 'FAILED'}")

    # POLISH — ground clears under a tile (#A), stone boulders are seeded & shipped (#F), stores
    # surface as stockpiles (#C/D), and a settled soul routes home to craft/research (#E).
    wpo = World().generate(seed=5)
    cgy, cgx = int(wpo.people[0]["y"]) if wpo.people else 40, int(wpo.people[0]["x"]) if wpo.people else 40
    wpo.veg_growth[cgy - 1:cgy + 2, cgx - 1:cgx + 2] = 0.9
    wpo._clear_ground(cgx, cgy)
    cleared_ok = float(wpo.veg_growth[cgy, cgx]) == 0.0 and float(wpo.veg_growth[cgy + 1, cgx]) == 0.0
    stone_ok = len(wpo.stone_nodes) > 0 and "stone" in wpo.snapshot()
    # Stockpiles payload: a granary + a home larder both show up.
    wpo.people = []
    syp, sxp = np.argwhere((wpo.water == WATER_NONE) & (wpo.biome == B["grassland"]))[0]
    wpo._add_person(int(sxp), int(syp), name="Keeper")
    kp = wpo.people[0]; kp["home_struct"] = "s_h"; kp["home"] = (int(kp["x"]), int(kp["y"]))
    kp["store"] = {"food": 7, "wood": 3}
    wpo.granary = {"store": {"food": 12}, "x": int(sxp) + 5, "y": int(syp)}
    piles = wpo._stockpiles_payload()
    stock_ok = any(s["communal"] for s in piles) and any(not s["communal"] and s["total"] == 10 for s in piles)
    # Craft-at-home: a settled soul far from home is sent homeward; near home it stays put.
    kp["x"], kp["y"] = int(kp["home"][0]) + 20, int(kp["home"][1])
    away = wpo._homeward_if_away(kp)
    kp["x"], kp["y"] = kp["home"]
    homebody = wpo._homeward_if_away(kp)
    craft_home_ok = away is not None and away[0] == "wander" and homebody is None
    polish_ok = cleared_ok and stone_ok and stock_ok and craft_home_ok
    print(f"  polish test: ground-cleared={cleared_ok}, stone-nodes={len(wpo.stone_nodes)}>0={stone_ok}, "
          f"stockpiles={stock_ok}, craft-routes-home={craft_home_ok} -> {'OK' if polish_ok else 'FAILED'}")

    # SITING — autonomous builds keep a buffer from water (flood-shy) and off the trees, while the
    # god (avoid=0) may still build anywhere; and _clear_ground wipes the species, not just growth.
    wsi = World().generate(seed=5)
    land = np.argwhere(wsi.water == WATER_NONE)
    # A real shoreline tile: land with water within a tile. Centre the footprint on it so the
    # build sits right by the bank — a flood-shy soul refuses it.
    _lbp = BLUEPRINTS["leaf_shelter"]; _lbw, _lbh = len(_lbp["layout"][0]), len(_lbp["layout"])
    shore = None
    for idx in range(0, len(land), max(1, len(land) // 8000)):
        ly0, lx0 = land[idx]
        if wsi._water_within(int(lx0), int(ly0), 1):
            shore = (int(lx0), int(ly0)); break
    near_rej = shore and wsi._blueprint_tasks("leaf_shelter", shore[0] - _lbw // 2, shore[1] - _lbh // 2,
                                              avoid=WATER_BUILD_BUFFER)[0] is None
    dry = None
    for idx in range(0, len(land), max(1, len(land) // 6000)):
        ly2, lx2 = land[idx]
        if not wsi._water_within(int(lx2), int(ly2), WATER_BUILD_BUFFER + 3) \
                and wsi.veg_sp[ly2, lx2] not in WOOD_IDS:
            dry = (int(lx2), int(ly2)); break
    dry_ok = dry and wsi._blueprint_tasks("leaf_shelter", dry[0], dry[1], avoid=WATER_BUILD_BUFFER)[0] is not None
    # Blanket the spot with trees: a soul PREFERS a clearing (strict siting passes it over) but
    # the relaxed fallback still builds there, and clearing the footprint wipes the trees off.
    for yy in range(dry[1] - 1, dry[1] + 4):
        for xx in range(dry[0] - 1, dry[0] + 4):
            if wsi._in(xx, yy):
                wsi.veg_sp[yy, xx] = WOOD_IDS[0]; wsi.veg_growth[yy, xx] = 0.9
    tree_preferred_away = wsi._blueprint_tasks("leaf_shelter", dry[0], dry[1], avoid=WATER_BUILD_BUFFER)[0] is None
    god_anywhere = wsi._blueprint_tasks("leaf_shelter", dry[0], dry[1], avoid=0)[0] is not None
    wsi._clear_ground(dry[0], dry[1])
    sp_cleared = int(wsi.veg_sp[dry[1], dry[0]]) == VEG_NONE
    # A soul can SAY why a near-water spot won't do (reasoning about placement).
    reason_water = wsi._build_reason("leaf_shelter", shore[0] - _lbw // 2, shore[1] - _lbh // 2,
                                     WATER_BUILD_BUFFER) if shore else None
    reason_ok = reason_water is not None
    # Self-heal: a granary stranded in the water is hauled to dry ground (and its marker moves).
    wet_xy = None
    if shore:
        for dyy, dxx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            ny, nx = shore[1] + dyy, shore[0] + dxx
            if wsi._in(nx, ny) and wsi.water[ny, nx] != WATER_NONE:
                wet_xy = (nx, ny); break
    relocated = False
    if wet_xy:
        wsi.granary = {"store": {"food": 3}, "x": wet_xy[0], "y": wet_xy[1]}
        wsi.structures = [{"id": "s_g", "kind": "granary", "x": wet_xy[0], "y": wet_xy[1], "by": "band", "t": 0}]
        wsi._relocate_granary_if_stranded()
        gxn, gyn = wsi.granary["x"], wsi.granary["y"]
        relocated = (wsi.water[gyn, gxn] == WATER_NONE and (gxn, gyn) != wet_xy
                     and any(s["kind"] == "granary" and (s["x"], s["y"]) == (gxn, gyn) for s in wsi.structures))
    siting_ok = bool(near_rej and dry_ok and tree_preferred_away and god_anywhere and sp_cleared
                     and reason_ok and relocated)
    print(f"  siting test: near-water-refused={bool(near_rej)}, inland-ok={bool(dry_ok)}, "
          f"tree-preferred-away={tree_preferred_away}, god-anywhere={god_anywhere}, species-cleared={sp_cleared}, "
          f"reason=\"{reason_water}\", granary-relocated={relocated} "
          f"-> {'OK' if siting_ok else 'FAILED'}")

    # PLANNING — a soul WEIGHS where to build: a spot by water beats a parched one, and a spot
    # with room beats one crowding a neighbour (the scorer behind deliberate settlement layout).
    parched = None
    for idx in range(0, len(land), max(1, len(land) // 6000)):
        ly3, lx3 = land[idx]
        if not wsi._water_within(int(lx3), int(ly3), SITE_WATER_IDEAL):
            parched = (int(lx3), int(ly3)); break
    water_ok = bool(shore and parched
                    and wsi._score_site(shore[0], shore[1], [], False)
                    > wsi._score_site(parched[0], parched[1], [], False))
    crowded = wsi._score_site(shore[0], shore[1], [(shore[0], shore[1])], False)   # nd=0, cramped
    roomy = wsi._score_site(shore[0], shore[1], [(shore[0] + 10, shore[1])], False)  # spaced
    spacing_ok = roomy > crowded
    planning_ok = water_ok and spacing_ok
    print(f"  planning test: prefers-water={water_ok}, prefers-room (roomy {roomy:.1f}>cramped {crowded:.1f})={spacing_ok} "
          f"-> {'OK' if planning_ok else 'FAILED'}")

    # DISMANTLE — upgrading pulls down the old home so no abandoned husks linger.
    fake = {"id": "b_old", "bp": "leaf_shelter", "name": "Leaf Shelter", "ox": 0, "oy": 0, "by": "t",
            "tasks": [{"x": 300, "y": 300, "code": 2, "layer": "block", "cost": ["leaves", 1], "done": True},
                      {"x": 301, "y": 300, "code": 1, "layer": "roof", "cost": ["leaves", 1], "done": True}],
            "done": True, "communal": False}
    wsi.sites.append(fake); wsi.blocks[(300, 300)] = 2; wsi.roofs.add((301, 300))
    wsi._dismantle_site("b_old")
    dismantle_ok = ((300, 300) not in wsi.blocks and (301, 300) not in wsi.roofs
                    and not any(s["id"] == "b_old" for s in wsi.sites))
    print(f"  dismantle test: old-home-torn-down={dismantle_ok} -> {'OK' if dismantle_ok else 'FAILED'}")

    # Discovery + water bottle: once the band works out the leaf flask, a housed person makes
    # one, fills it at the water, and can then drink from the pack far from any river — the
    # crafting fix for the thirst problem, end to end through the real body loop.
    wf = World().generate(seed=5); wf.people = []
    fy, fx = np.argwhere((wf.water == WATER_NONE) & (wf.biome == B["grassland"]))[0]
    wf._add_person(int(fx), int(fy), name="Tinker")
    tk = wf.people[0]
    learned = mind.learn_recipe(tk, wf.known_recipes, "leaf_flask", wf.clock)   # band discovers it
    tk["inv"] = {"leaves": 6, "fiber": 6}                # raw stock to fashion flask + rope
    made = wf._craft_known(tk, "rope") and wf._craft_known(tk, "leaf_flask")
    has_flask = tk["inv"].get("leaf_flask", 0) >= 1
    wf._fill_containers(tk)                              # as if standing at the water
    filled = tk["inv"].get("water", 0) >= 1
    # Now far from water and thirsty, the body should choose to drink from the pack.
    tk["thirst"] = 0.9; tk["home_struct"] = "s"; tk["delib_cd"] = 0; wf.clock = 5000
    e3, d3, t3, s3, fi3, le3, lx3, ly3, _, _ = wf._perceive(tk["x"], tk["y"])
    d3[:] = False                                       # pretend no water in reach
    act, _ = wf._person_decide(tk, e3, d3, t3, s3, fi3, le3, False, lx3, ly3)
    bottle_ok = learned and has_flask and filled and act == "drink_pack"
    print(f"  water-bottle test: discovered={learned} crafted={has_flask} filled={filled} "
          f"drinks-from-pack-away-from-water={act == 'drink_pack'} -> {'OK' if bottle_ok else 'FAILED'}")

    # Berry test (P3): forage a safe bush for food + lore; learn a poison bush and then shun it.
    wbz = World().generate(seed=5); wbz.people = []
    byb, bxb = np.argwhere((wbz.water == WATER_NONE) & (wbz.biome == B["grassland"]))[0]
    wbz._add_person(int(bxb), int(byb), name="Berryer")
    bz = wbz.people[0]; wbz.clock = 1000.0
    good = {"x": int(bxb), "y": int(byb), "poison": False, "ripe_t": 0.0}
    wbz.berry_bushes = [good]; wbz._rebuild_berry_index()
    bz["hunger"], bz["thirst"], bz["fatigue"] = 0.6, 0.05, 0.05
    bz["inv"] = {}; bz["intention"] = None; bz["delib_cd"] = 0
    e, d, t, s, fi, le, lx, ly, _, _ = wbz._perceive(bz["x"], bz["y"])
    bact, _ = wbz._person_decide(bz, e, d, t, s, fi, le, False, lx, ly)
    wbz._forage_bush(bz, good)                              # execute the pick
    berry_food = bz["inv"].get("food", 0) >= 1
    learned_good = bz["berry_lore"].get(f"{bxb},{byb}") == "good"
    not_ripe_now = not wbz._bush_ripe(good)                 # picked → must re-ripen
    # A known poison bush: force the sickness, confirm it's logged bad and then shunned.
    poison = {"x": int(bxb) + 1, "y": int(byb), "poison": True, "ripe_t": 0.0}
    wbz.berry_bushes = [poison]; wbz._rebuild_berry_index()
    bz["illness"] = None; bz["immune"] = {}
    import random as _r
    got_sick = False
    for _ in range(40):                                    # poison is probabilistic — force a bout
        bz["illness"] = None
        if wbz._maybe_poison(bz):
            got_sick = True; break
    wbz._learn_bush(bz, poison, bad=True)
    shuns = wbz._nearest_bush(bz) is None                   # the only bush is known-bad → ignored
    berry_ok = (bact == "forage_berry" and berry_food and learned_good and not_ripe_now
                and got_sick and shuns)
    print(f"  berry test: forages={bact == 'forage_berry'} food+={berry_food} learns-good={learned_good} "
          f"re-ripens={not_ripe_now} poison-sickens={got_sick} shuns-known-bad={shuns} "
          f"-> {'OK' if berry_ok else 'FAILED'}")

    # Hunt/fish/cook test (P4): spear a rabbit for meat, cook it at the hearth, and confirm raw
    # flesh can sicken while a cooked meal can't.
    whz = World().generate(seed=5); whz.people = []; whz.animals = []
    hyy, hxx = np.argwhere((whz.water == WATER_NONE) & (whz.biome == B["grassland"]))[0]
    whz._add_person(int(hxx), int(hyy), name="Hunter")
    hz = whz.people[0]; hz["inv"] = {"crude_spear": 1}; whz.clock = 1000.0
    whz._add_animal("rabbit", int(hxx) + 1, int(hyy))     # game one step away
    hunt_act = None
    for _ in range(200):                                  # pursue & strike until the kill lands
        a = whz._hunt(hz)
        if a is None:
            break
        hunt_act = a[0]
        if a[1] is None:                                  # adjacent strike — resolve it
            pid = hz.pop("_prey", None)
            prey = next((an for an in whz.animals if an["id"] == pid), None)
            if prey and whz.rng.random() < HUNT_KILL_SPEAR:
                whz.animals = [an for an in whz.animals if an["id"] != pid]
                hz["inv"]["meat"] = hz["inv"].get("meat", 0) + HUNT_MEAT_YIELD["rabbit"] + 1
        if hz["inv"].get("meat", 0) > 0:
            break
    got_meat = hz["inv"].get("meat", 0) > 0
    hz["hearth"] = True                                   # cook the catch at the hearth
    raw_before = hz["inv"].get("meat", 0)
    whz._cook_at_home(hz)
    cooked = hz["inv"].get("cooked_meat", 0) >= 1 and hz["inv"].get("meat", 0) == raw_before - 1
    # A cooked meal never sickens; force-test the raw gamble independently.
    hz["illness"] = None; hz["immune"] = {}
    before = whz._eat_cooked(hz); cooked_safe = before and hz.get("illness") is None
    hz["inv"]["meat"] = 5; hz["illness"] = None; hz["immune"] = {}
    raw_sick = any((hz.update({"illness": None}) or whz._maybe_taint(hz)) for _ in range(60))
    hunt_ok = got_meat and cooked and cooked_safe and raw_sick
    print(f"  hunt/cook test: hunts={hunt_act == 'hunt'} got-meat={got_meat} cooks={cooked} "
          f"cooked-safe={cooked_safe} raw-can-sicken={raw_sick} -> {'OK' if hunt_ok else 'FAILED'}")

    # Spoilage / preserving / seasonal / pests test (P5).
    wp5 = World().generate(seed=5); wp5.people = []
    pyy, pxx = np.argwhere((wp5.water == WATER_NONE) & (wp5.biome == B["grassland"]))[0]
    wp5._add_person(int(pxx), int(pyy), name="Keeper")
    kp = wp5.people[0]; kp["hearth"] = True
    kp["inv"] = {"meat": 10}; wp5._tick_spoilage(2.0)            # raw flesh rots fast (shelf ~1.2d)
    spoils = kp["inv"].get("meat", 0) < 10
    kp["inv"] = {"meat": 6}; wp5._cook_at_home(kp)              # a big haul gets DRIED, not just cooked
    preserves = kp["inv"].get("dried_meat", 0) >= 1
    d0 = kp["inv"].get("dried_meat", 0); wp5._tick_spoilage(30.0)
    dried_keeps = kp["inv"].get("dried_meat", 0) == d0          # preserved goods never spoil
    seasonal = (mind.SEASON_STOCK_MULT["autumn"] > mind.SEASON_STOCK_MULT["spring"] * 1.5)
    kp["store"] = {"food": 40}; raided = False                  # a fat larder draws vermin
    for _ in range(40):
        kp["store"]["food"] = 40
        wp5._tick_pests()
        if kp["store"].get("food", 40) < 40:
            raided = True; break
    p5_ok = spoils and preserves and dried_keeps and seasonal and raided
    print(f"  spoilage/season test: spoils={spoils} preserves={preserves} dried-keeps={dried_keeps} "
          f"seasonal-stockpile={seasonal} larder-raided={raided} -> {'OK' if p5_ok else 'FAILED'}")

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
