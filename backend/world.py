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
}

# The dwelling ladder — ascending comfort. Once a soul has any roof and is kitted out,
# their standing "life project" is to climb this: a draughty leaf lean-to → a snug timber
# hut → a roomy cabin. Each rung is built tile-by-tile from forageable wood & thatch (no
# station needed), so it's pure, visible material progress with the machinery that already
# works. (The deeper crafting tree — stations, metal, furniture — is a later rung that
# needs station-proximity; this is the foundation those layers build on.)
DWELLING_LADDER = ["leaf_shelter", "hut", "cabin"]
MONUMENT_BP = "gathering"          # the communal status build raised once a soul tops the ladder

# ─── Renown — social standing (Phase 2) ──────────────────────────────────────
# A soul's STANDING in the band: it grows from socially-visible achievements (raising a fine
# home, a communal monument, teaching a craft, giving freely) and fades slowly if not renewed,
# so status must be earned and maintained. Ambition turns standing into a pursued goal — the
# monument project — so settled, driven souls compete to leave a mark, not just nest.
RENOWN_GAIN = dict(dwelling=0.07, monument=0.55, teach=0.10, gift=0.05)
RENOWN_DECAY = 0.012               # fraction of standing shed per game-day (≈ half-life ~8 weeks)
AMBITION_MONUMENT = 0.55           # a soul this ambitious will undertake a monument for the band
PLY_WOOD_STOCK = 14                # a woodcutter plying their trade stocks timber up to this

# ─── Generations & lineage (Phase 4) ─────────────────────────────────────────
# A band that breeds true: bonded adults have children who inherit a blend of their parents'
# nature (but NOT their knowledge — culture must be taught afresh each generation, so it can
# grow or be lost), grow from dependent childhood to a calling of their own, and inherit home
# and a share of a parent's standing. This is where character becomes lineage and lineage,
# slowly, becomes culture. Ages are in game-days (DAYS_PER_YEAR == 60).
ADULT_AGE = 16 * DAYS_PER_YEAR     # childhood ends ~16 yrs: full capability + a vocation
BREED_MIN_AGE = 18 * DAYS_PER_YEAR
BREED_MAX_AGE = 45 * DAYS_PER_YEAR
BREED_COOLDOWN_DAYS = 14.0         # game-days between a mother's children (births must outpace attrition)
BOND_WARMTH = 0.30                 # mutual sentiment at which two adults pair off
POP_CAP = 36                       # ceiling on band size (performance + ecology)
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
RAW_FLESH_SICKEN = 0.16            # chance a raw meat/fish meal brings on tainted_gut
COOKED_HUNGER_RELIEF = 0.55        # a cooked meal is the most filling thing there is
COOKED_SATIETY = 0.22
RAW_HUNGER_RELIEF = 0.35           # raw flesh fills you too — but it's a gamble
RAW_SATIETY = 0.12

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
        self.berry_bushes: list[dict] = []  # scattered berry bushes (some poisonous), P3
        self._berry_index: dict[tuple[int, int], dict] = {}  # (x,y)->bush, rebuilt on load/seed
        self.log: list[dict] = []        # recent god actions / notable events
        # Craft knowledge. Everyone is born knowing the basics (STARTER_RECIPES); the make-
        # shift survival crafts (water flask, etc.) must be DISCOVERED by an individual and
        # then SPREAD soul to soul by teaching. `known_recipes` is the band-wide union (for
        # the catalog/UI), but who-knows-what is now personal (p["recipes"]). Every first
        # making and every failed experiment is written into the Ledger of Making.
        self.known_recipes: set[str] = set(crafting.STARTER_RECIPES)
        self.ledger: list[dict] = []     # the Ledger of Making — discoveries + dead ends
        self.speed = 1.0                 # fast-forward multiplier (1×/2×/4×), set from the UI
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
        self._seed_berry_bushes()
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
                        has_spear = p["inv"].get("crude_spear", 0) > 0
                        if self.rng.random() < (HUNT_KILL_SPEAR if has_spear else HUNT_KILL_BARE):
                            self.animals = [a for a in self.animals if a["id"] != pid]
                            yld = HUNT_MEAT_YIELD.get(prey["sp"], 2) + (1 if has_spear else 0)
                            p["inv"]["meat"] = p["inv"].get("meat", 0) + yld
                            self._note("hunt", f"{p['name']} brought down a {prey['sp']}.")
                            self.version += 1
            elif action == "fish":
                if drinkable[ly, lx]:
                    rate = FISH_CATCH_ROD if p["inv"].get("fishing_rod", 0) else FISH_CATCH_BARE
                    if self.rng.random() < rate * dt_game_min:
                        p["inv"]["fish"] = p["inv"].get("fish", 0) + 1
                        self.version += 1
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
                # If nothing's underway yet, start the bootstrap axe (survival crafts start
                # their own item in the decide step). Then tick the active craft's timer;
                # the item is only granted when the work is finished.
                if not p.get("craft") and p["inv"].get("axe", 0) < 1:
                    self._begin_craft(p, "crude_axe")
                self._advance_craft(p, dt_game_min)
            elif action == "found_site":
                self._found_site(p, p.pop("next_bp", "leaf_shelter"), communal=p.pop("next_communal", False))
            elif action == "build_block":
                self._build_next_block(p)
            # Seeking and wandering move the body; acting-in-place does not.
            if action in ("seek_food", "seek_water", "seek_wood", "seek_stone",
                          "seek_fiber", "seek_leaves", "seek_berry", "hunt", "fish",
                          "haul", "wander", "socialize"):
                self._move_person(p, movedir)

            # Health couples to the physiological RESERVES (never to comfort). Any reserve in
            # the danger zone erodes hp — the deeper, and the more reserves at once, the faster.
            # A sound body heals, but only up to its VITALITY ceiling (min of satiety & stamina),
            # so a chronically hungry or exhausted soul's hp is dragged down and slowly declines
            # even when it isn't outright starving — the malnutrition/exhaustion coupling.
            res = (p["hydration"], p["satiety"], p["stamina"])
            deficit = sum(max(0.0, PERSON["hp_danger"] - r) for r in res)
            vitality = min(p["satiety"], p["stamina"])
            if deficit > 0:
                p["hp"] = max(0.0, p["hp"] - PERSON["starve_dmg"] * deficit * dt_game_min)
            elif min(res) > PERSON["hp_safe"]:
                p["hp"] = min(vitality, p["hp"] + PERSON["heal"] * dt_game_min)

            if p["hp"] <= 0 or p["age"] > PERSON["max_age"]:
                dead.append(p)

        # Social pass: people in sight of one another notice, gossip, trade — and adults pair
        # off. Then the band may bear children. This is where reputation, the barter economy
        # and now lineage all emerge.
        self._tick_minds_social()
        self._tick_reproduction()

        for p in dead:
            if p["age"] > PERSON["max_age"]:
                cause = "old age"
            elif p.get("illness") and self.clock >= p["illness"]["onset_t"]:
                cause = p["illness"]["d"].replace("_", " ")   # the sickness took them
            else:                                            # name the reserve that gave out first
                cause = min((("thirst", p.get("hydration", 1.0)), ("hunger", p.get("satiety", 1.0)),
                             ("exhaustion", p.get("stamina", 1.0))), key=lambda kv: kv[1])[0]
            self._note("death", f"{p['name']} died of {cause}.")
            self._bequeath(p)                                # home + a share of standing pass to kin
            # Those who knew the dead carry it: a heavy, durable memory.
            for q in self.people:
                if q is p:
                    continue
                if p["id"] in q.get("rel", {}) or mind._manhattan(p, q) <= PERSON["vision"]:
                    mind.remember(q, f"{p['name']} died of {cause}", 0.95, "death", self.clock)
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
                        # A toolmaker (or anyone) beside a band-mate who lacks a piece of gear
                        # they carry a SPARE of hands it over — how crafted goods spread when
                        # they aren't barter goods. This is the toolmaker's social role.
                        for gid, _h, _r in self._GEAR:
                            if giver.get("inv", {}).get(gid, 0) >= 2 and taker.get("inv", {}).get(gid, 0) == 0:
                                ev = mind.give(giver, taker, gid, self.clock)
                                if ev:
                                    self._note("social", ev)
                                break

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

    def _seek_person(self, p):
        """Step direction toward the nearest other living person (for a social/trade goal),
        or None if alone or already beside someone. Stops adjacent so the social pass can
        run an encounter rather than walking onto them."""
        best, bd = None, 1e9
        for q in self.people:
            if q is p:
                continue
            d = abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])
            if d < bd:
                best, bd = q, d
        if best is None or bd <= 1:
            return None
        return (int(np.sign(best["x"] - p["x"])), int(np.sign(best["y"] - p["y"])))

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
        for q in self.people:
            if q is p:
                continue
            d = abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])
            if q.get("inv", {}).get("food", 0) <= 1 and d < nd and d <= PERSON["vision"] * 3:
                needy_id, needy_name, nd = q["id"], q["name"], d
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
        voc = None if is_child else mind.vocation(p)        # the soul's calling (division of labour)
        p["vocation"] = voc
        needs_hearth = (not is_child and p.get("home_struct") is not None
                        and self._person_knows(p, "campfire") and not p.get("hearth"))
        return {
            "needs_gear": needs_gear,
            "needs_hearth": needs_hearth,
            "project": proj,
            "vocation": voc,
            "clock": self.clock, "night": night, "season": self.season(),
            "weather": self.weather, "time_str": f"{int(self.time_of_day()):02d}:00",
            "others_exist": len(self.people) > 1,
            "alive_ids": tuple(names.keys()), "_names": names,
            "fav_id": fav_id, "fav_name": fav_name,
            "foe_id": foe_id, "foe_name": foe_name, "foe_mag": foe_mag,
            "needy_id": needy_id, "needy_name": needy_name,
            "nearby": nearby or "no one",
            # What the band still hasn't figured out, and how hard-pressed this soul is —
            # so a curious, recently-thirsty person is the keenest to invent.
            "unsolved": unsolved,
            "unsolved_problems": [crafting.SURVIVAL_DISCOVERIES[r] for r in unsolved],
            "hardship": min(1.0, max(p.get("thirst", 0), p.get("fatigue", 0) * 0.6)),
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
            return "rest", None
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
        if kind in ("socialize", "befriend"):
            move = self._seek_toward(p, target) if target else self._seek_person(p)
            if move is not None:
                return "socialize", move
            return self._idle(p)          # already beside them — the social pass does the rest
        if kind == "explore":
            p["last_explore_t"] = self.clock
            # Curiosity is leashed to home range: range out, but turn back before straying
            # past easy return to known water — wonder shouldn't be a death sentence.
            hx, hy = p["home"]
            if abs(hx - x) + abs(hy - y) > EXPLORE_LEASH:
                return "wander", (int(np.sign(hx - x)), int(np.sign(hy - y)))
            return "wander", self._explore_dir(p)
        if kind == "avoid" and target:
            t = next((q for q in self.people if q["id"] == target), None)
            if t is not None and (t["x"] != x or t["y"] != y):
                return "wander", (int(np.sign(x - t["x"])), int(np.sign(y - t["y"])))
        return self._idle(p)

    def _seek_toward(self, p, target_id):
        """Step toward a specific person by id (or None if gone/adjacent)."""
        t = next((q for q in self.people if q["id"] == target_id), None)
        if t is None:
            return None
        d = abs(t["x"] - p["x"]) + abs(t["y"] - p["y"])
        if d <= 1:
            return None
        return (int(np.sign(t["x"] - p["x"])), int(np.sign(t["y"] - p["y"])))

    def _idle(self, p):
        """Drift home if strayed, else amble — the resting state of a mind between aims."""
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) > 6:
            return "wander", (int(np.sign(hx - p["x"])), int(np.sign(hy - p["y"])))
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

        # First project: a crude axe — cheap, and it makes every later chop yield more.
        if inv.get("axe", 0) < 1:
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
            act = self._pursue_building(p, proj["bp"], getters)
            if act:
                return act
        # The status project: an ambitious soul whose own home is fine raises a communal
        # monument for the band — a visible bid for lasting standing (Phase 2). If a half-built
        # hall was orphaned by its raiser's death, adopt and finish it rather than starting anew.
        if proj and proj.get("kind") == "monument":
            if self._person_site(p) is None:
                orphan = self._orphaned_monument()
                if orphan is not None:
                    p["site"] = orphan["id"]
            act = self._pursue_building(p, proj["bp"], getters, communal=True)
            if act:
                return act

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
            return "haul", (int(np.sign(hx - p["x"])), int(np.sign(hy - p["y"])))
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

    def _deposit_home(self, p):
        """Resting at home, a soul banks the surplus food/water it is carrying above its travel
        reserve into the larder — so a good forage outlives the day. Only survival consumables
        are stored (building stock and gear stay on-person where the build logic expects them)."""
        inv, store = p["inv"], p.setdefault("store", {})
        for key, keep in STORE_KEEP.items():
            spare = inv.get(key, 0) - keep
            if spare > 0:
                store[key] = store.get(key, 0) + spare
                inv[key] = keep
                if inv[key] <= 0:
                    inv.pop(key, None)

    def _prefer_store(self, p, want):
        """Decide whether to fall back on the larder rather than forage. Only when the store
        actually holds it, home is within ranging distance, AND home is no farther than any
        remembered wild spot — so a needy soul is never marched PAST nearer water/food to the
        larder (that detour, overriding a closer known spring, quietly cost thirst deaths)."""
        store = p.get("store", {})
        if want == "food":
            stocked = store.get("food", 0) > 0
        else:
            stocked = store.get("safe_water", 0) + store.get("water", 0) > 0
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
        keys = ["food"] if want == "food" else ["safe_water", "water"]   # boiled water first
        if not any(store.get(k, 0) > 0 for k in keys):
            return None
        hx, hy = p["home"]
        if abs(hx - p["x"]) + abs(hy - p["y"]) > 1:                      # still on the way home
            return "haul", (int(np.sign(hx - p["x"])), int(np.sign(hy - p["y"])))
        for k in keys:                                                   # at the larder — withdraw one
            if store.get(k, 0) > 0:
                store[k] -= 1
                if store[k] <= 0:
                    store.pop(k, None)
                p["inv"][k] = p["inv"].get(k, 0) + 1
                if want == "food":
                    return "eat", None
                return ("drink_safe" if k == "safe_water" else "drink_pack"), None
        return None

    def _pursue_building(self, p, bp_name, getters, communal: bool = False):
        """Raise a building from a blueprint tile by tile: found the footprint at home, then
        forage each tile's material and lay it. Returns a body action, or None when there's
        nothing to do this beat (between steps, or just finished). Shared by the first
        lean-to, every dwelling-ladder upgrade, and the communal monument."""
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
                return "found_site", None
            return "haul", (int(np.sign(hx - x)), int(np.sign(hy - y)))
        task = self._site_next_task(site)
        if task is None:                                       # all tiles placed → finish it
            self._finish_site(p, site)
            return None
        item, qty = task["cost"]
        if inv.get(item, 0) >= qty:                            # have the material — go lay it
            if max(abs(task["x"] - x), abs(task["y"] - y)) <= 2:
                return "build_block", None
            return "haul", (int(np.sign(task["x"] - x)), int(np.sign(task["y"] - y)))
        return getters.get(item, getters["wood"])()

    def _current_dwelling_bp(self, p):
        """The blueprint name of the soul's current finished home, or None if they've no roof
        yet. Looked up from the site `home_struct` points at (finished sites are kept)."""
        sid = p.get("home_struct")
        if not sid:
            return None
        for s in self.sites:
            if s["id"] == sid:
                return s.get("bp")
        return None

    def _orphaned_monument(self):
        """An in-progress communal monument no living soul is still raising (its builder died),
        free for another to adopt and finish — so a half-built hall is never abandoned forever."""
        live_sites = {q.get("site") for q in self.people}
        for s in self.sites:
            if s["bp"] == MONUMENT_BP and not s["done"] and s["id"] not in live_sites:
                return s
        return None

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
        mon = {"kind": "monument", "bp": MONUMENT_BP,
               "why": "raise a gathering hall — a place for us all, and a name that lasts"}
        has_real_home = i >= DWELLING_LADDER.index("hut")
        mine_inprog = any(s["id"] == p.get("site") and s["bp"] == MONUMENT_BP and not s["done"]
                          for s in self.sites)
        if mine_inprog:
            return mon                                         # I'm the one building it — keep at it
        band_has_hall = any(s["bp"] == MONUMENT_BP and s["done"] for s in self.sites)
        # Only a hall someone LIVING is still raising counts as taken; one whose builder died is
        # orphaned and may be adopted (handled in the executor), so it never blocks the band.
        live_sites = {q.get("site") for q in self.people}
        someone_building = any(s["bp"] == MONUMENT_BP and not s["done"] and s["id"] in live_sites
                               for s in self.sites)
        if (has_real_home and not band_has_hall and not someone_building
                and mind._trait(p, "ambition") >= AMBITION_MONUMENT):
            return mon                                         # an ambitious soul undertakes (or adopts) it
        # Otherwise keep climbing the dwelling ladder to a snugger home.
        if i + 1 < len(DWELLING_LADDER):
            nxt = DWELLING_LADDER[i + 1]
            return {"kind": "dwelling", "bp": nxt,
                    "why": f"raise a finer home — a proper {BLUEPRINTS[nxt]['name'].lower()}"}
        return None

    # Survival gear the band has discovered, in priority order, with what each does and the
    # raw it ultimately comes from (rope is made from fiber on the spot).
    _GEAR = (("leaf_flask", "water", "leaves"), ("forage_sack", "sack", "fiber"),
             ("sleeping_mat", "mat", "fiber"))

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
                self._begin_craft(p, rid)                      # have everything — start it (takes time)
                return "craft", None
            need = crafting.missing(inv, rid)                  # what's short — go get it
            if "rope" in need and inv.get("rope", 0) < need["rope"]:
                if inv.get("fiber", 0) >= 3:
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
            return "haul", (int(np.sign(hx - x)), int(np.sign(hy - y)))
        cap = PERSON["inv_cap"] + (6 if inv.get("forage_sack", 0) else 0)
        load = min(STORE_KEEP["food"] + PROVISION_LOAD, cap)   # carry the reserve plus a load to bank
        if inv.get("food", 0) < load:
            berry = self._berry_seek(p)                # a ripe bush near home is the best larder-filler
            if berry:
                return berry
            here = bool(edible[ly, lx]) and self.veg_growth[y, x] > PERSON["gather_min"]
            return self._seek(p, x, y, here, edible, lx, ly, "food", "gather", "seek_food")
        if abs(hx - x) + abs(hy - y) > 1:                      # loaded — bring it home to the larder
            return "haul", (int(np.sign(hx - x)), int(np.sign(hy - y)))
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
            # Game first — a carcass is the richest haul. Then cast for fish if water's at hand.
            if inv.get("meat", 0) < MEAT_STOCK:
                hunt = self._hunt(p)
                if hunt:
                    return hunt
            if inv.get("fish", 0) < MEAT_STOCK and (drinkable[ly, lx] or self._nearest_local(drinkable, lx, ly)):
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

    def _blueprint_tasks(self, name, ox, oy, occupied=None):
        """Turn a blueprint at origin (ox,oy) into placement tasks + the home core tile, or
        (None, None) if the footprint doesn't fit (off-map, over water, or OVERLAPPING an
        existing building/site — which is why shelters used to grow inside one another). Each
        task carries its own material cost so blueprints can mix wood, thatch and leaves.
        Blocks are laid first, then roof tiles. A 'C' core lays no block but is roofed/home."""
        bp = BLUEPRINTS.get(name)
        if not bp:
            return None, None
        occupied = occupied if occupied is not None else self._occupied_tiles()
        layout = bp["layout"]
        roof_cost = bp.get("roof_cost", ROOF_COST)
        blocks, roof, core = [], [], None
        for dy, row in enumerate(layout):
            for dx, ch in enumerate(row):
                tx, ty = ox + dx, oy + dy
                if ch == GLYPH_CORE:
                    if not self._in(tx, ty) or self.water[ty, tx] != WATER_NONE \
                            or (tx, ty) in occupied:
                        return None, None
                    roof.append({"x": tx, "y": ty, "code": int(BLOCK_FLOOR), "layer": "roof",
                                 "cost": list(roof_cost), "done": False})
                    core = (tx, ty)
                    continue
                code = BLOCK_CHARS.get(ch, BLOCK_EMPTY)
                if code == BLOCK_EMPTY:
                    continue
                if not self._in(tx, ty) or self.water[ty, tx] != WATER_NONE \
                        or (tx, ty) in occupied:
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

    @staticmethod
    def _site_offsets():
        """Footprint origins to try, spiralling outward from home so a site lands as close
        as it can but steps away ring by ring when the near ground is taken."""
        offs = [(0, 0)]
        for r in range(1, 7):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r:
                        offs.append((dx, dy))
        return offs

    def _found_site(self, p, name: str = "leaf_shelter", communal: bool = False):
        """Reserve a building footprint near the person's home. Tries the chosen blueprint
        (falling back to the always-cheap leaf shelter) over a few offsets so a tree/edge
        doesn't block it forever; on failure sets a cooldown before retrying. A `communal`
        site (a monument) is NOT the builder's home, so it doesn't move their home anchor."""
        bx, by = p["home"]
        occupied = self._occupied_tiles()
        cands = [name] + (["leaf_shelter"] if name != "leaf_shelter" else [])
        for cand in cands:
            bp = BLUEPRINTS[cand]
            bw, bh = len(bp["layout"][0]), len(bp["layout"])
            for off in self._site_offsets():
                ox, oy = bx - bw // 2 + off[0], by - bh // 2 + off[1]
                # Non-overlapping footprint only (so shelters no longer grow inside one
                # another). We deliberately DON'T force homes far apart: the band settles
                # tight on the waterside, and pushing a home inland to make room is what
                # kills people on the commute to drink. The spiral finds the nearest free
                # spot, which stays by the bank.
                tasks, core = self._blueprint_tasks(cand, ox, oy, occupied)
                if not tasks:
                    continue
                site = {"id": "b_" + uuid.uuid4().hex[:8], "bp": cand, "name": bp["name"],
                        "ox": int(ox), "oy": int(oy), "by": p["name"],
                        "insul": float(bp.get("insulation", 1.0)),
                        "tasks": tasks, "done": False, "t": round(self.clock, 1)}
                site["communal"] = bool(communal)
                self.sites.append(site)
                p["site"] = site["id"]
                if not communal:                         # a home moves the builder's anchor; a monument doesn't
                    home = core or next(((t["x"], t["y"]) for t in tasks if t["code"] == BLOCK_FLOOR), (bx, by))
                    p["home"] = (int(home[0]), int(home[1]))
                self.version += 1
                verb = "began a" if communal else "marked out a"
                self._note("build", f"{p['name']} {verb} {bp['name'].lower()}.")
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

    def _earn_renown(self, p, amount: float, why: str) -> None:
        """Raise a soul's social standing for a visible deed, and lodge it as a proud memory."""
        p["renown"] = p.get("renown", 0.0) + amount
        mind.remember(p, why, min(0.95, 0.5 + amount), "renown", self.clock)

    def _finish_site(self, p, site):
        site["done"] = True
        self.version += 1
        if BLUEPRINTS.get(site["bp"], {}).get("communal"):
            # A monument, not a home: it doesn't house the builder, but it crowns them — the
            # band now has a shared landmark and its raiser wins lasting renown.
            self._note("build", f"{p['name']} finished a {site['name'].lower()} for the band.")
            self._earn_renown(p, RENOWN_GAIN["monument"],
                              f"raised a {site['name'].lower()} for us all — a name that will last")
            return
        p["home_struct"] = site["id"]
        p["insul"] = site.get("insul", 1.0)     # how well the finished home holds heat/cold
        self._note("build", f"{p['name']} finished building a {site['name'].lower()}.")
        mind.remember(p, f"raised my own {site['name'].lower()} — a home at last", 0.85,
                      "build", self.clock)
        self._earn_renown(p, RENOWN_GAIN["dwelling"], f"raised a fine {site['name'].lower()} of my own")

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

    def _nearest_prey(self, p, max_d=HUNT_VISION):
        """The nearest huntable animal (rabbit/deer) within `max_d`, or None."""
        x, y = p["x"], p["y"]
        best = None; best_d = max_d + 1
        for a in self.animals:
            if a["sp"] not in ("rabbit", "deer"):
                continue
            d = abs(a["x"] - x) + abs(a["y"] - y)
            if d < best_d:
                best, best_d = a, d
        return best

    def _hunt(self, p, max_d=HUNT_VISION):
        """Pursue the nearest game within `max_d`; a body action toward it, or a strike when
        adjacent (the kill is resolved in the action handler). None when there's nothing to chase."""
        prey = self._nearest_prey(p, max_d)
        if prey is None:
            return None
        dx, dy = prey["x"] - p["x"], prey["y"] - p["y"]
        if abs(dx) + abs(dy) <= 1:
            p["_prey"] = prey["id"]                 # stash the quarry for the strike handler
            return "hunt", None
        return "hunt", (int(np.sign(dx)), int(np.sign(dy)))

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
        """Tend the hearth fire: turn a measure of raw meat/fish into a safe, cooked meal. Runs
        each rest-tick beside the home hearth, like boiling water (P1b)."""
        inv = p["inv"]
        for raw, done in (("meat", "cooked_meat"), ("fish", "cooked_fish")):
            if inv.get(raw, 0) > 0:
                inv[raw] -= 1
                if inv[raw] <= 0:
                    inv.pop(raw, None)
                inv[done] = inv.get(done, 0) + 1

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

    def _nearest_bush(self, p):
        """The best ripe bush worth foraging within reach: nearest one this soul doesn't KNOW to
        be poisonous (a known-good bush is preferred, an unknown one is a gamble worth taking).
        Returns the bush dict or None."""
        lore = p.get("berry_lore", {})
        x, y = p["x"], p["y"]
        best = None; best_score = None
        for b in self.berry_bushes:
            d = abs(b["x"] - x) + abs(b["y"] - y)
            if d > BERRY_SEEK_RANGE or not self._bush_ripe(b):
                continue
            tag = lore.get(f"{b['x']},{b['y']}")
            if tag == "bad":
                continue                                  # known poison — shun it
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
        return "seek_berry", (int(np.sign(dx)), int(np.sign(dy)))

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
        p["hp"] = max(0.0, p["hp"] - DISEASE[ill["d"]]["hp"] * dt_game_min)
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

    def _begin_craft(self, p, rid: str) -> bool:
        """Start crafting `rid` if nothing's already underway: pay the inputs now and set a
        timer. Returns True if a craft is active afterwards (so the caller holds position)."""
        if p.get("craft"):
            return True
        inv = p["inv"]
        if rid == "crude_axe":                                  # bootstrap tool, predates recipes
            if inv.get("wood", 0) < BUILD["axe_wood"]:
                return False
            inv["wood"] -= BUILD["axe_wood"]
            out, qty = "axe", 1
        else:
            if rid in crafting.SURVIVAL_DISCOVERIES and not self._person_knows(p, rid):
                return False                                     # can't make what they haven't worked out
            if not crafting.can_craft(inv, rid, stations=(), tools=None):
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

    def _advance_craft(self, p, dt_game_min: float) -> bool:
        """Tick an in-progress craft; grant the item and clear the state when it finishes.
        Returns True on the tick it completes."""
        c = p.get("craft")
        if not c:
            return False
        c["left"] = max(0.0, c["left"] - dt_game_min)
        if c["left"] > 0:
            return False
        p["inv"][c["out"]] = p["inv"].get(c["out"], 0) + c["qty"]
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
            if trust < 0.4 or self.rng.random() > 0.5:
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
            "berries": [{"x": b["x"], "y": b["y"], "ripe": self._bush_ripe(b)} for b in self.berry_bushes],
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
                "berry_bushes": self.berry_bushes,
                "log": self.log, "version": self.version,
                "known_recipes": sorted(self.known_recipes),
                "ledger": self.ledger,
                "speed": self.speed,
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
