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
import json
import os
import time
import uuid

import numpy as np

# ─── Dimensions & clock ───────────────────────────────────────────────────
W = 128
H = 128

# Bumped whenever the saved-world layout changes in a way older saves can't
# satisfy. A save stamped with a different (or missing) schema is treated as
# incompatible and regenerated on load, so a world written by a broken/older
# build self-heals on the next launch instead of staying frozen forever.
SCHEMA = 2
CHUNK = 16                      # chunk size for later level-of-detail / streaming
SEA_LEVEL = 0.36                # elevation below this is ocean
MOUNTAIN_LEVEL = 0.78           # elevation above this reads as mountain/rock

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
    hunger_rate=0.00045, thirst_rate=0.00093, fatigue_rate=0.00069,
    t_hunger=0.50, t_thirst=0.45, t_rest=0.70,        # need thresholds to act on
    eat_bite=0.05, food_value=2.5,                     # graze speed / hunger restored per unit
    drink_rate=0.05, rest_rate=0.04,                   # thirst/fatigue relieved per min
    inv_cap=8, gather_min=0.30,                         # carry capacity / tile richness to gather
    starve_dmg=0.0008, heal=0.0006,                    # hp lost when a need maxes / regained when sated
)
EDIBLE_PLANTS = {"grass", "oak", "reeds", "palm", "shrub"}   # plants people can forage
NAMES_M = ("Aren", "Bram", "Cael", "Doran", "Eli", "Finn", "Garreth", "Holt",
           "Ivo", "Joss", "Korin", "Lugh", "Mato", "Niall", "Osric", "Pell")
NAMES_F = ("Ada", "Bel", "Cyra", "Dara", "Esme", "Fern", "Greta", "Hana",
           "Isla", "Juno", "Kira", "Lena", "Maeve", "Nara", "Orla", "Petra")


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
        self.log: list[dict] = []        # recent god actions / notable events
        self.version = 0                 # bumped on any mutation, for render diffing
        self.rng = np.random.default_rng()
        # Tile fields (allocated in generate()/load()).
        self.elevation = self.biome = self.soil = self.moisture = None
        self.water = self.veg_sp = self.veg_growth = None

    # ── generation ──────────────────────────────────────────────────────────
    def generate(self, seed: int | None = None) -> "World":
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy % (2**31))
        self.rng = np.random.default_rng(self.seed)
        rng = self.rng

        elev = _fractal_noise(H, W, 6, rng)
        # Pull the borders down a touch so the map tends to sit in an ocean frame.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        edge = np.minimum.reduce([xx, yy, (W - 1 - xx), (H - 1 - yy)]) / (min(W, H) / 2)
        elev = elev * (0.55 + 0.45 * np.clip(edge, 0, 1))
        elev -= elev.min(); elev /= (elev.max() + 1e-9)
        self.elevation = elev.astype(np.float32)

        moist = _fractal_noise(H, W, 5, np.random.default_rng(self.seed ^ 0x9E3779B9))
        # Latitude wetness: a damp temperate belt, drier toward the hot middle band.
        lat = np.abs(yy / (H - 1) - 0.5) * 2
        moist = np.clip(moist * 0.7 + (0.5 - np.abs(lat - 0.45)) * 0.6, 0, 1)
        self.moisture = moist.astype(np.float32)

        self.water = np.where(elev < SEA_LEVEL, WATER_OCEAN, WATER_NONE).astype(np.uint8)
        self._carve_lakes_and_rivers()
        # Damp ground near any water.
        self._dampen_near_water()

        self.biome = self._classify_biomes()
        # Soil: fertile where moist & low, poor on rock/desert/snow; plus noise.
        soil = np.clip(0.35 + self.moisture * 0.5 - elev * 0.25, 0, 1)
        soil += (_fractal_noise(H, W, 4, np.random.default_rng(self.seed ^ 0x12345)) - 0.5) * 0.2
        for bad in ("rock", "mountain", "snow", "desert", "ocean"):
            soil[self.biome == B[bad]] *= 0.35
        self.soil = np.clip(soil, 0, 1).astype(np.float32)

        self.veg_sp = np.zeros((H, W), np.uint8)
        self.veg_growth = np.zeros((H, W), np.float32)
        self._seed_initial_vegetation()
        self._seed_initial_wildlife(count=130)
        self._seed_initial_people(count=6)

        self.clock = 8 * 60.0            # start at 08:00 on day 0
        self._last_eco = self.clock
        self.version += 1
        self._note("world", f"A new world took shape (seed {self.seed}).")
        return self

    def _carve_lakes_and_rivers(self):
        elev = self.elevation
        # Lakes: interior local minima that aren't ocean.
        nbr_min = np.full_like(elev, 1.0)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nbr_min = np.minimum(nbr_min, np.roll(np.roll(elev, dy, 0), dx, 1))
        depression = (elev <= nbr_min + 0.002) & (elev >= SEA_LEVEL) & (elev < SEA_LEVEL + 0.18)
        self.water[depression] = WATER_LAKE
        # Rivers: from a handful of high, wet springs, follow steepest descent to water.
        springs = np.argwhere((elev > 0.62) & (self.moisture > 0.5))
        if len(springs):
            picks = self.rng.choice(len(springs), size=min(12, len(springs)), replace=False)
            for idx in picks:
                self._trace_river(int(springs[idx][0]), int(springs[idx][1]))

    def _trace_river(self, y: int, x: int):
        elev = self.elevation
        for _ in range(400):
            if self.water[y, x] in (WATER_OCEAN, WATER_LAKE):
                return
            self.water[y, x] = WATER_RIVER
            best, by, bx = elev[y, x], y, x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and elev[ny, nx] < best:
                        best, by, bx = elev[ny, nx], ny, nx
            if (by, bx) == (y, x):       # stuck in a pit → make it a lake, stop
                self.water[y, x] = WATER_LAKE
                return
            y, x = by, bx

    def _dampen_near_water(self):
        wet = (self.water != WATER_NONE).astype(np.float32)
        near = wet.copy()
        for _ in range(3):               # spread the influence a few tiles out
            acc = near.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                acc = np.maximum(acc, np.roll(np.roll(near, dy, 0), dx, 1) * 0.7)
            near = acc
        self.moisture = np.clip(self.moisture + near * 0.35, 0, 1).astype(np.float32)

    def _annual_temperature(self) -> np.ndarray:
        """Mean annual temperature field in [0,1] from latitude and elevation."""
        yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
        lat = np.abs(yy / (H - 1) - 0.5) * 2         # 0 at equator-ish middle, 1 at poles
        temp = 0.92 - lat * 0.62 - self.elevation * 0.42
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
        # Wet lowlands become swamp.
        bm[(elev < SEA_LEVEL + 0.08) & (moist > 0.7) & (self.water == WATER_NONE)] = B["swamp"]
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

    def _suitability(self, species: int) -> np.ndarray:
        """Per-tile growth suitability in [-1,1] for a plant *right now* (season-aware)."""
        p = PLANTS[species]
        temp = self.temperature_field()
        biome_ok = np.isin(self.biome, [B[b] for b in p["biomes"]]).astype(np.float32)
        t_ok = _band(temp, *p["t"])
        m_ok = _band(self.moisture, *p["m"])
        suit = biome_ok * np.minimum(t_ok, m_ok) * (0.4 + 0.6 * self.soil)
        # No land plants in water.
        suit[self.water != WATER_NONE] = 0
        # Below 0 means actively dying conditions (out of comfort band on a live tile).
        return (suit * 2 - (biome_ok * 0.0)).astype(np.float32) - (1 - np.minimum(t_ok, m_ok)) * 0.15

    def _seed_initial_vegetation(self):
        for sp in PLANTS:
            suit = self._suitability(sp)
            chance = np.clip(suit, 0, 1) * PLANTS[sp]["spread"] * 8
            place = (self.rng.random((H, W)) < chance) & (self.veg_sp == 0)
            self.veg_sp[place] = sp
            self.veg_growth[place] = self.rng.random((H, W))[place].astype(np.float32) * 0.6 + 0.2

    def _seed_initial_wildlife(self, count: int):
        land = np.argwhere((self.water == WATER_NONE) & (self.biome != B["ocean"]))
        if not len(land):
            return
        weights = {"rabbit": 0.62, "deer": 0.31, "wolf": 0.07}
        for _ in range(count):
            sp = self.rng.choice(list(weights), p=list(weights.values()))
            y, x = land[self.rng.integers(len(land))]
            self._add_animal(sp, int(x), int(y), age=self.rng.integers(0, 60))

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

    def temperature_field(self) -> np.ndarray:
        """Current temperature field: annual mean shifted by season and time of day."""
        base = self._annual_temperature()
        season_off = {"spring": 0.0, "summer": 0.18, "autumn": -0.02, "winter": -0.22}[self.season()]
        diurnal = np.cos((self.time_of_day() / 24 - 0.5) * 2 * np.pi) * -0.06  # cool nights
        cold = -0.10 if self.weather in ("rain", "storm", "snow") else 0.0
        return np.clip(base + season_off + diurnal + cold, 0, 1).astype(np.float32)

    # ── stepping ────────────────────────────────────────────────────────────────
    def step(self, dt_real_sec: float):
        """Advance the world by `dt_real_sec` of wall-clock time. Movement/wildlife run
        every call; heavy ecology batches once per game-hour to stay cheap."""
        dt_game_min = dt_real_sec * GAME_SEC_PER_REAL_SEC / 60.0
        self.clock += dt_game_min
        self._update_weather()
        self._tick_wildlife(dt_game_min)
        self._tick_people(dt_game_min)
        if self.clock - self._last_eco >= 60.0:                    # one game-hour elapsed
            steps = int((self.clock - self._last_eco) // 60.0)
            for _ in range(min(steps, 6)):                         # cap catch-up bursts
                self._tick_ecology()
            self._last_eco = self.clock
        self.version += 1

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

    def _tick_ecology(self):
        """One game-hour of plant growth, spread, soil and moisture dynamics."""
        # Rain refills moisture; sun/heat dries it. Diffuse gently toward equilibrium.
        if self.weather in ("rain", "storm"):
            self.moisture = np.clip(self.moisture + 0.04 * self.weather_intensity, 0, 1)
        elif self.weather == "clear":
            dry = 0.012 + 0.02 * np.clip(self.temperature_field() - 0.5, 0, 1)
            self.moisture = np.clip(self.moisture - dry, 0, 1)
        # Keep water-adjacent ground from drying out.
        self._dampen_tick()

        grown_any = False
        for sp in PLANTS:
            mask = self.veg_sp == sp
            if not mask.any():
                continue
            suit = self._suitability(sp)
            rate = PLANTS[sp]["grow"]
            delta = np.where(suit > 0, suit * rate, suit * 0.05)   # thrive vs. wither
            self.veg_growth[mask] = self.veg_growth[mask] + delta[mask]
            # Death: growth fell to/below zero.
            dead = mask & (self.veg_growth <= 0)
            self.veg_sp[dead] = VEG_NONE
            self.veg_growth[dead] = 0
            self.veg_growth = np.clip(self.veg_growth, 0, 1)
            # Spread: mature plants seed an empty, suitable neighbour.
            mature = (self.veg_sp == sp) & (self.veg_growth > 0.75)
            if mature.any():
                self._spread(sp, mature, suit)
                grown_any = True
            # Soil: growing plants slowly draw it down.
            self.soil[self.veg_sp == sp] = np.clip(self.soil[self.veg_sp == sp] - 0.0008, 0, 1)
        # Fallow soil slowly recovers.
        fallow = self.veg_sp == VEG_NONE
        self.soil[fallow] = np.clip(self.soil[fallow] + 0.0004, 0, 1)
        if grown_any:
            pass

    def _spread(self, sp: int, mature: np.ndarray, suit: np.ndarray):
        empty = (self.veg_sp == VEG_NONE) & (suit > 0.15) & (self.water == WATER_NONE)
        chance = PLANTS[sp]["spread"]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = np.roll(np.roll(mature, dy, 0), dx, 1)
            cand = src & empty & (self.rng.random((H, W)) < chance)
            self.veg_sp[cand] = sp
            self.veg_growth[cand] = 0.05
            empty &= ~cand

    def _dampen_tick(self):
        wet = self.water != WATER_NONE
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj = np.roll(np.roll(wet, dy, 0), dx, 1)
            self.moisture[adj] = np.maximum(self.moisture[adj], 0.6)

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
            "inv": {},                                   # carried goods, e.g. {"food": 3}
            "home": (int(x), int(y)),                    # anchor: idle wandering drifts back here
            "action": "wander",                          # current body behaviour (for the renderer)
        })

    def _seed_initial_people(self, count: int):
        """Settle a small founding band together on hospitable ground *within reach of
        water* — people drink, and thirst is the fastest need, so a dry start is a
        death sentence."""
        habitable = np.isin(self.biome, [B["grassland"], B["forest"], B["savanna"], B["beach"]])
        # Tiles a short walk (~5) from a drink: dilate the land-beside-water mask.
        watery = self.water != WATER_NONE
        near_water = watery.copy()
        for _ in range(5):
            acc = near_water.copy()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                acc |= np.roll(np.roll(near_water, dy, 0), dx, 1)
            near_water = acc
        land = self.water == WATER_NONE
        good = np.argwhere(land & habitable & near_water)
        if not len(good):
            good = np.argwhere(land & near_water)        # any watered land
        if not len(good):
            good = np.argwhere(land)                      # last resort: anywhere dry
        if not len(good):
            return
        cy, cx = good[self.rng.integers(len(good))]
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
        # Perception fields, computed once per tick and shared by everyone.
        edible = np.zeros((H, W), bool)
        for sp, info in PLANTS.items():
            if info["name"] in EDIBLE_PLANTS:
                edible |= (self.veg_sp == sp)
        edible &= (self.veg_growth > 0.12)
        watery = self.water != WATER_NONE
        drinkable = np.zeros((H, W), bool)               # land tiles bordering water (drink spots)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            drinkable |= np.roll(np.roll(watery, dy, 0), dx, 1)
        drinkable &= (self.water == WATER_NONE)

        dead = []
        for p in self.people:
            p["age"] += dt_day
            p["hunger"] = min(1.0, p["hunger"] + PERSON["hunger_rate"] * dt_game_min)
            p["thirst"] = min(1.0, p["thirst"] + PERSON["thirst_rate"] * dt_game_min)
            p["fatigue"] = min(1.0, p["fatigue"] + PERSON["fatigue_rate"] * dt_game_min)

            action, movedir = self._person_decide(p, edible, drinkable, night)
            p["action"] = action
            x, y = p["x"], p["y"]
            if action == "eat":
                g = float(self.veg_growth[y, x])
                if edible[y, x] and g > 0.12:                    # graze the tile
                    bite = min(g, PERSON["eat_bite"] * dt_game_min)
                    self.veg_growth[y, x] = g - bite
                    p["hunger"] = max(0.0, p["hunger"] - bite * PERSON["food_value"])
                elif p["inv"].get("food", 0) > 0:                # eat from the pack
                    p["inv"]["food"] -= 1
                    p["hunger"] = max(0.0, p["hunger"] - 0.35)
            elif action == "drink":
                p["thirst"] = max(0.0, p["thirst"] - PERSON["drink_rate"] * dt_game_min)
            elif action == "rest":
                p["fatigue"] = max(0.0, p["fatigue"] - PERSON["rest_rate"] * dt_game_min)
            elif action == "gather":
                g = float(self.veg_growth[y, x])
                if edible[y, x] and g > PERSON["gather_min"] and p["inv"].get("food", 0) < PERSON["inv_cap"]:
                    take = min(g - 0.2, 0.3)
                    self.veg_growth[y, x] = g - take
                    p["inv"]["food"] = p["inv"].get("food", 0) + 1
            # Seeking and wandering move the body; acting-in-place does not.
            if action in ("seek_food", "seek_water", "wander"):
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

    def _person_decide(self, p, edible, drinkable, night):
        """Pick the most pressing body action. Returns (action, movedir|None).
        Priority: thirst → hunger → rest → opportunistic gathering → wander."""
        x, y, v = p["x"], p["y"], PERSON["vision"]
        if p["thirst"] >= PERSON["t_thirst"]:
            if drinkable[y, x]:
                return "drink", None
            d = self._nearest_dir(x, y, drinkable, v)
            if d:
                return "seek_water", d
        if p["hunger"] >= PERSON["t_hunger"]:
            if (edible[y, x] and self.veg_growth[y, x] > 0.12) or p["inv"].get("food", 0) > 0:
                return "eat", None
            d = self._nearest_dir(x, y, edible, v)
            # Food in sight → head for it; none in sight → range outward to find new
            # grazing (NOT back home — the local patch is what's exhausted).
            return "seek_food", d
        if p["fatigue"] >= PERSON["t_rest"] or (night and p["fatigue"] > 0.35):
            return "rest", None
        # Opportunistic top-ups: sip or nibble while standing on a resource so needs
        # never build to a crisis when supplies are close at hand.
        if drinkable[y, x] and p["thirst"] > 0.25:
            return "drink", None
        if edible[y, x] and self.veg_growth[y, x] > 0.12 and p["hunger"] > 0.30:
            return "eat", None
        if (edible[y, x] and self.veg_growth[y, x] > PERSON["gather_min"]
                and p["inv"].get("food", 0) < PERSON["inv_cap"]):
            return "gather", None
        # Idle: drift back toward home if we've strayed, else amble.
        hx, hy = p["home"]
        if abs(hx - x) + abs(hy - y) > 6:
            return "wander", (int(np.sign(hx - x)), int(np.sign(hy - y)))
        return "wander", None

    def _nearest_dir(self, x: int, y: int, mask: np.ndarray, v: int):
        """Step direction toward the nearest True tile within a vision box, or None."""
        y0, y1 = max(0, y - v), min(H, y + v + 1)
        x0, x1 = max(0, x - v), min(W, x + v + 1)
        sub = mask[y0:y1, x0:x1]
        if not sub.any():
            return None
        ys, xs = np.nonzero(sub)
        gx, gy = xs + x0, ys + y0
        k = int(np.argmin(np.abs(gx - x) + np.abs(gy - y)))
        return (int(np.sign(gx[k] - x)), int(np.sign(gy[k] - y)))

    def _move_person(self, p, direction):
        """One step (people can't walk onto water). None → a random amble."""
        if direction and (direction[0] or direction[1]):
            sx, sy = int(np.sign(direction[0])), int(np.sign(direction[1]))
        else:
            sx, sy = int(self.rng.integers(-1, 2)), int(self.rng.integers(-1, 2))
        nx, ny = p["x"] + sx, p["y"] + sy
        if self._in(nx, ny) and self.water[ny, nx] == WATER_NONE:
            p["x"], p["y"] = nx, ny

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

    def snapshot(self) -> dict:
        """Full world for an initial render (tile layers as base64 uint8 grids) + entities."""
        return {
            "w": W, "h": H, "version": self.version, "seed": self.seed,
            "clock": round(self.clock, 1), "day": self.day(), "time": round(self.time_of_day(), 2),
            "season": self.season(), "weather": self.weather,
            "biomes": BIOMES, "sea_level": int(SEA_LEVEL * 255),
            "plants": {sp: PLANTS[sp]["name"] for sp in PLANTS},
            "layers": {
                "elevation": self._b64u8((self.elevation * 255).astype(np.uint8)),
                "biome": self._b64u8(self.biome),
                "water": self._b64u8(self.water),
                "veg_sp": self._b64u8(self.veg_sp),
                "veg_growth": self._b64u8((self.veg_growth * 255).astype(np.uint8)),
            },
            "animals": self.animals,
            "people": self.people,
        }

    def tick_state(self) -> dict:
        """Light per-tick payload for the live renderer (no heavy tile layers — those
        come once via snapshot(); ticks just move time, weather and entities)."""
        return {
            "version": self.version, "clock": round(self.clock, 1), "day": self.day(),
            "time": round(self.time_of_day(), 2), "season": self.season(),
            "weather": self.weather, "census": self.census(),
            "animals": self.animals, "people": self.people,
        }

    def census(self) -> dict:
        counts = {}
        for a in self.animals:
            counts[a["sp"]] = counts.get(a["sp"], 0) + 1
        veg = {PLANTS[sp]["name"]: int((self.veg_sp == sp).sum()) for sp in PLANTS}
        return {"animals": counts, "vegetation": veg, "people": len(self.people)}

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
            ppl = (f"People: {len(self.people)} alive ({roster})"
                   + (f"; struggling: {', '.join(distress[:6])}" if distress else "; all faring well") + ". ")
        else:
            ppl = "People: none yet — the land is unpeopled. "
        return (
            f"THE WORLD — a {W}×{H} land you and he preside over as gods. "
            f"Day {self.day()}, {self.time_of_day():.0f}:00, {self.season()}, weather {self.weather}. "
            f"Terrain: {land} land tiles, {water} water. Wildlife: {animals}. Flora: {plants}. "
            f"{ppl}"
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
                veg_growth=self.veg_growth,
            )
            meta = {
                "schema": SCHEMA,
                "seed": self.seed, "clock": self.clock, "last_eco": self._last_eco,
                "weather": self.weather, "weather_intensity": self.weather_intensity,
                "weather_until": self._weather_until, "animals": self.animals,
                "people": self.people, "log": self.log, "version": self.version,
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
            # Guard against arrays saved at a different grid size.
            if self.biome.shape != (H, W):
                print(f"[world] saved grid {self.biome.shape} != {(H, W)}; regenerating")
                return False
            self.seed = meta.get("seed", 0); self.clock = meta.get("clock", 0.0)
            self._last_eco = meta.get("last_eco", self.clock)
            self.weather = meta.get("weather", "clear")
            self.weather_intensity = meta.get("weather_intensity", 0.0)
            self._weather_until = meta.get("weather_until", 0.0)
            self.animals = meta.get("animals", []); self.people = meta.get("people", [])
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

    print(f"generated 128×128 world (seed {w.seed}) in {gen_ms:.0f} ms")
    print(f"  water tiles: {water_tiles}  ({water_tiles*100//(W*H)}%)")
    print(f"  biomes: {dist}")
    print(f"  start census: {w.census()}")
    print(f"  founding band: {[p['name'] for p in w.people]}")

    # Simulate ~20 game-days (1 step = 1 real sec = 24 game-sec → 3600 steps/game-day).
    # Watch populations and vegetation across a couple of season turns.
    days = 8
    steps = days * 3600
    print("\n  day  season   weather  rabbit deer wolf  ppl  vegtiles  ms/step")
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
                  f"{c['people']:>3}  {sum(c['vegetation'].values()):>8}  {ms:>6.2f}", flush=True)
    sim_s = time.time() - t0
    if w.people:
        sample = w.people[0]
        print(f"  survivor sample — {sample['name']}: hunger {sample['hunger']:.2f} "
              f"thirst {sample['thirst']:.2f} fatigue {sample['fatigue']:.2f} hp {sample['hp']:.2f} "
              f"inv {sample['inv']} doing '{sample['action']}'")

    # Death test: a person stranded on barren rock (no food/water in reach) must die.
    rock = np.argwhere(w.biome == B["rock"])
    if len(rock):
        ry, rx = rock[0]
        w._add_person(int(rx), int(ry), name="Stranded")
        before = len(w.people)
        for _ in range(3 * 3600):                  # up to 3 game-days
            w.step(dt_real_sec=1.0)
            if not any(p["name"] == "Stranded" for p in w.people):
                break
        gone = not any(p["name"] == "Stranded" for p in w.people)
        print(f"  death test: stranded person {'died as expected' if gone else 'SURVIVED (unexpected)'} "
              f"(pop {before}->{len(w.people)})")
    print(f"\nsimulated {steps} steps in {sim_s:.2f}s "
          f"({steps/sim_s:.0f} steps/s, {sim_s*1000/steps:.2f} ms/step avg)")

    # Exercise a couple of god actions, then persistence round-trips.
    w.sculpt(64, 64, 6, 0.25, by="test")
    w.add_water(40, 40, 4, "lake", by="test")
    w.spawn_animal(64, 64, "deer", n=5, by="test")
    w.plant(64, 64, "oak", radius=3, by="test")
    w.spawn_person(64, 64, n=3, by="test")
    snap = w.snapshot()
    print(f"\nsnapshot ok: {len(snap['layers'])} layers, {len(snap['animals'])} animals, "
          f"{len(snap['people'])} people, "
          f"~{sum(len(v) for v in snap['layers'].values())//1024} KB packed")

    w.save()
    w2 = World()
    ok = w2.load()
    print(f"persistence round-trip: {'OK' if ok and w2.day() == w.day() else 'FAILED'} "
          f"(reloaded day {w2.day()}, clock {w2.clock:.0f})")
    print(f"\ndigest preview:\n{w.digest()}")
    sys.exit(0)
