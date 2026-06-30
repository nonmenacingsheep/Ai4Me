# The World → Civilization: Implementation Spec
### Companion to `WORLD_CIVILIZATION_ROADMAP.md` — concrete data structures, integration points, algorithms, and near-buildable code

> **Read the roadmap first.** That document is the *why* and the *what* (foundations F1–F4, the era ladder, milestones M0–M11, the research). **This** document is the *how*: the actual hook points in the existing code, real Python data structures and stubs that match the live schemas, the tech-tree extension as real recipe rows, the god-action/endpoint wiring, the render contract, and the M0 milestone spec'd to near-code. Everything here is keyed to the **real shapes** read out of `world.py` / `crafting.py` / `server.py` as of `SCHEMA = 7`.

---

## 1. The integration map — exactly where things hook in

### 1.1 The tick (`World.step`, `world.py:1269`)
Current order each call:
```
clock += dt ; _update_weather ; _tick_wildlife ; _tick_people ; _tick_reactors
hourly:  _tick_ecology_active ; _tick_spoilage
daily:   _tick_pests ; _tick_governance ; _decay_footfall
season:  _festival
version += 1
```
**Insert new stages in this exact spot** (keep the cheap-per-tick / slow-per-area discipline):
```python
def step(self, dt_real_sec):
    ...
    self._tick_wildlife(dt_game_min)
    self._tick_people(dt_game_min)          # MICRO agents (unchanged)
    self._tick_cohorts(dt_game_min)         # NEW F2: macro population, cheap vectorized
    self._tick_reactors(dt_game_min)
    if self.clock - self._last_eco >= 60.0:
        self._tick_ecology_active()
        self._tick_networks()               # NEW F3: utility flood-fill (hourly is plenty)
        self._last_eco = self.clock
    if self.clock - self._last_pest >= 1440.0:   # daily block
        self._tick_pests()
        self._tick_governance()
        self._decay_footfall()
        self._harden_roads()                # NEW F3: desire-paths → dirt roads
        self._tick_settlements()            # NEW F1: founding, zoning fill, public works, era advance
        self._tick_economy_macro()          # NEW: prices, trade routes, migration
        self._last_pest = self.clock
    ...
```
**Why these cadences:** cohorts are cheap but touch population every tick for smooth growth/movement of aggregate flows; networks/roads/settlement-planning change slowly, so daily/hourly keeps the 4M-tile map cheap. Never add a per-tick whole-grid loop.

### 1.2 Persistence (`World.save` / `load`, `SCHEMA`)
Every new persisted field below must be added to `save()`/`load()` and **`SCHEMA` bumped (7 → 8)**. New entity lists serialize as plain dicts (numpy → use the existing `_json_safe`). Test: delete `~/.ai4me/world.*` → clean gen; load a `SCHEMA 7` save → self-heal/regen.

### 1.3 Serialization to the renderer (`snapshot()` 5631, `tick_state()` 5696, `view()` 5663)
- `snapshot()` ships once (full world, tile layers downsampled to `OVERVIEW_MAX=256`). **Add:** `settlements`, `roads`, `buildings`, `zones`, and optional `overlays` (power/water/sewage/traffic fields, packed via `_b64u8` like tile layers).
- `tick_state()` ships ~6/s (light). **Add only what moves:** road/building *deltas* by `version`, aggregate cohort counts, vehicle/flow markers. Keep it small (the existing `_HEAVY_PERSON_KEYS` strip is the precedent).
- `view(x0,y0,x1,y1,step)` streams crisp tiles for the zoomed window — **add a parallel `view_civic()`** returning roads/buildings/zones clipped to the window for the street-zoom.

### 1.4 God actions (`_apply_world_action` `server.py:2037`, `_WORLD_ARGS` 2079)
The dispatch is a flat `if tool == "...":` chain calling a `World` method, mirrored for Aitha's directives via `_WORLD_ARGS`. **Add new tools the same way** (§7): `zone`, `road`, `found`, `decree`. Each gets a `World.<verb>(...)` method that bumps `self.version` and calls `self._note(...)`, plus a `_WORLD_ARGS` row so Aitha can use it too.

### 1.5 Endpoints (`server.py:1096+`)
Every world route is gated by `capabilities.world` and uses `await asyncio.to_thread(...)` + `broadcast({"type":"world_changed", ...})`. **Add** `GET /api/world/settlement/{id}` (the civic inspector) and `GET /api/world/view_civic` following that exact template.

---

## 2. New modules (file plan)

```
backend/
  settlement.py   # F1  Settlement, Cohort, founding, zoning, institutions, public works
  networks.py     # F3  RoadGraph, UtilityNetwork (power/water/sewage), HPA* pathfinder
  buildings.py    # F4  Building entity, floors/capacity, RCI demand→fill, land value
  citygen.py      # §2 roadmap: procedural road skeleton, block subdivision, zoning layout
  production.py    # Era 5+: production/supply chains + logistics flows
  # world.py gains: self.settlements/cohorts/roads/buildings/networks + the new _tick_* stages
  # crafting.py gains: new RECIPE rows, STRUCTURE_KINDS, TECH_LADDER (§6)
```
All of these stay **pure Python (+numpy), no LLM, deterministic on `self.rng`**, like `crafting.py`/`mind.py`, so each is unit-testable headless (`python settlement.py`).

---

## 3. Concrete data structures (field-accurate to the live schemas)

### 3.1 `Settlement` (F1) — `backend/settlement.py`
```python
class Settlement:
    """A first-class town the World owns a list of. Promotes today's _origin/home-cluster
    into an object that owns territory, population (micro ids + macro cohorts), a treasury,
    zoning, infrastructure graphs, and an institution that commissions public works."""
    def __init__(self, sid, name, cx, cy, founded):
        self.id = sid                      # "town_" + uuid hex[:8]
        self.name = name
        self.cx, self.cy = int(cx), int(cy)        # centroid (recomputed from members)
        self.founded = float(founded)              # clock at founding
        self.chunks = set()                        # owned (cx,cy) CHUNK cells = territory
        self.micro_ids = set()                     # person["id"]s simulated in full (F2)
        self.cohorts = []                          # [Cohort] — the aggregated bulk (F2)
        self.households = []                        # [{lot, building_id, size, occ_mix}]
        self.treasury = 0                          # coin (ties into money_invented economy)
        self.stores = {}                           # aggregate granary {item: qty}
        self.zones = {}                            # (x,y) -> "R"/"C"/"I"/"civic" (sparse)
        self.land_value = {}                       # (x,y) -> 0..1 (drives upgrade/high-rise)
        self.roads = None                          # RoadGraph handle (or shared world graph view)
        self.util = {}                             # "power"/"water"/"sewage" -> UtilityNetwork
        self.era = 0                               # index into ERAS (see §6)
        self.leader_id = None                      # highest-renown soul → headman/mayor
        self.council = []                          # top-N renowned (era-gated)
        self.laws = []                             # [{name, rule_id, since}]
        self.tax_rate = 0.0
        self.works_queue = []                      # [PublicWork] the authority will build
        self.demand = {"R": 0.0, "C": 0.0, "I": 0.0}   # RCI pressure (CS-style)

    def population(self):                          # micro + macro
        return len(self.micro_ids) + sum(c.n for c in self.cohorts)

    def to_dict(self):  ...                        # JSON for snapshot()/inspector
    @classmethod
    def from_dict(cls, d):  ...                    # load() restore
```

### 3.2 `Cohort` (F2) — the macro population (this is the scaling unlock)
```python
class Cohort:
    """A statistical bundle of people the sim does NOT tick individually. One Cohort stands
    for `n` souls of a rough age band + occupation, with aggregate needs/output. Ticked by
    cheap vectorized math per game-hour. A cohort PROMOTES a representative to a micro person
    when it enters the active region or produces a notable; micro DEMOTES back to a cohort
    when it leaves. Transitions carry forward summaries so nothing is lost (lossless enough)."""
    __slots__ = ("n", "age_mean", "occ", "needs", "output", "mood", "x", "y")
    def __init__(self, n, age_mean, occ, x, y):
        self.n = int(n)                # head count (respect PEOPLE_PER_AGENT scale)
        self.age_mean = float(age_mean)
        self.occ = occ                 # "farmer"/"laborer"/"crafter"/"merchant"/...
        self.needs = {"food": 0.0, "water": 0.0, "shelter": 0.0}
        self.output = {}               # goods produced per game-day (feeds production.py)
        self.mood = 0.6                # aggregate contentment (folk-psychology summary)
        self.x, self.y = int(x), int(y)
```
**Demographics tick (vectorized, deterministic):**
```python
def _tick_cohorts(self, dt):
    days = dt / 1440.0
    for s in self.settlements:
        for c in s.cohorts:
            food = s.stores.get("food", 0)
            birth = ERA_FERTILITY[s.era] * c.n * days * (1.0 if food > c.n else 0.3)
            death = ERA_MORTALITY[s.era] * c.n * days * (1.0 if food > c.n else 2.5)
            c.n = max(0, c.n + int(self.rng.poisson(max(0, birth)))
                              - int(self.rng.poisson(max(0, death))))
            # consume from aggregate stores; produce per occupation → s.stores / production.py
            self._cohort_consume_produce(s, c, days)
        s.cohorts = [c for c in s.cohorts if c.n > 0]
        self._maybe_split_settlement(s)     # migration / daughter founding (§5.4)
```
This is the fix for the "band trends to extinction" problem from prior notes — **tune survival at the cohort/demographic layer, not by hand-tuning 7 individuals.**

### 3.3 `Building` (F4) — `backend/buildings.py`
```python
class Building:
    """A placed structure as an ENTITY (the blocks/roofs layer becomes its render+collision
    projection). Floors/capacity model verticality without a 3D engine: a skyscraper is one
    footprint with many floors + a height render cue."""
    def __init__(self, bid, kind, footprint, by):
        self.id = bid                  # "b_" + uuid
        self.kind = kind               # "house"/"shop"/"factory"/"townhall"/"tower"/...
        self.func = "R"                # R/C/I/civic
        self.footprint = footprint     # [(x,y), ...] tiles (also written into self.blocks)
        self.floors = 1                # >1 => taller render; capacity scales with floors
        self.cap_res = 0               # residents it can hold (cohort math)
        self.cap_job = 0               # jobs it provides
        self.services = {"road": False, "power": False, "water": False, "sewage": False}
        self.condition = 1.0
        self.by = by; self.built_t = None
    def capacity(self):                # land value + era + floors
        return dict(res=self.cap_res * self.floors, job=self.cap_job * self.floors)
```
Keep the existing **`site` dict** (`{id,name,ox,oy,by,done,tasks:[{x,y,code,done}]}`) as the *construction* representation; on completion, register a `Building` over the same footprint. The `DWELLING_LADDER`/`BLUEPRINTS` machinery already builds the tiles — you're adding an entity layer on top, not replacing it.

### 3.4 `RoadGraph` + `UtilityNetwork` (F3) — `backend/networks.py`
```python
class RoadGraph:
    """Roads as a sparse tile layer PLUS an adjacency graph. Edge types by era:
    dirt < gravel < cobble < paved < asphalt ; parallel types: rail, canal, sea, air.
    Same graph serves pathfinding (HPA*) AND carries utilities (CS2 pattern)."""
    def __init__(self):
        self.tiles = {}                # (x,y) -> road tier int (0=none)
        self.kind = {}                 # (x,y) -> "road"/"rail"/"canal"
        self.nodes = {}                # intersections/endpoints -> id
        self.adj = {}                  # node_id -> [(node_id, cost, kind)]
        self.dirty = True              # rebuild adj when tiles change
    def add(self, x, y, tier, kind="road"): ...
    def cost(self, a, b, agent):       # CS2: time + comfort + money + behavior
        ...

class UtilityNetwork:
    """Power / clean water / sewage as a flood-fill from sources along the road/pipe graph,
    with capacity vs demand. Generalizes the existing power_pole/POWER_RADIUS relay."""
    def __init__(self, kind, road: RoadGraph):
        self.kind = kind               # "power" | "water" | "sewage"
        self.road = road
        self.sources = []              # [{x,y,capacity}]
        self.served = {}               # (x,y) -> served? (rebuilt each _tick_networks)
        self.demand = 0; self.supply = 0
    def recompute(self, buildings):
        # BFS from each source along road tiles (utilities ride roads); decrement capacity
        # per served building; a building is served iff reached AND supply not exhausted.
        ...
```

---

## 4. Algorithms (real skeletons, not hand-waving)

### 4.1 Desire-paths → dirt roads (F3, builds on `footfall` + `_decay_footfall`)
The `footfall` dict (`(x,y)->wear`) already exists and decays daily. Harden the hottest tiles into dirt roads (the Physarum/desire-path result):
```python
ROAD_PROMOTE_WEAR = 40.0     # wear at which a trodden tile becomes a dirt road
ROAD_DEMOTE_WEAR  = 4.0      # a dirt road with no upkeep/footfall below this reverts

def _harden_roads(self):
    for (x, y), wear in list(self.footfall.items()):
        if wear >= ROAD_PROMOTE_WEAR and self.roads.tiles.get((x, y), 0) == 0:
            if self.water[y, x] in FORDABLE_WATER:        # don't pave deep water
                self.roads.add(x, y, tier=1)              # dirt
                self._note("road", f"a path wore into a road at ({x},{y})")
    # planned roads (Era 3+) are laid by the Planning Authority via citygen, not here
    self.roads.dirty = True
```
Planned roads (cobble/paved) come from `citygen` once a settlement has an institution + era ≥ Classical.

### 4.2 HPA\* over chunks (F3) — scalable pathfinding
The map already partitions into `CHUNK=64` cells. Build the abstract graph once per road-`version`:
```
1. For each CHUNK, find road tiles crossing its borders -> "portal" nodes.
2. Connect portals within a chunk by intra-chunk A* (cached).
3. Connect adjacent chunks' shared-border portals.
Query(start, goal):
   - insert start/goal into their chunks' portal sets (temporary edges),
   - A* on the abstract portal graph (tiny),
   - refine each abstract edge to tile path lazily (cache hot edges),
   - cost = ROAD_TIER_SPEED + comfort + toll (CS2 cost model).
Flow fields: for many-agents-one-destination (market/factory/station), compute one
   Dijkstra distance field over road tiles to that target; every agent just descends it.
```
Use HPA\* for any road travel and inter-settlement trips; keep the existing local grid movement for off-road foraging. (Research: HPA\* cuts large-grid path time >95% vs A\*.)

### 4.3 Utility flood-fill (F3) — plumbing & power, unified
```python
def _tick_networks(self):
    by_xy = {xy: b for b in self.buildings for xy in b.footprint}
    for s in self.settlements:
        for kind, net in s.util.items():
            net.served.clear(); net.supply = sum(src["capacity"] for src in net.sources)
            budget = net.supply
            for src in net.sources:                     # BFS along road tiles from each source
                for (x, y) in self._bfs_road(src["x"], src["y"], net.road):
                    b = by_xy.get((x, y))
                    if b and budget > 0:
                        b.services[kind] = True; net.served[(x, y)] = True; budget -= 1
            net.demand = sum(1 for b in s.buildings_in(s) if b.func in ("R","C","I"))
            # feedback: no clean water -> illness (reuse waterborne illness); no power -> no high-rise
```
**Growth-gate:** a zone/lot only upgrades when its needed services are present (next section). A cut/overloaded source browns out downstream — set `b.services[kind]=False` for anything past the exhausted budget (simple cascade).

### 4.4 RCI demand → building fill (F4, CS2/SimCity-style)
```python
def _fill_zones(self, s):
    for (x, y), z in s.zones.items():
        if (x, y) in self._occupied: continue
        if s.demand[z] <= 0: continue
        if not self._lot_serviced(s, x, y, need=ZONE_NEEDS[z]):  # road + water + (power for C/I)
            continue
        b = self._raise_building(s, x, y, func=z)     # drops a site; band/cohort builds it
        s.demand[z] -= 1
    self._upgrade_buildings(s)                         # land value + services -> add floors

def _upgrade_buildings(self, s):
    for b in s.buildings_in(s):
        lv = s.land_value.get((b.cx, b.cy), 0.0)
        if all(b.services.values()) and lv > FLOOR_THRESH[s.era] and b.floors < ERA_MAX_FLOORS[s.era]:
            b.floors += 1                              # this is how skyscrapers emerge (modern era)
```
`ERA_MAX_FLOORS` is 1 until the **Modern** era unlocks `steel_beam` construction, then rises — so skyscrapers can *only* appear once tech + land value + full services coincide.

### 4.5 Procedural city skeleton (`citygen.py`, Era 3+) — Parish & Müller, terrain-aware
```python
def lay_city(world, s):
    """Grow a designed layout FROM the emergent desire-paths, not a stamped template."""
    seeds = high_footfall_roads_near(world, s.cx, s.cy)        # start from where people walk
    # 1) arterials: extend roads following a tensor field = blend of
    #    (a) terrain gradient (follow contours, avoid steep/water; bridge at fords),
    #    (b) toward resources/water/existing town centers (global goals),
    #    with local constraints (snap near-parallel roads, close short loops).
    arterials = grow_streamlines(world, seeds, field=tensor_field(world, s))
    # 2) secondary streets fill between arterials until block size < TARGET_BLOCK.
    streets = subdivide_until(arterials, TARGET_BLOCK)
    # 3) blocks -> lots: recursive split with road frontage (Manor-Lords burgage rule).
    lots = parcel_blocks(streets, frontage=MIN_FRONTAGE)
    # 4) zoning: center=civic/C, ring=R, periphery/near-rail/near-resource=I.
    for lot in lots: s.zones[lot.anchor] = zone_for(lot, s)
    # 5) infra pass: lay water mains + sewers + power along the new roads; place civic anchors
    #    (square, town hall, market, water tower, later substation/sewage works).
    lay_utilities_along(world, s, streets)
```
Keep each step a pure function over tile fields + `s`, re-runnable as the city expands (call from `_tick_settlements` when `s.demand` outruns zoned lots).

### 4.6 Daughter-settlement site selection (§5 of roadmap)
```python
def _score_site(self, x, y):
    if self.water[y, x] != WATER_NONE: return -1
    near_water = 1.0 if self._adjacent_water(x, y) or self._dist_to_water(x, y) < 12 else 0.2
    soil = float(self.soil[y, x])
    buildable = self._flat_land_fraction(x, y, r=8)
    resources = self._resource_proximity(x, y)        # ore/wood/berry within range
    crowding  = -0.5 * self._settlement_proximity(x, y)   # don't found on top of a town
    return 1.4*near_water + 1.1*soil + 0.8*buildable + 0.7*resources + crowding

def _maybe_split_settlement(self, s):
    if s.population() < ERA_SPLIT_POP[s.era]: return
    if s.stores.get("food", 0) < s.population(): return      # only a SURPLUS town colonizes
    spot = self._best_scouted_site(s)                        # scored ring around s, beyond leash
    if spot and self._score_site(*spot) > SITE_MIN:
        self._found_settlement(*spot, parent=s, seed_pop=ERA_SEED_POP[s.era])
```

---

## 5. Tech-tree extension (`crafting.py`) — real rows in the existing format

Rows are `(out_id, qty, {inputs}, station, tool_cap, tier)`. The ladder today tops out at `reactor` (tier 7). Add eras as new rows + extend `STRUCTURE_KINDS`/`STATION_KINDS` + append to `TECH_LADDER`. Define an **`ERAS`** ordering (used by `Settlement.era`):

```python
ERAS = ("band","neolithic","bronze","iron","classical","medieval",
        "industrial","modern","contemporary")
```

**New stations/structures:**
```python
STATION_KINDS += ("granary","mill","brick_kiln","sawmill","blast_furnace",
                  "machine_shop","water_tower","pump","sewage_works","substation","trainyard")
STRUCTURE_KINDS += ("aqueduct","sewer","road","bridge","wall","tower","townhall","market_hall",
                    "factory","warehouse","rail","station","port","dock","ship","car","truck",
                    "tram","subway","airport","plane","skyscraper","powerplant","dam")
```

**Representative new recipe rows (extend `_RECIPE_ROWS`):**
```python
# ── K. Neolithic / Agriculture (tier 1-2) ──
("hoe",            1, {"stick":2, "stone":1},                None,        None, 1),
("plow",           1, {"plank":2, "iron_ingot":1},           "workbench", "saw", 2),
("seed_grain",     4, {"grain":1},                           None,        None, 1),
("flour",          2, {"grain":3},                           "mill",      None, 2),
("granary",        1, {"plank":12,"stone":6},                "workbench", "hammer", 2),  # structure
# ── L. Classical urbanism (tier 3-4) ──
("brick",          4, {"clay":3},                            "brick_kiln",None, 3),
("aqueduct",       1, {"brick":12,"stone":8},                "workbench", "chisel", 4),  # structure: water main
("sewer",          1, {"brick":8,"stone":6},                 "workbench", "chisel", 4),  # structure: sewage
("paved_road",     2, {"stone":4,"sand":2},                  None,        "hammer", 3),  # structure: road tier 3
("cart_axle",      1, {"plank":2,"iron_ingot":1},            "workbench", "saw", 3),
# ── M. Industrial (tier 5-6) ──
("steam_engine",   1, {"steel_beam":2,"copper_pipe":2,"gear":4}, "machine_shop","hammer",6),
("rail",           4, {"steel_beam":1,"plank":2},            "machine_shop","hammer",6),  # structure: rail edge
("locomotive",     1, {"steam_engine":1,"steel_beam":6,"gear":6}, "machine_shop","hammer",6),
("ship_hull",      1, {"steel_beam":8,"plank":12},           "sawmill",   "saw", 6),
("factory",        1, {"brick":20,"steel_beam":8,"glass":6}, "machine_shop","hammer",6),  # structure
# ── N. Modern (tier 7) ──
("concrete_beam",  1, {"concrete":2,"steel_beam":1},         "machine_shop","hammer",7),
("automobile",     1, {"steel_beam":2,"electric_motor":1,"glass":2,"rubber":4}, "factory","hammer",7),
("elevator",       1, {"electric_motor":2,"steel_beam":2,"gear":4}, "factory","hammer",7),
("skyscraper_core",1, {"concrete_beam":20,"steel_beam":20,"glass":30,"elevator":2}, "factory","hammer",7),  # structure
("transformer",    1, {"copper_coil":4,"steel_plate":2},     "factory",   "hammer",7),     # substation core
# ── O. Contemporary / continental (tier 8) ──  (bump max tier; ERAS handles gating)
("turbofan",       1, {"steel_beam":4,"electric_motor":2,"gear":6}, "factory","hammer",8),
("airliner",       1, {"turbofan":2,"steel_beam":12,"glass":8},     "factory","hammer",8),
("rail_highspeed", 4, {"steel_beam":2,"concrete":2,"copper_wire":1},"factory","hammer",8),  # structure
```
Append all new `out_id`s to **`TECH_LADDER`** in era order so the rule-body/mind climbs them when comfortable (the existing ladder pattern). Add any new raws (`rubber`, `copper_pipe`) to `RAW` with a source + tool. **The engine (`can_craft`/`do_craft`) needs no change** — it already gates on station-in-reach + tool-capability, which is why the smithy/factory `station` fields above "just work."

---

## 6. New god actions + endpoints (follow the exact existing pattern)

**`World` methods** (each bumps `self.version`, calls `self._note`, mirrors `add_water`/`place_power` style):
```python
def zone(self, x, y, r, kind, by="him"):           # paint R/C/I/civic into nearest settlement
def lay_road(self, x, y, r, kind="road", by="him"):# god-drawn road/rail/canal tiles
def found_settlement(self, x, y, name=None, by="him"):
def decree(self, sid, rule, by="him"):             # enact a law / set tax / queue a public work
```
**Dispatch rows in `_apply_world_action`** (`server.py:2041` chain) + **`_WORLD_ARGS`**:
```python
if tool == "zone":  w.zone(x, y, max(1,r), str(spec.get("kind","R")), by=by);  return f"{by} zoned {spec.get('kind')} at ({x},{y})"
if tool == "road":  w.lay_road(x, y, max(1,r), str(spec.get("kind","road")), by=by); return f"{by} laid {spec.get('kind')} at ({x},{y})"
if tool == "found": w.found_settlement(x, y, str(spec.get("name") or '').strip() or None, by=by); return f"{by} founded a settlement at ({x},{y})"
# _WORLD_ARGS additions:
"zone": ("x","y","r","kind"), "road": ("x","y","r","kind"), "found": ("x","y","name"),
```
**Endpoints** (clone the `/api/world/...` template — capability gate + `to_thread` + broadcast):
```python
@app.get("/api/world/settlement/{sid}")   # civic inspector: pop, era, treasury, demand, laws, works
@app.get("/api/world/view_civic")          # roads/buildings/zones/overlays clipped to a window
```

---

## 7. Render contract (`renderer/app.js` + `index.html`)

`app.js` already: loads `snapshot()`, streams `view()` on zoom, draws tile layers from base64 `Uint8Array`s, draws `blocks`/`roofs`/`people`/`paths`, runs the speed control + double-click inspector, and builds the god-tools menu. **Additions (incremental, one per milestone):**
- **Roads/rail:** draw `roads` tiles by tier (width/color), rail as ties; from `snapshot.roads` + `view_civic`.
- **Buildings & height:** draw `buildings` footprints; **skyscraper height cue** = darker drop-shadow + a floor-count tag (`"12▮"`), scaled by `building.floors`. No 3D needed.
- **Zones overlay:** translucent R/C/I/civic wash, toggle in god-tools.
- **Infra overlays:** power/water/sewage/traffic as heat-tinted overlays from packed `overlays` fields (decode like tile layers), CS2-style toggles.
- **Settlement labels:** name + population + era badge at each `settlement.centroid`; click → civic inspector (new endpoint).
- **God-tools menus (new `<section class="wmenu">` blocks in `index.html`):** *Zoning* (R/C/I/civic brush), *Roads & Transport* (road/rail/canal/bridge), *Public Works/Decrees* (per-settlement), *Settlements* (list + inspector).

The renderer must stay a **projection of sim state** (never a source of truth) — exactly as `blocks`/`roofs` already are.

---

## 8. Milestone M0, spec'd to near-code (the first concrete step)

**Goal:** introduce the `Settlement` object with *zero behavior change* — pure refactor so everything after has a civic home. Acceptance: one settlement wraps today's band; canary distributions unchanged; old save self-heals.

1. `backend/settlement.py`: add `Settlement` (§3.1) with `to_dict`/`from_dict`. No `Cohort` yet.
2. `world.py __init__`: `self.settlements = []`.
3. `generate()` (after `_seed_initial_people`, ~line 829): create the founding settlement around `_origin`:
   ```python
   s = settlement.Settlement("town_"+uuid.uuid4().hex[:8], self._name_town(),
                             *self._origin, self.clock)
   s.micro_ids = {p["id"] for p in self.people}
   s.cx, s.cy = self._people_centroid()
   self.settlements = [s]
   ```
4. **Migrate, don't duplicate:** point the existing communal `granary` at `s.stores` (or keep `granary` and have `s.stores` alias it for now); when a soul gets a `home_struct`, add it to the nearest `s.micro_ids` and recompute `s.cx,cy` from members.
5. `save()`/`load()`: serialize `self.settlements` via `to_dict`/`from_dict`. **Bump `SCHEMA` 7→8.** In `load()`, if a `SCHEMA 7` save has people but no settlements, *reconstruct* one settlement from the existing band (self-heal) instead of regenerating the whole world.
6. `snapshot()`: add `"settlements": [s.to_dict() for s in self.settlements]`.
7. `census()`: add `"settlements": len(self.settlements)`.
8. **Self-test** (extend the `if __name__ == "__main__":` block in `world.py`): generate a world, assert `len(w.settlements) == 1`, assert `w.settlements[0].population() == len(w.people)`, save+load, assert it round-trips and the settlement survived.
9. **Canary:** run the headless sim 3× for N game-days vs a stashed baseline; assert survival/population distributions unchanged (M0 must be behavior-neutral).

Then proceed to **M1 (Cohorts/LOD)** → **M2 (road graph)** → … exactly as the roadmap's §6 milestone list, each with its acceptance test.

---

## 9. Testing & safety discipline (do this every milestone)

- **Headless self-tests:** `python world.py`, `python mind.py`, and new `python settlement.py`/`networks.py` run assert-based tests on import — extend them per milestone (the repo already does this; e.g. the berry/hunt/bottle tests around `world.py:6692+`).
- **Canary = 3-run distributions vs a stashed baseline.** This sim's determinism is sensitive; never judge by a single run. A change is safe iff survival/population/era-progress distributions don't regress.
- **Isolated runs:** second backend, **isolated `USERPROFILE`** (not bash `HOME`), spare port; **run solo** — parallel canaries share `~/.ai4me/world.*` and corrupt each other.
- **Perf budget per system:** assert `_tick_cohorts`, `_tick_networks`, `_harden_roads`, HPA\* queries each stay under a fixed ms budget at target population; vehicles/logistics are **flows, not entities**.
- **Renderer ≠ API:** verify the World tab in the Electron front-end (the project run flow / preview tools), not just curl — `app.js` isn't exercised by endpoint tests.
- **SCHEMA discipline:** any new persisted field → bump `SCHEMA`, test clean-gen AND old-save self-heal.
- **Offline guarantee:** before merging any era, confirm everything runs with **no model present** (LLM is enrichment for notables only; `mind.py`'s `heuristic_goal` fallback is the pattern).
- **Fast-forward to verify emergence:** new systems "fire 0" in a short canary because they need a mature band/city — use the speed control / longer headless runs to confirm they actually fire.

---

### One-paragraph handoff
Add five pure-Python modules (`settlement`, `networks`, `buildings`, `citygen`, `production`), wire their `_tick_*` stages into `World.step` at the cadences in §1.1, and serialize their state through `snapshot`/`tick_state`/`view` and the cloned god-action/endpoint patterns in §6. Build the four foundations as milestones **M0–M4** (Settlement object → Cohort LOD → road graph + HPA\* → utility flood-fill + zoning/buildings), each behavior-verified by 3-run canary distributions and headless self-tests, bumping `SCHEMA` whenever you persist new fields. Then walk the era ladder by appending the §5 recipe rows and turning on `citygen` from the Classical era, always shipping rule-based emergence first (offline) and letting the LLM enrich only the notable few — the same body/mind philosophy this sim already lives by, now scaled from a band to a continent.
