# The World → Civilization Roadmap
### A foolproof, handoff-ready plan to evolve the Ai4Me World sim from a caveman band into a modern, multi-settlement civilization with designed cities, infrastructure, and continental transport

> **Who this is for:** a fresh Claude Code chat working in the `Ai4Me` repo (`E:\Downloads\Ai4Me`). Read this whole document first, then read the files it points at, then work the **Milestones** in order. Every milestone is independently shippable and verifiable. Do not skip the **Working Discipline** section — this codebase has hard-won rules that will bite you if ignored.

---

## 0. Orientation — read this before touching anything

### 0.1 What the World already is (ground truth, from the code)
The World is a **top-down god-sim** living in its own tab. The architecture is a deliberate **body/mind split** (Project Sid / PIANO-style):

- **`backend/world.py`** (~6.8k lines) — the **body / data layer**. Pure Python + numpy, *no LLM*, deterministic. Owns:
  - A **2048×2048 tile grid** (`W = H = 2048`) of numpy fields: `elevation`, `biome` (13 biomes), `soil`, `moisture`, `water` (river/lake/ocean/shallow), `veg_sp`, `veg_growth`.
  - A **clock** (`GAME_SEC_PER_REAL_SEC = 8.0`, 1 game-day ≈ 3 real hours), **seasons** (`DAYS_PER_SEASON = 15`, 4 seasons), and **weather**.
  - **Cost model that makes scale possible:** ecology only runs inside an **active region** around the people (`ACTIVE_MAX = 768` tiles/side, `CHUNK = 64`); dormant chunks **fast-forward growth on revisit** (`_chunk_eco`, `ECO_CATCHUP_CAP`). The whole-world snapshot is downsampled to `OVERVIEW_MAX = 256`/side. **Never write a per-tick whole-grid loop.**
  - **People** (`self.people`): per-soul dicts ticked every beat in pure Python (`_tick_people`) — needs, perception (`p["seen"]`, no omniscience), obstacle-aware movement + river fording, foraging/hunting/fishing, illness, exposure, predation, building, death, birth/lineage.
  - **Things people build:** `self.blocks` (sparse `(x,y)->block code`: floor/wall/door/window/fence/leaf), `self.roofs` (set), `self.sites` (buildings under construction, per-tile tasks), `self.structures`, `self.station_objs` (workbench/furnace/…), `self.granary` (shared store), `self.decor`, and **`self.footfall`** (`(x,y)->wear`, where feet fall most, worn into cosmetic **paths** — this is the embryo of roads).
  - **Buildings are data:** `BLUEPRINTS` are glyph grids (`"WDW"/"WFW"/"WWW"`), built tile-by-tile. `DWELLING_LADDER = ["leaf_shelter","hut","cabin"]`. Communal builds already exist: gathering hall, workshop, storehouse, **smithy**, **well**, inn, watchtower, **market**.
  - **Power grid (modern-era seed):** `POWER_SOURCES = ("generator","reactor")`, `power_pole` relays, `POWER_RADIUS = 7`, electrified-home shelter bonus, reactor meltdown stakes, and **awe/reverse-engineering** (curious souls study a god-placed generator and learn `copper_coil`).
  - **God-action API** (`world_action` / the `_act_*` dispatch around line 1975+): the *same* calls the UI brush tools make AND Aitha's `<world>/<sculpt>/<spawn>` directives route into. Actions: `eat/drink/rest/forage_berry/gather/hunt/fish/chop/mine/craft/found_site/build_block/...`.
  - **Persistence:** `~/.ai4me/world.npz` (tile arrays) + `~/.ai4me/world.json` (clock/entities/log). **`SCHEMA = 7`** — bump it whenever the save layout changes incompatibly; old saves self-heal by regenerating.

- **`backend/mind.py`** (~1.3k lines) — the **mind / inner life**, layered on the body. Also runs **offline with no model**:
  - **Memory stream** (Stanford generative-agents recipe: recency × importance × relevance, token-overlap instead of embeddings).
  - **Relationships** (per-person trust/sentiment), **reputation via gossip**, **emergent barter economy** (bilateral, trust-gated, no global prices), **goals**.
  - **LLM is strictly enrichment** (reflection, naturalistic goals, speech) and **every touch-point degrades to a rule-based fallback** (`heuristic_goal`). **The body never waits on the mind.**

- **`backend/crafting.py`** — the **tech tree as data**. `RAW`, `ITEMS`, **128 `RECIPES`** (tiers 0–7), `TOOL_PROVIDES` (capabilities), `STATION_KINDS`, `STRUCTURE_KINDS`, `TECH_LADDER`. **Current ceiling = the nuclear age:** `concrete`, `steel_beam`, `turbine`, `uranium_fuel`, `reactor`. There is already a `cart` (tier 1) and `concrete`/`steel_beam` — primitives the later eras reuse.

- **`backend/projects.py`** — self-authored "life projects" (open-vocabulary goals a comfortable soul pursues).
- **`backend/server.py`** — FastAPI; world endpoints: `GET /api/world`, `/api/world/view`, `/api/world/person/{id}`, `POST /api/world/action`, `/api/world/templates*`, `/api/world/reset`, `/api/world/speed`, `/api/world/ledger`, `/api/world/recipes`.
- **`renderer/{index.html,app.js,style.css}`** — the World tab UI: canvas, HUD, speed control, double-click inspector, and the **God-tools panel** (Terrain / Water / Flora / Wildlife / Power / People / Crafting / Ledger / Templates / Settings). The World is **off by default** (Settings → Behavior → Capabilities → "The World").

### 0.2 The vision (your target end-state)
A world that **naturally evolves** from a stone-age band to a modern civilization:
- **Multiple settlements** founded organically and growing into **intelligently designed cities** — roads, **plumbing (water + sewage)**, **electricity**, zoning/districts, eventually **steel-frame skyscrapers**.
- **Production & supply chains** feeding urban growth.
- **Governance/institutions** that plan public works.
- **Transportation networks between settlements** — paved roads + **cars**, **railways + trains**, canals/**ports + ships**, and finally **air travel** enabling **continental travel**.
- All of it **emergent and rule-driven first** (works offline), with the **LLM as enrichment** for the notable few — exactly the existing design philosophy, scaled up.

### 0.3 The honest gap analysis — why this is a big lift
Five hard truths the plan is built around:

1. **No settlement as a first-class entity.** Today there is `_origin`, per-soul homes, a granary, and communal buildings raised on individual need. A *city* needs a civic object that owns population, territory, zoning, a treasury, infrastructure, and a planning authority. **This is prerequisite #1.**
2. **No agent scaling (LOD).** All 7 people are fully ticked. A city of thousands cannot be N full agents (Project Sid hit compute walls past ~1000 even on a server; Cities: Skylines 1 capped at 65k; Transport Fever uses a **1 agent : 4 people** abstraction). You need **two-tier simulation: micro agents (the notable + on-camera few) over macro cohorts (everyone else as statistics).** **Prerequisite #2.**
3. **The world is 2D top-down; "skyscraper" has no meaning yet.** Blocks are a flat `(x,y)->code` layer. Verticality must be modeled as **building entities with floors/capacity + a height render cue**, not literal stacked tiles. **Design decision baked into the building refactor.**
4. **No network substrate.** Roads, power, water, sewage, transit all want a **graph**. The winning pattern (Cities: Skylines II) is **roads carry utilities** — one graph, many uses (pathfinding + utility flood-fill). The existing `footfall` paths and `power_pole` relay are the seeds. **Prerequisite #3.**
5. **Tech ceiling stops at the reactor.** The ladder must extend through **agriculture → classical urbanism → medieval → industrial → modern → contemporary**, and crucially each tier must **unlock infrastructure & city systems, not just items.**

### 0.4 What we learned from comparable projects (steal these ideas)
- **Project Sid / PIANO (Altera)** — 10–1000+ agents form **specialized roles, follow & amend collective rules, and transmit culture/religion**; the **Cognitive Controller** keeps concurrent modules coherent; **social density drives cultural spread** (towns generate more "memes" than rural areas); 500-agent runs worked via **geographically distributed towns + migration**, *not* synchronized global interaction. → Your mind layer already mirrors PIANO; the lesson is **specialization, governance, and culture are emergent if you give agents roles + a shared rule object + dense settlements.**
- **Stanford Generative Agents** — memory/reflection/planning recipe (already implemented in `mind.py`).
- **Procedural city generation (Parish & Müller 2001 "CityEngine"; tensor-field streets, Chen et al. 2008; L-systems; agent-growth, Lechner 2003)** — cities = **road network first, then parcels/lots between roads, then buildings**. Use **global goals + local constraints**; trace roads along terrain/tensor fields; subdivide blocks into lots.
- **Slime-mold / desire-path networks (Physarum)** — **near-optimal transport networks emerge** from agents reinforcing used paths and decaying unused ones. Your `footfall` layer is literally this. → **Harden high-footfall paths into roads automatically.**
- **Manor Lords** — **organic growth along roads** via flexible **burgage plots** (one plot = one family = workers + a home industry). → Great model for the medieval/organic-growth era and for the **plot = population+production unit** abstraction.
- **Anno 1800 / Workers & Resources** — **production chains** (raw → intermediate → finished) + **logistics** (warehouses, an "invisible" transport layer). → Backbone of the industrial economy.
- **Cities: Skylines II** — **no hard agent cap (hardware-bound)**; **pathfinding cost = time + comfort + money + behavior**; **utilities flow through roads**; **utilities are growth-gates** (zones stall when capacity < demand). → Adopt the cost-based pathfinding and the **capacity/demand growth-gate** loop.
- **Macro/micro ABM scaling** — replace identical individuals with **aggregated cohorts/continuum models**; proven 10×+ speedups. → Your LOD layer.
- **HPA\* / hierarchical pathfinding + flow fields** — partition the grid into clusters; path on the abstract graph then refine. >95% time reduction vs A\* on large grids; flow fields share one path among many agents. → Required once roads + vehicles exist on a 2048² map.

---

## 1. Architecture-first: the four foundations (build these before any "era")

These are load-bearing. Eras 1–7 assume they exist. Build them as their own milestones (see §3), keep them **deterministic, offline, vectorized, and canary-safe.**

### F1 — Settlement & Civic layer (`backend/settlement.py`, new)
A first-class **`Settlement`** object the world owns a list of:
```
id, name, founded_clock, centroid(x,y), territory (chunk set / radius),
population_register (micro ids + macro cohorts), households,
treasury (coin), stores (aggregate granary), zones (see F3),
infrastructure graphs (road/power/water/sewage — see F3),
institutions (leader, laws[], tax_rate, public_works_queue),
tech_level (era index), demand/supply ledger.
```
- **Founding:** a settlement is born when a cluster of homes + a communal building + a population threshold coheres around a locus (promote today's `_origin`/home-cluster logic). **Daughter settlements** form when population pressure + a scouted good site (water + soil + resources + buildable land) exceed thresholds (site-selection scorer over the tile fields).
- **The Planning Authority** lives here: once a settlement has a **leader/institution** (built on the existing **renown** system — highest-renown soul becomes headman → council → mayor by era), it can **commission public works** (roads, wells→aqueducts, walls, later sewers/power) that individuals wouldn't build alone. This is the bridge from "souls build huts" to "a city is *designed*."

### F2 — Population LOD: two-tier simulation
- **Micro agents** — full body (`world.py`) + mind (`mind.py`). Keep for: everyone inside the **active region**, all **notables** (leaders, named specialists, anyone the camera/inspector is on, anyone in an LLM relationship). Target budget: ~**150–400 micro agents** max at once.
- **Macro cohorts** — households/districts represented as **statistics** (counts by age/occupation, aggregate needs, aggregate production/consumption, birth/death/migration rates). Ticked cheaply per game-hour with vectorized math. A cohort can **promote a representative to micro** when it enters the active region or produces a notable (Sid-style migration without global sync).
- **Agent scale knob** (`PEOPLE_PER_AGENT`, like Transport Fever's 1:4) so a rendered "person" can stand for several. Macro→micro and micro→macro transitions must be **lossless enough** (carry forward inventory/needs summaries) and **deterministic**.
- **Acceptance:** a settlement of 2,000 "people" runs at the same tick cost as today's band, because only the micro set is fully simulated.

### F3 — The Network Substrate: roads as the universal graph (`backend/networks.py`, new)
One graph to rule them all (the Cities: Skylines II pattern):
- **Roads** are real entities (a sparse layer like `blocks`, plus an adjacency **graph** of road segments/nodes). Tiers by era: **dirt path → gravel → cobble → paved → asphalt highway**; plus **rail** and **canal** as parallel edge types.
- **Auto-emergence (early eras):** promote high-`footfall` tiles into **dirt paths** automatically (slime-mold/desire-path rule: reinforce on use, decay unused). Planned roads (later eras) are laid by the Planning Authority (F1) using procedural layout (§2, Era 3+).
- **Utilities ride the graph:** **power, clean water, and sewage** are **flood-fill propagations along the road/pipe graph** from sources (power plant, water tower/pump, treatment) to connected buildings, with **capacity vs demand**. Generalize the existing `power_pole`/`POWER_RADIUS` relay into this. **Buildings connect to the nearest road** to receive services.
- **Capacity/demand growth-gate:** a zone/plot only upgrades when its needed services (jobs access, water, power, sewage) are met — the core loop that makes cities grow *intelligently* instead of sprawling randomly.
- **Pathfinding at scale:** implement **HPA\*** over the road graph (cluster by `CHUNK`), with **flow fields** for many-agents-same-destination (markets, factories, stations). Cost = **time + comfort + money** (CS2). This replaces per-agent A\* on the raw grid for anything traveling roads.

### F4 — Buildings as entities + verticality (`backend/buildings.py`, refactor over `blocks`/`sites`)
- Promote a building from "a set of placed tiles" to a **`Building` entity**: footprint, blueprint id, **floors/levels**, **capacity** (residents, jobs), **function** (residential/commercial/industrial/civic), service connections (road/power/water/sewage), upkeep, condition. Keep the tile `blocks`/`roofs` layer as the **render + collision** projection of the entity.
- **Verticality without a 3D engine:** a skyscraper is **one footprint with N floors and a tall render cue** (height shading / drop shadow / a "12▮" floor count), housing/employing many via the LOD cohort math. This is how 2D/2.5D city-builders fake height. (Optionally later: an isometric or layered render, but **do not block the roadmap on a renderer rewrite**.)
- **Zoning:** the Planning Authority marks tiles/parcels as **R/C/I/civic** zones; macro demand (RCI-style) fills zones with buildings over time, gated by services (F3). Organic early growth uses **Manor-Lords-style plots along roads**; planned later growth uses **procedural blocks** (§2).

---

## 2. The procedural city engine (used from Era 3 onward)

When a settlement gains a Planning Authority and the right era, switch from purely organic accretion to **designed layout**:

1. **Road skeleton** — generate arterials with a **tensor-field / L-system hybrid** seeded by terrain (follow contours, avoid water/steep, bridge rivers at fords), then **secondary streets** filling between arterials (Parish & Müller "global goals + local constraints"). Snap to existing emergent desire-path roads so the design *grows from* where people already walk.
2. **Block subdivision** — the road loops define **city blocks**; subdivide each block into **lots/parcels** (recursive split, Manor-Lords burgage-style frontage rules: each lot fronts a road).
3. **Zoning assignment** — assign lots R/C/I/civic by distance-to-center, services, and demand (center = dense commercial/civic; ring = residential; periphery/near-resources/near-rail = industrial).
4. **Infrastructure pass** — lay water mains + sewers + power lines **under/along the roads** (F3); place civic anchors (square, market, town hall, later: water tower, power substation, sewage works, fire/clinic).
5. **Fill over time** — macro demand raises buildings on lots, **upgrading floors** as the era + services + land value rise (low huts → townhouses → mid-rise → **skyscrapers** in high-value, fully-serviced, modern-era core).

Keep each step a pure function over tile fields + the settlement object so it's testable headless and re-runnable as a city expands.

---

## 3. The era ladder — what unlocks, what to build, where in the code

Each era is gated by **tech + population + institutions** (Civ-style, but emergent). Extend `crafting.py` `RECIPES`/`TECH_LADDER` with the new tiers and wire unlocks into the settlement `tech_level`. **Bump `SCHEMA` whenever you add persisted fields.**

| Era | Trigger (emergent) | Headline unlocks | Key new systems | Primary files |
|---|---|---|---|---|
| **0. Band (now)** | — | foraging, huts, smithy, money invention | (exists) | — |
| **1. Neolithic / Agriculture** | food surplus + settled cluster | **farming** (hoe/plow, fields, crops, irrigation ditches), **animal domestication** (pasture, livestock), pottery/storage at scale, **permanent settlement** | F1 settlement births; surplus → **population boom** → LOD (F2) kicks in; emergent **dirt roads** from footfall (F3) | `world.py` (farm tiles/seasons), `crafting.py`, `settlement.py` |
| **2. Bronze / Iron** | smithy + ore trade | metallurgy (already have smithy/forge), **division of labor** (occupations register), walls/fortification, **wells → first water works** | occupations in F1 register; specialization (Sid roles); first **planned core** around the hall | `crafting.py`, `settlement.py`, `mind.py` (roles) |
| **3. Classical / Antiquity** | leader/council + treasury | **urban grid planning**, **aqueducts + plumbing**, **paved roads**, **sewers**, marketplaces+currency (have money), monuments, **written laws** | **Procedural city engine (§2)** turns on; **utility networks (F3)**; governance/laws on the **renown** system | `networks.py`, `settlement.py`, procedural-layout module |
| **4. Medieval** | trade between settlements | **districts/zoning**, **guilds**, **burgage-plot organic growth**, markets, inter-settlement **trade routes** (carts on roads), castles | multi-settlement world; **trade-route logistics**; RCI-lite demand | `settlement.py`, `networks.py`, economy in `mind.py` |
| **5. Industrial** | coal + capital + dense city | **factories + production/supply chains** (Anno/W&R), **steam power**, **railways + trains between cities**, **canals/ports**, mass urbanization, **pollution** | supply chains; **rail edges** in the transport graph; logistics fleets (cohort flows); pollution/health feedback | `crafting.py` (machines), `networks.py` (rail), `production.py` (new) |
| **6. Modern** | electrification + steel + autos | **electric grid** (extend existing power), **automobiles + road networks**, **steel-frame skyscrapers**, **water/sewage utility grid citywide**, telecom, mass transit (tram/bus/subway) | vehicles as agents/flows on road graph (HPA\*/flow fields); **zoning→high-rise** via land value + services; utility growth-gates | `networks.py`, `buildings.py` (floors→skyscrapers), `production.py` |
| **7. Contemporary / Continental** | multiple mature cities | **highways**, **high-speed rail**, **airports + air travel**, **shipping ports**, globalized economy, **continental travel** | inter-continent transport graph over the existing **oceans/continents** (map already has them); migration & trade between continents; airline/shipping flows | `networks.py`, `settlement.py`, world map overview |

**Design rule for every era:** ship the **rule-based emergence first** (works offline), then let the **LLM enrich** the notable few (leaders naming a law, an inventor describing a machine, a mayor justifying a public work). Never gate world progress on a model being present.

---

## 4. Cross-cutting subsystem specs (the deep ends)

### 4.1 Economy, money & markets
You already have **money invention** (`money_invented`, a soul mints the first coin) and barter in `mind.py`. Extend: **local prices** from supply/demand per market, **wages/employment** (jobs in F4 buildings pay coin), **a treasury + taxation** (F1 institutions), **banks/credit** (later), and **inter-settlement trade** (price arbitrage drives caravans/trains). Keep it **bilateral & emergent** — no global price oracle; markets aggregate local trades.

### 4.2 Production & supply chains (`backend/production.py`, new, Era 5+)
Model chains as data (Anno/W&R): `recipe@building: inputs[] -> outputs[] @rate, needs[workers,power]`. Logistics move goods along the **road/rail graph** (F3) via **aggregate flows** (don't simulate every crate; simulate throughput with occasional rendered vehicles for life). Warehouses/markets buffer. A factory pulls inputs within its catchment; shortfall throttles output (growth-gate again).

### 4.3 Governance & institutions
Build on **renown + gossip**. Progression: **headman** (highest renown) → **council** (top-N) → **elected mayor** (era-gated) → multi-settlement **polity**. Institutions hold a **treasury** (taxes), pass **laws** (norms with enforcement — Sid showed agents follow & amend collective rules; the existing "first law = granary contribution" idea generalizes), and run the **public-works queue** (what makes cities *designed*). Laws/decisions are rule-chosen with **LLM authoring the wording/justification** as enrichment.

### 4.4 Demographics & migration
Macro cohorts carry **age structure, fertility, mortality, occupation**. **Migration** flows toward opportunity (jobs, food, safety, low rent) and away from collapse (Sid's distributed-towns + migration model). Migration is also how **micro↔macro** and **settlement↔settlement** populations move. Tune so population is **self-sustaining per era** (the current band trends to extinction — fix this at the cohort level with era-appropriate birth/death rates, *not* by hand-tuning 7 individuals).

### 4.5 Transportation network (the user's headline want)
- **Substrate:** the road/rail/canal/sea/air **graph** in F3.
- **Eras:** foot/desire-paths (0–2) → carts on roads + ports (3–4) → **canals + railways + steamships** (5) → **cars on paved road networks + trams/subway** (6) → **highways + high-speed rail + airports + container shipping** (7, **continental travel**).
- **Movement:** intra-city agents use **HPA\* + flow fields** with **cost = time + comfort + money** (CS2). Inter-city/continental movement is **aggregate flow** (passengers/goods per route per tick) with a few rendered vehicles for flavor. **Vehicles are mostly statistics, selectively embodied.**
- **Network effects:** travel time between settlements drives **trade, migration, and growth** — a rail link makes two towns one labor market.

### 4.6 Infrastructure / utilities (plumbing + electricity)
Generalize the existing power grid into the **F3 utility flood-fill**: sources (well/water-tower/pump → mains; campfire/furnace → none; generator/reactor/power-plant → grid; outhouse/sewer → treatment) push service **along the road/pipe graph** to connected buildings, with **capacity vs demand** and **failure propagation** (a cut main or overloaded substation browns out downstream — ORNL-style cascade, kept simple). Health/comfort/land-value feed back from service coverage (no clean water → disease, building on the existing **waterborne illness**; no power → no high-rise).

### 4.7 The mind at civilizational scale
- Keep **`mind.py` per-micro-agent**. Macro cohorts get a **"folk psychology" summary** (aggregate mood/trust/culture), not per-soul minds.
- **Culture/memes/religion** spread between settlements via **migration + trade contact + gossip** (Sid: density-driven). Represent culture as **traits/memes on settlements** that bias behavior and diffuse.
- **LLM budget:** only **notables** get reflection/speech/goal-authoring. A `WORLD_MIND_MODEL` already exists separate from chat — respect that knob and the offline fallback everywhere.

### 4.8 Rendering & UX (don't let it block the sim)
- **Multi-zoom:** keep the `OVERVIEW_MAX = 256` downsample for the world map; add a **mid (settlement)** and **close (street)** zoom that draw roads, zones, building footprints/heights, and **infra overlays** (toggle power/water/sewage/traffic like CS2).
- **God tools:** add menus for **Zoning**, **Roads/Transport**, **Public Works/Decrees**, and a **Settlement inspector** (population, era, treasury, demand). The god can nudge; the civilization should run without nudging.
- **Verticality cue:** height shading + floor count is enough for skyscrapers in 2.5D.

---

## 5. Working discipline — the rules this codebase enforces (do not violate)

1. **Body = deterministic, offline, vectorized.** No LLM calls, no blocking I/O, no per-tick whole-grid loops in `world.py`. Heavy work stays inside the **active region** and uses numpy array ops.
2. **Mind = enrichment only.** Every LLM touch-point must have a **rule-based fallback**; the body never waits on the mind. Reuse the `heuristic_goal` pattern.
3. **Respect the cost model.** New per-entity work scales with entity count, not tiles. New per-area work runs on the **slow ecology cadence** and only in active chunks. Add a **performance budget** assertion per new system (e.g. ≤ X ms/tick at N agents).
4. **Canary testing is how you judge changes.** This sim's canary determinism is sensitive — **judge by 3-run distributions against a stashed baseline**, not a single run. A change is "safe" if survival/population/era-progress distributions don't regress. (`python world.py` / `python mind.py` run headless self-tests — extend them.)
5. **Saves & schema.** Any new persisted field → **bump `SCHEMA`** and handle load of older saves (regenerate or migrate). Test: delete `~/.ai4me/world.*` and confirm a clean world generates; load an old save and confirm self-heal.
6. **Isolated runs.** Verify on a second backend with an **isolated `USERPROFILE`** (not bash `HOME`) on a spare port; **parallel canaries share the save path and corrupt each other — run solo.**
7. **Frequent triggers.** New behaviors need a **frequent enough trigger to actually fire** in a canary window, or you'll think they're broken. Many systems "fire 0 in a short canary" because they need a mature band — **fast-forward** to verify (speed control / longer headless run).
8. **Renderer ≠ API.** API/curl tests don't exercise `app.js`. Verify UI via the Electron front-end (the project's run flow / preview tools), not just endpoints.
9. **Build on what exists.** Promote, don't replace: `footfall`→roads, `power_pole`→utility graph, communal-build-on-need→public-works queue, renown→governance, `_origin`/home-cluster→settlement, barter→markets. Reuse `BLUEPRINTS`/`sites` for new buildings.

---

## 6. Milestones — the critical path (each is shippable + verifiable)

Work top-to-bottom. **M0–M4 are the foundations; do not start the eras until they pass canary.** Each milestone lists an **acceptance test**.

> Legend: 🧱 foundation · 🌍 era · 🔬 acceptance test

- **M0 🧱 Settlement object (F1).** Introduce `Settlement`, born from the existing home-cluster/`_origin`. Migrate population/granary/communal-builds to belong to it. *No behavior change yet.* 🔬 *A loaded world shows exactly one settlement wrapping today's band; canary distributions unchanged; old save self-heals after `SCHEMA` bump.*
- **M1 🧱 Population LOD (F2).** Add macro cohorts + micro/macro promotion + `PEOPLE_PER_AGENT`. 🔬 *Seed a settlement of 1,000; tick cost ≈ today's band; micro set stays ≤ budget; a cohort promotes to a micro agent when entering the active region and demotes on leaving, conserving aggregate needs.*
- **M2 🧱 Road graph + emergent paths (F3 part 1).** Roads as entities + adjacency graph; auto-harden high-`footfall` tiles into dirt roads (decay unused). 🔬 *After a fast-forward, well-trodden routes between home/water/grove become persistent dirt roads; isolated wear fades.*
- **M3 🧱 HPA\* pathfinding (F3 part 2).** Hierarchical pathfinding over the road graph + flow fields for shared destinations; route everyday travel through it. 🔬 *Path computation on a 2048² map with roads is >5× faster than the current grid A\* for cross-settlement trips; agents prefer roads (lower cost).*
- **M4 🧱 Building entities + zoning + utility flood-fill (F4 + F3 part 3).** `Building` with floors/capacity/function; RCI-style zoning; generalize power into road-carried **power/water/sewage** with capacity/demand growth-gates. 🔬 *A zoned area fills only when serviced; cutting a water main browns out downstream buildings; a building gains floors when land value + services rise.*
- **M5 🌍 Era 1 — Agriculture.** Farming/irrigation/domestication; surplus → cohort population boom; settlement founding rule. 🔬 *A surplus band founds a daughter settlement at a scored good site; population grows via cohorts instead of trending to extinction.*
- **M6 🌍 Era 2–3 — Specialization + Classical urbanism.** Occupations register; governance (headman→council) on renown; **turn on the procedural city engine (§2)** for planned cores; aqueduct/sewer/paved roads. 🔬 *A mature settlement lays a road skeleton, subdivides blocks, zones them, and runs water/sewers under roads; a leader commissions a public work no individual would build.*
- **M7 🌍 Era 4 — Medieval + multi-settlement economy.** Districts/guilds; **inter-settlement trade routes** (carts on roads); price-driven caravans; migration between towns. 🔬 *Two settlements with complementary surpluses establish a recurring trade flow; a rail-less travel-time change shifts migration/growth.*
- **M8 🌍 Era 5 — Industrial.** Production/supply chains (`production.py`); factories; coal/steam; **railways + trains** between cities; ports/canals; pollution. 🔬 *A multi-stage supply chain delivers a finished good to market via rail; a rail link merges two towns' labor markets (measurable growth).*
- **M9 🌍 Era 6 — Modern.** Citywide electric grid; **automobiles on road networks**; **steel-frame skyscrapers** (floors scale with land value + full services); mass transit. 🔬 *A fully-serviced, high-land-value core grows high-rises; car traffic routes via HPA\*/flow fields with congestion; brownouts cap high-rise growth.*
- **M10 🌍 Era 7 — Contemporary / Continental.** Highways, high-speed rail, **airports + air travel**, container shipping; **continental travel** over the map's existing oceans/continents; globalized economy. 🔬 *Passengers/goods flow between settlements on **different continents** via air/sea routes; a new continent gets colonized by migration and joins the trade network.*
- **M11 ✨ Mind-at-scale polish.** Notable-only LLM enrichment for leaders/inventors/mayors; culture/meme/religion diffusion between cities. 🔬 *With no model present, everything above still runs (offline); with a model, notables author laws/inventions/speeches and a cultural trait visibly diffuses along trade/migration links.*

**Parallelization:** M2/M3 (networks) can proceed alongside M1 (LOD) after M0. Everything from M5 on depends on M0–M4. Rendering/UX (4.8) and god-tools can be grown incrementally per milestone.

---

## 7. Top risks & how to defuse them

- **Performance blow-up at scale** → enforce LOD (F2) + active-region discipline + per-system tick budgets; profile every milestone; vehicles/logistics as **flows, not entities**.
- **Determinism/canary drift** → keep all randomness on seeded RNG; judge by **3-run distributions vs baseline**; extend headless self-tests per milestone.
- **Save/schema breakage** → bump `SCHEMA`, test old-save self-heal, test clean-gen, every milestone.
- **Over-scripting (kills emergence)** → prefer **rules + thresholds + agent incentives** over hard-coded city plans; the procedural engine should *grow from* emergent desire-paths and demand, not stamp a fixed template.
- **Renderer becomes the bottleneck** → keep 2.5D height cues; never block sim milestones on a 3D rewrite.
- **LLM-dependence creep** → audit that every new behavior has an offline rule path before merging.
- **Population not self-sustaining** → fix at the **cohort/demographic** layer per era, not by tuning individuals.

---

## 8. Research bibliography (what to read & what to take)

**Agent civilizations / minds**
- Project Sid: Many-agent simulations toward AI civilization — https://arxiv.org/abs/2411.00114 · GitHub: https://github.com/altera-al/project-sid · PIANO review: https://andlukyane.com/blog/paper-review-piano — *roles, governance, culture/religion transmission, distributed-towns + migration scaling, Cognitive Controller for coherence.*
- Stanford Generative Agents — (already implemented in `mind.py`) — *memory/reflection/planning.*

**Procedural cities & roads**
- Parish & Müller, "Procedural Modeling of Cities" / CityEngine (2001); tensor-field streets (Chen et al. 2008); agent-growth (Lechner 2003) — overview & slides: https://phiresky.github.io/procedural-cities/presentation.html · paper: https://github.com/phiresky/procedural-cities/blob/master/paper.md — *roads-first, global goals + local constraints, block→lot subdivision.*
- Slime-mold / Physarum transport networks (desire paths): https://www.crl.epi.dendai.ac.jp/projects/robot-coop/3528/ · https://cargocollective.com/sagejenson/physarum — *emergent near-optimal networks from reinforce-on-use / decay-unused (your `footfall`).*

**City-builder design patterns**
- Manor Lords burgage plots / organic growth: https://wiki.hoodedhorse.com/Manor_Lords/Burgage_plot — *plot = family = workers + home industry; growth along roads.*
- Anno 1800 production chains: https://anno1800.fandom.com/wiki/Production_chains — *raw→intermediate→finished + invisible logistics.*
- Cities: Skylines II — Traffic AI: https://www.paradoxinteractive.com/games/cities-skylines-ii/features/traffic-ai · Electricity & Water (utilities through roads): https://www.paradoxinteractive.com/games/cities-skylines-ii/features/electricity-water · no agent cap: https://80.lv/articles/cities-skylines-2-doesn-t-have-limit-for-people-it-can-track — *cost-based pathfinding (time+comfort+money+behavior); utilities ride roads; capacity/demand growth-gates.*

**Scaling & pathfinding**
- Large-scale ABM scaling / macro-micro (continuum & aggregation): https://www.researchgate.net/publication/226301178_Large_Scale_Agent-Based_Modelling_A_Review_and_Guidelines_for_Model_Scaling — *aggregate cohorts; 10×+ speedups; Transport Fever 1:4 agent scale.*
- HPA\* near-optimal hierarchical pathfinding: https://www.researchgate.net/publication/228785110_Near_optimal_hierarchical_path-finding_HPA · NavMesh HNA\*: https://www.cs.upc.edu/~npelechano/Pelechano_HNAstar_prePrint.pdf — *cluster the grid, path abstract then refine, >95% time cut; flow fields for crowds.*

**Tech-tree / era design**
- Civilization eras & tech/civics trees: https://civilization.fandom.com/wiki/Era_(Civ6) · tech-tree analysis: https://gamestudies.org/1201/articles/tuur_ghys — *gate eras by tech **and** civics; "interlocking vines," not a single line.*

**Utility-network modeling & cascades**
- ORNL interdependent grid/water cascade simulation: https://www.ornl.gov/news/what-100000-simulations-reveal-about-our-power-grid — *coupled networks; failure propagation (keep it simple in-sim).*

---

### TL;DR for the implementing chat
Build the **four foundations first** — *Settlement object, Population LOD, the road-graph network substrate (with utilities riding roads + HPA\* pathfinding), and Building entities with zoning/floors.* Then walk the **era ladder** (Agriculture → Classical urbanism → Medieval trade → Industrial rail → Modern cars/skyscrapers/grids → Contemporary air/sea/continental), extending `crafting.py`'s tech tree and turning on the **procedural city engine** from the Classical era. Keep everything **deterministic, offline, vectorized, and canary-verified**, with the **LLM as enrichment for notables only** — exactly the body/mind philosophy this sim already lives by, scaled up. Cities should **grow from emergent desire-paths and service-gated demand**, not from stamped templates. Ship one milestone at a time; each has an acceptance test in §6.
