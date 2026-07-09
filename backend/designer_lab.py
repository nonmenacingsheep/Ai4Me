"""
designer_lab.py — a PLAYGROUND for the designer AI.

A separate, blank world where the designer lays out a whole town from its blueprints (and any
LLM-authored designs), so its town-building abilities can be exercised and tweaked in isolation —
no survival sim, no band to keep alive, no waiting for a prosperous settlement to emerge. Build,
look, tweak, rebuild.

    python designer_lab.py [village|town|city] [--seed N] [--authored] [--html PATH] [--empty]

It prints a report of what the designer made and writes a self-contained HTML picture of the town
you can open in a browser. Everything here only USES World's public API (generate / build_town /
apply_authored_building / snapshot), so nothing in the survival core is touched — a safe sandbox.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import world as W


# ── the blank canvas ─────────────────────────────────────────────────────────
def blank_world(seed: int = 1):
    """A clean, flat, grassy world with a small lake near the centre and nothing else — no
    wildlife, no band, no ore/berries. A pure canvas for the designer to build on. (Generates a
    real world first, then flattens it, so every array/dtype/index the engine expects is valid.)"""
    w = W.World().generate(seed)
    H, Wd = W.H, W.W
    # Flatten the terrain to a uniform grassy plain, well above sea level so it's all buildable.
    w.elevation[:] = 0.55
    w.biome[:] = W.B["grassland"]
    w.soil[:] = 0.5
    w.moisture[:] = 0.5
    w.water[:] = W.WATER_NONE
    w.veg_sp[:] = W.VEG_NONE
    w.veg_growth[:] = 0.0
    cx, cy = w._origin
    w.water[cy - 3:cy + 4, cx + 16:cx + 24] = W.WATER_LAKE     # a little lake off to one side
    # Wipe every living/built thing and its indices — a truly empty stage.
    w.people, w.animals = [], []
    w.sites, w.structures = [], []
    w.blocks, w.roofs, w.roads, w.decor = {}, set(), {}, {}
    w.station_objs, w.footfall = {}, {}
    w.companies, w.settlements = [], []
    w.ore_nodes, w._ore_index = [], {}
    w.stone_nodes = []
    w.berry_bushes, w._berry_index = [], {}
    w.granary = {"store": {}, "x": None, "y": None}
    w.authored_blueprints = list(getattr(w, "authored_blueprints", []))   # keep any (empty by default)
    return w


# A tiny, valid AUTHORED design to demonstrate the authored-blueprint path (the "immune system"
# validates it before it can be placed). A 3×3 hut-like home with a bed inside — the shape the
# LLM would emit. Swap/extend this to test the designer's authored forms without a live model.
_SAMPLE_AUTHORED = {
    "name": "Lab Cottage",
    "function": "home",
    "purpose": "a snug little home the designer dreamed up",
    "layout": ["WDW", "WbW", "WWW"],
}


def run(tier: str = "town", seed: int = 1, populate: bool = True, authored: bool = False):
    """Build on a blank world and return (world, report). tier village|town|city → a whole
    settlement (build_town); tier castle|keep → raise the GRAND KEEP alone (the masonry showpiece).
    With `authored`, first feed the designer a validated LLM-style design so it's raised too."""
    w = blank_world(seed)
    injected = None
    if authored:
        injected = w.apply_authored_building(dict(_SAMPLE_AUTHORED), by="the Lab")
    cx, cy = w._origin
    if tier in ("castle", "keep", "grand"):
        return w, _build_grand(w, cx, cy, populate)
    summary = w.build_town(cx, cy, tier=tier, populate=populate)
    return w, report(w, summary, injected)


def _build_grand(w, cx, cy, populate: bool):
    """Raise the GRAND KEEP on the blank canvas — the designer's most complex structure, built
    from the masonry palette (stone/brick/pillar/glass/marble/gate). A few folk take up residence."""
    site = w._stamp_building("castle", cx, cy, communal=True)
    if site is None:
        return {"error": "the keep would not fit here"}
    if populate:
        floors = [(t["x"], t["y"]) for t in site["tasks"]
                  if t.get("code") == W.BLOCK_MARBLE and t.get("layer") == "block"]
        for (fx, fy) in floors[:6]:
            q = w._add_person(int(fx), int(fy), age=float(w.rng.integers(W.ADULT_AGE + 200, W.ADULT_AGE + 1400)))
            q["home"], q["home_struct"], q["insul"] = (int(fx), int(fy)), site["id"], site.get("insul", 1.0)
    rep = report(w)
    mats = {}
    for t in site["tasks"]:
        if t.get("layer") == "block":
            mats[W.BLOCK_NAMES.get(t["code"], "?")] = mats.get(W.BLOCK_NAMES.get(t["code"], "?"), 0) + 1
    rep["grand"] = {"name": site.get("name"), "tiles": len(site["tasks"]), "materials": mats}
    return rep


# ── the report ───────────────────────────────────────────────────────────────
def report(w, summary=None, injected=None) -> dict:
    """What the designer actually made — buildings by role, homes, roads, farms, furniture,
    companies, and the town's identity — the numbers you tune the designer against."""
    by_func = {}
    homes = 0
    for s in w.sites:
        if not s.get("done"):
            continue
        fn = w._bp_function(s.get("bp")) or ""
        if s.get("communal") and fn:
            by_func[fn] = by_func.get(fn, 0) + 1
        elif not s.get("communal"):
            homes += 1
    wheat_id = next((k for k, v in W.PLANTS.items() if v["name"] == "wheat"), None)
    wheat = int((w.veg_sp == wheat_id).sum()) if wheat_id is not None else 0
    decor_kinds = {}
    for k in w.decor.values():
        decor_kinds[k] = decor_kinds.get(k, 0) + 1
    settle = (w._band_settlement() or {}) if hasattr(w, "_band_settlement") else {}
    return {
        "tier_summary": summary,
        "town_name": settle.get("name"),
        "square": settle.get("square_name"),
        "character": settle.get("character"),
        "era": w._civilization_era(),
        "homes": homes,
        "civic_by_function": by_func,
        "roads": len(w.roads),
        "farm_tiles": wheat,
        "furniture": decor_kinds,
        "companies": [{"name": c.get("name"), "kind": c.get("kind")} for c in w.companies],
        "people": len(w.people),
        "authored_used": injected,
        "authored_designs": [ab.get("name") for ab in w.authored_blueprints],
    }


def print_report(rep: dict):
    if rep.get("grand"):
        gr = rep["grand"]
        print("\n── the designer's grand keep ───────────────────────────────────")
        print(f"  {gr['name']} — {gr['tiles']} tiles of masonry")
        print(f"  materials: {gr['materials']}")
        print("────────────────────────────────────────────────────────────────")
        return
    print("\n── the designer's town ─────────────────────────────────────────")
    if rep.get("town_name"):
        print(f"  {rep['town_name']}" + (f" · {rep['square']}" if rep.get('square') else "")
              + (f" — {rep['character']}" if rep.get('character') else ""))
    print(f"  era: {rep['era']}   homes: {rep['homes']}   roads: {rep['roads']}   "
          f"farm tiles: {rep['farm_tiles']}   folk: {rep['people']}")
    print(f"  civic buildings by role: {rep['civic_by_function'] or 'none'}")
    print(f"  furniture placed: {rep['furniture'] or 'none'}")
    if rep["companies"]:
        print(f"  companies: {', '.join(c['name'] for c in rep['companies'])}")
    if rep.get("authored_used"):
        print(f"  authored design raised: {rep['authored_used']} "
              f"(designs on hand: {rep['authored_designs']})")
    print("────────────────────────────────────────────────────────────────")


# ── the picture ──────────────────────────────────────────────────────────────
_BLOCK_RGB = {1: "#ab8456", 2: "#78522d", 3: "#c49e5c", 4: "#96c4d6", 5: "#96785a", 6: "#4e8a40",
              7: "#8a8a84", 8: "#b2b0a8", 9: "#96cde1", 10: "#e0dcd0", 11: "#966e46", 12: "#a65c4a"}
_DECOR_DOT = {"bed": "#c2563f", "table": "#8a6038", "chair": "#9a6e3e", "chest": "#6e4a26",
              "cairn": "#b4b0a8", "obelisk": "#b9b3a6", "totem": "#c8a23a", "statue": "#c9c3b6",
              "fountain": "#5fb6e6", "arch": "#b3ada1"}


def render_html(w, path: str, title: str = "Designer's Town"):
    """Write a self-contained HTML picture of the build — every placed block, road, water tile,
    furniture and folk in the region, drawn on a canvas with a legend. Open it in any browser."""
    marks = list(w.blocks.keys()) + list(w.roads.keys()) + list(w.decor.keys())   # every drawn tile
    xs = [t[0] for t in marks] or [w._origin[0]]
    ys = [t[1] for t in marks] or [w._origin[1]]
    pad = 3
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    wtiles, htiles = x1 - x0 + 1, y1 - y0 + 1
    # Tile codes: 0 grass, 'w' water, 'r' road, or a block code 1..6.
    grid = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            if w.blocks.get((x, y)):
                row.append(w.blocks[(x, y)])
            elif (x, y) in w.roads:
                row.append("r")
            elif 0 <= y < W.H and 0 <= x < W.W and w.water[y, x] != W.WATER_NONE:
                row.append("w")
            else:
                row.append(0)
        grid.append(row)
    decor = [[t[0] - x0, t[1] - y0, k] for t, k in w.decor.items()
             if x0 <= t[0] <= x1 and y0 <= t[1] <= y1]
    people = [[p["x"] - x0, p["y"] - y0] for p in w.people
              if x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1]
    labels = [[s["ox"] - x0, s["oy"] - y0, (s.get("name") or s.get("bp") or "")]
              for s in w.sites if s.get("done") and s.get("communal")
              and x0 <= s["ox"] <= x1 and y0 <= s["oy"] <= y1]
    rep = report(w)
    payload = {"grid": grid, "decor": decor, "people": people, "labels": labels,
               "blockRGB": _BLOCK_RGB, "decorDot": _DECOR_DOT, "w": wtiles, "h": htiles}
    html = _HTML_TEMPLATE.replace("__TITLE__", title) \
        .replace("__SUBTITLE__", _subtitle(rep)) \
        .replace("__DATA__", json.dumps(payload))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def _subtitle(rep):
    bits = [rep["era"], f"{rep['homes']} homes", f"{rep['roads']} road tiles"]
    if rep["civic_by_function"]:
        bits.append(", ".join(f"{n}× {k}" for k, n in rep["civic_by_function"].items()))
    if rep["farm_tiles"]:
        bits.append(f"{rep['farm_tiles']} farm tiles")
    return " · ".join(bits)


_HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  body{margin:0;background:#12140f;color:#cdd6c8;font:14px/1.5 system-ui,sans-serif;padding:18px}
  h1{font-size:18px;margin:0 0 2px;color:#e8c76a} .sub{color:#8ba888;margin-bottom:14px;font-size:13px}
  #wrap{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start}
  canvas{background:#1c2417;border:1px solid #2c3a24;border-radius:8px;image-rendering:pixelated}
  .legend{font-size:12px} .legend div{display:flex;align-items:center;gap:7px;margin:3px 0}
  .sw{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid #0004}
</style></head><body>
<h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div>
<div id="wrap"><canvas id="c"></canvas>
<div class="legend" id="leg"></div></div>
<script>
const D = __DATA__;
const TS = Math.max(6, Math.min(16, Math.floor(760 / Math.max(D.w, D.h))));
const cv = document.getElementById('c'); cv.width = D.w*TS; cv.height = D.h*TS;
const g = cv.getContext('2d');
for(let y=0;y<D.h;y++)for(let x=0;x<D.w;x++){
  const c = D.grid[y][x]; let col='#2a3a20';
  if(c==='w') col='#2f6fb0'; else if(c==='r') col='#9c7c56';
  else if(c && D.blockRGB[c]) col=D.blockRGB[c];
  g.fillStyle=col; g.fillRect(x*TS,y*TS,TS,TS);
}
for(const [x,y,k] of D.decor){ g.fillStyle=D.decorDot[k]||'#e8c76a';
  g.beginPath(); g.arc(x*TS+TS/2,y*TS+TS/2,Math.max(1.5,TS*0.28),0,7); g.fill(); }
for(const [x,y] of D.people){ g.fillStyle='#f2e9d8'; g.strokeStyle='#0008';
  g.beginPath(); g.arc(x*TS+TS/2,y*TS+TS/2,Math.max(2,TS*0.32),0,7); g.fill(); g.stroke(); }
g.fillStyle='#e8c76a'; g.font=`bold ${Math.max(8,TS*0.9)}px system-ui`; g.textAlign='center';
for(const [x,y,t] of D.labels){ g.fillStyle='#00000088';
  const w=g.measureText(t).width+6; g.fillRect(x*TS+TS/2-w/2,y*TS-11,w,12);
  g.fillStyle='#f4e8c0'; g.fillText(t, x*TS+TS/2, y*TS-2); }
const leg=document.getElementById('leg');
const items=[['#9c7c56','road'],['#78522d','wall'],['#ab8456','floor'],['#c49e5c','door'],
  ['#96c4d6','window'],['#4e8a40','leaf wall'],['#2f6fb0','water'],['#c2563f','bed'],
  ['#8a6038','table'],['#f2e9d8','folk']];
leg.innerHTML='<b>Legend</b>'+items.map(([c,n])=>
  `<div><span class="sw" style="background:${c}"></span>${n}</div>`).join('');
</script></body></html>"""


def commission(request: str, seed: int = 1):
    """The PROMPTABLE designer in the lab: ask for a building in words, it designs (offline
    templates), sites and raises it on a blank canvas, and returns (world, plan)."""
    import mind
    w = blank_world(seed)
    design = mind.commission_offline(request)
    rep = w.commission_build(design, request=request, by="the god")
    return w, rep


if __name__ == "__main__":
    tier = "town"
    seed, authored, html_path, populate, request = 1, False, None, True, None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("village", "town", "city", "castle", "keep", "grand"):
            tier = a
        elif a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); i += 1
        elif a == "--authored":
            authored = True
        elif a == "--empty":
            populate = False
        elif a == "--request" and i + 1 < len(args):
            request = args[i + 1]; i += 1
        elif a == "--html" and i + 1 < len(args):
            html_path = args[i + 1]; i += 1
        i += 1
    if request:                                            # the promptable designer
        print(f'the god asks for: "{request}"…')
        w, rep = commission(request, seed=seed)
        if rep.get("ok"):
            print(f'\n  the designer says: "{rep["say"]}"')
            print(f'  → raised {rep["name"]} at ({rep["where"][0]}, {rep["where"][1]}) — '
                  f'{rep["tiles"]} tiles: {rep["materials"]}')
        else:
            print(f'  the designer set it aside — {rep.get("reason")}')
        out = html_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "designer_commission.html")
        render_html(w, out, title=rep.get("name") or "Commission")
        print(f"\npicture written → {out}\n(open it in a browser)")
        sys.exit(0)
    print(f"building a {tier} on a blank canvas (seed {seed})…")
    w, rep = run(tier=tier, seed=seed, populate=populate, authored=authored)
    print_report(rep)
    out = html_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"designer_town_{tier}.html")
    render_html(w, out, title=(rep.get("town_name") or f"The designer's {tier}"))
    print(f"\npicture written → {out}\n(open it in a browser)")
