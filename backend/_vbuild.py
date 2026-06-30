import numpy as np
from world import World, ADULT_AGE, WATER_NONE, BLOCK_DOOR
w = World().generate(seed=5)
# an inland tile with water within reach (so reachability passes)
home = None
ys, xs = np.nonzero(w.water != WATER_NONE)
cx, cy = int(xs[len(xs)//3]), int(ys[len(ys)//3])
for r in range(2, 14):
    for dx in range(-r, r+1):
        for dy in range(-r, r+1):
            t = (cx+dx, cy+dy)
            if (w._in(*t) and w.water[t[1], t[0]] == WATER_NONE
                    and w._water_within(*t, 6) and not w._water_within(*t, 1)):
                home = t; break
        if home: break
    if home: break
assert home, "no home tile"
for bp in ("leaf_shelter", "hut", "cabin", "gathering"):
    w.people = []; w._add_person(home[0], home[1], name="B"); p = w.people[0]
    p["age"] = ADULT_AGE + 5; p["home"] = home; p["inv"] = {"wood": 99}; p["skills"] = {"building": 0.5}
    w.sites = []
    before = len(w.sites)
    try:
        w._found_site(p, bp, communal=(bp == "gathering"))
        ok = len(w.sites) > before
        hasdoor = ok and any(t.get("code") == BLOCK_DOOR for t in w.sites[-1]["tasks"]) if bp != "leaf_shelter" else "n/a"
        print(f"{bp:14} sited={ok} tiles={len(w.sites[-1]['tasks']) if ok else 0} has_door={hasdoor}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{bp:14} ERROR {type(e).__name__}: {e}")
