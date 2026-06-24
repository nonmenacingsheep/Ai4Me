"""
Crafting — the World's item/recipe registry and a tiny rule-based crafting engine.

This is *content + mechanics*, deliberately decoupled from the live body loop in
world.py. It defines:
  • RAW    — gatherable raw materials (where they come from, what tool they need).
  • ITEMS  — every item that can exist (raws + crafted goods + stations + structures),
             each with a display name, category, icon and tier.
  • RECIPES — 128 recipes forming a shallow tech tree: primitive tools → workbench →
             cooking → pottery/glass → smelting → metalwork → textiles → buildings.
  • a small engine (can_craft / do_craft / available / missing) that works on a plain
             inventory dict, gated by which crafting STATIONS are in reach and which
             TOOL capabilities the crafter holds.
  • TECH_LADDER — a modest, ordered set of goals an autonomous "body" can climb when
             survival is comfortable (the rule-based people still only chase a handful;
             gods, the UI and the future LLM "mind" can use the full registry).

Nothing here calls an LLM or touches numpy — it's pure Python so it can be unit-tested
on its own (`python crafting.py`) and imported cheaply by both world.py and server.py.
"""

# ════════════════════════════════════════════════════════════════════════════
#  Raw materials — what the world yields, and what's needed to harvest it.
#    tool = the TOOL CAPABILITY required to gather (None = bare hands).
#    source = a short hint of where it's found (used by UI / digest / future minds).
# ════════════════════════════════════════════════════════════════════════════
RAW = {
    "wood":       dict(name="Wood",        icon="🪵", tool="axe",     source="felled trees"),
    "stone":      dict(name="Stone",       icon="🪨", tool=None,      source="rock & mountain"),
    "flint":      dict(name="Flint",       icon="🦴", tool=None,      source="rocky ground & gravel beds"),
    "fiber":      dict(name="Plant Fiber", icon="🌾", tool=None,      source="grass, reeds & shrubs"),
    "leaves":     dict(name="Leaves",      icon="🍃", tool=None,      source="trees & shrubs (gathered in armfuls)"),
    "clay":       dict(name="Clay",        icon="🟤", tool="shovel",  source="riverbanks & swamps"),
    "sand":       dict(name="Sand",        icon="🏖️", tool="shovel",  source="beaches & desert"),
    "berry":      dict(name="Berries",     icon="🫐", tool=None,      source="shrubs & forest"),
    "grain":      dict(name="Wild Grain",  icon="🌾", tool=None,      source="grassland & savanna"),
    "herb":       dict(name="Herbs",       icon="🌿", tool=None,      source="meadows & forest floor"),
    "mushroom":   dict(name="Mushroom",    icon="🍄", tool=None,      source="forest & swamp"),
    "egg":        dict(name="Egg",         icon="🥚", tool=None,      source="nests"),
    "meat":       dict(name="Raw Meat",    icon="🥩", tool="spear",   source="hunted game"),
    "fish":       dict(name="Fish",        icon="🐟", tool="rod",     source="rivers, lakes & sea"),
    "hide":       dict(name="Hide",        icon="🟫", tool="knife",   source="hunted game"),
    "water":      dict(name="Water",       icon="💧", tool=None,      source="rivers, lakes & wells"),
    "uranium_ore": dict(name="Uranium Ore", icon="🟢", tool="pickaxe", source="deep mountain veins"),
    "copper_ore": dict(name="Copper Ore",  icon="🟧", tool="pickaxe", source="surface veins in hills"),
    "tin_ore":    dict(name="Tin Ore",     icon="⬜", tool="pickaxe", source="hill & mountain veins"),
    "iron_ore":   dict(name="Iron Ore",    icon="⬛", tool="pickaxe", source="mountain rock"),
    "gold_ore":   dict(name="Gold Ore",    icon="🟨", tool="pickaxe", source="rare mountain veins"),
    "coal":       dict(name="Coal",        icon="◼️", tool="pickaxe", source="exposed seams"),
}

# ════════════════════════════════════════════════════════════════════════════
#  Stations & structures (outputs that aren't carried — they're placed/built).
# ════════════════════════════════════════════════════════════════════════════
STATION_KINDS = ("workbench", "campfire", "furnace", "kiln", "forge",
                 "loom", "tannery", "anvil", "well")
STRUCTURE_KINDS = ("shelter", "stone_wall", "brick_house", "stone_house", "windmill",
                   "generator", "power_pole", "reactor")

# Tools provide CAPABILITIES; a recipe that needs e.g. "hammer" is satisfied by any
# held item that provides it (so a steel hammer works wherever a crude one would).
TOOL_PROVIDES = {
    "crude_axe": ["axe"], "wooden_axe": ["axe"], "stone_axe": ["axe"],
    "copper_axe": ["axe"], "bronze_axe": ["axe"], "iron_axe": ["axe"], "steel_axe": ["axe"],
    "crude_pickaxe": ["pickaxe"], "wooden_pickaxe": ["pickaxe"], "stone_pickaxe": ["pickaxe"],
    "copper_pickaxe": ["pickaxe"], "bronze_pickaxe": ["pickaxe"], "iron_pickaxe": ["pickaxe"],
    "steel_pickaxe": ["pickaxe"],
    "crude_hammer": ["hammer"], "stone_hammer": ["hammer"], "iron_hammer": ["hammer"],
    "crude_knife": ["knife"], "stone_knife": ["knife"], "copper_knife": ["knife"],
    "wooden_shovel": ["shovel"], "iron_shovel": ["shovel"],
    "wooden_hoe": ["hoe"], "iron_hoe": ["hoe"],
    "crude_spear": ["spear"], "bow": ["spear"],
    "fishing_rod": ["rod"], "fishing_net": ["rod"],
    "saw": ["saw"], "chisel": ["chisel"],
}


# ════════════════════════════════════════════════════════════════════════════
#  Recipes — (output_id, qty, {inputs}, station, tool_cap, tier).
#    station = crafting station that must be in reach (None = anywhere/handheld).
#    tool    = a TOOL CAPABILITY the crafter must hold (None = none needed).
#    The output_id doubles as the recipe id; every output is unique.
# ════════════════════════════════════════════════════════════════════════════
_RECIPE_ROWS = [
    # ── A. Primitive — bare hands, no station (tier 0) ─────────────────────────
    ("stick",          2, {"wood": 1},                       None, None, 0),
    ("crude_axe",      1, {"wood": 2},                        None, None, 0),
    ("crude_pickaxe",  1, {"wood": 1, "stone": 2},            None, None, 0),
    ("crude_hammer",   1, {"stick": 1, "stone": 2},           None, None, 0),
    ("crude_knife",    1, {"flint": 1, "stick": 1},           None, None, 0),
    ("crude_spear",    1, {"stick": 1, "flint": 1},           None, None, 0),
    ("digging_stick",  1, {"stick": 1},                       None, None, 0),
    ("rope",           1, {"fiber": 3},                       None, None, 0),
    ("torch",          2, {"stick": 1, "fiber": 1},           None, None, 0),
    ("flint_shard",    2, {"flint": 1},                       None, None, 0),
    ("basket",         1, {"fiber": 4},                       None, None, 0),
    # ── Make-shift survival kit — the hand-figured early crafts a band must DISCOVER
    #    (see SURVIVAL_DISCOVERIES). Each has a deliberately distinct ingredient-type set
    #    so a correct guess of *what it's made of* identifies it. Solving water/food/rest
    #    out here is what lets people range from the riverbank and live. ──────────────
    ("leaf_flask",     1, {"leaves": 3, "rope": 1},           None, None, 0),  # carry water
    ("forage_sack",    1, {"fiber": 4, "rope": 1},            None, None, 0),  # carry more food
    ("sleeping_mat",   1, {"leaves": 4, "fiber": 2},          None, None, 0),  # rest well anywhere
    ("campfire",       1, {"wood": 3, "stone": 2, "leaves": 1}, None, None, 0),  # station: cooking
    ("bow",            1, {"stick": 2, "rope": 1},            None, None, 0),
    ("arrow",          4, {"stick": 1, "flint": 1},           None, None, 0),
    ("workbench",      1, {"wood": 4, "stick": 2},            None, None, 0),  # station

    # ── B. Workbench — shaped wood, stone tools (tier 1) ───────────────────────
    ("plank",          2, {"wood": 1},                        "workbench", None, 1),
    ("wooden_handle",  2, {"plank": 1},                       "workbench", None, 1),
    ("wooden_axe",     1, {"plank": 2, "stick": 1},           "workbench", None, 1),
    ("wooden_pickaxe", 1, {"plank": 2, "stick": 1},           "workbench", None, 1),
    ("wooden_shovel",  1, {"plank": 1, "stick": 1},           "workbench", None, 1),
    ("wooden_hoe",     1, {"plank": 1, "stick": 1},           "workbench", None, 1),
    ("stone_axe",      1, {"stone": 3, "stick": 2, "rope": 1}, "workbench", None, 1),
    ("stone_pickaxe",  1, {"stone": 3, "stick": 2, "rope": 1}, "workbench", None, 1),
    ("stone_hammer",   1, {"stone": 2, "stick": 1, "rope": 1}, "workbench", None, 1),
    ("stone_knife",    1, {"flint": 2, "stick": 1},           "workbench", None, 1),
    ("chisel",         1, {"flint": 2, "stick": 1},           "workbench", None, 1),
    ("saw",            1, {"flint": 3, "plank": 1},           "workbench", None, 1),
    ("fishing_rod",    1, {"stick": 2, "rope": 1},            "workbench", None, 1),
    ("fishing_net",    1, {"rope": 4},                        "workbench", None, 1),
    ("ladder",         1, {"plank": 2, "stick": 4},           "workbench", None, 1),
    ("cart",           1, {"plank": 6, "stick": 2, "rope": 2}, "workbench", "saw", 1),
    ("wooden_bucket",  1, {"plank": 3},                       "workbench", None, 1),
    ("wooden_chest",   1, {"plank": 6},                       "workbench", None, 1),
    ("wooden_door",    1, {"plank": 4},                       "workbench", None, 1),
    ("furnace",        1, {"stone": 8, "clay": 4},            "workbench", None, 1),  # station

    # ── C. Campfire — cooking & preserving (tier 1) ────────────────────────────
    ("cooked_meat",    1, {"meat": 1},                        "campfire", None, 1),
    ("cooked_fish",    1, {"fish": 1},                        "campfire", None, 1),
    ("roasted_berry",  1, {"berry": 2},                       "campfire", None, 1),
    ("roasted_grain",  1, {"grain": 2},                       "campfire", None, 1),
    ("mushroom_skewer", 1, {"mushroom": 2, "stick": 1},       "campfire", None, 1),
    ("boiled_egg",     1, {"egg": 1, "water": 1},             "campfire", None, 1),
    ("herbal_tea",     1, {"herb": 2, "water": 1},            "campfire", None, 1),
    ("cooked_stew",    2, {"meat": 1, "herb": 1, "water": 1}, "campfire", None, 1),
    ("dried_meat",     1, {"meat": 2},                        "campfire", None, 1),
    ("dried_fish",     1, {"fish": 2},                        "campfire", None, 1),
    ("flour",          2, {"grain": 3},                       "campfire", None, 1),
    ("flatbread",      2, {"flour": 2, "water": 1},           "campfire", None, 1),
    ("pemmican",       1, {"dried_meat": 1, "berry": 1},      "campfire", None, 1),
    ("smoked_fish",    1, {"fish": 2, "charcoal": 1},         "campfire", None, 1),
    ("trail_ration",   2, {"flatbread": 1, "dried_meat": 1, "berry": 1}, "campfire", None, 1),
    ("charcoal",       2, {"wood": 3},                        "campfire", None, 1),

    # ── D. Kiln — pottery, brick & glass (tier 2) ──────────────────────────────
    ("kiln",           1, {"stone": 6, "clay": 8},            "workbench", None, 2),  # station
    ("brick",          1, {"clay": 2},                        "kiln", None, 2),
    ("clay_pot",       1, {"clay": 3},                        "kiln", None, 2),
    ("clay_bowl",      2, {"clay": 2},                        "kiln", None, 2),
    ("clay_jug",       1, {"clay": 4},                        "kiln", None, 2),
    ("clay_tile",      2, {"clay": 2},                        "kiln", None, 2),
    ("glass",          1, {"sand": 2},                        "kiln", None, 2),
    ("glass_pane",     1, {"glass": 2},                       "kiln", None, 2),
    ("bottle",         1, {"glass": 1},                       "kiln", None, 2),
    ("terracotta",     1, {"clay": 3},                        "kiln", None, 2),
    ("crucible",       1, {"clay": 4},                        "kiln", None, 2),
    ("ceramic_cup",    2, {"clay": 1},                        "kiln", None, 2),
    ("flower_pot",     1, {"clay": 2},                        "kiln", None, 2),
    ("oil_lamp",       1, {"clay": 2, "fiber": 1},            "kiln", None, 2),
    ("brick_block",    1, {"brick": 4},                       "kiln", None, 2),
    ("mortar",         2, {"sand": 2, "clay": 1},             "kiln", None, 2),

    # ── E. Furnace — smelting ore into ingots & stock (tier 2-3) ───────────────
    ("copper_ingot",   1, {"copper_ore": 2, "charcoal": 1},   "furnace", None, 2),
    ("tin_ingot",      1, {"tin_ore": 2, "charcoal": 1},      "furnace", None, 2),
    ("bronze_ingot",   2, {"copper_ingot": 3, "tin_ingot": 1}, "furnace", None, 3),
    ("iron_ingot",     1, {"iron_ore": 2, "charcoal": 2},     "furnace", None, 3),
    ("gold_ingot",     1, {"gold_ore": 2, "charcoal": 1},     "furnace", None, 3),
    ("steel_ingot",    1, {"iron_ingot": 2, "charcoal": 3},   "furnace", None, 4),
    ("copper_wire",    4, {"copper_ingot": 1},                "furnace", None, 2),
    ("copper_plate",   1, {"copper_ingot": 2},                "furnace", None, 2),
    ("iron_plate",     1, {"iron_ingot": 2},                  "furnace", None, 3),
    ("steel_plate",    1, {"steel_ingot": 2},                 "furnace", None, 4),
    ("nails",          8, {"iron_ingot": 1},                  "furnace", None, 3),
    ("forge",          1, {"brick_block": 4, "iron_ingot": 2}, "workbench", None, 3),  # station

    # ── F. Forge — metal tools, weapons & armour (tier 3-4, needs a hammer) ─────
    ("anvil",          1, {"iron_ingot": 4},                  "forge", "hammer", 3),  # station
    ("copper_axe",     1, {"copper_ingot": 2, "stick": 1},    "forge", "hammer", 2),
    ("copper_pickaxe", 1, {"copper_ingot": 3, "stick": 2},    "forge", "hammer", 2),
    ("copper_knife",   1, {"copper_ingot": 1, "stick": 1},    "forge", "hammer", 2),
    ("bronze_axe",     1, {"bronze_ingot": 2, "plank": 1},    "forge", "hammer", 3),
    ("bronze_pickaxe", 1, {"bronze_ingot": 3, "plank": 1},    "forge", "hammer", 3),
    ("bronze_sword",   1, {"bronze_ingot": 2, "wooden_handle": 1}, "forge", "hammer", 3),
    ("bronze_shield",  1, {"bronze_ingot": 3, "plank": 2},    "forge", "hammer", 3),
    ("iron_axe",       1, {"iron_ingot": 2, "wooden_handle": 1}, "forge", "hammer", 3),
    ("iron_pickaxe",   1, {"iron_ingot": 3, "wooden_handle": 1}, "forge", "hammer", 3),
    ("iron_shovel",    1, {"iron_ingot": 1, "wooden_handle": 1}, "forge", "hammer", 3),
    ("iron_hoe",       1, {"iron_ingot": 2, "wooden_handle": 1}, "forge", "hammer", 3),
    ("iron_hammer",    1, {"iron_ingot": 2, "wooden_handle": 1}, "forge", "hammer", 3),
    ("iron_sword",     1, {"iron_ingot": 2, "wooden_handle": 1, "leather": 1}, "forge", "hammer", 3),
    ("iron_shield",    1, {"iron_plate": 2, "plank": 1},      "forge", "hammer", 3),
    ("iron_chain",     3, {"iron_ingot": 2},                  "forge", "hammer", 3),
    ("steel_axe",      1, {"steel_ingot": 2, "wooden_handle": 1}, "forge", "hammer", 4),
    ("steel_pickaxe",  1, {"steel_ingot": 3, "wooden_handle": 1}, "forge", "hammer", 4),
    ("steel_sword",    1, {"steel_ingot": 3, "leather": 1, "wooden_handle": 1}, "forge", "hammer", 4),
    ("steel_armor",    1, {"steel_plate": 4, "leather": 2},   "forge", "hammer", 4),
    ("helmet",         1, {"iron_plate": 2, "leather": 1},    "forge", "hammer", 3),
    ("chainmail",      1, {"iron_chain": 5, "leather": 2},    "forge", "hammer", 4),
    ("plow",           1, {"iron_plate": 1, "plank": 3, "wooden_handle": 1}, "forge", "hammer", 3),
    ("horseshoe",      2, {"iron_ingot": 1},                  "forge", "hammer", 3),
    ("lantern",        1, {"iron_ingot": 1, "glass_pane": 1}, "forge", "hammer", 3),
    ("gear",           2, {"steel_ingot": 1},                 "forge", "hammer", 4),

    # ── G. Loom & tannery — textiles & leather (tier 2-3) ──────────────────────
    ("loom",           1, {"plank": 6, "rope": 2},            "workbench", None, 2),  # station
    ("tannery",        1, {"plank": 4, "stone": 2},           "workbench", None, 2),  # station
    ("leather",        1, {"hide": 2},                        "tannery", "knife", 2),
    ("thread",         2, {"fiber": 2},                       "loom", None, 2),
    ("cloth",          1, {"thread": 4},                      "loom", None, 2),
    ("linen",          1, {"fiber": 6},                       "loom", None, 2),
    ("tunic",          1, {"cloth": 3},                       "loom", None, 2),
    ("cloak",          1, {"cloth": 2, "leather": 1},         "loom", None, 3),
    ("backpack",       1, {"leather": 3, "rope": 1},          "tannery", None, 3),
    ("waterskin",      1, {"leather": 2},                     "tannery", None, 2),
    ("boots",          1, {"leather": 2, "thread": 2},        "tannery", None, 3),
    ("sail",           1, {"cloth": 6, "rope": 2},            "loom", None, 3),

    # ── H. Building, furniture & advanced (tier 3-5) ───────────────────────────
    ("well",           1, {"stone": 10, "brick": 6},          "workbench", None, 3),  # station
    ("stone_wall",     1, {"stone": 6},                       "workbench", None, 2),  # structure
    ("brick_house",    1, {"brick_block": 8, "plank": 6, "wooden_door": 1}, "workbench", "hammer", 4),  # structure
    ("stone_house",    1, {"stone": 12, "plank": 8, "wooden_door": 1}, "workbench", "hammer", 4),  # structure
    ("table",          1, {"plank": 5},                       "workbench", "saw", 2),
    ("chair",          1, {"plank": 3},                       "workbench", "saw", 2),
    ("bed",            1, {"plank": 4, "cloth": 2},           "workbench", "saw", 3),
    ("bookshelf",      1, {"plank": 6},                       "workbench", "saw", 3),
    ("paper",          2, {"fiber": 4},                       "workbench", None, 2),
    ("book",           1, {"paper": 5, "leather": 1},         "workbench", None, 3),
    ("wheel",          1, {"plank": 4, "iron_ingot": 1},      "workbench", "saw", 3),
    ("windmill",       1, {"plank": 12, "cloth": 4, "gear": 2, "iron_ingot": 3}, "workbench", "hammer", 5),  # structure

    # ── I. Electricity — wiring, power & light (the first rung of the MODERN era, tier 4-5) ──
    #    Built at the smithy's forge/furnace from metal stock. The generator is a power SOURCE,
    #    poles+wire conduct it, and the light/motor are the first things it runs. This is where a
    #    band crosses from craft into industry — the foundation the reactor era will build on.
    ("magnet",         1, {"iron_ingot": 1, "coal": 1},        "furnace", None, 4),
    ("copper_coil",    1, {"copper_wire": 3},                  "workbench", None, 4),
    ("generator",      1, {"copper_coil": 2, "magnet": 2, "iron_ingot": 2, "plank": 2}, "forge", "hammer", 5),  # structure: power source
    ("battery",        1, {"copper_plate": 2, "iron_plate": 1, "glass": 1}, "forge", None, 5),
    ("power_pole",     2, {"plank": 2, "copper_wire": 1},       "workbench", None, 5),  # structure: conductor
    ("light_bulb",     2, {"glass": 1, "copper_wire": 1},       "forge", None, 5),
    ("electric_motor", 1, {"copper_coil": 1, "magnet": 1, "iron_ingot": 1}, "forge", "hammer", 5),

    # ── J. Industry & the NUCLEAR age — concrete, turbines, and a working reactor (tier 5-7) ──
    #    The summit of the tree, and the literal end of the user's scene: a wooden band that climbs
    #    this far can raise its own reactor. Needs the smithy's forge + furnace and a hammer.
    ("concrete",       2, {"sand": 2, "stone": 2, "water": 1},  "kiln", None, 4),
    ("steel_beam",     1, {"steel_ingot": 2},                   "forge", "hammer", 5),
    ("turbine",        1, {"steel_beam": 2, "copper_coil": 2, "gear": 2}, "forge", "hammer", 6),
    ("uranium_fuel",   1, {"uranium_ore": 3, "steel_plate": 1}, "furnace", None, 6),
    ("reactor",        1, {"uranium_fuel": 2, "turbine": 1, "concrete": 8, "steel_beam": 4, "copper_coil": 4}, "forge", "hammer", 7),  # structure: the modern power source
]

# Categorise outputs so callers (UI, world.py, future minds) know how to treat each.
_TOOL_CATS = ("axe", "pickaxe", "hammer", "knife", "shovel", "hoe", "spear", "saw",
              "chisel", "rod", "bow", "arrow", "shard", "stick", "handle", "rope",
              "torch", "basket", "net", "ladder", "bucket", "nails", "wire", "chain",
              "plate", "ingot", "plow", "horseshoe", "gear", "wheel", "sword", "shield",
              "armor", "helmet", "chainmail", "lantern")


def _categorise(item_id: str) -> str:
    if item_id in RAW:
        return "raw"
    if item_id in STATION_KINDS:
        return "station"
    if item_id in STRUCTURE_KINDS:
        return "structure"
    if item_id in TOOL_PROVIDES:
        return "tool"
    foods = ("cooked", "roasted", "boiled", "dried", "smoked", "stew", "bread",
             "ration", "pemmican", "tea", "flour", "skewer")
    if any(k in item_id for k in foods):
        return "food"
    textiles = ("cloth", "linen", "thread", "tunic", "cloak", "boots", "leather",
                "waterskin", "backpack", "sail")
    if item_id in textiles or any(k in item_id for k in textiles):
        return "textile"
    furniture = ("table", "chair", "bed", "bookshelf", "chest", "door", "book",
                 "paper", "cart")
    if item_id in furniture:
        return "furniture"
    return "material"


def _display(item_id: str) -> str:
    return item_id.replace("_", " ").title()


# Build RECIPES (id → dict) and ITEMS (id → dict) from the rows above.
RECIPES: dict[str, dict] = {}
for _out, _qty, _inp, _st, _tool, _tier in _RECIPE_ROWS:
    RECIPES[_out] = dict(id=_out, out=_out, qty=_qty, inp=dict(_inp),
                         station=_st, tool=_tool, tier=_tier)

ITEMS: dict[str, dict] = {}
# Raws first (their names/icons are authored).
for _id, _meta in RAW.items():
    ITEMS[_id] = dict(id=_id, name=_meta["name"], icon=_meta["icon"],
                      cat="raw", tier=0, tool=_meta["tool"], source=_meta["source"])
# Then everything a recipe can produce, plus any input that wasn't already known.
for _r in RECIPES.values():
    for _id in [_r["out"], *_r["inp"].keys()]:
        if _id not in ITEMS:
            ITEMS[_id] = dict(id=_id, name=_display(_id), icon="",
                              cat=_categorise(_id), tier=RECIPES.get(_id, {}).get("tier", 0))

ICONS = {  # a few hand-picked glyphs so the UI isn't all blanks (rest fall back to ▪)
    "stick": "🪵", "plank": "🪵", "crude_axe": "🪓", "stone_axe": "🪓", "iron_axe": "🪓",
    "steel_axe": "🪓", "crude_pickaxe": "⛏️", "iron_pickaxe": "⛏️", "steel_pickaxe": "⛏️",
    "crude_hammer": "🔨", "iron_hammer": "🔨", "crude_knife": "🔪", "saw": "🪚",
    "bow": "🏹", "arrow": "🏹", "rope": "🪢", "torch": "🔥", "workbench": "🛠️",
    "furnace": "🏭", "kiln": "🧱", "forge": "⚒️", "anvil": "⚙️", "loom": "🧵",
    "tannery": "🟫", "well": "⛲", "brick": "🧱", "glass": "🪟", "clay_pot": "🏺",
    "copper_ingot": "🟧", "iron_ingot": "⬛", "steel_ingot": "🔩", "gold_ingot": "🟨",
    "bronze_ingot": "🟫", "cloth": "🧶", "leather": "🟫", "book": "📖", "bed": "🛏️",
    "table": "🪑", "chair": "🪑", "lantern": "🏮", "iron_sword": "⚔️", "steel_sword": "⚔️",
    "iron_shield": "🛡️", "windmill": "🌬️", "brick_house": "🏠", "stone_house": "🏘️",
    "generator": "🔌", "battery": "🔋", "power_pole": "🗼", "light_bulb": "💡",
    "electric_motor": "⚙️", "magnet": "🧲", "copper_coil": "🌀",
    "reactor": "☢️", "uranium_ore": "🟢", "uranium_fuel": "🟩", "concrete": "🧱",
    "steel_beam": "🏗️", "turbine": "🌀",
}
for _id, _ic in ICONS.items():
    if _id in ITEMS:
        ITEMS[_id]["icon"] = _ic


# ── Craft durations — how long a recipe takes, in GAME-MINUTES ────────────────
# The world runs on a scaled clock (≈8 game-seconds per real second, a game-day ≈ 3
# real hours); these are in-world minutes, so crafting obeys the same time as hunger,
# sleep and the day/night cycle. Simple hand-work is quick; a worked tool is an hour or
# two; smelting/forging is most of a day; textiles and clothing run into DAYS — a tunic
# is genuinely days of spinning, weaving and stitching, as it ought to be.
#
# Survival kit is kept deliberately short so a founding band can still kit out within a
# day and live — the balance canary (run the 8-day world.py sim) watches this.
_CRAFT_BASE_MIN = {0: 25, 1: 120, 2: 360, 3: 720, 4: 1440, 5: 2880}
_CRAFT_OVERRIDE = {
    # quick hand-work
    "stick": 8, "flint_shard": 6, "plank": 12, "rope": 18, "torch": 12,
    "digging_stick": 10, "arrow": 20, "thread": 30, "paper": 40,
    "crude_axe": 12, "crude_pickaxe": 18, "crude_hammer": 20, "crude_knife": 16,
    "crude_spear": 18, "basket": 40, "bow": 55,
    # survival kit a band must discover — short on purpose (keeps survival winnable; the
    # whole bootstrap chain must stay quick or the founding band starves at the balance edge)
    "leaf_flask": 12, "forage_sack": 15, "sleeping_mat": 18, "campfire": 70,
    "workbench": 90,
    # textiles, leather & clothing — measured in DAYS of patient work
    "linen": 720, "cloth": 1200, "leather": 960, "tunic": 2880, "cloak": 3600,
    "boots": 2160, "backpack": 1800, "waterskin": 720, "sail": 4320, "bed": 1500,
    "book": 1080,
}


def craft_minutes(rid: str) -> float:
    """In-world minutes to craft one batch of a recipe. Used by the body to make
    crafting take time (a worker holds station with a ⚙ until it's done)."""
    if rid in _CRAFT_OVERRIDE:
        return float(_CRAFT_OVERRIDE[rid])
    r = RECIPES.get(rid)
    return float(_CRAFT_BASE_MIN.get(r["tier"] if r else 0, 120))


# ════════════════════════════════════════════════════════════════════════════
#  Engine — operates on a plain inventory dict {item_id: count}.
#    stations: an iterable of station kinds currently in reach.
#    tools:    optional explicit set of tool capabilities; if None it's derived
#              from the inventory via TOOL_PROVIDES.
# ════════════════════════════════════════════════════════════════════════════
def tool_caps(inv: dict) -> set:
    """The set of tool capabilities an inventory confers (e.g. {'axe','hammer'})."""
    caps: set[str] = set()
    for item, n in inv.items():
        if n > 0:
            caps.update(TOOL_PROVIDES.get(item, ()))
    return caps


def recipe(rid: str) -> dict | None:
    return RECIPES.get(rid)


def missing(inv: dict, rid: str) -> dict:
    """Per-input shortfall {item: amount_short} for a recipe given an inventory."""
    r = RECIPES.get(rid)
    if not r:
        return {}
    return {k: need - inv.get(k, 0) for k, need in r["inp"].items() if inv.get(k, 0) < need}


def can_craft(inv: dict, rid: str, stations=(), tools=None) -> bool:
    """True if the recipe exists, its station is in reach, the crafter has the tool
    capability it needs, and the inventory holds every input."""
    r = RECIPES.get(rid)
    if not r:
        return False
    stations = set(stations)
    if r["station"] and r["station"] not in stations:
        return False
    if r["tool"]:
        caps = tool_caps(inv) if tools is None else set(tools)
        if r["tool"] not in caps:
            return False
    return all(inv.get(k, 0) >= need for k, need in r["inp"].items())


def do_craft(inv: dict, rid: str, stations=(), tools=None) -> bool:
    """Consume inputs and add the output to `inv` in place. Returns False (no change)
    if the recipe can't currently be made. Station/structure outputs are still added
    to inv as a token of '1 made' — the world layer decides where to place them."""
    if not can_craft(inv, rid, stations, tools):
        return False
    r = RECIPES[rid]
    for k, need in r["inp"].items():
        inv[k] -= need
        if inv[k] <= 0:
            del inv[k]
    inv[r["out"]] = inv.get(r["out"], 0) + r["qty"]
    return True


def available(inv: dict, stations=(), tools=None) -> list[str]:
    """Recipe ids craftable right now with this inventory/stations/tools."""
    return [rid for rid in RECIPES if can_craft(inv, rid, stations, tools)]


# ════════════════════════════════════════════════════════════════════════════
#  Discovery — recipes are the laws of physics, but a people doesn't KNOW them.
#  Knowledge is earned: an agent hypothesises what an item is made of, and a
#  correct guess of the ingredient *types* (counts don't matter — it's the idea
#  that's hard) identifies the recipe and unlocks it for the whole band. The body
#  uses an offline experiment loop; the LLM mind reasons its way to the answer.
#  Only this make-shift survival tier is hidden — the deeper tech tree is content
#  the gods/UI expose, not something the rule-based folk grope toward unaided.
# ════════════════════════════════════════════════════════════════════════════
# What every newborn band already knows in its bones (so existing behaviour stands).
# NOTE: crude_axe is NOT here — a band must WORK OUT the axe (knap a sharp edge) through the
# experience of chopping wood by hand; it then spreads by teaching like any craft. See world.py.
STARTER_RECIPES = ("stick", "crude_pickaxe", "rope", "workbench", "basket")

# The hidden make-shift crafts a band must work out for itself, each tagged with the
# survival problem it eases — so a mind can aim its experiments at what it lacks.
SURVIVAL_DISCOVERIES = {
    "leaf_flask":   "carrying water away from the river",
    "forage_sack":  "carrying more food at once",
    "sleeping_mat": "resting well far from home",
    "campfire":     "making fire to cook by",
}

# How much water a filled container holds (drinks-worth), by container item.
CONTAINER_WATER = {"leaf_flask": 4, "waterskin": 6, "clay_jug": 8, "wooden_bucket": 10}


def identify(input_types, candidates) -> str | None:
    """Given a guessed SET of ingredient types and the recipes still unknown to a band,
    return the recipe whose inputs are exactly those types — i.e. 'did this hunch hit a
    real make-shift craft?'. Counts and station are ignored: the insight is the materials."""
    want = frozenset(t for t in input_types if t)
    if not want:
        return None
    for rid in candidates:
        r = RECIPES.get(rid)
        if r and frozenset(r["inp"].keys()) == want:
            return rid
    return None


def discoverable(known) -> list[str]:
    """Make-shift survival recipes a band hasn't worked out yet."""
    return [rid for rid in SURVIVAL_DISCOVERIES if rid not in set(known)]


def catalog() -> dict:
    """The whole registry as plain JSON-able data — for GET /api/world/recipes, the
    god-tools UI, and the future LLM mind. Recipes are grouped by tier for display."""
    return {
        "items": ITEMS,
        "raw": RAW,
        "stations": list(STATION_KINDS),
        "structures": list(STRUCTURE_KINDS),
        "tool_provides": TOOL_PROVIDES,
        "recipes": list(RECIPES.values()),
        "tech_ladder": TECH_LADDER,
        "count": len(RECIPES),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Tech ladder — a modest, ordered list of goals an autonomous body can pursue
#  once survival is comfortable. world.py's rule-based people only chase the first
#  few (axe → workbench → shelter material); gods/UI/the future mind can use the
#  whole tree. Each rung is just a recipe id to aim for, cheapest/earliest first.
# ════════════════════════════════════════════════════════════════════════════
TECH_LADDER = [
    "crude_axe", "stick", "workbench", "plank", "stone_axe", "stone_pickaxe",
    "campfire", "cooked_meat", "kiln", "brick", "furnace", "charcoal",
    "copper_ingot", "copper_axe", "tin_ingot", "bronze_ingot", "bronze_axe",
    "forge", "iron_ingot", "iron_axe", "iron_pickaxe", "loom", "thread", "cloth",
    "tannery", "leather", "steel_ingot", "steel_axe", "brick_house", "windmill",
    # The modern era's first rung — electricity (needs a smithy's forge + a hammer):
    "magnet", "copper_coil", "generator", "power_pole", "light_bulb", "electric_motor",
    # …and its summit — concrete, turbines and a working reactor:
    "concrete", "steel_beam", "turbine", "uranium_fuel", "reactor",
]


# ════════════════════════════════════════════════════════════════════════════
#  Headless self-test:  python crafting.py
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"recipes: {len(RECIPES)}  |  items: {len(ITEMS)}  |  raw: {len(RAW)}")
    assert len(RECIPES) == 144, f"expected 144 recipes, got {len(RECIPES)}"

    # Discovery: a correct ingredient guess identifies a hidden survival craft; a wrong one
    # doesn't; and each discoverable recipe has a UNIQUE ingredient-type set (so a right
    # guess is unambiguous).
    cand = discoverable(STARTER_RECIPES)
    assert identify({"leaves", "rope"}, cand) == "leaf_flask", "leaves+rope should be a flask"
    assert identify({"wood", "stone", "leaves"}, cand) == "campfire"
    assert identify({"leaves"}, cand) is None, "a lone material shouldn't match"
    assert identify({"gold_ore", "rope"}, cand) is None, "nonsense shouldn't match"
    sets = [frozenset(RECIPES[r]["inp"]) for r in SURVIVAL_DISCOVERIES]
    assert len(sets) == len(set(sets)), "discoverable recipes need distinct ingredient sets"
    print("discovery: correct guesses identify hidden crafts, wrong ones don't  ✓")

    # Every recipe input must be a known item (no dangling references).
    bad = []
    for r in RECIPES.values():
        for k in r["inp"]:
            if k not in ITEMS:
                bad.append((r["id"], k))
    assert not bad, f"unknown inputs: {bad}"
    print("all recipe inputs resolve to known items  ✓")

    # Every non-raw, non-station/structure item should be makeable by some recipe.
    makeable = set(RECIPES)
    orphans = [i for i, m in ITEMS.items()
               if m["cat"] not in ("raw",) and i not in makeable]
    print(f"items with no recipe (raws excluded): {orphans or 'none'}")

    # Reachability: starting from infinite raws + every station + every tool cap,
    # can each recipe eventually be crafted? (Catches impossible intermediates.)
    raws = {k: 99 for k in RAW}
    all_stations = set(STATION_KINDS)
    all_caps = {c for caps in TOOL_PROVIDES.values() for c in caps}
    inv = dict(raws)
    made, changed = set(), True
    while changed:
        changed = False
        for rid in RECIPES:
            if rid in made:
                continue
            if can_craft(inv, rid, all_stations, all_caps):
                inv[rid] = 999            # once craftable, it's producible in quantity
                made.add(rid); changed = True
    unreachable = [r for r in RECIPES if r not in made]
    print(f"reachable recipes: {len(made)}/{len(RECIPES)}"
          + (f"  UNREACHABLE: {unreachable}" if unreachable else "  ✓"))
    assert not unreachable, unreachable

    # A concrete craft: from raws, build a workbench then a plank.
    p = {"wood": 10, "stick": 4, "fiber": 6}
    assert do_craft(p, "workbench"), "should craft a workbench from wood+stick"
    assert do_craft(p, "plank", stations={"workbench"}), "plank needs a workbench in reach"
    assert not do_craft({"wood": 10}, "plank"), "plank must fail with no workbench"
    print(f"craft demo ok — inv after workbench+plank: {p}")

    # Tool gating: a forge axe needs a hammer capability.
    f = {"copper_ingot": 2, "stick": 1}
    assert not can_craft(f, "copper_axe", stations={"forge"}), "should need a hammer"
    f["crude_hammer"] = 1
    assert can_craft(f, "copper_axe", stations={"forge"}), "hammer should unlock it"
    print("tool-capability gating ok  ✓")

    by_tier: dict[int, int] = {}
    for r in RECIPES.values():
        by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
    print(f"recipes by tier: {dict(sorted(by_tier.items()))}")
    print("ALL CRAFTING SELF-TESTS PASSED")
