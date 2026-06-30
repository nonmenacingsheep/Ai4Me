"""
crafting_full.py — the FULLY EXPANDED tech & recipe tree: Stone Age → Contemporary.

This is a drop-in DATA superset for crafting.py. It keeps the exact same row format
    (out_id, qty, {inputs}, station, tool_capability, tier)
so the existing engine (can_craft / do_craft / available / missing) needs NO changes —
it already gates on "a station of this kind in reach" + "a held tool providing this
capability", which is exactly what the deep chains below rely on.

What's here vs. the original crafting.py:
  • 12 eras (tier 0..11) instead of 0..7 — a continuous ladder caveman→modern→frontier.
  • Full production CHAINS, not just headline items: ore→ingot→plate→component→machine,
    fiber→thread→cloth→garment, clay→brick→cement→reinforced concrete, sand→glass→lens,
    crude_oil→fuels/plastics/rubber, quartz→silicon→wafer→chip→computer, and the
    transport/energy/construction trees that ride on them.
  • ~300 recipes, every one validated by the __main__ block: every input resolves to a
    RAW or an earlier output, every station/tool is declared, and the dependency graph is
    acyclic (topologically sortable). Run `python crafting_full.py` to check.

INTEGRATION (do under canary, see WORLD_CIVILIZATION_IMPLEMENTATION.md §9):
  1. Merge RAW / STATION_KINDS / STRUCTURE_KINDS / TOOL_PROVIDES / _RECIPE_ROWS / ERAS /
     TECH_LADDER into crafting.py (or import from here). IDs that already exist in
     crafting.py are intentionally preserved, so older saves/recipes still resolve.
  2. Bump world SCHEMA (new stations/structures may appear in saves).
  3. Gate each era behind Settlement.era so the band climbs it gradually (the rule-body
     and LLM mind still only pursue what's reachable + comfortable, as today).

Pure data + a validator. No numpy, no LLM, no game imports — testable on its own.
"""

# ════════════════════════════════════════════════════════════════════════════
#  ERAS — tier index == era band. Settlement.era indexes this; recipes carry the
#  tier at which they first become reachable.
# ════════════════════════════════════════════════════════════════════════════
ERAS = (
    "paleolithic",    # 0  bare hands, flint, fire
    "neolithic",      # 1  farming, pottery, weaving, polished stone, domestication
    "copper",         # 2  chalcolithic — first smelting, copper tools
    "bronze",         # 3  alloying, casting, the first cities
    "iron",           # 4  iron & early steel, classical glass/concrete, mechanics
    "medieval",       # 5  water/wind power, masonry, advanced textiles, paper
    "renaissance",    # 6  printing, optics, gunpowder, precision clockwork
    "industrial",     # 7  steam, coke, machine tools, rail, heavy chemistry
    "electric",       # 8  electricity grid, Bessemer steel, oil, internal combustion, telecom, aluminum
    "atomic",         # 9  plastics, electronics (tube→transistor), autos, aircraft, nuclear
    "information",    # 10 semiconductors, computers, jets, digital telecom, satellites
    "frontier",       # 11 renewables, composites, robotics, EVs, space ("and more")
)

# ════════════════════════════════════════════════════════════════════════════
#  RAW — gatherables. tool = capability needed to harvest (None = bare hands).
# ════════════════════════════════════════════════════════════════════════════
RAW = {
    # Paleolithic / always-available
    "wood":        dict(name="Wood",         icon="🪵", tool="axe",     source="felled trees"),
    "stone":       dict(name="Stone",        icon="🪨", tool=None,      source="rock & mountain"),
    "flint":       dict(name="Flint",        icon="🦴", tool=None,      source="gravel beds & chalk"),
    "fiber":       dict(name="Plant Fiber",  icon="🌾", tool=None,      source="grass, reeds & shrubs"),
    "leaves":      dict(name="Leaves",       icon="🍃", tool=None,      source="trees & shrubs"),
    "bone":        dict(name="Bone",         icon="🦴", tool=None,      source="carcasses & middens"),
    "berry":       dict(name="Berries",      icon="🫐", tool=None,      source="shrubs & forest"),
    "herb":        dict(name="Herbs",        icon="🌿", tool=None,      source="meadows & forest floor"),
    "mushroom":    dict(name="Mushroom",     icon="🍄", tool=None,      source="forest & swamp"),
    "egg":         dict(name="Egg",          icon="🥚", tool=None,      source="nests"),
    "meat":        dict(name="Raw Meat",     icon="🥩", tool="spear",   source="hunted game"),
    "fish":        dict(name="Fish",         icon="🐟", tool="rod",     source="rivers, lakes & sea"),
    "hide":        dict(name="Hide",         icon="🟫", tool="knife",   source="hunted game"),
    "fat":         dict(name="Tallow/Fat",   icon="🧈", tool="knife",   source="butchered game"),
    "water":       dict(name="Water",        icon="💧", tool=None,      source="rivers, lakes & wells"),
    # Neolithic (farmed / domesticated)
    "grain":       dict(name="Wild Grain",   icon="🌾", tool=None,      source="grassland & fields"),
    "vegetable":   dict(name="Vegetables",   icon="🥕", tool="hoe",     source="tilled plots"),
    "fruit":       dict(name="Fruit",        icon="🍎", tool=None,      source="orchards & vines"),
    "flax":        dict(name="Flax",         icon="🪻", tool="knife",   source="cultivated fields"),
    "cotton":      dict(name="Cotton",       icon="☁️", tool=None,      source="warm-climate fields"),
    "wool":        dict(name="Raw Wool",     icon="🐑", tool="knife",   source="shorn sheep"),
    "reed":        dict(name="Reeds",        icon="🌾", tool="knife",   source="marsh & riverbank"),
    "clay":        dict(name="Clay",         icon="🟤", tool="shovel",  source="riverbanks & swamps"),
    "sand":        dict(name="Sand",         icon="🏖️", tool="shovel",  source="beaches & desert"),
    "honey":       dict(name="Honey",        icon="🍯", tool=None,      source="hives"),
    "salt":        dict(name="Salt",         icon="🧂", tool="shovel",  source="salt flats & evaporation"),
    "limestone":   dict(name="Limestone",    icon="⬜", tool="pickaxe", source="sedimentary rock"),
    # Metals & minerals (mined)
    "copper_ore":  dict(name="Copper Ore",   icon="🟧", tool="pickaxe", source="surface veins in hills"),
    "tin_ore":     dict(name="Tin Ore",      icon="⬜", tool="pickaxe", source="hill & mountain veins"),
    "iron_ore":    dict(name="Iron Ore",     icon="⬛", tool="pickaxe", source="mountain rock"),
    "gold_ore":    dict(name="Gold Ore",     icon="🟨", tool="pickaxe", source="rare mountain veins"),
    "silver_ore":  dict(name="Silver Ore",   icon="⚪", tool="pickaxe", source="mountain veins"),
    "lead_ore":    dict(name="Lead Ore",     icon="🔘", tool="pickaxe", source="deep veins"),
    "zinc_ore":    dict(name="Zinc Ore",     icon="🔲", tool="pickaxe", source="hill veins"),
    "coal":        dict(name="Coal",         icon="◼️", tool="pickaxe", source="exposed seams"),
    "sulfur":      dict(name="Sulfur",       icon="🟡", tool="shovel",  source="volcanic & spring deposits"),
    "saltpeter":   dict(name="Saltpeter",    icon="🧂", tool="shovel",  source="cave & midden crusts"),
    "phosphate":   dict(name="Phosphate",    icon="🟫", tool="pickaxe", source="guano & rock beds"),
    "quartz":      dict(name="Quartz",       icon="💎", tool="pickaxe", source="silica veins"),
    "bauxite":     dict(name="Bauxite",      icon="🟥", tool="pickaxe", source="tropical laterite"),
    "nickel_ore":  dict(name="Nickel Ore",   icon="🪙", tool="pickaxe", source="mafic intrusions"),
    "chromium_ore":dict(name="Chromite",     icon="◾", tool="pickaxe", source="ultramafic rock"),
    "titanium_ore":dict(name="Ilmenite",     icon="⬛", tool="pickaxe", source="beach sands & rock"),
    "uranium_ore": dict(name="Uranium Ore",  icon="🟢", tool="pickaxe", source="deep mountain veins"),
    "crude_oil":   dict(name="Crude Oil",    icon="🛢️", tool="pickaxe", source="seeps & wells"),
    "latex":       dict(name="Latex",        icon="🥛", tool="knife",   source="tapped rubber trees"),
}

# ════════════════════════════════════════════════════════════════════════════
#  STATIONS — a crafting building/bench that must be in reach. STRUCTURES — placed,
#  non-carried outputs (shelters, civic works, infrastructure, vehicles, machines).
# ════════════════════════════════════════════════════════════════════════════
STATION_KINDS = (
    "campfire", "workbench", "drying_rack", "pottery_wheel", "kiln", "oven",
    "spinning_wheel", "loom", "tannery", "brewery", "well",
    "furnace", "bloomery", "crucible", "forge", "anvil", "glassworks",
    "watermill", "windmill", "sawmill", "grain_mill",
    "machine_shop", "lathe", "printing_press", "chemical_lab",
    "blast_furnace", "foundry", "refinery", "assembly_line",
    "electronics_lab", "cleanroom", "shipyard", "power_plant",
)
STRUCTURE_KINDS = (
    "leaf_shelter", "hut", "cabin", "longhouse", "brick_house", "stone_house",
    "apartment", "townhouse", "office_tower", "skyscraper",
    "gathering_hall", "workshop", "storehouse", "granary", "smithy", "inn",
    "watchtower", "market", "market_hall", "townhall", "temple", "library", "school",
    "hospital", "bank", "factory", "warehouse",
    "well_struct", "aqueduct", "water_tower", "water_treatment",
    "sewer", "sewage_works", "cistern",
    "dirt_road", "gravel_road", "cobble_road", "paved_road", "highway",
    "bridge", "canal", "rail", "tram_line", "subway",
    "wall", "gate", "tower", "lighthouse", "dam",
    "train_station", "port", "dock", "airport", "hangar", "bus_depot", "fuel_station",
    "windmill", "watermill", "generator", "power_pole", "substation",
    "powerplant", "hydro_dam", "wind_turbine", "solar_array", "reactor", "battery_bank",
)

# ════════════════════════════════════════════════════════════════════════════
#  TOOL_PROVIDES — held items → the capability they satisfy. A recipe needing
#  "saw" is met by any item providing it (a steel saw works wherever a crude one would).
# ════════════════════════════════════════════════════════════════════════════
TOOL_PROVIDES = {
    "crude_axe": ["axe"], "stone_axe": ["axe"], "copper_axe": ["axe"], "bronze_axe": ["axe"],
        "iron_axe": ["axe"], "steel_axe": ["axe"], "chainsaw": ["axe", "saw"],
    "crude_pickaxe": ["pickaxe"], "stone_pickaxe": ["pickaxe"], "copper_pickaxe": ["pickaxe"],
        "bronze_pickaxe": ["pickaxe"], "iron_pickaxe": ["pickaxe"], "steel_pickaxe": ["pickaxe"],
        "pneumatic_drill": ["pickaxe", "drill"],
    "crude_hammer": ["hammer"], "stone_hammer": ["hammer"], "bronze_hammer": ["hammer"],
        "iron_hammer": ["hammer"], "steel_hammer": ["hammer"],
    "crude_knife": ["knife"], "stone_knife": ["knife"], "copper_knife": ["knife"],
        "bronze_knife": ["knife"], "iron_knife": ["knife"], "steel_knife": ["knife"],
    "wooden_shovel": ["shovel"], "bronze_shovel": ["shovel"], "iron_shovel": ["shovel"], "steel_shovel": ["shovel"],
    "wooden_hoe": ["hoe"], "bronze_hoe": ["hoe"], "iron_hoe": ["hoe"], "steel_hoe": ["hoe"],
    "crude_spear": ["spear"], "bronze_spear": ["spear"], "iron_spear": ["spear"], "bow": ["spear"],
        "crossbow": ["spear"], "musket": ["spear"], "rifle": ["spear"],
    "fishing_rod": ["rod"], "fishing_net": ["rod"],
    "handsaw": ["saw"], "steel_saw": ["saw"],
    "chisel": ["chisel"], "steel_chisel": ["chisel"],
    "wrench": ["wrench"], "file": ["file"],
    "hand_drill": ["drill"], "power_drill": ["drill"],
    "soldering_iron": ["soldering"], "arc_welder": ["welding"],
}

# ════════════════════════════════════════════════════════════════════════════
#  RECIPES — (out_id, qty, {inputs}, station, tool_capability, tier).
#  Ordered roughly by tier so producers precede consumers (the validator confirms
#  every input resolves regardless of order, and that the graph is acyclic).
# ════════════════════════════════════════════════════════════════════════════
_RECIPE_ROWS = [

    # ══ TIER 0 — PALEOLITHIC: bare hands, flint, fire ════════════════════════
    ("stick",          2, {"wood": 1},                              None,        None,     0),
    ("cordage",        2, {"fiber": 3},                             None,        None,     0),  # twisted plant rope
    ("flint_shard",    2, {"flint": 1},                             None,        None,     0),  # struck cutting edge
    ("crude_axe",      1, {"flint_shard": 1, "stick": 1, "cordage": 1}, None,    None,     0),
    ("crude_knife",    1, {"flint_shard": 1, "stick": 1},           None,        None,     0),
    ("crude_pickaxe",  1, {"flint_shard": 1, "stick": 1, "stone": 1}, None,      None,     0),
    ("crude_hammer",   1, {"stone": 2, "stick": 1, "cordage": 1},   None,        None,     0),
    ("crude_spear",    1, {"flint_shard": 1, "stick": 2, "cordage": 1}, None,    None,     0),
    ("fishing_rod",    1, {"stick": 2, "cordage": 2},               None,        None,     0),
    ("torch",          2, {"stick": 1, "fat": 1, "fiber": 1},       None,        None,     0),
    ("campfire",       1, {"wood": 3, "stone": 4},                  None,        None,     0),  # station
    ("leaf_shelter",   1, {"leaves": 8, "stick": 4, "cordage": 2},  None,        None,     0),  # structure
    ("cooked_meat",    1, {"meat": 1},                              "campfire",  None,     0),
    ("cooked_fish",    1, {"fish": 1},                              "campfire",  None,     0),
    ("boiled_water",   1, {"water": 1},                             "campfire",  None,     0),  # safe to drink
    ("charcoal",       2, {"wood": 3},                              "campfire",  None,     0),  # smoldered; fuels hotter fires

    # ══ TIER 1 — NEOLITHIC: farming, pottery, weaving, polished stone ════════
    ("workbench",      1, {"wood": 4, "stick": 4, "cordage": 2},    None,        "axe",    1),  # station
    ("drying_rack",    1, {"stick": 6, "cordage": 3},               None,        None,     1),  # station
    ("pottery_wheel",  1, {"wood": 4, "stone": 2},                  "workbench", "axe",    1),  # station
    ("kiln",           1, {"clay": 8, "stone": 6},                  None,        None,     1),  # station
    ("oven",           1, {"clay": 6, "stone": 8},                  None,        None,     1),  # station
    ("spinning_wheel", 1, {"plank": 3, "stick": 2, "cordage": 2},   "workbench", "saw",    1),  # station
    ("loom",           1, {"plank": 4, "stick": 4, "cordage": 3},   "workbench", "saw",    1),  # station
    ("tannery",        1, {"plank": 4, "stick": 4, "stone": 2},     "workbench", "saw",    1),  # station
    ("brewery",        1, {"clay_pot": 4, "plank": 3},              "workbench", None,     1),  # station
    ("grain_mill",     1, {"stone": 8, "wood": 4},                  "workbench", "chisel", 1),  # station (hand quern)
    ("handsaw",        1, {"flint_shard": 2, "stick": 1, "cordage": 1}, "workbench", None, 1),
    ("chisel",         1, {"flint_shard": 1, "stick": 1, "stone": 1}, "workbench", None,   1),
    ("plank",          2, {"wood": 2},                              "workbench", "saw",    1),
    ("stone_axe",      1, {"stone": 2, "stick": 1, "cordage": 1},   "workbench", "chisel", 1),
    ("stone_pickaxe",  1, {"stone": 3, "stick": 1, "cordage": 1},   "workbench", "chisel", 1),
    ("stone_hammer",   1, {"stone": 3, "stick": 1, "cordage": 1},   "workbench", "chisel", 1),
    ("stone_knife",    1, {"stone": 1, "flint_shard": 1, "stick": 1}, "workbench", None,   1),
    ("wooden_shovel",  1, {"plank": 2, "stick": 1},                 "workbench", "saw",    1),
    ("wooden_hoe",     1, {"plank": 1, "stick": 2, "cordage": 1},   "workbench", "saw",    1),
    ("bone_needle",    2, {"bone": 1},                              "workbench", "knife",  1),
    ("fish_hook",      3, {"bone": 1},                              "workbench", "knife",  1),
    ("fishing_net",    1, {"cordage": 6, "bone_needle": 1},         "workbench", None,     1),
    ("bow",            1, {"stick": 2, "cordage": 2},               "workbench", "knife",  1),
    ("arrow",          4, {"stick": 1, "flint_shard": 1, "feather": 1}, "workbench", None, 1),
    ("feather",        2, {"egg": 1},                               None,        None,     1),  # (proxy: from fowl)
    ("basket",         1, {"reed": 4, "cordage": 1},                "workbench", None,     1),
    ("clay_pot",       2, {"clay": 3},                              "kiln",      None,     1),
    ("brick",          4, {"clay": 3, "sand": 1},                   "kiln",      None,     1),
    ("roof_tile",      4, {"clay": 2},                              "kiln",      None,     1),
    ("waterskin",      1, {"leather": 2, "cordage": 1},             "workbench", "knife",  1),
    ("thread",         2, {"flax": 2},                              "spinning_wheel", None, 1),
    ("wool_yarn",      2, {"wool": 2},                              "spinning_wheel", None, 1),
    ("cotton_thread",  2, {"cotton": 2},                            "spinning_wheel", None, 1),
    ("linen",          1, {"thread": 3},                            "loom",      None,     1),
    ("wool_cloth",     1, {"wool_yarn": 3},                         "loom",      None,     1),
    ("cotton_cloth",   1, {"cotton_thread": 3},                     "loom",      None,     1),
    ("cloth",          1, {"thread": 2, "wool_yarn": 1},            "loom",      None,     1),  # generic textile
    ("leather",        1, {"hide": 1},                              "tannery",   "knife",  1),
    ("tunic",          1, {"linen": 2, "thread": 1},                "workbench", None,     1),
    ("leather_boots",  1, {"leather": 2, "cordage": 1},             "workbench", "knife",  1),
    ("cloak",          1, {"wool_cloth": 2, "thread": 1},           "workbench", None,     1),
    ("flour",          2, {"grain": 3},                             "grain_mill", None,    1),
    ("dough",          1, {"flour": 2, "water": 1},                 "workbench", None,     1),
    ("bread",          2, {"dough": 1},                             "oven",      None,     1),
    ("stew",           2, {"meat": 1, "vegetable": 2, "water": 1},  "campfire",  None,     1),
    ("dried_meat",     2, {"meat": 2, "salt": 1},                   "drying_rack", None,   1),
    ("dried_fish",     2, {"fish": 2, "salt": 1},                   "drying_rack", None,   1),
    ("cheese",         1, {"milk": 2, "salt": 1},                   "workbench", None,     1),
    ("milk",           1, {"water": 1, "fat": 1},                   None,        None,     1),  # (proxy: from livestock)
    ("beer",           2, {"grain": 3, "water": 2},                 "brewery",   None,     1),
    ("mead",           2, {"honey": 2, "water": 2},                 "brewery",   None,     1),
    ("wine",           2, {"fruit": 4, "water": 1},                 "brewery",   None,     1),
    ("herbal_remedy",  2, {"herb": 3, "water": 1},                  "workbench", None,     1),
    ("soap",           2, {"fat": 2, "wood_ash": 1},               "workbench", None,     1),
    ("wood_ash",       2, {"charcoal": 1},                          "campfire",  None,     1),
    ("hut",            1, {"plank": 8, "roof_tile": 6, "cordage": 4}, "workbench", "saw",  1),  # structure
    ("granary",        1, {"plank": 12, "brick": 8, "roof_tile": 8}, "workbench", "hammer", 1),  # structure
    ("well_struct",    1, {"stone": 16, "brick": 8},                "workbench", "chisel", 1),  # structure

    # ══ TIER 2 — COPPER / CHALCOLITHIC: first smelting & metal tools ══════════
    ("furnace",        1, {"brick": 12, "stone": 8, "clay": 4},     "workbench", "chisel", 2),  # station (smelting)
    ("crucible",       2, {"clay": 4, "sand": 2},                   "kiln",      None,     2),  # also a station-grade vessel
    ("copper_ingot",   1, {"copper_ore": 2, "charcoal": 1},         "furnace",   None,     2),
    ("tin_ingot",      1, {"tin_ore": 2, "charcoal": 1},            "furnace",   None,     2),
    ("gold_ingot",     1, {"gold_ore": 2, "charcoal": 1},           "furnace",   None,     2),
    ("silver_ingot",   1, {"silver_ore": 2, "charcoal": 1},         "furnace",   None,     2),
    ("lead_ingot",     1, {"lead_ore": 2, "charcoal": 1},           "furnace",   None,     2),
    ("copper_plate",   2, {"copper_ingot": 1},                      "anvil",     "hammer", 2),
    ("copper_wire",    3, {"copper_ingot": 1},                      "anvil",     "hammer", 2),
    ("copper_axe",     1, {"copper_ingot": 2, "stick": 1},          "anvil",     "hammer", 2),
    ("copper_pickaxe", 1, {"copper_ingot": 2, "stick": 1},          "anvil",     "hammer", 2),
    ("copper_knife",   1, {"copper_ingot": 1, "stick": 1},          "anvil",     "hammer", 2),
    ("anvil",          1, {"copper_ingot": 4, "stone": 8},          "furnace",   "hammer", 2),  # station
    ("coin",           4, {"copper_ingot": 1},                      "anvil",     "hammer", 2),  # money (ties into money_invented)
    ("gold_coin",      4, {"gold_ingot": 1},                        "anvil",     "hammer", 2),
    ("jewelry",        1, {"gold_ingot": 1, "silver_ingot": 1},     "workbench", "hammer", 2),

    # ══ TIER 3 — BRONZE AGE: alloying, casting, durable tools/weapons ════════
    ("bronze_ingot",   1, {"copper_ingot": 3, "tin_ingot": 1},      "crucible",  None,     3),
    ("brass_ingot",    1, {"copper_ingot": 3, "zinc_ore": 1, "charcoal": 1}, "crucible", None, 3),
    ("bronze_plate",   2, {"bronze_ingot": 1},                      "anvil",     "hammer", 3),
    ("bronze_axe",     1, {"bronze_ingot": 2, "stick": 1},          "forge",     "hammer", 3),
    ("bronze_pickaxe", 1, {"bronze_ingot": 2, "stick": 1},          "forge",     "hammer", 3),
    ("bronze_hammer",  1, {"bronze_ingot": 2, "stick": 1},          "forge",     "hammer", 3),
    ("bronze_knife",   1, {"bronze_ingot": 1, "stick": 1},          "forge",     "hammer", 3),
    ("bronze_shovel",  1, {"bronze_plate": 1, "stick": 1},          "forge",     "hammer", 3),
    ("bronze_hoe",     1, {"bronze_plate": 1, "stick": 2},          "forge",     "hammer", 3),
    ("bronze_spear",   1, {"bronze_ingot": 1, "stick": 2},          "forge",     "hammer", 3),
    ("sword",          1, {"bronze_ingot": 2, "leather": 1},        "forge",     "hammer", 3),
    ("shield",         1, {"plank": 3, "bronze_plate": 1, "leather": 1}, "workbench", "hammer", 3),
    ("forge",          1, {"brick": 16, "stone": 12, "copper_ingot": 2}, "furnace", "hammer", 3),  # station
    ("cart",           1, {"plank": 6, "stick": 2, "cordage": 2, "wheel": 2}, "workbench", "saw", 3),
    ("wheel",          1, {"plank": 4, "bronze_plate": 1},          "workbench", "saw",    3),
    ("longhouse",      1, {"plank": 20, "roof_tile": 16, "brick": 8}, "workbench", "saw",  3),  # structure

    # ══ TIER 4 — IRON AGE / CLASSICAL: iron, glass, concrete, mechanics ══════
    ("bloomery",       1, {"brick": 16, "stone": 10, "clay": 6},    "workbench", "chisel", 4),  # station
    ("glassworks",     1, {"brick": 14, "stone": 8, "clay": 4},     "workbench", "chisel", 4),  # station
    ("iron_bloom",     1, {"iron_ore": 2, "charcoal": 2},           "bloomery",  None,     4),
    ("iron_ingot",     1, {"iron_bloom": 1},                        "forge",     "hammer", 4),
    ("iron_plate",     2, {"iron_ingot": 1},                        "anvil",     "hammer", 4),
    ("iron_rod",       2, {"iron_ingot": 1},                        "anvil",     "hammer", 4),
    ("iron_wire",      3, {"iron_ingot": 1},                        "anvil",     "hammer", 4),
    ("nails",          8, {"iron_ingot": 1},                        "anvil",     "hammer", 4),
    ("iron_axe",       1, {"iron_ingot": 2, "stick": 1},            "forge",     "hammer", 4),
    ("iron_pickaxe",   1, {"iron_ingot": 2, "stick": 1},            "forge",     "hammer", 4),
    ("iron_hammer",    1, {"iron_ingot": 2, "stick": 1},            "forge",     "hammer", 4),
    ("iron_knife",     1, {"iron_ingot": 1, "stick": 1},            "forge",     "hammer", 4),
    ("iron_shovel",    1, {"iron_plate": 1, "stick": 1},            "forge",     "hammer", 4),
    ("iron_hoe",       1, {"iron_plate": 1, "stick": 2},            "forge",     "hammer", 4),
    ("iron_spear",     1, {"iron_ingot": 1, "stick": 2},            "forge",     "hammer", 4),
    ("steel_saw",      1, {"steel_ingot": 1, "wood": 1},            "forge",     "hammer", 4),
    ("steel_chisel",   1, {"steel_ingot": 1, "wood": 1},            "forge",     "hammer", 4),
    ("file",           1, {"steel_ingot": 1, "wood": 1},            "forge",     "hammer", 4),
    ("wrench",         1, {"steel_ingot": 1},                       "forge",     "hammer", 4),
    ("steel_ingot",    1, {"iron_ingot": 2, "charcoal": 2},         "crucible",  None,     4),  # crucible steel
    ("glass",          2, {"sand": 3, "limestone": 1},              "glassworks", None,    4),
    ("glass_pane",     2, {"glass": 1},                             "glassworks", None,    4),
    ("bottle",         2, {"glass": 1},                             "glassworks", None,    4),
    ("mirror",         1, {"glass_pane": 1, "silver_ingot": 1},     "workbench", None,     4),
    ("quicklime",      2, {"limestone": 2},                         "kiln",      None,     4),
    ("mortar",         2, {"quicklime": 1, "sand": 2, "water": 1},  "workbench", None,     4),
    ("cement",         2, {"limestone": 2, "clay": 1, "coal": 1},   "kiln",      None,     4),
    ("concrete",       2, {"cement": 1, "sand": 2, "stone": 2, "water": 1}, "workbench", None, 4),
    ("gear",           2, {"iron_ingot": 1},                        "anvil",     "file",   4),
    ("axle",           1, {"iron_rod": 2},                          "forge",     "hammer", 4),
    ("chain",          2, {"iron_wire": 3},                         "anvil",     "hammer", 4),
    ("rope",           2, {"cordage": 4},                           "workbench", None,     4),  # heavy rigging rope
    ("pulley",         1, {"wood": 2, "iron_rod": 1, "rope": 1},    "workbench", "saw",    4),
    ("crossbow",       1, {"plank": 2, "iron_rod": 1, "rope": 1},   "workbench", "hammer", 4),
    ("leather_armor",  1, {"leather": 4, "iron_plate": 1},          "tannery",   "knife",  4),
    ("chainmail",      1, {"iron_wire": 8},                         "anvil",     "hammer", 4),
    ("aqueduct",       1, {"brick": 24, "concrete": 8, "stone": 16}, "workbench", "chisel", 4),  # structure (water main)
    ("sewer",          1, {"brick": 16, "concrete": 6},             "workbench", "chisel", 4),  # structure
    ("cobble_road",    4, {"stone": 6, "sand": 2},                  None,        "hammer", 4),  # structure
    ("bridge",         1, {"stone": 20, "plank": 12, "iron_ingot": 4}, "workbench", "hammer", 4),  # structure
    ("wall",           1, {"stone": 24, "mortar": 6},               "workbench", "chisel", 4),  # structure
    ("brick_house",    1, {"brick": 24, "plank": 10, "roof_tile": 12, "glass_pane": 2}, "workbench", "hammer", 4),  # structure
    ("stone_house",    1, {"stone": 30, "mortar": 10, "plank": 8, "roof_tile": 12}, "workbench", "chisel", 4),  # structure
    ("temple",         1, {"stone": 60, "mortar": 20, "plank": 16}, "workbench", "chisel", 4),  # structure

    # ══ TIER 5 — MEDIEVAL: water/wind power, masonry, paper, advanced metal ══
    ("watermill",      1, {"plank": 16, "gear": 4, "iron_ingot": 4, "stone": 12}, "workbench", "saw", 5),  # station/structure
    ("windmill",       1, {"plank": 20, "cloth": 6, "gear": 4, "iron_ingot": 4}, "workbench", "hammer", 5),  # station/structure
    ("sawmill",        1, {"plank": 12, "gear": 4, "iron_ingot": 3, "watermill": 1}, "workbench", "saw", 5),  # station
    ("blast_furnace",  1, {"brick": 30, "stone": 20, "iron_plate": 6}, "forge", "hammer", 5),  # station
    ("pig_iron",       2, {"iron_ore": 3, "coke": 2, "limestone": 1}, "blast_furnace", None, 5),
    ("coke",           2, {"coal": 3},                              "kiln",      None,     5),  # baked coal for hot smelting
    ("steel_plate",    2, {"steel_ingot": 1},                       "anvil",     "hammer", 5),
    ("steel_beam",     1, {"steel_ingot": 2},                       "forge",     "hammer", 5),
    ("steel_rod",      2, {"steel_ingot": 1},                       "anvil",     "hammer", 5),
    ("steel_wire",     3, {"steel_ingot": 1},                       "anvil",     "hammer", 5),
    ("spring",         2, {"steel_wire": 2},                        "anvil",     "file",   5),
    ("steel_axe",      1, {"steel_ingot": 2, "wood": 1},            "forge",     "hammer", 5),
    ("steel_pickaxe",  1, {"steel_ingot": 2, "wood": 1},            "forge",     "hammer", 5),
    ("steel_hammer",   1, {"steel_ingot": 2, "wood": 1},            "forge",     "hammer", 5),
    ("steel_knife",    1, {"steel_ingot": 1, "wood": 1},            "forge",     "hammer", 5),
    ("steel_shovel",   1, {"steel_plate": 1, "wood": 1},            "forge",     "hammer", 5),
    ("steel_hoe",      1, {"steel_plate": 1, "stick": 2},           "forge",     "hammer", 5),
    ("plow",           1, {"steel_plate": 2, "plank": 3},           "workbench", "hammer", 5),
    ("scythe",         1, {"steel_plate": 1, "stick": 2},           "forge",     "hammer", 5),
    ("paper",          3, {"reed": 2, "water": 1},                  "workbench", None,     5),
    ("parchment",      2, {"leather": 1},                           "tannery",   "knife",  5),
    ("book",           1, {"paper": 6, "leather": 1, "thread": 1},  "workbench", None,     5),
    ("candle",         3, {"fat": 1, "cordage": 1},                 "workbench", None,     5),
    ("lantern",        1, {"iron_plate": 1, "glass_pane": 1, "candle": 1}, "workbench", "hammer", 5),
    ("barrel",         1, {"plank": 6, "iron_rod": 2},              "workbench", "saw",    5),
    ("sail",           1, {"cloth": 6, "rope": 2},                  "loom",      None,     5),
    ("boat",           1, {"plank": 16, "rope": 4, "cloth": 2},     "workbench", "saw",    5),
    ("sailing_ship",   1, {"plank": 40, "steel_beam": 4, "sail": 4, "rope": 8}, "shipyard", "saw", 5),
    ("shipyard",       1, {"plank": 30, "steel_beam": 4, "stone": 10}, "workbench", "hammer", 5),  # station
    ("windmill_struct",1, {"windmill": 1},                          "workbench", None,     5),  # structure marker
    ("watermill_struct",1,{"watermill": 1},                         "workbench", None,     5),  # structure marker
    ("cathedral",      1, {"stone": 120, "mortar": 40, "glass_pane": 20, "steel_beam": 6}, "workbench", "chisel", 5),  # structure

    # ══ TIER 6 — RENAISSANCE: printing, optics, gunpowder, precision ═════════
    ("lathe",          1, {"steel_plate": 4, "gear": 6, "watermill": 1}, "machine_shop", "wrench", 6),  # station
    ("machine_shop",   1, {"steel_beam": 8, "steel_plate": 8, "gear": 8, "brick": 20}, "workbench", "wrench", 6),  # station
    ("printing_press", 1, {"steel_plate": 6, "gear": 8, "screw": 12, "plank": 10}, "machine_shop", "wrench", 6),  # station
    ("lens",           2, {"glass": 2},                            "lathe",     "file",   6),
    ("telescope",      1, {"lens": 2, "copper_plate": 2, "plank": 1}, "machine_shop", "file", 6),
    ("microscope",     1, {"lens": 3, "brass_ingot": 1, "iron_plate": 1}, "machine_shop", "file", 6),
    ("spectacles",     1, {"lens": 2, "copper_wire": 1},           "lathe",     "file",   6),
    ("clock",          1, {"gear": 8, "spring": 2, "brass_ingot": 1, "glass_pane": 1}, "machine_shop", "file", 6),
    ("compass",        1, {"iron_ingot": 1, "glass_pane": 1, "brass_ingot": 1}, "machine_shop", "file", 6),
    ("screw",          6, {"steel_rod": 1},                        "lathe",     "file",   6),
    ("bolt",           6, {"steel_rod": 1},                        "lathe",     "file",   6),
    ("nut",            6, {"steel_plate": 1},                      "lathe",     "file",   6),
    ("rivet",          8, {"steel_wire": 1},                       "machine_shop", "hammer", 6),
    ("bearing",        2, {"steel_plate": 1, "steel_wire": 1},     "lathe",     "file",   6),
    ("chemical_lab",   1, {"glass": 8, "copper_plate": 4, "brick": 12}, "workbench", "wrench", 6),  # station
    ("gunpowder",      2, {"saltpeter": 3, "sulfur": 1, "charcoal": 1}, "chemical_lab", None, 6),
    ("musket",         1, {"steel_beam": 1, "plank": 2, "spring": 1}, "machine_shop", "file", 6),
    ("cannon",         1, {"bronze_ingot": 8, "iron_plate": 4},    "foundry",   "hammer", 6),
    ("foundry",        1, {"brick": 30, "steel_beam": 6, "blast_furnace": 1}, "workbench", "hammer", 6),  # station
    ("printed_book",   2, {"paper": 8, "ink": 1},                  "printing_press", None, 6),
    ("ink",            2, {"charcoal": 1, "fat": 1, "water": 1},   "chemical_lab", None,   6),
    ("newspaper",      4, {"paper": 4, "ink": 1},                  "printing_press", None, 6),

    # ══ TIER 7 — INDUSTRIAL: steam, mass production, rail, chemistry ═════════
    ("steam_boiler",   1, {"steel_plate": 8, "copper_pipe": 4, "rivet": 12}, "machine_shop", "wrench", 7),
    ("copper_pipe",    2, {"copper_plate": 1},                     "machine_shop", "wrench", 7),
    ("steel_pipe",     2, {"steel_plate": 1},                      "machine_shop", "wrench", 7),
    ("piston",         2, {"steel_ingot": 1, "bearing": 1},        "lathe",     "file",   7),
    ("cylinder",       1, {"steel_plate": 2, "bearing": 1},        "lathe",     "file",   7),
    ("crankshaft",     1, {"steel_ingot": 2, "bearing": 2},        "lathe",     "file",   7),
    ("valve",          2, {"brass_ingot": 1, "spring": 1},         "lathe",     "file",   7),
    ("steam_engine",   1, {"steam_boiler": 1, "piston": 2, "crankshaft": 1, "valve": 2, "gear": 4}, "machine_shop", "wrench", 7),
    ("pump",           1, {"steel_plate": 2, "piston": 1, "valve": 2}, "machine_shop", "wrench", 7),
    ("sulfuric_acid",  2, {"sulfur": 2, "water": 1},               "chemical_lab", None,   7),
    ("nitric_acid",    2, {"saltpeter": 2, "sulfuric_acid": 1},    "chemical_lab", None,   7),
    ("fertilizer",     3, {"phosphate": 2, "nitric_acid": 1},      "chemical_lab", None,   7),
    ("dynamite",       2, {"nitric_acid": 1, "charcoal": 1, "paper": 1}, "chemical_lab", None, 7),
    ("dye",            3, {"herb": 2, "sulfuric_acid": 1},         "chemical_lab", None,   7),
    ("assembly_line",  1, {"steel_beam": 16, "gear": 12, "steam_engine": 1, "conveyor": 4}, "machine_shop", "wrench", 7),  # station
    ("conveyor",       2, {"steel_plate": 2, "gear": 2, "rubber_belt": 1}, "machine_shop", "wrench", 7),
    ("rubber",         2, {"latex": 3, "sulfur": 1},               "chemical_lab", None,   7),  # vulcanized
    ("rubber_belt",    2, {"rubber": 2, "cloth": 1},               "machine_shop", None,   7),
    ("rail",           4, {"steel_beam": 1, "plank": 2, "nails": 4}, "machine_shop", "hammer", 7),  # structure
    ("locomotive",     1, {"steam_engine": 1, "steel_beam": 12, "wheel_steel": 6, "boiler_plate": 4}, "assembly_line", "wrench", 7),
    ("wheel_steel",    2, {"steel_plate": 2, "bearing": 1},        "lathe",     "file",   7),
    ("boiler_plate",   2, {"steel_plate": 2, "rivet": 4},          "machine_shop", "hammer", 7),
    ("train_car",      1, {"steel_beam": 8, "plank": 12, "wheel_steel": 4}, "assembly_line", "wrench", 7),
    ("steamship",      1, {"steel_beam": 30, "steam_engine": 2, "boiler_plate": 8, "rivet": 40}, "shipyard", "wrench", 7),
    ("factory",        1, {"brick": 40, "steel_beam": 16, "glass_pane": 12, "concrete": 8}, "machine_shop", "hammer", 7),  # structure
    ("warehouse",      1, {"brick": 30, "steel_beam": 10, "plank": 20}, "machine_shop", "hammer", 7),  # structure
    ("paved_road",     4, {"concrete": 2, "stone": 4, "sand": 2},  None,        "hammer", 7),  # structure
    ("train_station",  1, {"brick": 40, "steel_beam": 12, "glass_pane": 16, "concrete": 10}, "machine_shop", "hammer", 7),  # structure
    ("dam",            1, {"concrete": 60, "steel_beam": 20, "steel_rod": 20}, "machine_shop", "wrench", 7),  # structure

    # ══ TIER 8 — ELECTRIC / 2nd INDUSTRIAL: power, oil, IC engine, telecom, Al
    ("magnet",         1, {"iron_ingot": 1, "coal": 1},            "furnace",   None,     8),
    ("copper_coil",    1, {"copper_wire": 4},                      "machine_shop", None,   8),
    ("insulated_wire", 2, {"copper_wire": 2, "rubber": 1},         "machine_shop", None,   8),
    ("generator",      1, {"copper_coil": 3, "magnet": 3, "steel_ingot": 2, "bearing": 2}, "machine_shop", "wrench", 8),  # structure: power source
    ("electric_motor", 1, {"copper_coil": 2, "magnet": 2, "steel_plate": 1, "bearing": 1}, "machine_shop", "wrench", 8),
    ("transformer",    1, {"copper_coil": 4, "steel_plate": 2, "insulated_wire": 4}, "machine_shop", "wrench", 8),  # substation core
    ("power_pole",     2, {"plank": 3, "insulated_wire": 2},       "workbench", None,     8),  # structure: conductor
    ("battery",        1, {"lead_ingot": 2, "sulfuric_acid": 1, "glass": 1}, "machine_shop", None, 8),
    ("light_bulb",     2, {"glass": 1, "copper_wire": 1, "tungsten_wire": 1}, "glassworks", None, 8),
    ("tungsten_wire",  2, {"iron_wire": 1, "coke": 1},             "foundry",   None,     8),  # (proxy hi-temp filament)
    ("soldering_iron", 1, {"copper_plate": 1, "iron_rod": 1, "electric_motor": 1}, "machine_shop", "wrench", 8),
    ("arc_welder",     1, {"transformer": 1, "insulated_wire": 4, "steel_plate": 2}, "machine_shop", "wrench", 8),
    ("telegraph",      1, {"copper_wire": 4, "magnet": 1, "battery": 1}, "machine_shop", "soldering", 8),
    ("telephone",      1, {"copper_coil": 1, "magnet": 1, "copper_wire": 2}, "machine_shop", "soldering", 8),
    ("alumina",        2, {"bauxite": 3, "sodium_hydroxide": 1},   "chemical_lab", None,   8),
    ("sodium_hydroxide",2,{"salt": 2, "water": 1},                "chemical_lab", None,   8),  # (electrolysis proxy)
    ("aluminum_ingot", 1, {"alumina": 2, "electricity": 2},        "foundry",   None,     8),  # electrolysis: needs power
    ("aluminum_sheet", 2, {"aluminum_ingot": 1},                   "machine_shop", "hammer", 8),
    ("electricity",    4, {"coal": 2, "steam_engine": 1, "generator": 1}, "power_plant", None, 8),  # the grid commodity
    ("power_plant",    1, {"brick": 50, "steel_beam": 30, "steam_boiler": 4, "generator": 4}, "machine_shop", "wrench", 8),  # structure/station
    ("gasoline",       2, {"crude_oil": 3},                        "refinery",  None,     8),
    ("diesel",         2, {"crude_oil": 3},                        "refinery",  None,     8),
    ("kerosene",       2, {"crude_oil": 3},                        "refinery",  None,     8),
    ("lubricant",      2, {"crude_oil": 2},                        "refinery",  None,     8),
    ("asphalt",        2, {"crude_oil": 2, "sand": 2, "stone": 2}, "refinery",  None,     8),
    ("naphtha",        2, {"crude_oil": 2},                        "refinery",  None,     8),  # plastics feedstock
    ("refinery",       1, {"steel_beam": 30, "steel_pipe": 20, "concrete": 16, "pump": 4}, "machine_shop", "wrench", 8),  # station/structure
    ("spark_plug",     2, {"steel_rod": 1, "copper_wire": 1, "porcelain": 1}, "machine_shop", "file", 8),
    ("porcelain",      2, {"clay": 2, "quartz": 1},                "kiln",      None,     8),
    ("combustion_engine",1,{"cylinder": 4, "piston": 4, "crankshaft": 1, "spark_plug": 4, "valve": 4}, "assembly_line", "wrench", 8),
    ("tire",           1, {"rubber": 3, "steel_wire": 1},          "assembly_line", None,  8),
    ("radiator",       1, {"copper_pipe": 4, "aluminum_sheet": 2}, "machine_shop", "wrench", 8),
    ("car_chassis",    1, {"steel_beam": 6, "steel_plate": 8, "bolt": 12}, "assembly_line", "wrench", 8),
    ("automobile",     1, {"car_chassis": 1, "combustion_engine": 1, "tire": 4, "glass_pane": 4, "battery": 1, "radiator": 1}, "assembly_line", "wrench", 8),
    ("truck",          1, {"car_chassis": 2, "combustion_engine": 1, "tire": 6, "steel_plate": 6}, "assembly_line", "wrench", 8),
    ("tram",           1, {"steel_beam": 10, "electric_motor": 2, "wheel_steel": 4, "glass_pane": 8}, "assembly_line", "wrench", 8),
    ("substation",     1, {"transformer": 4, "steel_beam": 8, "insulated_wire": 12}, "machine_shop", "wrench", 8),  # structure
    ("water_tower",    1, {"steel_beam": 16, "steel_plate": 12, "pump": 2}, "machine_shop", "wrench", 8),  # structure
    ("water_treatment",1, {"concrete": 30, "steel_pipe": 16, "pump": 4, "chemical_lab": 1}, "machine_shop", "wrench", 8),  # structure
    ("apartment",      1, {"brick": 40, "steel_beam": 16, "concrete": 12, "glass_pane": 16}, "machine_shop", "hammer", 8),  # structure

    # ══ TIER 9 — ATOMIC: plastics, electronics, vehicles, aircraft, nuclear ══
    ("plastic",        3, {"naphtha": 2},                          "chemical_lab", None,   9),
    ("synthetic_rubber",2,{"naphtha": 2, "sulfur": 1},            "chemical_lab", None,   9),
    ("fiberglass",     2, {"glass": 2, "plastic": 1},              "chemical_lab", None,   9),
    ("vacuum_tube",    2, {"glass": 1, "copper_wire": 1, "tungsten_wire": 1}, "electronics_lab", "soldering", 9),
    ("electronics_lab",1, {"steel_plate": 6, "glass": 8, "copper_wire": 8, "electric_motor": 1}, "machine_shop", "wrench", 9),  # station
    ("capacitor",      4, {"aluminum_sheet": 1, "plastic": 1},     "electronics_lab", "soldering", 9),
    ("resistor",       6, {"copper_wire": 1, "porcelain": 1},      "electronics_lab", "soldering", 9),
    ("circuit_board",  2, {"fiberglass": 1, "copper_plate": 1},    "electronics_lab", "soldering", 9),
    ("radio",          1, {"circuit_board": 1, "vacuum_tube": 2, "copper_coil": 1, "capacitor": 2}, "electronics_lab", "soldering", 9),
    ("television",     1, {"circuit_board": 2, "vacuum_tube": 4, "glass_pane": 2, "copper_coil": 2}, "electronics_lab", "soldering", 9),
    ("antibiotic",     2, {"mushroom": 3, "sulfuric_acid": 1, "herb": 2}, "chemical_lab", None, 9),
    ("propeller",      1, {"aluminum_sheet": 3, "steel_rod": 1},   "assembly_line", "wrench", 9),
    ("airframe",       1, {"aluminum_sheet": 12, "steel_beam": 4, "rivet": 40}, "assembly_line", "wrench", 9),
    ("airplane",       1, {"airframe": 1, "combustion_engine": 2, "propeller": 2, "tire": 3, "glass_pane": 4}, "assembly_line", "wrench", 9),
    ("diesel_engine",  1, {"cylinder": 6, "piston": 6, "crankshaft": 1, "valve": 8, "steel_plate": 4}, "assembly_line", "wrench", 9),
    ("diesel_locomotive",1,{"diesel_engine": 1, "generator": 1, "electric_motor": 4, "steel_beam": 16, "wheel_steel": 8}, "assembly_line", "wrench", 9),
    ("cargo_ship",     1, {"steel_beam": 60, "diesel_engine": 2, "steel_plate": 40, "rivet": 80}, "shipyard", "welding", 9),
    ("bus",            1, {"car_chassis": 2, "diesel_engine": 1, "tire": 6, "glass_pane": 12}, "assembly_line", "wrench", 9),
    ("reinforced_concrete",2,{"concrete": 2, "steel_rod": 2},     "machine_shop", None,    9),
    ("elevator",       1, {"electric_motor": 2, "steel_beam": 2, "gear": 4, "steel_wire": 8}, "machine_shop", "wrench", 9),
    ("uranium_fuel",   1, {"uranium_ore": 3, "steel_plate": 1, "nitric_acid": 1}, "chemical_lab", None, 9),
    ("turbine",        1, {"steel_beam": 2, "copper_coil": 2, "bearing": 4, "aluminum_sheet": 2}, "machine_shop", "wrench", 9),
    ("reactor",        1, {"uranium_fuel": 2, "turbine": 2, "reinforced_concrete": 16, "steel_beam": 12, "copper_coil": 6, "pump": 4}, "machine_shop", "welding", 9),  # structure
    ("office_tower",   1, {"reinforced_concrete": 30, "steel_beam": 40, "glass_pane": 60, "elevator": 2}, "machine_shop", "welding", 9),  # structure
    ("highway",        4, {"asphalt": 3, "reinforced_concrete": 2, "stone": 2}, None, "wrench", 9),  # structure
    ("airport",        1, {"reinforced_concrete": 80, "steel_beam": 40, "glass_pane": 40, "asphalt": 20}, "machine_shop", "welding", 9),  # structure
    ("port",           1, {"reinforced_concrete": 60, "steel_beam": 30, "crane": 2}, "machine_shop", "welding", 9),  # structure
    ("crane",          1, {"steel_beam": 12, "electric_motor": 2, "steel_wire": 12, "gear": 6}, "machine_shop", "wrench", 9),

    # ══ TIER 10 — INFORMATION: semiconductors, computers, jets, satellites ═══
    ("cleanroom",      1, {"steel_plate": 12, "glass_pane": 20, "plastic": 12, "electronics_lab": 1, "pump": 4}, "machine_shop", "wrench", 10),  # station
    ("silicon",        2, {"quartz": 3, "coke": 2, "electricity": 2}, "foundry", None,    10),
    ("silicon_wafer",  2, {"silicon": 1},                          "cleanroom", None,     10),
    ("transistor",     6, {"silicon_wafer": 1, "gold_ingot": 1},   "cleanroom", "soldering", 10),
    ("integrated_circuit",2,{"silicon_wafer": 1, "gold_ingot": 1, "plastic": 1}, "cleanroom", None, 10),
    ("microchip",      1, {"integrated_circuit": 4, "silicon_wafer": 1}, "cleanroom", None, 10),
    ("memory_chip",    2, {"integrated_circuit": 2, "silicon_wafer": 1}, "cleanroom", None, 10),
    ("processor",      1, {"microchip": 2, "gold_ingot": 1},       "cleanroom", None,     10),
    ("display_panel",  1, {"glass_pane": 2, "integrated_circuit": 2, "plastic": 2}, "cleanroom", None, 10),
    ("computer",       1, {"processor": 1, "memory_chip": 2, "circuit_board": 2, "display_panel": 1, "plastic": 4}, "electronics_lab", "soldering", 10),
    ("server",         1, {"processor": 4, "memory_chip": 8, "circuit_board": 4, "steel_plate": 2}, "electronics_lab", "soldering", 10),
    ("smartphone",     1, {"processor": 1, "memory_chip": 1, "display_panel": 1, "battery_li": 1, "aluminum_sheet": 1}, "cleanroom", None, 10),
    ("battery_li",     1, {"aluminum_sheet": 1, "plastic": 1, "nitric_acid": 1, "copper_wire": 1}, "electronics_lab", None, 10),
    ("jet_engine",     1, {"titanium_alloy": 4, "turbine": 2, "steel_beam": 4, "bearing": 8}, "assembly_line", "welding", 10),
    ("titanium_ingot", 1, {"titanium_ore": 2, "coke": 2, "electricity": 2}, "foundry", None, 10),
    ("titanium_alloy", 1, {"titanium_ingot": 2, "aluminum_ingot": 1}, "foundry", None,    10),
    ("airliner",       1, {"airframe": 2, "jet_engine": 2, "titanium_alloy": 4, "computer": 1, "glass_pane": 8}, "assembly_line", "welding", 10),
    ("helicopter",     1, {"airframe": 1, "jet_engine": 1, "steel_rod": 8, "computer": 1}, "assembly_line", "welding", 10),
    ("electric_locomotive",1,{"electric_motor": 6, "transformer": 2, "computer": 1, "steel_beam": 16, "wheel_steel": 8}, "assembly_line", "welding", 10),
    ("high_speed_train",1,{"electric_motor": 8, "aluminum_sheet": 20, "computer": 2, "transformer": 2, "wheel_steel": 8}, "assembly_line", "welding", 10),
    ("satellite",      1, {"aluminum_sheet": 8, "solar_panel": 4, "processor": 2, "radio": 2, "titanium_alloy": 2}, "cleanroom", "soldering", 10),
    ("solar_panel",    2, {"silicon_wafer": 2, "glass_pane": 1, "aluminum_sheet": 1, "copper_wire": 1}, "cleanroom", None, 10),
    ("steel_frame",    1, {"steel_beam": 8, "bolt": 16, "rivet": 16}, "machine_shop", "welding", 10),
    ("skyscraper",     1, {"steel_frame": 20, "reinforced_concrete": 40, "glass_pane": 120, "elevator": 4, "transformer": 2}, "machine_shop", "welding", 10),  # structure
    ("subway",         4, {"reinforced_concrete": 8, "steel_beam": 4, "rail": 2}, "machine_shop", "welding", 10),  # structure
    ("data_center",    1, {"server": 8, "transformer": 2, "reinforced_concrete": 20, "computer": 2}, "electronics_lab", "soldering", 10),  # structure

    # ══ TIER 11 — FRONTIER: renewables, robotics, EVs, composites ("and more")
    ("carbon_fiber",   2, {"plastic": 2, "naphtha": 1, "electricity": 2}, "chemical_lab", None, 11),
    ("composite_panel",1, {"carbon_fiber": 2, "aluminum_sheet": 1, "plastic": 1}, "assembly_line", None, 11),
    ("wind_turbine",   1, {"composite_panel": 8, "generator": 1, "steel_beam": 20, "gear": 6}, "assembly_line", "welding", 11),  # structure
    ("solar_array",    1, {"solar_panel": 16, "steel_beam": 8, "transformer": 1}, "assembly_line", "wrench", 11),  # structure
    ("grid_battery",   1, {"battery_li": 12, "transformer": 1, "steel_plate": 4}, "electronics_lab", "soldering", 11),
    ("battery_bank",   1, {"grid_battery": 8, "reinforced_concrete": 8}, "machine_shop", "wrench", 11),  # structure
    ("ev_battery",     1, {"battery_li": 8, "aluminum_sheet": 2, "computer": 1}, "electronics_lab", None, 11),
    ("electric_car",   1, {"car_chassis": 1, "electric_motor": 2, "ev_battery": 1, "tire": 4, "computer": 1, "composite_panel": 4}, "assembly_line", "welding", 11),
    ("robot_arm",      1, {"electric_motor": 4, "processor": 1, "steel_beam": 2, "bearing": 6, "sensor": 2}, "assembly_line", "soldering", 11),
    ("sensor",         4, {"integrated_circuit": 1, "plastic": 1, "copper_wire": 1}, "cleanroom", "soldering", 11),
    ("industrial_robot",1,{"robot_arm": 2, "computer": 1, "sensor": 4, "steel_frame": 1}, "assembly_line", "soldering", 11),
    ("automated_factory",1,{"factory": 1, "industrial_robot": 4, "conveyor": 8, "computer": 2}, "machine_shop", "welding", 11),  # structure
    ("rocket_engine",  1, {"titanium_alloy": 8, "turbine": 4, "composite_panel": 6, "computer": 2}, "assembly_line", "welding", 11),
    ("rocket",         1, {"rocket_engine": 2, "composite_panel": 20, "computer": 4, "titanium_alloy": 12}, "assembly_line", "welding", 11),
    ("spaceport",      1, {"reinforced_concrete": 120, "steel_frame": 40, "rocket": 1, "data_center": 1}, "machine_shop", "welding", 11),  # structure
    ("fusion_reactor", 1, {"titanium_alloy": 20, "composite_panel": 20, "grid_battery": 8, "computer": 8, "magnet": 16}, "machine_shop", "welding", 11),  # structure

    # ══ POWER TOOLS — later, faster versions of hand tools (close the capability set) ══
    ("hand_drill",     1, {"steel_rod": 2, "gear": 3, "wood": 1},  "machine_shop", "file",  6),  # brace-and-bit -> "drill"
    ("pneumatic_drill",1, {"steel_pipe": 2, "piston": 2, "valve": 2, "steel_plate": 2}, "machine_shop", "wrench", 8),  # rock drill -> pickaxe+drill
    ("chainsaw",       1, {"combustion_engine": 1, "chain": 1, "steel_plate": 2}, "assembly_line", "wrench", 9),  # -> axe+saw
    ("power_drill",    1, {"electric_motor": 1, "steel_plate": 1, "plastic": 1}, "electronics_lab", "wrench", 9),  # -> drill
    ("rifle",          1, {"steel_pipe": 1, "steel_plate": 1, "spring": 1, "plastic": 1}, "machine_shop", "file", 9),  # -> spear (ranged)
]


# ════════════════════════════════════════════════════════════════════════════
#  TECH_LADDER — the ordered spine of headline unlocks a Settlement climbs as its
#  `era` advances. (The full RECIPES set above is the breadth; this is the path.)
# ════════════════════════════════════════════════════════════════════════════
TECH_LADDER = [
    # paleolithic → neolithic
    "crude_axe", "campfire", "leaf_shelter", "charcoal",
    "workbench", "kiln", "loom", "tannery", "clay_pot", "brick", "linen", "leather", "bread", "hut", "granary",
    # copper → bronze
    "furnace", "copper_ingot", "anvil", "coin", "bronze_ingot", "forge", "sword", "cart", "longhouse",
    # iron / classical
    "bloomery", "iron_ingot", "steel_ingot", "glass", "concrete", "gear", "aqueduct", "brick_house",
    # medieval
    "watermill", "windmill", "blast_furnace", "steel_beam", "paper", "book", "sailing_ship", "cathedral",
    # renaissance
    "machine_shop", "lathe", "printing_press", "lens", "clock", "gunpowder",
    # industrial
    "steam_engine", "assembly_line", "rail", "locomotive", "factory", "paved_road", "train_station", "dam",
    # electric
    "generator", "electric_motor", "transformer", "electricity", "power_plant", "refinery", "gasoline",
    "combustion_engine", "automobile", "aluminum_ingot", "telephone", "substation", "water_treatment", "apartment",
    # atomic
    "plastic", "vacuum_tube", "circuit_board", "radio", "airplane", "diesel_locomotive", "elevator",
    "reactor", "office_tower", "highway", "airport", "port",
    # information
    "cleanroom", "silicon_wafer", "transistor", "microchip", "processor", "computer", "jet_engine",
    "airliner", "high_speed_train", "satellite", "solar_panel", "skyscraper", "subway", "data_center",
    # frontier
    "carbon_fiber", "wind_turbine", "solar_array", "battery_bank", "electric_car", "industrial_robot",
    "automated_factory", "rocket", "spaceport", "fusion_reactor",
]


# ─── self-validation (run `python crafting_full.py`) ──────────────────────────
def _build():
    recipes, dup = {}, []
    for out, qty, inp, st, tool, tier in _RECIPE_ROWS:
        if out in recipes:
            dup.append(out)
        recipes[out] = dict(out=out, qty=qty, inp=dict(inp), station=st, tool=tool, tier=tier)
    return recipes, dup


def _validate():
    recipes, dup = _build()
    raws = set(RAW)
    outs = set(recipes)
    tool_caps = {c for caps in TOOL_PROVIDES.values() for c in caps}
    stations = set(STATION_KINDS)
    errors, warns = [], []

    if dup:
        errors.append(f"duplicate output ids: {sorted(set(dup))}")

    for out, r in recipes.items():
        for ing in r["inp"]:
            if ing not in raws and ing not in outs:
                errors.append(f"{out}: input '{ing}' is neither a RAW nor a recipe output")
        if r["station"] is not None and r["station"] not in stations:
            errors.append(f"{out}: station '{r['station']}' not in STATION_KINDS")
        if r["tool"] is not None and r["tool"] not in tool_caps:
            errors.append(f"{out}: tool capability '{r['tool']}' not provided by any TOOL_PROVIDES item")

    # tool items must themselves be craftable (or raw) so a capability is reachable
    for item in TOOL_PROVIDES:
        if item not in outs and item not in raws:
            warns.append(f"tool '{item}' provides a capability but has no recipe")

    # cycle check via topological sort over input edges
    indeg = {o: 0 for o in outs}
    needs = {o: [i for i in recipes[o]["inp"] if i in outs] for o in outs}
    for o in outs:
        for i in needs[o]:
            indeg[o] += 1
    ready = [o for o in outs if indeg[o] == 0]
    seen = 0
    producedby = {}
    for o in outs:
        for i in needs[o]:
            producedby.setdefault(i, []).append(o)
    q = list(ready)
    while q:
        n = q.pop()
        seen += 1
        for dependent in producedby.get(n, []):
            indeg[dependent] -= 1
            if indeg[dependent] == 0:
                q.append(dependent)
    if seen != len(outs):
        stuck = [o for o in outs if indeg[o] > 0]
        errors.append(f"dependency CYCLE detected among {len(stuck)} items, e.g. {stuck[:8]}")

    # tier monotonicity: an input should not sit in a strictly higher tier than its output
    for out, r in recipes.items():
        for ing in r["inp"]:
            if ing in recipes and recipes[ing]["tier"] > r["tier"]:
                warns.append(f"{out} (t{r['tier']}) consumes higher-tier '{ing}' (t{recipes[ing]['tier']})")

    return recipes, errors, warns


if __name__ == "__main__":
    recipes, errors, warns = _validate()
    by_tier = {}
    for r in recipes.values():
        by_tier.setdefault(r["tier"], 0)
        by_tier[r["tier"]] += 1
    print("=" * 64)
    print(f"FULL TECH TREE — {len(RAW)} raws, {len(recipes)} recipes, "
          f"{len(STATION_KINDS)} stations, {len(STRUCTURE_KINDS)} structures")
    print("recipes per tier/era:")
    for t in sorted(by_tier):
        print(f"  tier {t:>2} {ERAS[t]:<13} {by_tier[t]:>3} recipes")
    print(f"TECH_LADDER spine: {len(TECH_LADDER)} milestones")
    print("-" * 64)
    for w in warns:
        print("WARN:", w)
    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(f"\n{len(errors)} validation error(s) — tree is NOT consistent.")
    print(f"\nOK — tree validates: every input resolves, all stations/tools declared, "
          f"graph is acyclic. ({len(warns)} soft warning(s).)")
