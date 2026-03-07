# Roguelike Design Notes

Arena-based ECS roguelike with Lean 4 formal verification of key invariants.

## Arena-Based ECS Architecture

The core insight: all game state lives in typed arenas. No `Box<dyn Component>`,
no `HashMap<TypeId, Vec<Box<dyn Any>>>`. Each component type gets its own
`Arena<T>`, and entities are just bundles of `Id<T>` handles.

```
World
  entities:    Arena<Entity>
  positions:   Arena<Position>
  healths:     Arena<Health>
  names:       Arena<Name>
  ais:         Arena<AiBehavior>
  items:       Arena<Item>
  renderables: Arena<Renderable>
```

An `Entity` is a struct of `Option<Id<T>>` fields — one per component type.
This is a "struct of arrays" layout but with explicit optional handles instead
of bitsets. Simple, cache-friendly, no dynamic dispatch.

### Why not generational arenas?

Generational indices solve the ABA problem (dangling ID reused after free+realloc).
But roguelikes have a simpler lifecycle: entities are created during level gen or
gameplay and freed in bulk on level transition. We can use **epoch-based
invalidation** instead — bump a generation counter per level, invalidate all IDs
from prior epochs. Simpler to verify formally.

If we later need per-entity free (e.g., item pickup destroys floor item), add a
`free_list: Vec<u32>` and a generation field to `Id<T>`. The Lean spec already
models arenas as lists, so extending to generational is straightforward.

## Entities and Components

### Component Types

| Component | Fields | Notes |
|-----------|--------|-------|
| `Position` | `x: i32, y: i32` | Grid coordinates |
| `Health` | `current: i32, max: i32` | Clamped to [0, max] |
| `Name` | `name: String` | Display name |
| `Renderable` | `glyph: char, fg: Color, bg: Color` | Terminal rendering |
| `BlocksTile` | (unit/marker) | Collision flag |
| `Viewshed` | `range: i32, visible: Vec<Position>` | FOV data |
| `AiBehavior` | `kind: AiKind` | Monster AI strategy |
| `CombatStats` | `power: i32, defense: i32` | Melee combat |
| `Item` | `kind: ItemKind` | Consumable/equipment |
| `InBackpack` | `owner: Id<Entity>` | Inventory relationship |
| `WantsToMelee` | `target: Id<Entity>` | Intent component |
| `WantsToMove` | `dest: Position` | Intent component |

### Entity Archetypes

- **Player**: Position, Health, Name, Renderable, Viewshed, CombatStats
- **Monster**: Position, Health, Name, Renderable, Viewshed, CombatStats, AiBehavior, BlocksTile
- **Item**: Position, Name, Renderable, Item
- **Carried Item**: Name, Item, InBackpack (no Position — it's in inventory)

## Map Generation

### Representation

```rust
struct Map {
    width: i32,
    height: i32,
    tiles: Vec<TileType>,  // row-major: tiles[y * width + x]
}

enum TileType { Wall, Floor, StairsDown }
```

The map is a flat array indexed by `(x, y)`. This is the canonical roguelike
representation — simple, fast spatial queries via index math.

### Generation Algorithm: BSP + Drunkard Hybrid

1. **BSP split** the map into 6-12 rectangular rooms (min 4x4).
2. **Connect rooms** with L-shaped corridors between BSP siblings.
3. **Drunkard's walk** from 2-3 random floor tiles for organic caves (30% of
   remaining walls, capped at 500 steps).
4. **Place stairs** in the room farthest from the player start (Dijkstra distance).
5. **Validate connectivity** via flood fill — if disconnected, tunnel to connect.

Step 5 is the key formal verification target (see below).

### Dungeon Depth

Each level increases difficulty: more monsters, better loot, harder generation
parameters. Depth is a simple `u32` on the `World`.

## Turn System

**Energy-based turns.** Each entity has an `energy: i32` field. On each tick:

1. All entities gain energy equal to their `speed` stat.
2. Entities with `energy >= 100` act (spend 100 energy per action).
3. Faster entities act more often — a speed-200 entity acts twice per tick.

For the initial version, simplify to **strict alternation**: player acts, then
all monsters act in arena order. Energy system is a later upgrade.

### Turn Flow (Simple Version)

```
loop {
    render(world)
    input = get_player_input()
    resolve_player_action(world, input)   // writes WantsToMove / WantsToMelee
    run_ai(world)                         // writes WantsToMove / WantsToMelee for monsters
    resolve_movement(world)               // processes WantsToMove
    resolve_combat(world)                 // processes WantsToMelee
    cleanup_dead(world)                   // remove entities with health <= 0
}
```

Systems process intent components (`WantsToMelee`, `WantsToMove`) and clear them
after resolution. This decouples decision from execution — important for
deterministic replay and formal reasoning.

## Combat Mechanics

### Melee Combat

Simple subtraction model:

```
damage = max(0, attacker.power - defender.defense + roll(-2..2))
defender.health.current -= damage
```

The random roll adds variance without making defense useless. At `power ==
defense`, hits deal 0-2 damage. Power advantage of +5 means 3-7 damage.

### Later Extensions

- **Ranged combat**: `WantsToRangedAttack { target, weapon }` intent.
- **Abilities**: `WantsToUseAbility { ability, target_pos }` — area effects, buffs.
- **Status effects**: `StatusEffect` component with duration countdown.

Keep the initial version to melee only. Prove combat invariants first, then extend.

## Formal Verification Targets (Lean 4)

The point isn't to verify everything — it's to verify the things that are
**hardest to get right by testing** and **most catastrophic when wrong**.

### Tier 1: Arena Safety (already started)

- `get_alloc`: allocating then looking up returns the value (partially proved).
- `get_bounds`: `arena.get(id) = some v` implies `id.index < arena.len`.
- `alloc_monotonic`: allocation never shrinks the arena.
- `alloc_preserves`: allocating a new item doesn't change existing items.

These are the foundation — every other invariant depends on arena correctness.

### Tier 2: Map Connectivity

- `connected`: for any two floor tiles `a` and `b`, there exists a path of
  adjacent floor tiles from `a` to `b`.
- `stairs_reachable`: the stairs-down tile is reachable from the player start.

This is the most valuable verification target. A disconnected map is a
softlock — the player literally cannot progress. Testing catches most cases,
but formal proof catches all of them.

**Approach**: Model the map as a graph in Lean. Prove that the generation
algorithm's corridor-connection + flood-fill-repair step guarantees
connectivity. The Lean spec doesn't need to model the full BSP — just the
invariant that the output satisfies `connected`.

### Tier 3: Game State Transitions

- `health_bounded`: `0 <= health.current <= health.max` after any combat
  resolution.
- `dead_removed`: after `cleanup_dead`, no entity has `health.current <= 0`.
- `turn_determinism`: given the same RNG seed and inputs, the game produces
  identical state sequences.

### Tier 4: Inventory Invariants

- `item_unique_location`: an item is either on the floor (has Position, no
  InBackpack) or in exactly one backpack (has InBackpack, no Position). Never
  both, never neither (unless being destroyed).
- `backpack_owner_exists`: `InBackpack.owner` always refers to a living entity.

### What NOT to verify

- Rendering (visual correctness is a human judgment).
- Input handling (platform-specific, better tested than proved).
- Performance properties (not expressible in Lean's logic).
- Randomness quality (statistical, not logical).

## Implementation Roadmap

1. **Arena + World struct** — expand `Arena<T>` with generational free, build
   `World` with component arenas.
2. **Map + generation** — `Map` type, BSP+drunkard generator, connectivity check.
3. **Player + movement** — input handling, `WantsToMove`, collision.
4. **FOV + rendering** — symmetric shadowcasting, terminal output.
5. **Monsters + AI** — spawn during mapgen, simple chase AI.
6. **Combat** — `WantsToMelee`, damage resolution, death cleanup.
7. **Items** — pickup, drop, use (healing potions).
8. **Dungeon depth** — stairs, level transitions, epoch invalidation.

Each step adds Lean specs for the new invariants alongside the Rust code.

## Terminal Rendering

Use raw ANSI escape codes — no TUI framework dependency. The map fits in an
80x50 terminal. Player is `@`, monsters are letters (`g`oblin, `o`rc, `D`ragon),
items are symbols (`!` potion, `/` weapon, `[` armor), walls are `#`, floors `.`.

## Open Questions

- **Save/load**: serialize World to JSON/bincode? Arena layout makes this trivial
  (just dump the vecs), but ID stability across save/load needs thought.
- **Proc-gen item names**: fun but scope creep for v0.1.
- **Multi-level memory**: keep old levels in memory or regenerate? Memory is
  cheap, regeneration means the player can't return to previous levels.
