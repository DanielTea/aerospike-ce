# The verification pipeline

One entry point. Five stages. Every test in the repository belongs to exactly
one of them, and the pipeline refuses to run if a test module belongs to none.

```bash
cd validate
python pipeline.py                      # every stage, screening depth, a few minutes
python pipeline.py --depth print        # at the voxel the part ships at, about an hour
python pipeline.py --stage physics      # one stage
python pipeline.py --file ../docs/print/regen-spike-75.3mf
python pipeline.py --json ../out/pipeline.json
python pipeline.py --list               # the stages and the modules each one owns
```

It exits non-zero if any gate fails, prints a line per gate, and will write the
whole report as JSON.

## Why the stages are in this order

`pytest` proves the maths. It does not prove the part. Every defect that has
actually shipped from this repository was downstream of a green suite:

| what shipped | which stage would have caught it |
|---|---|
| a cooling channel closed by decimation | watertight — the genus moved |
| 27 cm3 of copper attached to nothing inside the cowl | watertight — two positive-volume surfaces |
| injector orifices that stopped reaching their plenum | watertight — a sealed void |
| 6,334 zero-area triangles in a published file | slicing — a triangle with no normal |
| one pinch point in 24 million triangles | watertight caught it and could not say what it was — "not watertight, zero boundary edges"; slicing names it, and the Euler parity proves it |
| a constant-thickness hollow that narrows as it rises | printability — a support inside a sealed cavity that nothing can reach |
| 416 triangles with area in memory and none on disk | slicing — the only stage that reads the file rather than the mesh |

Each was invisible to the stage above it and obvious to the stage below, and
until now there was no stage below. So the gates run in the order the failure
propagates, and each stage is only meaningful if the one before it passed.

## Stage 1 — physics

The numbers. A wrong throat area is a wrong engine and no amount of mesh
checking will say so.

Runs `test_contour`, `test_combustion`, `test_cooling`, `test_injector`,
`test_engine_systems`, `test_optimise`, and then asks the solved design:

| gate | what it proves |
|---|---|
| chamber conditions | pressure, temperature and mass flow are positive and finite |
| delivered performance | thrust is positive and vacuum Isp exceeds sea level |
| injector sized | an injector was returned, with its element count and geometry |
| every wall has a circuit | `design_engine` returns **no** circuit rather than a bad one, so a missing circuit is a refusal to gate on |
| wall margin, per wall | peak wall temperature is under the material limit, and the coolant under its own |
| coolant can carry the heat | total heat load is inside the fuel's capacity |
| cycle life, per wall | thermal strain gives more cycles than the spec requires |
| startup, per wall | the transient peak does not exceed what steady state allows |

## Stage 2 — geometry

The shapes those numbers imply, and the seam the C# reads.

Runs `test_engine`, `test_shell`, `test_manifolds`, `test_interfaces`,
`test_feed_paths`, `test_printability`, `test_cooled_geometry`,
`test_build_plan`, and then checks the plan actually written:

| gate | what it proves |
|---|---|
| three parts | centrebody, cowl and head all present |
| profile, per part | finite, non-negative radii, with its point count and revolved volume |
| plan is versioned | the reader can refuse a version it does not know |
| plan is in millimetres | the unit is stated rather than assumed |
| plan covers every part | nothing silently dropped at the seam |
| narrowest feature is real | the voxel size has something to be derived from |
| plan round-trips | it survives JSON exactly, which is how the C# receives it |
| the C# reader compiles | the other side of the seam still builds. Skipped, out loud, when the SDK or the vendored PicoGK is not on the machine -- a gate that quietly disappears is worse than one that was never there |

## Stage 3 — printability

Whether a machine can build those shapes at all. Cheap -- it works on the
profiles and the features, with no mesh -- so it runs before the expensive
stages: a part nobody can build is wrong whether or not its mesh closes.

Runs `test_printability`, then, in the orientation the model derives rather
than one chosen:

| gate | what it proves |
|---|---|
| build direction is derived | both orientations are analysed and the better one is reported with the evidence. Building +x gives 24 findings; the other way up gives 174 |
| fits the build envelope | height and diameter against the process |
| nothing unsupportable | **the gate that matters.** A ray is cast down the build direction to ask whether a support column could actually reach each overhanging facet. The ones it can reach cost support material and a finishing operation; the ones it cannot are defects, because a support inside a sealed cavity stays there for ever |
| no feature / drainage / envelope finding | nothing under the process's minimum feature, no void powder cannot get out of, nothing outside the machine |
| printable as it stands | all of the above together |

Supportable is not the same as printable, and treating the two alike either
rejects every printable engine or accepts an unbuildable one. It is also why
the centrebody cavity closes on the axis in a cone: a constant-thickness offset
of the spike is the obvious hollowing and it is unbuildable, because an
internal void that narrows as it rises hangs material over nothing and there is
no way in to support it.

## Stage 4 — watertight

The solid those shapes close into.

Runs `test_mesh`, then meshes each part and asks:

| gate | what it proves | what it does not see |
|---|---|---|
| resolution | the voxel is at most a third of the narrowest feature | nothing, when it is not — the gate is skipped with a reason at screening depth rather than asserted at a resolution that cannot support it |
| closes | no boundary edges, no edge in more than two triangles | a pinch at a vertex |
| euler is even | two closed orientable surfaces sum to an even Euler characteristic, so an odd one **proves** the surface meets itself | an even number of pinches |
| is one solid | exactly one positive-volume surface: no fragment attached to nothing | |
| cavities drain | no negative-volume surface: no cavity that stays full of powder | |
| volume | the mesh and the field agree, by two different routes through the same field | a surface that drifted without changing volume |
| follows the field | no point on the mesh is further from the field than a voxel of the resolution the features demand | |
| keeps its shape | the mean distance from the field is under a twentieth of the thinnest feature | |

The volume gate is read as a length, not a percentage. The difference between
the divergence-theorem volume of the triangles and the occupancy integrated
cell by cell is divided by the surface area, which gives the uniform offset
that would account for it; that is the quantity that has to be small against
the voxel. A percentage is not comparable between a solid disc and a part with
392 channels in it, and a real loss — a dropped slab, a region meshed inside
out — is a whole feature, orders above the bound either way.

The last two gates exist because volume is a single number over a whole part
and a mesh can hold it exactly while the shape drifts. Decimation moves
triangles onto the chords of the curves they spanned, which takes as much off
one side as it adds to the other; the volume never notices and neither does the
genus. So the distance is asked directly, at the vertices and across the faces,
against the same field the volume came from.

They are read differently on purpose. Marching cubes cannot put a vertex inside
a sharp concave corner, so the worst point on any faithful mesh is about a voxel
out — the undecimated cowl reads 195 µm and reads 195 µm again after a ten-fold
decimation. The maximum therefore bounds the meshing, and the **mean** is what
moves: 12 µm undecimated, 18 µm at the decimation floor. That is the number that
would notice a part quietly going faceted.

Both are lower bounds rather than guarantees, and the reason is worth knowing:
the field is composed with `min` and `max`, and the maximum of two distance
functions is not a distance function — outside a concave seam it reads short.
The measurement is tight over the smooth ground decimation actually changes and
optimistic in the corners.

Unlike the resolution gate, these two **can** be asked of a written file, and
they are the questions worth asking of one: not what voxel it claims to have
been built at, but how far from the engine it actually is.

## Stage 5 — slicing

What a slicer reads and a topologist never looks at. This is the stage that was
missing, and every check in it passes `manifold_report` unharmed.

Runs `test_print_ready`, then, on the meshes or on a written file:

| gate | what it proves | why watertightness misses it |
|---|---|---|
| triangles have area | no zero-area triangle | by index it is an ordinary face: two neighbours per edge, correct genus, no volume contribution. A slicer has no normal to offset and no side to be inside of, and a plane of them is what Cura calls a missing or extraneous surface |
| one triangle per place | no duplicated face | a coincident pair keeps every edge count at two |
| winding is consistent | every interior edge is walked once in each direction | a flipped patch still closes; the slicer reads solid and void the wrong way round across it |
| surface never pinches | no vertex where two cones meet at a point | every edge is still in exactly two triangles; only the Euler parity betrays it, and only when the number of pinches is odd |
| solid is outward | positive enclosed volume, finite coordinates, indices in range | |
| file states its unit | 3MF says millimetre | STL does not say anything, and every slicer guesses |
| file carries every solid | all three objects arrived | |

### The gap between the mesh and the file

This stage is the only one that reads what was written rather than what was
meant, and that is not a formality. 3MF carries coordinates as decimal text, so
writing quantises; a triangle whose corners are closer together than the written
quantum has area right up until it is written and none afterwards.

It happened here. `mesh_solid.level_guard_mm` separates a marching-cubes vertex
from its sample by about 1.5e-4 mm, and the writer used four decimals -- a
tenth of a micron. The guard and the writer were arguing about the same decimal
place, and the first file built with the guard in place reported no zero-area
triangles anywhere in memory and came back off disk with 416.

The fix was not only more decimals. The part is now snapped onto the write grid
*before* it is checked and welded there, so the mesh the gate reads and the mesh
a slicer opens are the same mesh down to the last decimal. More decimals alone
would have made it rarer and left the gap open.

Given `--file`, **both** mesh stages read the written file instead of meshing:
every gate is then asked of exactly what a slicer would open, rather than of
what was in memory when it was written. That is the whole point of having a file
to check. Two gates are skipped with a reason rather than faked — a 3MF does not
carry the voxel it was built at, and it does not carry the field its volume
should be compared against.

This is the release gate:

```bash
python print_ready.py --spec ../spec/regen.json --out ../docs/print
python pipeline.py --file ../docs/print/regen-spike-75.3mf --json ../out/release.json
```

## Depths

| depth | voxel | takes | what it is for |
|---|---|---|---|
| `screen` | 0.6 mm | a few minutes | CI. Closure, connectivity, orientation and the whole suite. The features are not resolved at this voxel, so the resolution gate is skipped and the genus is reported rather than asserted: at 0.6 mm the cowl counts 709 handles and at 0.233 mm it counts 392, because most of what it is counting at 0.6 mm is aliasing. |
| `print` | narrowest / 3 | about an hour | the gate the release asset has to pass |

## What this pipeline does not check

It is a geometry and arithmetic gate, and it inherits every limit in
`CLAUDE.md`. It does not know whether the engine works.

- Nothing here is CFD, FEA or a thermochemical solve. `cooling_ref` stands on
  Bartz at plus or minus thirty percent; a design that only just passes the
  physics stage has not passed.
- Base flow behind the truncated plug is not modelled at all, and it is worth
  several percent of the thrust the physics stage reports as green.
- The slicing stage proves a slicer will accept the file. It does not prove the
  machine will build the part: no self-intersection test, no build-volume
  check against a named machine, no support or thermal simulation.
- Only the 3MF path is snapped to its own write grid. `export_cooled.py` and
  `mesh_export.export_assembly` write binary STL, whose float32 coordinates
  quantise to about 1.5e-5 mm out at the cowl's radius. That is ten times finer
  than the level guard's separation, so the failure the 3MF had is ten times
  less likely there rather than impossible, and nothing checks for it. Those
  files are inspection aids; the 3MF is the deliverable.
- A green pipeline means every gate that exists passed. The failures listed at
  the top of this page were all, at the time they shipped, outside every gate
  that existed.
