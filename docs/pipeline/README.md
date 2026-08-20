# The verification pipeline

One entry point. Four stages. Every test in the repository belongs to exactly
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
| one pinch point in 24 million triangles | slicing — the surface meeting itself |

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

## Stage 3 — watertight

The solid those shapes close into.

Runs `test_mesh`, then meshes each part and asks:

| gate | what it proves | what it does not see |
|---|---|---|
| resolution | the voxel is at most a third of the narrowest feature | nothing, when it is not — the gate is skipped with a reason at screening depth rather than asserted at a resolution that cannot support it |
| closes | no boundary edges, no edge in more than two triangles | a pinch at a vertex |
| euler is even | two closed orientable surfaces sum to an even Euler characteristic, so an odd one **proves** the surface meets itself | an even number of pinches |
| is one solid | exactly one positive-volume surface: no fragment attached to nothing | |
| cavities drain | no negative-volume surface: no cavity that stays full of powder | |
| volume | the mesh and the field agree, by two different routes through the same field | |

The volume gate is read as a length, not a percentage. The difference between
the divergence-theorem volume of the triangles and the occupancy integrated
cell by cell is divided by the surface area, which gives the uniform offset
that would account for it; that is the quantity that has to be small against
the voxel. A percentage is not comparable between a solid disc and a part with
392 channels in it, and a real loss — a dropped slab, a region meshed inside
out — is a whole feature, orders above the bound either way.

## Stage 4 — slicing

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

Given `--file`, all of it is re-derived from what is on disk rather than from
what was in memory, because the writer is the part a slicer actually sees.

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
- A green pipeline means every gate that exists passed. The failures listed at
  the top of this page were all, at the time they shipped, outside every gate
  that existed.
