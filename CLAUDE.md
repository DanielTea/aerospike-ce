# Working rules for this repository

You are working on a Computational Engineering model: a program that generates a
rocket nozzle from a specification. There is no interactive CAD. Nobody draws
anything. Read this before making changes.

## The one rule that matters

**You cannot see the geometry.** The PicoGK viewer is invisible to you. Never
claim a part "looks right". Verify through the channels you actually have:

1. `validate/pipeline.py` — the gate. Five stages, in the order a failure
   propagates: **physics**, **geometry**, **printability**, **watertight**,
   **slicing**. Every test module in the repo belongs to exactly one of them
   and the pipeline refuses to run if one belongs to none.
   `docs/pipeline/README.md` lists every gate and what it catches.
2. `out/contour.png`, `out/engine.png`, `out/engine_3d.png` — rendered plots you
   can read with your image tools.
3. `validate/verify_print_file.py` — reopens a written 3MF as a stranger would
   and re-derives everything from what is on disk.

If a change is not covered by one of these, add coverage before making it. If
you add a gate, add it to a stage; a check nobody runs is a check nobody has.

**Watertight is not the same as printable.** Four separate defects have shipped
past a mesh that reported watertight with the correct genus: a zero-area
triangle, a duplicated face, a patch wound inside out, and a vertex where the
surface pinches against itself. Every one of them is an ordinary face by index
— two neighbours per edge, so the edge arithmetic is content — and every one of
them is something a slicer reads and refuses. That is why there is a slicing
stage, and why "it is watertight" is not a claim that the part will print.

## Python owns the physics, C# owns the voxels

There is one implementation of every number and it lives in `validate/`. The C#
reads a build plan -- profiles as point lists, features as their parameters, all
in millimetres -- and voxelises it. It cannot disagree about geometry because it
is never told how to derive any.

This replaced a mirror. Two thirds of the C# used to be a hand-kept copy of the
Python maths, and divergence was named right here as the failure mode the
structure existed to prevent. The mirror never once caught an error in the
Python. Every bug the pairing found was a bug in the copy: a `SetRadius` call
that silently reset the step count, a CSV writing commas as both decimal point
and delimiter, a distance function costing twenty-two billion operations a part.
It charged a port every time the physics moved and defended against a failure
that never happened.

So do not put physics back into `model/`. If the C# needs a number, add it to
the plan. If it needs a *property* to make sense of a field, that field is on
the wrong side of the seam, and `test_build_plan.py` fails the build for exactly
that.

```bash
python validate/build_plan.py --spec spec/regen.json --out out/plan.json
dotnet run --project model -- out/plan.json --exit-when-done
```

## Order of work

The Python in `validate/` is the source of truth for the physics. The C# in
`model/` is a port. When changing engineering behaviour:

1. Change the relevant `validate/*_ref.py`. That is the source of truth.
2. Add or update a test in the matching `validate/test_*.py`.
3. Run `pytest` from inside `validate/`. Make it pass.
4. Run `python plot_contour.py --spec ../spec/demo.json --out ../out`, then
   `python plot_engine.py`, then `python mesh_export.py`.
5. **Actually look at the PNGs.** Read the images. Do not skip this.
5b. Run `python pipeline.py`. Not `pytest`: the suite is one gate out of four,
   and the three below it are where the defects that actually shipped were.
6. Only then port the change to `model/`.
7. `python build_plan.py`, then `dotnet run --project model -- out/plan.json`.
   There is nothing left to cross-check: the C# does not compute numbers, it
   reads them.

Step 7 needs the PicoGK native runtime. Without it you can still `dotnet build`
the project, which compile-checks everything, and cross-check the maths headless:
compile `EngineSpec.cs`, `GasDynamics.cs`, `PlugContour.cs` and
`EngineAssembly.cs` into a bare console project with no PicoGK reference. Those
four files have no native dependency, and they are where every number comes from.

Never edit the C# physics without a matching Python change. Divergence between
the two is the failure mode this structure exists to prevent.

## What goes where

| Path | Contains | Rule |
|---|---|---|
| `spec/*.json` | descriptive input | the only file a *user* edits |
| `validate/contour_ref.py` | authoritative nozzle physics | change here first |
| `validate/engine_ref.py` | authoritative assembly geometry | change here first |
| `validate/combustion_ref.py` | chamber conditions, performance | parametric, not CEA |
| `validate/injector_ref.py` | element sizing, face layout | |
| `validate/cooling_ref.py` | Bartz, fins, film, coatings, search | correlations, not CFD |
| `validate/structural_ref.py` | thermal strain and cycle life | screening, not FEA |
| `validate/transient_ref.py` | startup conduction through the wall | sequencing, not ignition |
| `validate/stability_ref.py` | chamber acoustics and screens | not a stability prediction |
| `validate/plug_flow_ref.py` | plug surface pressure, altitude | not method of characteristics |
| `validate/engine_design.py` | the whole pipeline, one spec in | derives the coolant split |
| `validate/interfaces_ref.py` | ports, bosses, the head joint | what to connect where |
| `validate/printability_ref.py` | overhangs, bridges, drainage | decides the build direction |
| `validate/optimise_ref.py` | evolutionary search over the spec | seeded, so it is testable |
| `validate/mesh_solid.py` | SDF + marching cubes, and how far a mesh strays from it | for anything not axisymmetric |
| `validate/mesh_export.py` | revolve to STL, cutaway PNG | exact, axisymmetric only |
| `validate/test_*.py` | invariants | add a test before adding physics |
| `validate/pipeline.py` | the gate: physics, geometry, watertight, slicing | every test belongs to a stage |
| `validate/plot_contour.py` | contour PNG | your eyes on the nozzle |
| `validate/plot_engine.py` | assembly PNG + area schedule | your eyes on the engine |
| `validate/plot_cooled.py` | section views through the field | your eyes on the channels |
| `validate/export_cooled.py` | cooled STLs + topology check | |
| `validate/print_ready.py` | the checked print files, full and compact | refuses rather than writes |
| `validate/verify_print_file.py` | reopens a written 3MF and re-derives everything | checks the writer, not the mesher |
| `validate/build_plan.py` | writes the plan C# reads | the seam; shapes only |
| `model/BuildPlan.cs` | reads the plan | refuses a version it does not know |
| `model/CooledGeometry.cs` | SDFs for every feature | mirror of `mesh_solid.py` |
| `model/SpikeGeometry.cs` | hands an implicit to PicoGK | forty lines, and rightly so |
| `model/Program.cs` | read plan, voxelise, export | no engineering logic |
| `out/` | generated artifacts | never commit, never hand-edit |

Physics never leaks into `SpikeGeometry.cs`. Geometry never leaks into
`PlugContour.cs`. If you find yourself computing a Mach number inside a voxel
routine, stop and move it.

`mesh_export.py` is a checking tool and a deliverable, not a second source of
truth. It revolves what `engine_ref.py` already decided. If it and
`SpikeGeometry.cs` disagree about a shape, both are wrong until `engine_ref.py`
says otherwise.

`mesh_solid.py` and `model/CooledGeometry.cs` carry the *same* distance
functions, deliberately. PicoGK renders any bounded implicit into voxels, so the
C# hands the kernel the identical formulation rather than booleaning several
hundred channel solids and keeping a second description of the same geometry in
step with the first.

## Supportable is not the same as printable

An overhang on an outer surface, or in a duct open at both ends, takes a support
that gets broken off afterwards. An overhang inside a sealed cavity takes a
support that stays there for ever, because nothing can reach in to remove it.
Treating the two alike either rejects every printable engine or accepts an
unbuildable one, so `printability_ref` casts a ray down the build direction and
asks whether a support column could actually reach the facet.

The corollary shaped the geometry: the centrebody cavity now closes on the axis
in a cone and is clamped so it never narrows faster than the process angle. A
constant-thickness offset of the spike is the obvious hollowing and it is
unbuildable -- an internal void that narrows as it rises hangs material over
nothing, and there is no way in to support it.

## Measure perpendicular, not radially

This has now caused two separate bugs, so it is worth stating as a rule. Where a
surface is inclined, a radial measurement is not a thickness and a radial
annulus is not a flow area.

- The throat is the sonic line from the cowl lip to the spike shoulder, at nu_e
  from radial. Read radially it measures 2.6 times too small.
- The cowl wall at the lip is 1.0 mm thick. Read radially it measures 2.99 mm,
  which cleared cooling channels to run somewhere they would have emerged
  straight through the taper.

`engine_ref._distance_to_polyline` and `EngineAssembly.DistanceToPolyline` are
the honest instruments. Reach for them.

## A construction that works at one parameter value is not verified

The corner fan that builds the cowl wall was tested at a contraction ratio of 3
and was correct there. At 6 the wall folded back on itself by two microns --
invisible, geometrically irrelevant, and enough to make every interpolation
along that wall return a different answer in Python than in C#. It surfaced only
because the two languages corrupted it differently, and it had already put a
0.9 mm error into the wall radius the thermal model was using.

Two things follow. Parametrise the geometry tests over the range the spec can
actually reach, not over the default. And when the C# and the Python disagree,
the disagreement is the finding: neither being right is the normal case.

## Meshing has a resolution floor, and it is a design constraint

Marching cubes needs about three samples across the narrowest feature. A 0.4 mm
cooling channel therefore needs 0.13 mm voxels, and a whole part at that size is
tens of millions of triangles before decimation. That is not a meshing detail,
it is a reason to prefer a channel the process can hold and an inspection can
resolve. `export_cooled.py` derives the voxel size from the narrowest feature,
checks the Euler characteristic against what the features imply, and only
decimates as far as the topology survives -- pushed further, quadric collapse
closes a cooling channel and leaves a mesh that still looks perfectly fine.

## A decimator in a hurry breaks the surface, and the gate blames the geometry

`fast_simplification` grows its error threshold as `(iteration + 3) ** agg`, so
a larger `agg` reaches the requested reduction by taking worse collapses near
the end. At the default 7 the centrebody comes back from keep 0.40 with two
edges in four triangles, two duplicated faces and a handle missing -- out of six
and a half million -- and its volume unchanged to three decimals. Nothing was
lost; the surface was folded. The ladder refused it, correctly, and the part
then shipped every one of its 8.2 million triangles.

At `agg = 3` the same rung is clean and so is keep 0.15, and the floor is 1.73
million triangles at the same genus and the same volume. The collapses that
broke the surface were never needed. `COLLAPSE_AGGRESSIVENESS` is 3 for that
reason, and higher values fail in a characteristic way worth recognising: agg 4
loses one handle, agg 5 loses 68, agg 6 and 7 turn the part into a hundred
pieces and add five percent to its volume.

The ladder now opens at the target rung and falls *back up* on failure, rather
than climbing from 0.85 every time. Both orders end in the same place; the
difference is eleven probes against one, and a probe on a 24-million-triangle
part is two minutes. The old order existed because the ladder used to stop at
the first failure -- which was the right response to a decimator that failed
early and often, and the wrong one once it stopped doing that.

## Topology is necessary and it is not sufficient

Watertight, one solid, no sealed void, right genus, volume to three decimals --
a mesh can hold every one of those while its shape drifts. Decimation moves
triangles onto the chords of the curves they spanned, and that takes as much off
one side as it adds to the other, so the volume never notices and the genus
cannot. Only distance notices.

`field_deviation` asks it: the signed field, sampled at the mesh's vertices and
across its faces, against the level the mesher actually took. Two numbers come
back and they are read differently.

- The **maximum** is about a voxel on any faithful mesh and stays there. Marching
  cubes cannot put a vertex inside a sharp concave corner, so the undecimated
  cowl reads 195 µm and reads 195 µm again after a ten-fold decimation. It
  bounds the meshing, not the decimation.
- The **mean** is what moves: 12 µm undecimated, 13 µm at keep 0.30, 18 µm at
  the floor. That is the number that would notice a part going quietly faceted.

Both are lower bounds, and the reason matters: the field is composed with `min`
and `max`, and the maximum of two distance functions is not a distance function
-- outside a concave seam it reads short. Tight over the smooth ground
decimation actually changes, optimistic in the corners. A measurement, not a
guarantee.

Sampling is budgeted and aimed. Evaluating the field costs one pass over the
profile per point and these profiles carry seventeen hundred segments, so asking
every vertex and centroid of the cowl takes a quarter of an hour. Half the
budget is a stride over the whole mesh and half goes to the largest triangles,
because chordal error grows with the triangle and making some triangles big is
the entire job of decimation.

## The print file is written twice

`full` and `compact`, from one meshing, differing only in that budget: 3 µm of
added rms drift for the reference and 12 µm for the compact one. Twelve microns
is a third of a layer and a fiftieth of the thinnest wall, so the compact file
is the same part to a printer and not to a measuring machine.

A coarser *voxel* is not the way to a smaller file and never will be. The 0.233
mm voxel is set by the 0.70 mm hot wall; below it the channels shred into
fragments, the watertight stage reports several hundred sealed voids, and what
comes out is not a cheaper engine but a broken one. Triangles come off by
collapse, and every collapse is checked.

## A print file is a claim, and it gets checked

`print_ready.py` writes 3MF, not STL: STL is a triangle soup with no units in
it, and every slicer guessing millimetres correctly is not the same as being
told. It refuses to write at all unless every part is watertight and no part has
a sealed void.

Sealed voids, not component counts. A part with internal cavities is not broken
-- the head disc is one solid plus two plenum surfaces, and those cavity
surfaces carry *negative* volume, so counting connected components rejects every
hollow part ever printed. What is actually broken is a cavity with no way out,
because it stays full of powder for ever. That check is what caught the injector
orifices no longer reaching their plenums after the manifolds were resized.

Two things that looked like geometry bugs and were not:

- Several hundred "sealed voids" in the cowl at a 0.7 mm test voxel. The channel
  is 0.8 mm wide. Testing a part below its own resolution floor shreds the
  channels into fragments and reports every fragment. Check the voxel before
  believing the finding.
- The head reporting three "components". One solid, two cavity surfaces.

## Marching cubes places its vertices in single precision

Two failures now, from the same cause at two scales.

The systematic one: a flat face normal to the axis lands on a lattice plane,
every sample there reads exactly zero, and marching cubes emits a sheet of
degenerate triangles with holes between them. `SURFACE_BIAS_MM` moves the level
a tenth of a micron off the zero set and that whole class goes away.

The sporadic one is not fixed by a bias, because it is not about where the
level is. Marching cubes interpolates in *index* units, in float32. Out at
index 1024 a float32 resolves about 6e-5 of an index, so a crossing at t = 2e-5
is placed exactly **on** the sample rather than beside it — and so is every
other edge into that sample. The weld then merges what marching cubes meant to
keep apart, and the surface pinches: one edge shared by four faces, no boundary
anywhere, an odd Euler characteristic, nothing whatsoever to see. The cowl did
it once in 24 million triangles, at a sample 95 nanometres above a channel
floor and so five below the level being meshed, and it took a lattice sweep to
prove it was arithmetic rather than geometry — sliding
the grid 0.08 mm made it vanish.

`hold_off_level` is the cure: no sample is allowed within a derived band of the
level, and samples inside it are pushed to the side they were already on, so no
cell changes classification and the surface moves by at most a micron. The band
comes from how far out the indices go and how fast the field can change between
neighbours, because those are the two things that set it — not from a constant
someone liked the look of.

The general lesson: **a fix that works at one grid alignment is not a fix.**
Sweep the lattice offset, the same way geometry gets swept over the parameter
range.

## Check the mesh on the grid the file is written on

A 3MF carries coordinates as decimal text, so writing quantises. Check in
memory and write afterwards and the two are different meshes: the first print
file built with the level guard reported no zero-area triangles anywhere in
memory and came back off disk with 416 of them.

The guard and the writer were arguing about the same decimal place --
`level_guard_mm` separates a vertex from its sample by about 1.5e-4 mm and the
writer used four decimals, a tenth of a micron. Both were individually correct.

Two rules follow. A writer's precision is part of the geometry, not a
formatting choice, and it has to sit *well under* the smallest separation the
mesher can produce rather than level with it. And quantise before checking, not
after: `snap_to_the_written_grid` rounds and then welds, so the vertices that
land on the same point merge and the faces that then have two identical corners
come out without opening the surface. What the gate reads is what a slicer
opens, down to the last decimal.

## The gate has a memory budget too, and it is easy to spend

Checking a mesh costs more than holding it. The cowl's print mesh is 900 MB;
one pass of the gate over it wanted 13.7 GB and the kernel killed the build
halfway through. None of that was the geometry -- it was numpy being asked for
the obvious thing four times over. `np.unique(edges, axis=0)` views seventy-two
million rows as a void dtype and sorts that, several copies deep, when the two
indices fit in one int64 and it could be an in-place 1-D sort. `v[f[:, 0]]`
gathers 1.7 GB of corners before any arithmetic happens, and four separate
routines wanted to do it.

So: pack indices into integers instead of sorting rows, count with `bincount`
instead of `unique`, do per-face vector maths a couple of million faces at a
time, and measure the peak on a mesh the size of the real one before running
anything that takes an hour. `resource.getrusage(...).ru_maxrss` and a
synthetic torus of the right size will tell you in ninety seconds what a print
build tells you in forty minutes.

## Voxel resolution

`geometry.voxel_size_mm` controls memory cubically. Halving it multiplies memory
roughly eightfold. Default to 0.2 mm while iterating. Do not drop below 0.05 mm
without being asked, and warn before you do.

## Scope

This used to stop at nozzle geometry. That restriction was lifted deliberately,
so the model now covers combustion conditions, delivered performance, injector
element sizing, and regenerative cooling as well. Do not reinstate the old limit
on the basis of a stale comment somewhere.

What matters instead is being straight about what each model *is*:

- `combustion_ref.py` is a parametric performance model with tabulated
  propellant data. There is no Gibbs minimisation and no thermochemical
  database. It is right about trends and about the identities; it is a fit in
  the wings. Real work substitutes CEA or RPA output through the spec, which
  every routine accepts as input for exactly that reason.
- `cooling_ref.py` stands on Bartz and Dittus-Boelter. Both are correlations.
  Bartz is quoted at plus or minus thirty percent and is being applied to an
  annular throat with a hydraulic diameter substituted for the throat diameter.
  A design that only just passes on Bartz has not passed.
- `injector_ref.py` sizes orifices and balances momentum. It does not model
  spray, vaporisation, or combustion stability. The chug check is a
  pressure-drop rule of thumb. Injectors get hot fired; this cannot substitute.

Those were once the whole model. It now also carries film cooling, thermal
barrier coatings, a startup transient, a thermostructural life estimate,
combustion stability screening, and thrust from the plug surface pressure
integrated against ambient. Each is a screening model of the same character:

- `film_effectiveness` is a slot correlation, and a real film is worse than any
  correlation because it is also burning. It is an upper bound.
- `structural_ref.py` is not finite elements. No creep, no ratchetting
  accumulation, no build-direction anisotropy, no residual stress from the
  print. Printed properties vary by more than the margins it reports.
- `transient_ref.py` has no axial conduction, no jacket heat capacity, no
  ignition overpressure, no two-phase coolant during priming.
- `stability_ref.py` computes where the chamber rings and applies the standard
  screens. There is no combustion response function and no eigenvalue problem.
  It cannot say an engine is stable. Nothing short of hot fire can.
- `plug_flow_ref.py` integrates the pressure the Angelino construction already
  gives. It is not a method of characteristics solve, and the base region behind
  a truncated spike is not modelled at all though it is worth several percent.

Still genuinely absent: base flow on the truncated plug, combustion response,
creep, and any real CFD. If a design needs one of those to close, say so instead
of tuning inputs until the report looks green.

`optimise_ref.py` ranks by Deb's rules rather than by a penalty function:
feasible beats infeasible, then smaller violation, then better objective. That
avoids inventing an exchange rate between "melted by 40 K" and "three seconds of
impulse", which is a trade nobody can make honestly. Keep it that way.

The habit that matters more than any scope line: when a model cannot make a
design work, report that. `engine_design.py` returns no cooling circuit at all
rather than the least-bad one, because a quietly returned melted engine is worse
than a refusal.

## Conventions

- All lengths in mm, all angles in radians internally, degrees only for display.
- PicoGK naming: `vox` for Voxels, `msh` for Mesh, `o` for objects, `f` for floats.
  Match the surrounding style when editing ShapeKernel-adjacent code.
- Every derived quantity gets logged through `Library.Log`. That log is how a
  future agent understands what happened.
