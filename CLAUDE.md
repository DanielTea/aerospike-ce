# Working rules for this repository

You are working on a Computational Engineering model: a program that generates a
rocket nozzle from a specification. There is no interactive CAD. Nobody draws
anything. Read this before making changes.

## The one rule that matters

**You cannot see the geometry.** The PicoGK viewer is invisible to you. Never
claim a part "looks right". Verify through the channels you actually have:

1. `validate/` — pytest assertions on the maths and on physical invariants.
2. `out/contour.png`, `out/engine.png`, `out/engine_3d.png` — rendered plots you
   can read with your image tools.
3. Mesh closure — `validate/mesh_export.py` reports watertightness, Euler
   characteristic, and mesh volume against the analytic volume. A solid that
   does not close fails here without anyone looking at it.

If a change is not covered by one of these, add coverage before making it.

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
| `validate/mesh_solid.py` | SDF + marching cubes | for anything not axisymmetric |
| `validate/mesh_export.py` | revolve to STL, cutaway PNG | exact, axisymmetric only |
| `validate/test_*.py` | invariants | add a test before adding physics |
| `validate/plot_contour.py` | contour PNG | your eyes on the nozzle |
| `validate/plot_engine.py` | assembly PNG + area schedule | your eyes on the engine |
| `validate/plot_cooled.py` | section views through the field | your eyes on the channels |
| `validate/export_cooled.py` | cooled STLs + topology check | |
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
