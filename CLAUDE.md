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
7. `dotnet run --project model -- spec/demo.json` and compare the logged numbers
   against the Python output. They must agree to at least four significant figures.

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
| `validate/cooling_ref.py` | Bartz, fins, channel search | correlations, not CFD |
| `validate/engine_design.py` | the whole pipeline, one spec in | derives the coolant split |
| `validate/mesh_solid.py` | SDF + marching cubes | for anything not axisymmetric |
| `validate/mesh_export.py` | revolve to STL, cutaway PNG | exact, axisymmetric only |
| `validate/test_*.py` | invariants | add a test before adding physics |
| `validate/plot_contour.py` | contour PNG | your eyes on the nozzle |
| `validate/plot_engine.py` | assembly PNG + area schedule | your eyes on the engine |
| `validate/plot_cooled.py` | section views through the field | your eyes on the channels |
| `validate/export_cooled.py` | cooled STLs + topology check | |
| `model/GasDynamics.cs` | isentropic + Prandtl-Meyer | mirror of the Python |
| `model/PlugContour.cs` | contour construction | mirror of the Python |
| `model/EngineAssembly.cs` | assembly profiles | mirror of `engine_ref.py` |
| `model/Combustion.cs` | chamber conditions | mirror of `combustion_ref.py` |
| `model/Injector.cs` | element sizing | mirror of `injector_ref.py` |
| `model/Cooling.cs` | cooling solve and search | mirror of `cooling_ref.py` |
| `model/CooledGeometry.cs` | SDFs for channels and orifices | mirror of `mesh_solid.py` |
| `model/SpikeGeometry.cs` | PicoGK voxel assembly | geometry only, no physics |
| `model/Program.cs` | orchestration, logging, export | no engineering logic |
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

Still genuinely absent, and worth saying so rather than pretending otherwise:
film cooling, thermal barrier coatings, transient startup, structural analysis
of a hot wall against a cold jacket, combustion stability, and any
method-of-characteristics treatment of the plug's altitude compensation. If a
design needs one of those to close, say so instead of tuning inputs until the
report looks green.

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
