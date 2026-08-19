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

1. Change `validate/contour_ref.py` (nozzle physics) or `validate/engine_ref.py`
   (assembly geometry).
2. Add or update a test in `validate/test_contour.py`, `validate/test_engine.py`,
   or `validate/test_mesh.py`.
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
| `validate/test_contour.py` | contour invariants | add a test before adding physics |
| `validate/test_engine.py` | assembly invariants | add a test before adding geometry |
| `validate/test_mesh.py` | mesh closure and topology | |
| `validate/plot_contour.py` | contour PNG | your eyes on the nozzle |
| `validate/plot_engine.py` | assembly PNG + area schedule | your eyes on the engine |
| `validate/mesh_export.py` | revolve to STL, cutaway PNG | fallback path, no PicoGK needed |
| `model/GasDynamics.cs` | isentropic + Prandtl-Meyer | mirror of the Python |
| `model/PlugContour.cs` | contour construction | mirror of the Python |
| `model/EngineAssembly.cs` | assembly profiles | mirror of `engine_ref.py` |
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

## Voxel resolution

`geometry.voxel_size_mm` controls memory cubically. Halving it multiplies memory
roughly eightfold. Default to 0.2 mm while iterating. Do not drop below 0.05 mm
without being asked, and warn before you do.

## Things that are out of scope

Do not add combustion modelling, injector design, propellant selection, cooling
channel sizing, or thrust prediction. This model generates nozzle geometry for
printed demonstrators. If asked to extend into those areas, say so and stop.

The line is between *geometry* and *the engineering that would size it*. An
annular chamber volume set by a stated contraction ratio is geometry, and is in.
A chamber sized from a chosen propellant and chamber pressure is not. The head
disc is a blank closure for the same reason: give it an orifice pattern and it
becomes injector design, which is out. When a request straddles the line, build
the geometry, say plainly which part you did not build, and why.

## Conventions

- All lengths in mm, all angles in radians internally, degrees only for display.
- PicoGK naming: `vox` for Voxels, `msh` for Mesh, `o` for objects, `f` for floats.
  Match the surrounding style when editing ShapeKernel-adjacent code.
- Every derived quantity gets logged through `Library.Log`. That log is how a
  future agent understands what happened.
