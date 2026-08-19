# Working rules for this repository

You are working on a Computational Engineering model: a program that generates a
rocket nozzle from a specification. There is no interactive CAD. Nobody draws
anything. Read this before making changes.

## The one rule that matters

**You cannot see the geometry.** The PicoGK viewer is invisible to you. Never
claim a part "looks right". Verify through the two channels you actually have:

1. `validate/` — pytest assertions on the maths and on physical invariants.
2. `out/contour.png` — a rendered plot you can read with your image tools.

If a change is not covered by either channel, add coverage before making it.

## Order of work

The Python in `validate/` is the source of truth for the physics. The C# in
`model/` is a port. When changing engineering behaviour:

1. Change `validate/contour_ref.py`.
2. Add or update a test in `validate/test_contour.py`.
3. Run `pytest` from inside `validate/`. Make it pass.
4. Run `python plot_contour.py --spec ../spec/demo.json --out ../out`.
5. **Actually look at `out/contour.png`.** Read the image. Do not skip this.
6. Only then port the change to `model/PlugContour.cs` or `model/GasDynamics.cs`.
7. `dotnet run --project model -- spec/demo.json` and compare the logged numbers
   against the Python output. They must agree to at least four significant figures.

Never edit the C# physics without a matching Python change. Divergence between
the two is the failure mode this structure exists to prevent.

## What goes where

| Path | Contains | Rule |
|---|---|---|
| `spec/*.json` | descriptive input | the only file a *user* edits |
| `validate/contour_ref.py` | authoritative physics | change here first |
| `validate/test_contour.py` | invariants | add a test before adding physics |
| `model/GasDynamics.cs` | isentropic + Prandtl-Meyer | mirror of the Python |
| `model/PlugContour.cs` | contour construction | mirror of the Python |
| `model/SpikeGeometry.cs` | PicoGK voxel assembly | geometry only, no physics |
| `model/Program.cs` | orchestration, logging, export | no engineering logic |
| `out/` | generated artifacts | never commit, never hand-edit |

Physics never leaks into `SpikeGeometry.cs`. Geometry never leaks into
`PlugContour.cs`. If you find yourself computing a Mach number inside a voxel
routine, stop and move it.

## Voxel resolution

`geometry.voxel_size_mm` controls memory cubically. Halving it multiplies memory
roughly eightfold. Default to 0.2 mm while iterating. Do not drop below 0.05 mm
without being asked, and warn before you do.

## Things that are out of scope

Do not add combustion modelling, injector design, propellant selection, cooling
channel sizing, or thrust prediction. This model generates nozzle geometry for
printed demonstrators. If asked to extend into those areas, say so and stop.

## Conventions

- All lengths in mm, all angles in radians internally, degrees only for display.
- PicoGK naming: `vox` for Voxels, `msh` for Mesh, `o` for objects, `f` for floats.
  Match the surrounding style when editing ShapeKernel-adjacent code.
- Every derived quantity gets logged through `Library.Log`. That log is how a
  future agent understands what happened.
