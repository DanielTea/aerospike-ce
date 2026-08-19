# aerospike-ce

A minimal Computational Engineering model in the LEAP 71 style: a JSON spec goes
in, a complete aerospike engine geometry comes out. No sketches, no feature tree,
no mouse.

Three parts, all derived: a hollow centrebody carrying the Angelino plug contour,
a cowl forming the outer side of the annular chamber and of the throat, and a
head disc that closes the chamber and provides the mounting face.

Built on [PicoGK](https://picogk.org) and the LEAP 71 ShapeKernel, both Apache-2.0.

## What this is and is not

**Is:** a working skeleton of the paradigm. Descriptive input, codified
engineering knowledge, generated voxel geometry, printable STL, and a validation
harness that lets a coding agent iterate without seeing the model.

**Is not:** Noyron. LEAP 71's actual value is thousands of hours of propulsion
knowledge encoded into a proprietary model. This repo contains one textbook
nozzle contour method. That gap is the whole ballgame, and no amount of
scaffolding closes it. What this gives you is the shape of the workflow, so you
can start filling it with your own domain knowledge.

**Also not:** a machine. This generates nozzle geometry for printed
demonstrators. Combustion, injectors, cooling, materials, and testing are all out
of scope and deliberately excluded from the model.

## Platform

PicoGK ships native runtimes for **win-x64** and **osx-arm64** only. Linux is not
supported out of the box. Check `vendor/PicoGK/native/` after cloning.

## Setup

```bash
# 1. clone, with the LEAP 71 stack
#    PicoGK and ShapeKernel are submodules under vendor/, pinned to
#    PicoGK v2.3.0 and ShapeKernel v2.1.0.
git clone --recurse-submodules https://github.com/DanielTea/aerospike-ce.git
cd aerospike-ce

#    already cloned without --recurse-submodules?
#    git submodule update --init --recursive

# 2. .NET 9 SDK
#    https://dotnet.microsoft.com/download

# 3. install the PicoGK runtime for your platform
#    https://github.com/leap71/PicoGK/releases
#    macOS: install the .pkg. Windows: run the installer.

# 4. python side -- needs Python 3.9 or newer (matplotlib 3.8 floor).
#    A system python older than that will fail to resolve requirements.txt.
cd validate
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 15 tests, all should pass
cd ..

# 5. build and run
dotnet run --project model -- spec/demo.json
```

The viewer window opens, STLs land in `out/`.

## The loop

```
edit spec/demo.json
        |
        v
pytest                              ->  invariants hold
        |
        v
python plot_contour.py              ->  out/contour.png     <- look at this
python plot_engine.py               ->  out/engine.png      <- and this
python mesh_export.py               ->  out/engine_3d.png   <- and this
                                        out/*.stl
        |
        v
dotnet run --project model          ->  voxel model, viewer, out/*.stl
        |
        v
slicer
```

The PNG step exists because a coding agent cannot see the 3D viewer. Skipping it
is how you end up with a confidently generated part that is quietly wrong.

There are two paths to STL. `validate/mesh_export.py` revolves the profiles into
a triangle mesh in pure Python: no native dependency, and it self-checks
watertightness, Euler characteristic, and mesh volume against the analytic
volume. `dotnet run` voxelises the same profiles in PicoGK, which is the real
pipeline and the one that can do boolean work the mesher cannot. Use the Python
path when you have no PicoGK runtime, or as a cross-check when you do.

## The chamber

The plug contour only describes the supersonic surface. Feeding it needs an
annular chamber, and the cowl wall through the contraction is the one piece of
this model that is genuinely easy to get wrong.

A straight cone from the chamber radius down to the lip looks entirely
reasonable. It pinches the duct to about 0.47 of the throat area roughly half a
millimetre aft of the spike shoulder, so the engine chokes there instead of at
the throat it was designed around, and the expansion ratio the spike was built
for is never reached. Nothing about the drawing looks wrong.

So the wall is solved rather than drawn. A monotonically decreasing area schedule
is prescribed from chamber to throat, and at each station the wall is placed at
the offset distance that delivers that area, with the offset direction fanning
about the shoulder corner. `out/engine.png` plots the resulting area
distribution, measured back off the built geometry rather than off the schedule.

The self-check: the last station of that schedule lands on the cowl lip to
machine precision, though nothing in the construction forces it. It falls out of
the area schedule and the Angelino contour agreeing about where the throat is.
That is `test_contraction_lands_on_the_lip`, and it is the assembly's counterpart
to `test_spike_closes_on_the_axis`.

## The nozzle physics

`validate/contour_ref.py` carries the full derivation of the Angelino
approximate method. The result:

```
alpha(M) = nu_e - nu(M) + mu(M)
xi       = (1 - sqrt(1 - sin(alpha) * M * eps(M) / eps_e)) / sin(alpha)
x        = r_e * xi * cos(alpha)
r        = r_e * (1 - xi * sin(alpha))
```

The self-check that proves it: at `M = M_e` the radicand vanishes, `xi = M_e`,
and `r = 0`. The ideal spike closes exactly on the axis. That is
`test_spike_closes_on_the_axis`, and it is the assertion to trust when you change
anything.

One subtlety worth knowing before you touch this: the throat is not a radial
annulus. It is the sonic line running from the cowl lip to the spike shoulder,
inclined at `nu_e` from the radial direction, and the lip sits *downstream* of
the shoulder rather than level with it. Measuring the throat radially at the lip
station gives 138 mm2 where the real area is 353 mm2. `PlugContour.lip_x` carries
the lip position and `throat_area_from_geometry` reproduces `pi*r_e^2/eps` from
it, which is what proves the placement.

Reference: G. Angelino, "Approximate Method for Plug Nozzle Design", AIAA
Journal 2(10), 1964.

## Suggested first changes

1. Sweep `truncate_fraction` from 1.0 down to 0.15 and watch the length collapse
   while the wall Mach barely moves. That is why no real engine flies a full spike.
2. Set `gamma` to 1.40 and re-run. That is your cold-gas demonstrator.
3. Sweep `contraction_ratio` from 2.0 to 5.0 and watch `out/engine.png`. The
   chamber grows, the area schedule steepens, and the throat does not move.
4. Drop `wall_thickness_mm` to 1.0 and raise it to 4.0. At 4.0 the naive offset
   at the spike shoulder folds over completely; the erosion in `engine_ref.py`
   prunes it, and `test_wall_is_never_thinner_than_specified` is what stops that
   pruning from quietly eating the wall.
5. Then the real exercise: derive `contraction_ratio` and `converging_length_mm`
   from an actual requirement rather than accepting them as input. That is the
   step where this stops being a shape generator and starts being a model.

## License

Apache-2.0, matching the upstream LEAP 71 stack. See `LICENSE`.

PicoGK and LEAP 71 ShapeKernel are vendored as submodules and remain under their
own Apache-2.0 licences and copyright.
