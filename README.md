# aerospike-ce

A minimal Computational Engineering model in the LEAP 71 style: a JSON spec goes
in, an aerospike plug-nozzle geometry comes out. No sketches, no feature tree, no
mouse.

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
# 1. .NET 9 SDK
#    https://dotnet.microsoft.com/download

# 2. clone the LEAP 71 stack into vendor/
git init
mkdir -p vendor
git submodule add https://github.com/leap71/PicoGK.git vendor/PicoGK
git submodule add https://github.com/leap71/LEAP71_ShapeKernel.git vendor/LEAP71_ShapeKernel
git submodule update --init --recursive

# 3. install the PicoGK runtime for your platform
#    https://github.com/leap71/PicoGK/releases
#    macOS: install the .pkg. Windows: run the installer.

# 4. python side
cd validate
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
python validate/plot_contour.py     ->  out/contour.png    <- look at this
        |
        v
pytest                              ->  invariants hold
        |
        v
dotnet run --project model          ->  out/*.stl
        |
        v
slicer
```

The PNG step exists because a coding agent cannot see the 3D viewer. Skipping it
is how you end up with a confidently generated part that is quietly wrong.

## The physics

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

Reference: G. Angelino, "Approximate Method for Plug Nozzle Design", AIAA
Journal 2(10), 1964.

## Suggested first changes

1. Sweep `truncate_fraction` from 1.0 down to 0.15 and watch the length collapse
   while the wall Mach barely moves. That is why no real engine flies a full spike.
2. Set `gamma` to 1.40 and re-run. That is your cold-gas demonstrator.
3. Add a hollow interior to the spike with a wall thickness parameter. PicoGK
   offsets make this a two-line change, and it is where voxel modelling starts
   paying for itself over b-rep.
4. Then the real exercise: replace the straight cowl with something derived from
   an actual requirement rather than a constant.
