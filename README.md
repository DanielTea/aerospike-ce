# aerospike-ce

A Computational Engineering model in the LEAP 71 style: a JSON spec goes in, a
complete regeneratively cooled aerospike engine comes out. No sketches, no
feature tree, no mouse.

Three printed parts, all derived: a hollow centrebody carrying the Angelino plug
contour, a cowl forming the outer side of the annular chamber and of the throat,
and a head disc that closes the chamber, carries the injector and provides the
mounting face. Both walls are regeneratively cooled by the fuel.

```bash
python validate/engine_design.py --spec spec/regen.json
```

prints the whole design: throat geometry, chamber conditions, thrust and Isp,
the injector pattern, a cooling circuit for each wall with its margins, the
thermal strain and cycle life of each hot wall, the startup transient, the
chamber acoustic modes, and the thrust coefficient against altitude.

Built on [PicoGK](https://picogk.org) and the LEAP 71 ShapeKernel, both Apache-2.0.

## The printable engine

Three solids in one 3MF, 20.75 kg in GRCop-42, meshed at 0.233 mm with all 556
cooling channels and 96 injector orifices in it. Every part watertight, no
cavity without a way out, nothing a slicer refuses, and the distance from the
field it was meshed from measured rather than assumed.

It is written twice, from one meshing, differing only in how much shape error
decimation was allowed to spend:

| | triangles | size | rms from the field |
|---|---|---|---|
| `regen-spike-75.3mf` | 15,428,890 | 191 MB | 6.9 – 13.0 µm |
| `regen-spike-75-compact.3mf` | 5,488,032 | 72 MB | 12.1 – 18.4 µm |

Twelve microns is a third of a layer at 30 µm and a fiftieth of the thinnest
wall, so the compact file is the same part to a printer. Both are built and
gated by the
[`print file` workflow](https://github.com/DanielTea/aerospike-ce/actions/workflows/print-file.yml)
and downloadable from its latest green run. See
[docs/print](docs/print/README.md) for what is checked, what is not, and what is
wrong with the head manifolds.

## What this is and is not

**Is:** a working skeleton of the paradigm. Descriptive input, codified
engineering knowledge, generated voxel geometry, printable STL, and a validation
harness that lets a coding agent iterate without seeing the model.

**Is:** a model that will tell you your design does not work. `spec/demo.json`
cannot be regeneratively cooled by its own fuel flow at any channel geometry, and
the model reports that rather than returning the least-bad circuit. Getting a
refusal out of a generator is the point of building one.

**Is not:** Noyron. LEAP 71's actual value is thousands of hours of propulsion
knowledge encoded into a proprietary model. This repo contains one textbook
nozzle contour method. That gap is the whole ballgame, and no amount of
scaffolding closes it. What this gives you is the shape of the workflow, so you
can start filling it with your own domain knowledge.

**Also not:** a machine. Every model here is a screening model, and the
honest description of each is in `CLAUDE.md`. `combustion_ref.py` is a
parametric fit with tabulated propellant data, not a Gibbs minimisation --
substitute CEA or RPA output through the spec, which every routine accepts for
exactly that reason. The cooling stands on Bartz, quoted at plus or minus thirty
percent and applied here to an annular throat. The injector sizes orifices; it
does not model spray, vaporisation or combustion stability.

Genuinely absent: base flow behind the truncated plug (worth several percent),
any combustion response function, creep, and any real CFD. If a design needs one
of those to close, that is worth saying rather than tuning inputs until the
report looks green.

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

# 3. the PicoGK native runtime is vendored in the submodule -- there is
#    nothing to install. The csproj copies it next to the binary.
#    vendor/PicoGK/native/<rid>/

# 4. python side -- needs Python 3.9 or newer (matplotlib 3.8 floor).
#    A system python older than that will fail to resolve requirements.txt.
cd validate
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 301 tests, all should pass
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
python pipeline.py                  ->  physics       the numbers
                                        geometry      the shapes, and the seam
                                        printability  can a machine build them
                                        watertight    the solid they close into
                                        slicing       what a slicer reads
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
python print_ready.py               ->  the 3MF, or a refusal
python verify_print_file.py         ->  the same gates, re-derived from disk
        |
        v
slicer
```

`pipeline.py` is the gate, and `pytest` is one stage of it. Every test module in
the repository belongs to exactly one stage and the pipeline refuses to run if
one belongs to none. [docs/pipeline](docs/pipeline/README.md) lists every gate,
what it proves, and which shipped defect it exists because of.

The PNG step exists because a coding agent cannot see the 3D viewer. Skipping it
is how you end up with a confidently generated part that is quietly wrong.

There are two paths to STL. `validate/mesh_export.py` revolves the profiles into
a triangle mesh in pure Python: no native dependency, and it self-checks
watertightness, Euler characteristic, and mesh volume against the analytic
volume. `dotnet run` voxelises the same profiles in PicoGK, which is the real
pipeline and the one that can do boolean work the mesher cannot. Use the Python
path when you have no PicoGK runtime, or as a cross-check when you do.

## Two specs

`spec/demo.json` is the geometry demonstrator: 30 mm exit radius, uncooled, the
smallest thing that shows the paradigm. `spec/regen.json` is a working engine at
75 mm, 25 bar, LOX/methane, about 7.6 kN at sea level, with both walls cooled.

The gap between them is instructive. Ask for cooling on the demo spec and you get:

```
heat load 241 kW exceeds what the fuel can absorb (189 kW). No channel geometry
fixes that: the engine needs more mass flow, a higher chamber pressure, film
cooling, or to be larger.
```

That is not a modelling failure, it is why small engines are ablative or film
cooled and large ones are regenerative. Heat load grows with wetted area while
coolant capacity grows with mass flow, and the ratio only comes right with size
and chamber pressure. `regen.json` also runs at mixture ratio 2.4 against a c*
peak of 3.4: deliberately fuel-rich, buying coolant mass flow at the cost of
about 4 s of specific impulse. Every regeneratively cooled engine makes that
trade.

## Cooling

Channels are not sized by picking a coolant velocity. On a small engine that is
actively misleading: velocity fixes the flow area, the flow area fixes the
channel count, and a low count on a fixed circumference leaves lands wider than
the channels, at which point the fin efficiency of the ribs decides the wall
temperature no matter how fast the coolant goes. Chasing velocity produced a
0.077 mm deep channel and a 236 bar pressure drop on a 20 bar chamber.

So `cooling_ref.py` searches instead. It packs the circumference at each
candidate width and land, sweeps depth, hot wall and material, and keeps only
what survives on wall temperature, coolant temperature, pressure drop and
turbulence. The winner for `regen.json` lands at a land-to-channel ratio of about
one, which is where real regen chambers sit.

The fuel split between the two circuits is derived, not specified. The
centrebody carries the chamber wall *and* the whole plug surface, so it needs
about 55 percent; an even split starves it while the cowl sits comfortable.

## The injector

Momentum-balanced unlike doublets on the annular face. The two streams do not
carry equal momentum, so splitting the included angle evenly throws the resultant
about 11 degrees off axis, which streaks the chamber and drives one side of the
wall hot. The angles are solved to cancel the transverse components instead:

```
a_fuel = atan2( m_ox sin(total),  m_fuel + m_ox cos(total) )
```

The heavier stream takes the shallower angle, which is the physical result, and
the resultant lands on the axis to machine precision rather than by tuning.

An annular chamber gives an injector very little radial room, so the pattern is
usually constrained by the face rather than by the flow. `fit_injector_to_face`
searches element count and impingement geometry and reports what fits, or says
plainly that nothing does.

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

## What else the model knows

**Film cooling.** Eight percent of the fuel runs along the wall. It is what lets
the chamber be long enough for a sane characteristic length and still be
coolable, and it is not free: that fuel does not burn at the design mixture
ratio, so it is charged at a reduced combustion efficiency and shows up as lost
specific impulse.

**Thermal barrier coatings.** Off by default. YSZ is about 1.0 W/(m K) against
290 for GRCop-42, so a tenth of a millimetre is worth three millimetres of
copper -- but the coating surface then runs at 2000 K where the bare metal ran
at 980, and the design search rejects any coating that would spall.

**Thermal strain and life.** A regen hot wall is *expected* to exceed yield
every firing; requiring it to stay elastic would reject every chamber ever
flown. What matters is the plastic strain per cycle, so the criterion is
Coffin-Manson life against the cycles you need, not elasticity.

**Startup.** The wall diffuses heat in about ten milliseconds, far faster than
any pressure ramp, so the wall is not the lag. The sequencing is: gas arriving a
quarter second before the coolant overshoots the wall by several hundred kelvin
and takes it past its limit. That is a valve sequencing requirement, not a
preference.

**Stability screening.** Chamber acoustics -- first tangential, longitudinal,
radial -- plus L\*, residence time and injection stiffness. This is where the
model discovers that the contraction ratio is not a free parameter: with the
chamber length it sets L\*, which has to land between about 0.6 and 1.5 m or the
propellant leaves before it has burned.

**Altitude compensation.** The ideal bell thrust coefficient is the wrong number
for a plug, and wrong in the direction that matters. Integrating the pressure
the Angelino construction already gives on the spike surface, and stopping where
the surface falls below ambient because the plume detaches there, shows the plug
beating the bell at sea level and converging with it at altitude. That
truncation *is* the compensation mechanism, expressed geometrically.

## What to connect where

The engine is plumbed, not just shaped. `engine_design.py` prints the whole port
schedule -- bore, fitting size, flow, pressure and temperature for every
connection -- but the routing is worth stating plainly, because regenerative
cooling makes it less obvious than it looks:

| port | fluid | goes where |
|---|---|---|
| `fuel_in_cowl` | fuel | radial boss near the throat. Feeds the cowl jacket; the fuel then runs *forward* through the cowl channels to the injector |
| `fuel_in_spike` | fuel | axial, down the centre bore. Runs aft to the spike feed ports, then forward through the spike channels |
| `ox_in` | oxidiser | axial into the dome. Does no cooling, stays cold |
| `igniter` | gas | radial into the chamber annulus. Radial because the centre bore is carrying fuel |
| `pc_tap` | - | chamber pressure, opposite the igniter |
| `t_coolant_out`, `p_coolant_out` | - | head manifold instrumentation |

**There is no coolant outlet.** The fuel *is* the coolant: it leaves both jackets
into the injector manifold and gets burned. A port taking it out of the engine
would be throwing away the propellant along with the heat. Both fuel legs are
split in the ratio the two jackets actually need, which the thermal model
derives -- the centrebody carries the chamber wall *and* the whole plug surface,
so it takes about 55 percent, and an even split starves it.

The head joint is sized against the separating load rather than drawn: chamber
pressure over the sealed annulus plus the thrust reaction, with enough preload
that the joint never gaps. Working that through is what caught the flange being
half a millimetre too narrow to take its own fasteners.

## Printability, and which way up

Powder-bed fusion builds upward, so every downward face has to hold itself up
and every internal void needs a way for the powder out. `printability_ref.py`
checks both and picks the build direction; on this engine it is head down, spike
up, and that is not a matter of taste. Built that way the cowl and centrebody
outer surfaces both narrow as they rise, and the cooling channels run parallel
to the build direction, so several hundred of them are vertical tunnels with no
roof at all. Inverted, all of that inverts with it.

The distinction that matters is whether a support can *reach* a facet. An
overhang on an outer surface takes a support that gets broken off afterwards. An
overhang inside a sealed cavity takes one that stays there for ever. So the
checker casts a ray down the build direction and asks whether a column could
stand there.

That is what forced the centrebody's cavity to change shape. A constant-thickness
offset of the spike is the obvious way to hollow it, and it is unbuildable: an
internal void that narrows as it rises hangs material over nothing, which was 124
unsupported facets at the shoulder plus a flat internal ceiling 30 mm across. The
cavity now closes on the axis in a cone and is clamped so it never narrows faster
than the process angle, propagated backward from the tip so the wall only ever
gains material. It costs about ten percent more mass, which is what printability
costs and is worth seeing.

## Optimisation

```bash
python validate/optimise_ref.py --spec spec/regen.json --generations 25     --objective isp_sl --thrust-floor 7000 --out spec/optimised.json
```

Nine variables across the nozzle, chamber, operating point and structure, driven
by a `(mu + lambda)` evolution strategy with self-adaptive step sizes. Gradients
are the wrong tool here: ask for slightly more chamber pressure and the cooling
search may find nothing at all, so the space has cliffs rather than slopes, and
much of it is simply infeasible.

Constraints use Deb's rules rather than penalty weights -- feasible beats
infeasible, then smaller violation, then better objective. That avoids inventing
an exchange rate between "melted by 40 K" and "three seconds of impulse", and it
lets the search start infeasible and walk itself in.

Cooling closing, the wall surviving its cycles, the part being printable and the
joint bolting up are all constraints, not terms to be traded away.

## Meshing, and where it runs out

Two paths. `mesh_export.py` revolves a meridional profile: exact, cheap, and
incapable of anything that varies with angle. `mesh_solid.py` evaluates a signed
distance field and runs marching cubes, which handles the channels and orifices.
`model/CooledGeometry.cs` carries the same distance functions and hands them
straight to PicoGK, which renders any bounded implicit into voxels.

Marching cubes needs about three samples across the narrowest feature, so the
voxel size is derived from the part rather than taken as input. That makes the
narrowest feature a cost the whole engine pays: left free, the cooling search
picks a 0.4 mm channel, which forces 0.133 mm voxels everywhere. Pinning the
process floor at 0.7 mm gives 0.233 mm instead -- five times fewer voxels --
for 170 K of wall temperature the design had to spare. The field is built one
slab at a time, so memory is bounded by the slab rather than by the resolution.

The check that matters is the Euler characteristic against what the features
imply -- one handle per cooling channel, one per injector orifice, on top of the
base topology of the revolved profile. A channel that has broken through its
wall, merged with its neighbour, or been closed by over-aggressive decimation
all leave a mesh that is still watertight and still looks right; the genus is
what catches them.

Watertight is not the same as printable, and the gap is wider than it looks.
Four separate defects have shipped past a mesh reporting watertight at the
correct genus: a triangle with no area, a duplicated face, a patch wound inside
out, and a vertex where the surface pinches against itself. Each is an ordinary
face by index -- two neighbours on every edge, so the edge arithmetic is content
-- and each is something a slicer reads and refuses. That is what the pipeline's
slicing stage is for.

Marching cubes also places its vertices in single precision, in *index* units.
Out at index 1024 that resolves about 6e-5 of an index, so a crossing a few
nanometres off a sample is placed exactly on it, and so is every other edge into
that sample; the weld then merges what the mesher meant to keep apart. The cowl
did that once in 24 million triangles. `hold_off_level` keeps every sample a
derived distance clear of the level, on the side it was already on, so no cell
changes classification and the surface moves by at most a micron.

Decimation then runs as far as the topology survives and no further. On the head
that is 5 percent of the triangles at exactly the right genus; at 3 percent the
quadric collapse closes an orifice and the result is rejected.

The channel geometry is *also* verified against the distance field directly in
`test_cooled_geometry.py` -- channel count round the circumference, hot wall
thickness, the wall never breached -- because that check is exact and costs
nothing, and a mesh is a poor instrument for a question the field can answer.

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
5. Sweep `operation.mixture_ratio` on `regen.json` from 2.2 to 3.4 and watch the
   cooling margins collapse as the specific impulse rises. That trade is the
   whole reason the demo runs fuel-rich.
6. Raise `cooling.min_channel_width_mm` from 0.7 back to 0.4 and watch the
   whole engine's voxel size fall from 0.233 mm to 0.133 mm. The channel gets
   better at cooling and worse at existing; the trade is manufacturing against
   thermal margin, and it is decided in the spec rather than in the mesher.
7. Then the real exercise: derive `contraction_ratio`, `converging_length_mm` and
   the mixture ratio from an actual requirement rather than accepting them as
   input. That is the step where this stops being a shape generator and starts
   being a model.

## License

Apache-2.0, matching the upstream LEAP 71 stack. See `LICENSE`.

PicoGK and LEAP 71 ShapeKernel are vendored as submodules and remain under their
own Apache-2.0 licences and copyright.
