# The print file

A checked, print-resolution model of the whole engine: three solids in one 3MF,
dimensioned in millimetres, with every cooling channel, injector orifice and
mounting hole in it.

> **The published v0.1.0 asset is superseded.** It was built before the slicing
> gates existed and carries **6,334 triangles with no area** -- 793 on the
> centrebody, 5,300 on the cowl, 241 on the head. Every part in it is watertight
> at the right genus, which is exactly the problem: a zero-area triangle breaks
> no edge pairing, contributes no volume and moves no Euler characteristic, so
> nothing in the gate at the time could see it. A slicer can, and reports it as
> missing or extraneous surfaces. Rebuild rather than download it.

It is not a file in the tree. It is 409 MB, GitHub refuses a single file over
100, and a generated artifact of that size does not belong in history anyway:
the thing worth versioning is the generator, which is here.

**Download:** the latest green run of the
[`print file` workflow](https://github.com/DanielTea/aerospike-ce/actions/workflows/print-file.yml)
carries it as an artifact named `print-file`. The job builds it, runs the
release gate on the written file, and only then uploads, so anything on that
page passed its own checks; a run whose gate failed has a `release-report`
saying why and no print file at all. Artifacts keep for 30 days, and rebuilding
is one button.

A release asset would do the same job, and is the better home once there is a
version worth naming; the workflow exists because it needs no permission to
publish anything. It takes about half an hour on a GitHub runner -- roughly
twice as fast as the four-core machine it was developed on, for identical
triangle counts, genus and volumes.

```bash
gh workflow run "print file"                    # once this is on the default branch
git tag print-$(git rev-parse --short HEAD) && git push origin --tags
git push -f origin HEAD:build-print-file        # where tags are not allowed
```

Build it locally with:

```bash
cd validate
python print_ready.py --spec ../spec/regen.json --out ../docs/print
python pipeline.py --file ../docs/print/regen-spike-75.3mf --json ../out/release.json
```

The first command builds it, refusing to write rather than writing something
plausible. The second is the release gate: it reopens the file as a stranger
would and re-derives every property from what is on disk, across all five
stages. Checking the meshes in memory and then writing them leaves the writer
-- the part a slicer actually sees -- unchecked, and that gap is not
theoretical: it hid 416 zero-area triangles in the first build of this file
that had none in memory.

## What is in it

Meshed at 0.233 mm, the voxel the narrowest feature asks for. Every part
watertight, no boundary or non-manifold edges, no cavity without a way out, no
zero-area triangle, no duplicated face, no patch wound inside out and no vertex
where the surface pinches -- checked on the grid the file is written on.

| part | triangles | genus | volume | mass | decimation |
|---|---|---|---|---|---|
| centrebody | 8,221,176 | 328 | 402.49 cm3 | 3.524 kg | keep 0.50 |
| cowl | 20,357,084 | 393 | 392.84 cm3 | 3.440 kg | keep 0.85 |
| head | 7,726,426 | 281 | 1574.58 cm3 | 13.787 kg | keep 0.70 |

36,304,686 triangles, 20.75 kg, 409 MB. The equivalent binary STL would be
1815 MB, and would not say what unit it was in.

Vertices are written to six decimal places -- a nanometre -- rather than four,
and the mesh is snapped onto that grid and welded there *before* it is checked.
Four decimals is a tenth of a micron, which is far under any printer and not
under the mesher: `mesh_solid.level_guard_mm` separates a marching-cubes vertex
from its sample by about 1.5e-4 mm, so writing at 1e-4 rounded away the
separation the guard exists to create. That is the 416.

The head is two thirds of the mass because it is a 46 mm slab of copper,
thickened to contain its manifolds. If mass matters, that is the first place to
look, and it is a design change rather than a meshing one.

Decimation stops where the topology does, and on this engine that is early. The
centrebody survives halving; the head reaches 70 percent and the cowl only 85,
because both now carry 1.2 mm transfer passages and a quadric collapse closes
them before it closes anything else. Every step is checked and the result is
the last one that survived, so the file is as large as the geometry honestly
is: a regeneratively cooled engine at a resolution that holds its 556 cooling
channels *and* its feed paths is tens of millions of triangles.

## Why 3MF and not STL

STL is a triangle soup with no units in it. Every slicer guesses millimetres and
is usually right, which is not the same as being told; a part that silently
arrives at a twenty-fifth of its size is a wasted build. 3MF states its unit,
carries each part as a named object, and zips -- which matters when the geometry
has several hundred cooling channels in it.

## What is checked before it is written

`print_ready.py` refuses to write rather than writing something plausible, and
[`pipeline.py`](../pipeline/README.md) applies the same gates as its `watertight`
and `slicing` stages:

- **Watertight.** No boundary edges, no non-manifold edges.
- **No sealed void.** Every internal cavity has a way out.
- **One solid.** No second positive-volume surface -- no fragment attached to
  nothing.
- **Nothing a slicer refuses.** No zero-area triangle, no duplicated face, no
  patch wound inside out, no vertex where the surface pinches against itself.
  All four pass every check in the list above them: they are ordinary faces by
  index, with two neighbours on every edge.
- **Resolution.** The voxel is derived from the narrowest feature in the part,
  not chosen. Marching cubes needs about three samples across a feature.
- **Volume.** The mesh and the field agree, by two different routes through the
  same field -- the divergence theorem over the triangles against the occupancy
  integrated cell by cell.
- **Decimation that stops.** Quadric collapse reads a sub-millimetre channel as
  noise against a smooth wall and closes it, leaving a mesh that is still
  watertight and still looks like an engine. Every step is checked against the
  topology and the volume, and the result is the last one that survived.

The slicing gates are there because the first published file did not have them.
It carried 6,334 triangles with no area -- 0.017 percent of 36.3 million, and
enough for a slicer to report missing surfaces -- through a build that reported
every part watertight at the right genus. A zero-area triangle contributes no
volume, breaks no edge pairing and changes no Euler characteristic; there was
nothing in the gate that could see it.

The sealed-void check is the one worth explaining. A part with internal cavities
is not broken -- this head disc is one solid plus two plenum surfaces, and those
cavity surfaces carry negative volume, so counting connected components would
reject every hollow part ever printed. What is actually broken is a cavity with
no way out, because it stays full of powder for ever: unaccounted mass that can
shake loose later, and in a manifold it blocks the thing the cavity exists for.
That check is what caught the injector orifices no longer reaching their plenums
after the manifolds were resized.

## Which way up

Head down, spike up, building along +x. That is derived, not chosen, and
inverting it inverts all of it: built this way both outer surfaces narrow as
they rise, and the cooling channels run parallel to the build direction, so
several hundred of them are vertical tunnels with no roof at all.

The 24 overhangs that do need support are all on the cowl's outer surface, where
a support can be reached and broken off afterwards. There are no unsupportable
facets -- nothing needing a support inside a sealed cavity, which would stay
there for ever.

## Printing it

Copper alloy, laser powder-bed fusion. The thermal model assumes GRCop-42
(8756 kg/m3, 290 W/(m K)); the hot walls run at 763 K and 844 K against a
1350 K melting point, and that margin is a GRCop margin. Printing it in
stainless or Inconel does not give you this engine.

Both hot walls are 0.7 mm thick with 0.8 x 1.2 mm channels behind them. That is
at the floor of what powder-bed fusion holds and what an inspection resolves,
and it is why `cooling.min_channel_width_mm` is pinned in the spec rather than
left to the search, which would take 0.4 mm and a 0.133 mm voxel for the whole
engine.

## The feed paths

Worth reading before you connect anything, because the routing is not obvious
and an earlier version of this model got it comprehensively wrong.

The fuel does two jobs in series, which is what "regenerative" means:

```
fuel  -> cowl jacket   -.
                         >- head fuel manifold -> 48 fuel orifices
fuel  -> bore -> spike -'

ox    -> ox dome -> 48 radial ports -> 48 oxidiser orifices
```

There is no coolant outlet. The fuel *is* the coolant; it leaves both jackets
into the injector manifold and is burned. A port taking it out of the engine
would throw away the propellant along with the heat.

| inlet | where | flow |
|---|---|---|
| fuel, cowl jacket | radial boss, x = −26.4, r = 118 | 414 g/s, 37.5 bar, 111 K, AN-8 |
| fuel, spike jacket | axial on the axis, x = −203 | 553 g/s, 37.5 bar, 111 K, AN-10 |
| oxidiser | 6 radial bores through the rim, x = −168.6, r = 122.8 | 6 × 387 g/s, 37.5 bar, 90 K, AN-6 |

The oxidiser goes in through the rim rather than the end face because the dome
sits outboard: an axial bore on the end face at a radius clear of the orifice
rings misses the dome entirely and runs on into the injector.

### Why the head is laid out the way it is

The injector's two orifice rings are 7 mm apart radially — set by the
impingement geometry, not chosen. Two manifolds each straddling its own ring,
with a wall between them, leaves each one 4 mm wide, which forces them 40 mm
deep to carry the flow, which fills the disc and leaves nowhere for the
passages that feed them. That arrangement ran at 13–15 m/s and distributed
badly.

So only the fuel manifold straddles its ring. The oxidiser dome sits outboard
and reaches its ring through a ring of 48 radial ports — a coaxial post, built
from the primitives this model already has. Both are then wide and shallow:

| plenum | section | velocity |
|---|---|---|
| fuel manifold | 21.8 × 13.1 mm, r 71–84 | 8.0 m/s |
| ox dome | 17.1 × 17.1 mm, r 103–120 | 7.0 m/s |

against 36.5 and 22.2 m/s at the orifices, which is the ratio that governs how
evenly a ring distributes.

The cowl's coolant arrives at r 99 and has to reach r 78. It cannot cross the
dome — a plenum is a continuous ring, so anything inside its radial band goes
through it, and fuel crossing the oxidiser dome is the one failure this layout
exists to prevent. It runs axially down the corridor inboard of the dome, then
turns in through the one window where the manifold has begun and the dome has
not. The spike's coolant arrives close to the bore, where the manifold already
reaches, so it goes straight in.

Each jacket has a groove across its joint face collecting all of its channels,
so no channel dead-ends against a head that only has holes in 48 places.

### What was wrong before

Four defects, each of which produced a part that was watertight, drained, and
looked entirely correct:

- The oxidiser dome was packed outward from the bore to make it fit the disc,
  landing at r 98–117 while its own orifices sat at r 88. The dome fed nothing,
  and the orifices pointed at the radius the *fuel* manifold occupies.
- Orifice depth was read off the diamond section's extreme tip as though it
  applied at every radius. It does not, and the holes stopped 5 mm short.
- The port schedule named three inlets and the geometry cut none of them.
- The cowl's inlet ring was seated on where its feed ports *end* rather than on
  where the metal is, so it landed 2.1 mm proud of a tapering surface. Its boss
  touched the cowl only at the forward edge, and hollowing the plenum inside it
  cut that last connection — leaving 27 cm3 of copper attached to nothing,
  inside a part that passed every gate it was given.

The last one is why `print_ready.py` now refuses a part in more than one piece.
Watertightness cannot see it: both pieces are closed. The sealed-void check
cannot see it either, because a loose fragment is solid rather than hollow.

## What this is not

A verified flight part. Every model behind these numbers is a screening model:
Bartz is quoted at plus or minus thirty percent, the structural life is not
finite elements, and nothing short of a hot fire settles combustion stability.
`CLAUDE.md` states what each model is and is not, and the list of what is
genuinely absent -- base flow behind the truncated plug, combustion response,
creep, any real CFD.
