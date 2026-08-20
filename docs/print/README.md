# The print file

A checked, print-resolution model of the whole engine: three solids in one 3MF,
dimensioned in millimetres, with every cooling channel, injector orifice and
mounting hole in it.

**Download:** [regen-spike-75.3mf](https://github.com/DanielTea/aerospike-ce/releases/download/v0.1.0/regen-spike-75.3mf)
(238 MiB, from the [v0.1.0 release](https://github.com/DanielTea/aerospike-ce/releases/tag/v0.1.0))

It is a release asset rather than a file in the tree because it is 249 MB, and
a generated artifact of that size does not belong in git history. The thing
worth versioning is the generator, which is here.

Regenerate with:

```bash
cd validate
python print_ready.py --spec ../spec/regen.json --out ../docs/print
python verify_print_file.py ../docs/print/regen-spike-75.3mf
```

The first command builds it. The second reopens the written file as a stranger
would and re-derives every property from what is on disk, because checking the
meshes in memory and then writing them leaves the writer -- the part a slicer
actually sees -- unchecked.

## What is in it

Meshed at 0.233 mm, the voxel the narrowest feature asks for. Every part
watertight, no boundary or non-manifold edges, no cavity without a way out.

| part | triangles | genus | volume | mass |
|---|---|---|---|---|
| centrebody | 8,205,076 | 328 | 402.9 cm3 | 3.53 kg |
| cowl | 12,149,086 | 394 | 405.3 cm3 | 3.55 kg |
| head | 1,487,276 | 101 | 1679.5 cm3 | 14.71 kg |

Genus is the check worth reading. The head's 101 is derived, not observed:
one handle for the central bore, 48 for each plenum ring joined by its
48 orifices, and four for the mounting holes. A single blind orifice would
show up here as 100 -- while the part stayed watertight, drained, and looked
entirely correct. The cowl's 394 and the centrebody's 328 track their 392 and
164 channels.

The head is two thirds of the engine's 21.8 kg because it is a 46 mm slab of
copper, thickened to contain its manifolds. If mass matters, that is the first
place to look, and it is a design change rather than a meshing one.

Decimation stops where the topology does. The channelled parts survive halving
and not much more -- at 0.30 the quadric collapse closes a cooling channel --
whereas the head goes to 15 percent. That is why the file is large: a
regeneratively cooled engine at a resolution that actually holds its 556
channels is tens of millions of triangles, and there is no honest way to make
it small.

## Why 3MF and not STL

STL is a triangle soup with no units in it. Every slicer guesses millimetres and
is usually right, which is not the same as being told; a part that silently
arrives at a twenty-fifth of its size is a wasted build. 3MF states its unit,
carries each part as a named object, and zips -- which matters when the geometry
has several hundred cooling channels in it.

## What is checked before it is written

`print_ready.py` refuses to write rather than writing something plausible:

- **Watertight.** No boundary edges, no non-manifold edges.
- **No sealed void.** Every internal cavity has a way out.
- **Resolution.** The voxel is derived from the narrowest feature in the part,
  not chosen. Marching cubes needs about three samples across a feature.
- **Decimation that stops.** Quadric collapse reads a sub-millimetre channel as
  noise against a smooth wall and closes it, leaving a mesh that is still
  watertight and still looks like an engine. Every step is checked against the
  topology and the volume, and the result is the last one that survived.

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
