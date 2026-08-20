# The print file

A checked, print-resolution model of the whole engine: three solids in one 3MF,
dimensioned in millimetres, with every cooling channel, injector orifice and
mounting hole in it.

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

## The head manifolds, and what is wrong with them

Worth stating plainly, because it is the weakest part of this engine and it is
not visible in the geometry.

The injector's two orifice rings sit 7 mm apart radially -- that spacing is set
by the impingement geometry, not chosen -- and each needs its own manifold
behind it. Two manifolds 7 mm apart, with a 3 mm wall between them, leaves each
one 4 mm wide. Buying the flow area back in depth gives a 40 mm deep, 4 mm wide
slot, and even then:

| manifold | section | velocity | its orifices |
|---|---|---|---|
| fuel | 40 x 4.0 mm | 14.6 m/s | 36.5 m/s |
| oxidiser | 40 x 4.0 mm | 13.0 m/s | 22.2 m/s |

A feed manifold wants to be slow compared with the orifices it feeds, because
the dynamic pressure it does not recover shows up as a variation in orifice
pressure drop around the ring. At 13 m/s against 22 the oxidiser ring is at
about a third of orifice dynamic pressure, which is poor distribution. The model
reports the velocity the geometry actually has rather than the 4 m/s it was
asked for, and says why.

The real fix is not a bigger manifold, because there is nowhere to put one. It
is coaxial posts -- the oxidiser dome sitting behind the fuel manifold with the
ox orifices running through it in tubes -- which is what production engines do
and what this model does not yet have the geometry for.

Two things that had to be got right first, and were not:

- The manifolds were originally packed outward from the bore to make them fit
  the disc, which put the oxidiser dome at r 98-117 mm while its own orifices
  sat at r 88. The dome fed nothing, and the orifices pointed at the radius the
  *fuel* manifold occupies. A few millimetres more reach and the engine would
  have plumbed oxidiser into the fuel manifold. Both parts are watertight either
  way; only the sealed-void check noticed, as 325 cm3 of trapped powder.
- The section is a diamond, so how far aft it reaches depends on the radius at
  which you meet it. Reading the extreme tip as though it applied everywhere put
  the orifices 5 mm short of it.

## What this is not

A verified flight part. Every model behind these numbers is a screening model:
Bartz is quoted at plus or minus thirty percent, the structural life is not
finite elements, and nothing short of a hot fire settles combustion stability.
`CLAUDE.md` states what each model is and is not, and the list of what is
genuinely absent -- base flow behind the truncated plug, combustion response,
creep, any real CFD.
