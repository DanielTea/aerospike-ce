"""
Export the cooled engine as STL, with cooling channels and injector orifices.

    python export_cooled.py --spec ../spec/regen.json --out ../out

Uses the signed-distance mesher, because channels and orifices vary with angle
and the revolve cannot express them. Reports the manifold checks for each part;
the Euler characteristic is the one that catches a channel that has broken
through its wall, which produces a mesh that is watertight and wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from engine_design import design_engine, report
from mesh_export import manifold_report, mesh_volume, write_binary_stl
from mesh_solid import (
    build_mesh_streaming,
    centrebody_channels,
    cowl_channels,
    decimate,
    feed_ports,
    injector_holes,
)


def expected_genus(part: str, design) -> int:
    """
    Genus the part should have, from what was cut into it.

    A revolved profile that clears the axis is a torus, genus 1; one that
    touches it is a sphere, genus 0. Every through feature adds a handle: each
    cooling channel now runs from the head face to its own feed port, and each
    injector orifice passes through the head disc.
    """
    if part == "cowl":
        c = design.circuits["cowl"]
        return 1 + (c.channel.n_channels if c else 0)
    if part == "centrebody":
        c = design.circuits["centrebody"]
        return 0 + (c.channel.n_channels if c else 0)
    if part == "head":
        return 1 + (2 * design.injector.n_elements if design.injector else 0)
    raise ValueError(part)


def components(verts, faces) -> int:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                   shape=(len(verts), len(verts)))
    return int(connected_components(g, directed=False)[0])


def genus_of(verts, faces) -> int:
    rep = manifold_report(verts, faces)
    return (2 * components(verts, faces) - rep["euler"]) // 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="0 picks a size that puts three samples across the "
                         "narrowest feature")
    ap.add_argument("--decimate", type=float, default=0.08,
                    help="fraction of triangles to KEEP; 0 disables. The result "
                         "is only used if it preserves watertightness and genus")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    d = design_engine(spec)
    a = d.assembly
    os.makedirs(args.out, exist_ok=True)
    print(report(d))
    print()

    cuts, ports = {}, {}
    if d.circuits["cowl"]:
        cuts["cowl"] = cowl_channels(a, d.circuits["cowl"].channel)
        ports["cowl"] = feed_ports(a, cuts["cowl"])
    if d.circuits["centrebody"]:
        cuts["centrebody"] = centrebody_channels(a, d.circuits["centrebody"].channel)
        ports["centrebody"] = feed_ports(a, cuts["centrebody"])
    holes = injector_holes(a, d.injector) if d.injector else []

    # three samples across the narrowest thing in the part, or it will not survive
    narrowest = min([c.width_mm for c in cuts.values()]
                    + [c.hot_wall_mm for c in cuts.values()]
                    + ([min(h.diameter_mm for h in holes)] if holes else []))
    voxel = args.voxel if args.voxel > 0 else round(narrowest / 3.0, 3)
    print(f"narrowest feature {narrowest:.2f} mm -> {voxel:.3f} mm voxels "
          f"({narrowest / voxel:.1f} samples across)")
    print()

    name = spec.get("name", "engine")
    for part in ("cowl", "centrebody", "head"):
        t0 = time.time()
        v, f = build_mesh_streaming(
            a.profiles[part], voxel_mm=voxel,
            channels=[cuts[part]] if part in cuts else [],
            ports=[ports[part]] if part in ports else [],
            holes=holes if part == "head" else [])

        rep = manifold_report(v, f)
        g = genus_of(v, f)
        want = expected_genus(part, d)
        mv = mesh_volume(v, f)
        av = a.profiles[part].revolved_volume
        raw_tris = len(f)

        note = "ok" if g == want else f"MISMATCH (expected {want})"
        print(f"{part:11s} {time.time() - t0:6.1f}s  tris {raw_tris:9d}  "
              f"watertight {str(rep['watertight']):5s}  comps {components(v, f):4d}  "
              f"genus {g:5d}  {note}")

        if args.decimate > 0.0 and g == want:
            # Walk from the most aggressive reduction upward and take the first
            # that still has the right topology. Pushed too far, quadric
            # collapse merges across a channel and closes it -- the mesh stays
            # plausible and one cooling passage has silently gone.
            for keep in (0.03, 0.05, 0.08, 0.12, 0.20, 0.35):
                v2, f2 = decimate(v, f, keep)
                g2 = genus_of(v2, f2)
                rep2 = manifold_report(v2, f2)
                dv = abs(mesh_volume(v2, f2) - mv) / abs(mv)
                good = g2 == want and rep2["watertight"] and dv < 0.01
                print(f"{'':11s} keep {keep:4.2f} -> {len(f2):9d} tris  "
                      f"genus {g2:5d}  watertight {str(rep2['watertight']):5s}  "
                      f"drift {dv * 100:5.2f}%  {'accepted' if good else 'rejected'}")
                if good:
                    v, f = v2, f2
                    break

        path = os.path.join(args.out, f"{name}_{part}.stl")
        write_binary_stl(path, v, f, f"{name}_{part}")
        print(f"{'':11s} vol {mesh_volume(v, f) / 1000:7.2f} cm3 of {av / 1000:7.2f} solid "
              f"-> {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
