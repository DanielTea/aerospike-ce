"""
Export the cooled engine as STL, with cooling channels and injector orifices.

    python export_cooled.py --spec ../spec/regen.json --out ../out

Uses the signed-distance mesher, because channels and orifices vary with angle
and the revolve cannot express them.

Voxel size is derived, not chosen: marching cubes needs about three samples
across the narrowest feature, and on this engine that is the 0.4 mm cooling
channel.

Two things are checked, because they fail differently.

Volume, against a value computed analytically -- the revolved profile less the
channel and port volumes integrated along the wall. This is the check that
catches under-resolution, since a channel the mesher cannot resolve simply is
not removed, and it needs no assumptions about topology.

Topology, where the prediction is unambiguous. The head disc is one handle per
orifice on a bored disc and comes out exact. The channelled parts are reported
rather than asserted: a channel open at one end is a pocket and adds no genus, a
channel open at both is a tunnel and adds one, and which of those a given
channel is depends on where its wall ran out -- so the number is informative but
not a pass criterion.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import time

import numpy as np

from engine_design import design_engine, report
from mesh_export import manifold_report, mesh_volume, write_binary_stl
from mesh_solid import (
    build_mesh_streaming,
    centrebody_channels,
    cowl_channels,
    decimate,
    expected_channel_volume,
    expected_port_volume,
    feed_ports,
    injector_holes,
    stream_mesh_to_stl,
)


def expected_genus(part: str, design) -> int:
    """
    Genus the head disc should have: one handle per injector orifice on top of
    the bored disc's own. The channelled parts are not predicted here; see the
    module docstring on why their topology is reported rather than asserted.
    """
    if part == "head":
        return 1 + (2 * design.injector.n_elements if design.injector else 0)
    return -1


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
                    help="0 derives it from the narrowest feature")
    ap.add_argument("--verify-voxel", type=float, default=0.22,
                    help="resolution for the in-memory volume check; the full "
                         "part at inspection resolution does not fit in RAM "
                         "welded, so it is streamed instead")
    ap.add_argument("--decimate", type=float, default=0.08,
                    help="fraction of triangles to KEEP for the head; 0 disables")
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

    def voxel_for(part: str) -> tuple[float, float]:
        """
        Resolution from the part's own narrowest feature, not the engine's.

        The head carries 1.3 mm orifices and the centrebody 0.4 mm channels; one
        global voxel size means either the head is meshed eight times finer than
        it needs or the channels are lost. Memory and time both go as the cube,
        so this is not a small saving.
        """
        feats = []
        if part in cuts:
            feats += [cuts[part].width_mm, cuts[part].hot_wall_mm]
        if part == "head" and holes:
            feats.append(min(h.diameter_mm for h in holes))
        n = min(feats) if feats else 1.0
        return (args.voxel if args.voxel > 0 else round(n / 3.0, 3)), n

    name = spec.get("name", "engine")
    for part in ("cowl", "centrebody", "head"):
        voxel, narrowest = voxel_for(part)
        ch = [cuts[part]] if part in cuts else []
        pt = [ports[part]] if part in ports else []
        hl = holes if part == "head" else []

        solid = a.profiles[part].revolved_volume
        want = solid
        for c in ch:
            want -= expected_channel_volume(c)
        for c, q in zip(ch, pt):
            want -= expected_port_volume(c, q)
        for h in hl:
            want -= h.count * math.pi * (h.diameter_mm / 2) ** 2 * \
                a.structure.head_thickness_mm

        print(f"{part:11s} narrowest feature {narrowest:.2f} mm -> {voxel:.3f} mm "
              f"voxels ({narrowest / voxel:.1f} samples across)")

        # ---- verification pass, welded, at a resolution that fits in memory --
        t0 = time.time()
        v, f = build_mesh_streaming(a.profiles[part], voxel_mm=args.verify_voxel,
                                    channels=ch, ports=pt, holes=hl)
        rep = manifold_report(v, f)
        mv = mesh_volume(v, f)
        g = genus_of(v, f)
        err = abs(mv - want) / abs(want)
        want_g = expected_genus(part, d)
        gtxt = f"{g}" if want_g < 0 else f"{g} (expected {want_g}) {'ok' if g == want_g else 'MISMATCH'}"
        print(f"{part:11s} verify @{args.verify_voxel:.2f}mm {time.time() - t0:5.0f}s  "
              f"tris {len(f):8d}  watertight {str(rep['watertight']):5s}  "
              f"flat tris {rep['degenerate_faces']:6d}  "
              f"comps {components(v, f):3d}  genus {gtxt}")
        print(f"{'':11s} volume {mv / 1000:8.2f} cm3 against {want / 1000:8.2f} expected "
              f"({solid / 1000:.2f} solid less features)  -> {err * 100:.2f}%")

        # A part whose topology is provable and whose verified mesh decimates
        # cleanly is better shipped as that mesh than as a fresh stream: the
        # streamed one is larger, and nothing has checked it.
        path = os.path.join(args.out, f"{name}_{part}.stl")
        shipped = False
        if args.decimate > 0.0 and want_g >= 0 and g == want_g:
            for keep in (0.03, 0.05, 0.08, 0.12, 0.20):
                v2, f2 = decimate(v, f, keep)
                rep2 = manifold_report(v2, f2)
                if genus_of(v2, f2) == want_g and rep2["watertight"] \
                        and rep2["degenerate_faces"] == 0:
                    dv = abs(mesh_volume(v2, f2) - want) / abs(want)
                    write_binary_stl(path, v2, f2, f"{name}_{part}")
                    print(f"{'':11s} decimated to {len(f2):8d} tris at keep {keep:.2f}, "
                          f"genus and watertightness intact, volume {dv * 100:.2f}%")
                    print(f"{'':11s} -> {os.path.basename(path)} "
                          f"({os.path.getsize(path) / 1e6:.0f} MB), verified")
                    shipped = True
                    break
        del v, f

        # ---- otherwise stream at inspection resolution ----------------------
        if not shipped:
            t0 = time.time()
            tris = stream_mesh_to_stl(path, a.profiles[part], voxel_mm=voxel,
                                      channels=ch, ports=pt, holes=hl)
            print(f"{'':11s} streamed @{voxel:.3f}mm {time.time() - t0:5.0f}s  "
                  f"tris {tris:9d}  -> {os.path.basename(path)} "
                  f"({os.path.getsize(path) / 1e6:.0f} MB)")
        print()


if __name__ == "__main__":
    main()
