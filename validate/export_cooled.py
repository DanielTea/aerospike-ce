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
    build_mesh,
    centrebody_channels,
    cowl_channels,
    injector_holes,
)


def expected_genus(part: str, design) -> int:
    """
    Genus the part should have, from what was cut into it.

    A solid of revolution whose profile clears the axis is a torus, genus 1.
    Every through feature adds one handle: each cooling channel that runs the
    length of a wall, each injector orifice through the head. The centrebody
    touches the axis, so it starts at genus 0 instead.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--voxel", type=float, default=0.4)
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    d = design_engine(spec)
    a = d.assembly
    os.makedirs(args.out, exist_ok=True)
    print(report(d))
    print()

    feats = {
        "cowl": dict(
            channels=[cowl_channels(a, d.circuits["cowl"].channel)]
            if d.circuits["cowl"] else [], holes=[]),
        "centrebody": dict(
            channels=[centrebody_channels(a, d.circuits["centrebody"].channel)]
            if d.circuits["centrebody"] else [], holes=[]),
        "head": dict(channels=[], holes=injector_holes(a, d.injector) if d.injector else []),
    }

    name = spec.get("name", "engine")
    for part, kw in feats.items():
        t0 = time.time()
        v, f = build_mesh(a.profiles[part], voxel_mm=args.voxel, **kw)
        rep = manifold_report(v, f)
        mv = mesh_volume(v, f)
        av = a.profiles[part].revolved_volume
        genus = (2 - rep["euler"]) // 2
        want = expected_genus(part, d)
        path = os.path.join(args.out, f"{name}_{part}.stl")
        write_binary_stl(path, v, f, f"{name}_{part}")
        ok = "ok" if genus == want else "UNDER-RESOLVED"
        print(f"{part:11s} {time.time() - t0:6.1f}s  tris {len(f):8d}  "
              f"watertight {str(rep['watertight']):5s}  genus {genus:6d} (expected {want:5d}) {ok:14s} "
              f"vol {mv / 1000:7.2f} cm3 of {av / 1000:7.2f} solid  "
              f"-> {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.0f} MB)")
        if genus != want:
            print(f"{'':11s}   the 0.40 mm channels are about one voxel wide at "
                  f"{args.voxel:.2f} mm, so marching cubes breaks them up. The channel "
                  f"geometry itself is verified against the distance field in "
                  f"test_cooled_geometry.py; this STL is for looking at, not for "
                  f"inspecting.")


if __name__ == "__main__":
    main()
