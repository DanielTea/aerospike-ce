"""
The build plan: everything C# needs to voxelise, and nothing else.

    python build_plan.py --spec ../spec/regen.json --out ../out/plan.json

This is the seam between the two languages. Python owns every number -- physics,
sizing, geometry, the lot -- and writes down the resulting shapes. C# owns the
voxel kernel and reads them. Neither reimplements the other.

Why the seam moved here
-----------------------
The C# used to mirror the Python: two implementations of the same maths, kept in
step by hand, with divergence named in CLAUDE.md as the failure mode the whole
structure existed to prevent. Two thirds of the C# was that mirror. It never
caught an error in the Python -- every bug the pairing found was a bug in the
C# -- and it cost a port every time the physics moved.

A plan file removes the duplication instead of policing it. There is one place a
throat area is computed. The C# cannot disagree about geometry because it is not
told how to derive any, only what the answer was.

What crosses the boundary
-------------------------
Shapes and dimensions, in millimetres. Profiles as point lists, features as
their parameters. No propellants, no correlations, no solver settings. If C#
would need a physical property to use a field, that field is in the wrong file.

The plan is versioned. A reader that does not recognise the version should stop
rather than guess, because a silently misread plan builds a plausible engine
that is not the one that was designed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, is_dataclass

import numpy as np

PLAN_VERSION = 1


def _points(x, r) -> dict:
    """A meridional polyline, rounded to a micron -- well under any voxel."""
    return {"x": [round(float(v), 4) for v in np.asarray(x)],
            "r": [round(float(v), 4) for v in np.asarray(r)]}


def _f(v) -> float:
    return round(float(v), 6)


def build_plan(spec: dict, design=None) -> dict:
    """
    Everything needed to build the voxel model, derived from one spec.

    Deliberately flat and explicit. A reader should not have to know what a
    contraction ratio is to place a channel.
    """
    from engine_design import design_engine
    from manifold_ref import design_manifolds, geometry_features
    from mesh_solid import (
        centrebody_channels,
        cowl_channels,
        feed_ports,
        injector_holes,
    )

    d = design if design is not None else design_engine(spec)
    a = d.assembly
    md = design_manifolds(d)
    gf = geometry_features(d, md)

    plan: dict = {
        "plan_version": PLAN_VERSION,
        "name": spec.get("name", "engine"),
        "units": "mm",
        "voxel_size_mm": _f(spec.get("geometry", {}).get("voxel_size_mm", 0.2)),
        "build_direction": "+x",
        "parts": {},
        "summary": {
            "thrust_sea_level_n": _f(d.chamber.thrust_sea_level),
            "thrust_vacuum_n": _f(d.chamber.thrust_vacuum),
            "isp_sea_level_s": _f(d.chamber.isp_sea_level),
            "isp_vacuum_s": _f(d.chamber.isp_vacuum),
            "chamber_pressure_bar": _f(d.chamber.chamber_pressure / 1e5),
            "overall_length_mm": _f(a.overall_length),
            "overall_diameter_mm": _f(2.0 * a.overall_radius),
        },
    }

    for name, profile in a.profiles.items():
        plan["parts"][name] = {
            "profile": _points(profile.x, profile.r),
            "channels": [], "ports": [], "holes": [],
            "bosses": [], "lugs": [], "plenums": [],
        }

    # ---- cooling channels and their feed ports ----
    for part, maker in (("cowl", cowl_channels), ("centrebody", centrebody_channels)):
        circuit = d.circuits.get(part)
        if circuit is None:
            continue
        cut = maker(a, circuit.channel)
        port = feed_ports(a, cut)
        plan["parts"][part]["channels"].append({
            "wall": _points(cut.wall_x, cut.wall_r),
            "count": int(cut.n_channels),
            "width_mm": _f(cut.width_mm),
            "height_mm": _f(cut.height_mm),
            "hot_wall_mm": _f(cut.hot_wall_mm),
            "land_mm": _f(cut.land_mm),
            "x_start_mm": _f(cut.x_start),
            "x_end_mm": _f(cut.x_end),
            "outward": bool(cut.outward),
        })
        plan["parts"][part]["ports"].append({
            "x_mm": _f(port.x_at), "diameter_mm": _f(port.diameter_mm),
            "count": int(port.count), "r_lo_mm": _f(port.r_lo),
            "r_hi_mm": _f(port.r_hi), "phase_rad": _f(port.phase),
        })

    # ---- injector orifices ----
    if d.injector is not None:
        for h in injector_holes(a, d.injector):
            plan["parts"]["head"]["holes"].append({
                "radius_mm": _f(h.radius_mm), "diameter_mm": _f(h.diameter_mm),
                "count": int(h.count), "x_start_mm": _f(h.x_start),
                "x_end_mm": _f(h.x_end), "phase_rad": _f(h.phase),
            })

    # ---- manifolds, their boss rings, and the mounts ----
    for part, feats in gf.items():
        for b in feats.get("bosses", []):
            plan["parts"][part]["bosses"].append({
                "x_mm": _f(b.x_at), "r_inner_mm": _f(b.r_inner),
                "r_outer_mm": _f(b.r_outer), "half_x_mm": _f(b.half_x)})
        for pl in feats.get("plenums", []):
            plan["parts"][part]["plenums"].append({
                "x_mm": _f(pl.x_at), "r_inner_mm": _f(pl.r_inner),
                "half_x_mm": _f(pl.half_x), "half_r_mm": _f(pl.half_r)})
        for lg in feats.get("lugs", []):
            plan["parts"][part]["lugs"].append({
                "count": int(lg.count), "x_mm": _f(lg.x_at),
                "half_x_mm": _f(lg.half_x), "r_inner_mm": _f(lg.r_inner),
                "r_outer_mm": _f(lg.r_outer),
                "half_width_deg": _f(lg.half_width_deg),
                "phase_rad": _f(lg.phase)})
        for h in feats.get("holes", []):
            plan["parts"][part]["holes"].append({
                "radius_mm": _f(h.radius_mm), "diameter_mm": _f(h.diameter_mm),
                "count": int(h.count), "x_start_mm": _f(h.x_start),
                "x_end_mm": _f(h.x_end), "phase_rad": _f(h.phase)})

    return plan


def narrowest_feature(plan: dict) -> float:
    """
    The smallest dimension anything in the plan has.

    The voxel size has to resolve it -- about three samples across -- and a
    reader that voxelises coarser than this will produce a part with its
    cooling channels quietly missing.
    """
    smallest = math.inf
    for part in plan["parts"].values():
        for c in part["channels"]:
            smallest = min(smallest, c["width_mm"], c["height_mm"], c["hot_wall_mm"])
        for h in part["holes"]:
            smallest = min(smallest, h["diameter_mm"])
        for p in part["ports"]:
            smallest = min(smallest, p["diameter_mm"])
    return smallest if smallest < math.inf else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../out/plan.json")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="override the spec's voxel size, for quick builds")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    plan = build_plan(spec)
    if args.voxel > 0.0:
        plan["voxel_size_mm"] = _f(args.voxel)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)

    narrow = narrowest_feature(plan)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1024:.0f} kB)")
    for name, part in plan["parts"].items():
        bits = [f"{len(part['profile']['x'])} profile pts"]
        for k in ("channels", "ports", "holes", "bosses", "lugs", "plenums"):
            if part[k]:
                bits.append(f"{len(part[k])} {k}")
        print(f"  {name:11s} {', '.join(bits)}")
    print(f"narrowest feature {narrow:.2f} mm -> wants voxels of about "
          f"{narrow / 3.0:.3f} mm; plan says {plan['voxel_size_mm']:.3f} mm")


if __name__ == "__main__":
    main()
