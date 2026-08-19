"""
Section views of the cooled engine.

    python plot_cooled.py --spec ../spec/regen.json --out ../out

Slices through the signed distance field rather than renders of the mesh. A
render shows an outer skin; the things worth checking here are all inside it --
whether the channels sit in the wall or through it, whether they merge as the
spike closes, and whether the orifice rings clear the face. Each panel is a cut
plane, so what you see is the material.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from engine_design import design_engine  # noqa: E402
from mesh_solid import (  # noqa: E402
    _channel_sdf,
    _hole_sdf,
    _polygon_sdf,
    centrebody_channels,
    cowl_channels,
    injector_holes,
)


def _field_slice_axial(profiles, cuts, holes, x0, x1, r0, r1, n=900):
    """Material in the theta = 0 half plane."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    rs = np.linspace(r0, r1, n // 2, dtype=np.float32)
    X, R = np.meshgrid(xs, rs, indexing="ij")
    Y, Z = R, np.zeros_like(R)
    TH = np.zeros_like(R)

    solid = np.full(X.shape, np.inf, dtype=np.float32)
    for p in profiles:
        d = _polygon_sdf(X, R, np.asarray(p.x, np.float32), np.asarray(p.r, np.float32))
        solid = np.minimum(solid, d)
    for c in cuts:
        solid = np.maximum(solid, -_channel_sdf(X, R, TH, c))
    for h in holes:
        solid = np.maximum(solid, -_hole_sdf(X, Y, Z, h))
    return xs, rs, solid


def _field_slice_transverse(profiles, cuts, holes, x_at, rmax, n=900):
    """Material in a plane normal to the axis."""
    ys = np.linspace(-rmax, rmax, n, dtype=np.float32)
    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.sqrt(Y * Y + Z * Z).astype(np.float32)
    TH = np.arctan2(Z, Y).astype(np.float32)
    X = np.full(Y.shape, x_at, dtype=np.float32)

    solid = np.full(X.shape, np.inf, dtype=np.float32)
    for p in profiles:
        d = _polygon_sdf(X, R, np.asarray(p.x, np.float32), np.asarray(p.r, np.float32))
        solid = np.minimum(solid, d)
    for c in cuts:
        solid = np.maximum(solid, -_channel_sdf(X, R, TH, c))
    for h in holes:
        solid = np.maximum(solid, -_hole_sdf(X, Y, Z, h))
    return ys, solid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../out")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    d = design_engine(spec)
    a, c, ch = d.assembly, d.assembly.contour, d.chamber
    os.makedirs(args.out, exist_ok=True)

    cuts = []
    if d.circuits["cowl"] is not None:
        cuts.append(cowl_channels(a, d.circuits["cowl"].channel))
    if d.circuits["centrebody"] is not None:
        cuts.append(centrebody_channels(a, d.circuits["centrebody"].channel))
    holes = injector_holes(a, d.injector) if d.injector else []
    profiles = list(a.profiles.values())

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.25, wspace=0.22)

    # ---- 1. whole engine in axial section ----
    ax = fig.add_subplot(gs[0, :])
    x0 = a.head_x - a.structure.head_thickness_mm - 3
    x1 = float(a.inner_wall_x[-1]) + 3
    xs, rs, fld = _field_slice_axial(profiles, cuts, holes, x0, x1, 0.0,
                                     a.flange_radius + 3, n=1400)
    ax.pcolormesh(xs, rs, (fld < 0).T, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
    ax.pcolormesh(xs, -rs, (fld < 0).T, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
    for sgn in (1, -1):
        ax.plot([c.lip_x, c.x[0]], [sgn * c.exit_radius, sgn * c.r[0]],
                color="#c00", lw=1.6, zorder=5)
    ax.axhline(0, color="#999", lw=0.7, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title(
        f"{spec['name']}  |  {ch.thrust_sea_level:.0f} N sl / {ch.thrust_vacuum:.0f} N vac   "
        f"Isp {ch.isp_sea_level:.1f}/{ch.isp_vacuum:.1f} s   "
        f"pc {ch.chamber_pressure / 1e5:.0f} bar  MR {ch.mixture_ratio:.1f}   "
        f"{a.overall_length:.0f} x {2 * a.overall_radius:.0f} mm", fontsize=11)
    ax.grid(alpha=0.15)

    # ---- 2. throat close-up, both channel rings in section ----
    ax = fig.add_subplot(gs[1, 0])
    xs, rs, fld = _field_slice_axial(profiles, cuts, holes,
                                     c.lip_x - 26, c.lip_x + 14,
                                     a.shoulder_radius - 14, a.chamber_outer_radius + 9,
                                     n=900)
    ax.pcolormesh(xs, rs, (fld < 0).T, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
    ax.plot([c.lip_x, c.x[0]], [c.exit_radius, c.r[0]], color="#c00", lw=2, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("throat: channels either side of the gap", fontsize=10)

    # ---- 3. transverse cut through the chamber ----
    ax = fig.add_subplot(gs[1, 1])
    x_cut = 0.5 * (a.head_x + a.converging_start_x)
    ys, fld = _field_slice_transverse(profiles, cuts, holes, x_cut,
                                      a.chamber_outer_radius + 8, n=900)
    ax.pcolormesh(ys, ys, (fld < 0).T, cmap="Blues", shading="auto", vmin=0, vmax=1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    n_cowl = d.circuits["cowl"].channel.n_channels if d.circuits["cowl"] else 0
    n_cb = d.circuits["centrebody"].channel.n_channels if d.circuits["centrebody"] else 0
    ax.set_title(f"chamber section at x={x_cut:.0f} mm\n{n_cowl} cowl + {n_cb} centrebody channels",
                 fontsize=10)

    # ---- 4. injector face ----
    ax = fig.add_subplot(gs[1, 2])
    x_face = a.head_x - 0.5 * a.structure.head_thickness_mm
    ys, fld = _field_slice_transverse(profiles, [], holes, x_face,
                                      a.flange_radius + 4, n=900)
    ax.pcolormesh(ys, ys, (fld < 0).T, cmap="Greys", shading="auto", vmin=0, vmax=1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    if d.injector:
        i = d.injector
        ax.set_title(f"injector face: {i.n_elements} doublets\n"
                     f"fuel {i.d_fuel_mm:.2f} mm, ox {i.d_ox_mm:.2f} mm", fontsize=10)
    else:
        ax.set_title("injector face: no design fits", fontsize=10)

    out = os.path.join(args.out, "engine_cooled.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
