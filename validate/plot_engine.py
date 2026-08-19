"""
Render the full engine assembly to PNG.

Companion to plot_contour.py, which only ever showed the plug surface. This one
shows the parts, the duct, and the area schedule that produced the cowl wall.

    python plot_engine.py --spec ../spec/demo.json --out ../out

Look at out/engine.png after any change in engine_ref.py. The tests catch wrong
numbers; the picture catches geometry that is numerically fine and physically
absurd, which is a different failure and a common one.
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

from engine_ref import assembly_from_spec, measured_flow_area  # noqa: E402

PART_STYLE = {
    "centrebody": ("#1f4e79", "spike / centrebody"),
    "cowl":       ("#8c3b00", "cowl"),
    "head":       ("#4a4a4a", "head closure + flange"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/demo.json")
    ap.add_argument("--out", default="../out")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    a = assembly_from_spec(spec)
    c = a.contour
    os.makedirs(args.out, exist_ok=True)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ---- meridional section, mirrored ----
    for name, p in a.profiles.items():
        colour, label = PART_STYLE[name]
        xs = np.append(p.x, p.x[0])
        rs = np.append(p.r, p.r[0])
        ax1.fill(xs, rs, color=colour, alpha=0.30)
        ax1.plot(xs, rs, color=colour, lw=1.4, label=label)
        ax1.fill(xs, -rs, color=colour, alpha=0.30)
        ax1.plot(xs, -rs, color=colour, lw=1.4)

    # the throat: the sonic line from the lip to the shoulder
    for sgn in (1, -1):
        ax1.plot([c.lip_x, c.x[0]], [sgn * c.exit_radius, sgn * c.r[0]],
                 color="#c00", lw=2.2, zorder=5,
                 label="throat (sonic line)" if sgn == 1 else None)
    ax1.plot([c.lip_x], [c.exit_radius], "o", color="#c00", ms=6, zorder=6)
    ax1.plot([c.lip_x], [-c.exit_radius], "o", color="#c00", ms=6, zorder=6)

    ax1.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [mm]   (flow left to right)")
    ax1.set_ylabel("r [mm]")
    ax1.set_title(
        f"{spec.get('name', 'engine')}   |   eps={spec['nozzle']['expansion_ratio']:g}  "
        f"Me={c.exit_mach:.3f}  CR={a.chamber_area / c.throat_area:.2f}  "
        f"At={c.throat_area:.1f} mm2   "
        f"L={a.overall_length:.1f} mm  D={2 * a.overall_radius:.1f} mm"
    )
    # outside the axes: an in-plot legend sits exactly on top of the flange
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
               fontsize=9, ncol=4, frameon=False)
    ax1.grid(alpha=0.2)

    # ---- flow area, measured from the built geometry ----
    x_area, areas = measured_flow_area(a, n=500)
    ax2.plot(x_area, areas / c.throat_area, color="#1f4e79", lw=2,
             label="measured A / A_t")
    ax2.axhline(1.0, color="#c00", ls="--", lw=1.2, label="A_t")
    ax2.axhline(a.chamber_area / c.throat_area, color="#888", ls=":", lw=1.2,
                label="contraction ratio")
    ax2.plot([c.lip_x], [1.0], "o", color="#c00", ms=6)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("A / A_t")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0.0, a.chamber_area / c.throat_area * 1.15)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.25)

    out_png = os.path.join(args.out, "engine.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")

    total = sum(p.revolved_volume for p in a.profiles.values())
    print(f"exit Mach            {c.exit_mach:.4f}")
    print(f"throat area          {c.throat_area:.4f} mm2")
    print(f"lip station          x={c.lip_x:.4f} mm  r={c.exit_radius:.4f} mm")
    print(f"cowl wall ends at    x={a.outer_wall_x[-1]:.4f} mm  r={a.outer_wall_r[-1]:.4f} mm")
    print(f"chamber outer radius {a.chamber_outer_radius:.4f} mm")
    print(f"contraction ratio    {a.chamber_area / c.throat_area:.4f}")
    print(f"min measured A/At    {areas.min() / c.throat_area:.5f}")
    print(f"overall length       {a.overall_length:.3f} mm")
    print(f"overall diameter     {2 * a.overall_radius:.3f} mm")
    for name, p in a.profiles.items():
        print(f"volume {name:12s}  {p.revolved_volume / 1000.0:8.3f} cm3")
    print(f"volume {'TOTAL':12s}  {total / 1000.0:8.3f} cm3")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
