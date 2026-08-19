"""
Render the contour to PNG and CSV.

This is the eyes of the loop. Claude Code cannot see the PicoGK viewer, but it
can read a PNG. Run this after any change to the physics, then look at
out/contour.png before touching the C# model.

    python validate/plot_contour.py --spec spec/demo.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from contour_ref import build_contour, to_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/demo.json")
    ap.add_argument("--out", default="../out")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    n = spec["nozzle"]
    c = build_contour(
        expansion_ratio=n["expansion_ratio"],
        exit_radius_mm=n["exit_radius_mm"],
        gamma=spec["gas"]["gamma"],
        n_points=n.get("contour_points", 300),
        truncate_fraction=n.get("truncate_fraction", 1.0),
    )

    os.makedirs(args.out, exist_ok=True)
    to_csv(c, os.path.join(args.out, "contour.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # meridional section, both sides, plus the cowl lip
    ax1.plot(c.x, c.r, color="#1f4e79", lw=2, label="plug contour")
    ax1.plot(c.x, [-r for r in c.r], color="#1f4e79", lw=2)
    ax1.axhline(c.exit_radius, color="#999", ls=":", lw=1)
    ax1.axhline(-c.exit_radius, color="#999", ls=":", lw=1)
    # The lip sits at c.lip_x, not at x = 0. Drawing it at the origin makes the
    # throat look like a radial annulus; it is inclined at nu_e, and the area
    # difference between the two readings is a factor of about 2.5.
    ax1.plot([c.lip_x], [c.exit_radius], "o", color="#c00", ms=7, label="cowl lip")
    ax1.plot([c.lip_x], [-c.exit_radius], "o", color="#c00", ms=7)
    ax1.plot([c.x[0], c.lip_x], [c.r[0], c.exit_radius], color="#c00", lw=1.5,
             label="throat (sonic line)")
    ax1.plot([c.x[0], c.lip_x], [-c.r[0], -c.exit_radius], color="#c00", lw=1.5)
    ax1.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("r [mm]")
    ax1.set_title(
        f"plug nozzle  |  eps={n['expansion_ratio']:g}  "
        f"Me={c.exit_mach:.3f}  nu_e={math.degrees(c.throat_angle):.1f} deg  "
        f"L={c.length:.1f} mm  At={c.throat_area:.1f} mm2"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(c.x, c.mach, color="#c05000", lw=2)
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("Mach on wall")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out_png = os.path.join(args.out, "contour.png")
    fig.savefig(out_png, dpi=130)

    print(f"exit Mach          {c.exit_mach:.4f}")
    print(f"throat angle       {math.degrees(c.throat_angle):.3f} deg")
    print(f"spike length       {c.length:.3f} mm")
    print(f"tip radius         {c.r[-1]:.4f} mm")
    print(f"throat area        {c.throat_area:.3f} mm2")
    print(f"lip station        x={c.lip_x:.4f} mm")
    print(f"throat slant       {c.throat_slant_length:.4f} mm")
    print(f"throat area (geom) {c.throat_area_from_geometry:.3f} mm2")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
