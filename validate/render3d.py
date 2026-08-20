"""
Render the complete engine in 3D, cut open.

    python render3d.py --spec ../spec/regen.json --out ../out

A closed revolve shows an outer skin and hides everything that took the work:
the cooling channels in both walls, the feed ports that supply them, the
injector orifice rings. So a quadrant is removed from the distance field before
meshing. The cut faces are real geometry rather than a viewer clipping plane,
which means the channels appear in true section wherever the cut crosses them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from engine_design import design_engine  # noqa: E402
from manifold_ref import design_manifolds, geometry_features  # noqa: E402
from shell_ref import shell_features  # noqa: E402
from mesh_solid import (  # noqa: E402
    build_mesh_streaming,
    centrebody_channels,
    cowl_channels,
    decimate,
    feed_ports,
    injector_holes,
)

COLOUR = {"centrebody": "#6f9ec9", "cowl": "#c8803f", "head": "#9a9a9a"}


_FEATURE_CACHE: dict = {}


def _features(design):
    """Manifold, mount and shell geometry, computed once per design."""
    key = id(design)
    if key not in _FEATURE_CACHE:
        gf = geometry_features(design, design_manifolds(design))
        shell, _, _ = shell_features(design)
        for part, feats in shell.items():
            gf.setdefault(part, {}).update(feats)
        _FEATURE_CACHE[key] = gf
    return _FEATURE_CACHE[key]


def build_parts(design, voxel_mm: float, sweep_deg: float, target_tris: int,
                x_window=None, keep_sector=None, decimate_to=True):
    a = design.assembly
    # sweep 0 means take nothing away: the closed engine, for exterior views
    sector = (0.0, math.radians(sweep_deg)) if sweep_deg > 0.0 else None
    parts = {}
    for part in ("cowl", "centrebody", "head"):
        ch, pt, hl = [], [], []
        if part == "cowl" and design.circuits.get("cowl"):
            ch = [cowl_channels(a, design.circuits["cowl"].channel)]
            pt = [feed_ports(a, ch[0])]
        if part == "centrebody" and design.circuits.get("centrebody"):
            ch = [centrebody_channels(a, design.circuits["centrebody"].channel)]
            pt = [feed_ports(a, ch[0])]
        if part == "head" and design.injector is not None:
            hl = injector_holes(a, design.injector)

        # manifolds, their boss rings and the mounting lugs
        gf = _features(design).get(part, {})
        hl = list(hl) + list(gf.get("holes", []))

        t0 = time.time()
        try:
            v, f = build_mesh_streaming(
                a.profiles[part], voxel_mm=voxel_mm,
                channels=ch, ports=pt, holes=hl,
                bosses=gf.get("bosses"), lugs=gf.get("lugs"),
                plenums=gf.get("plenums"), ribs=gf.get("ribs"),
                legs=gf.get("legs"),
                cut_sector=None if keep_sector else sector,
                keep_sector=keep_sector, x_window=x_window)
        except ValueError:
            # a detail window can miss a part entirely -- the head disc is
            # nowhere near the throat -- which is not an error, just an absence
            continue
        raw = len(f)
        # Decimation is what makes a 0.5 mm channel disappear: quadric collapse
        # sees a narrow slot as noise against a smooth wall. The detail view
        # meshes a narrow wedge instead, so the count is low enough to keep
        # every triangle and the channels survive.
        if decimate_to and raw > target_tris:
            v, f = decimate(v, f, target_tris / raw)
        parts[part] = (v, f)
        print(f"  {part:11s} {time.time() - t0:5.0f}s  {raw:8d} -> {len(f):7d} tris",
              flush=True)
    return parts


def _shade(tri: np.ndarray, colour: str, elev: float, azim: float) -> np.ndarray:
    """
    Per-facet Lambertian shading, computed here rather than left to matplotlib.

    Its own shading path wants a facecolor array shaped to the faces and quietly
    broadcasts wrong when handed a single colour, which is the difference
    between a lit solid and a flat silhouette. Doing it explicitly also lets the
    light track the camera, so the cut faces stay readable in both views.
    """
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ln[ln < 1e-20] = 1.0
    n /= ln[:, None]

    e, a = math.radians(elev + 25.0), math.radians(azim + 35.0)
    light = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    lam = np.abs(n @ light)                      # abs: both faces of a cut are lit

    base = np.array(to_rgb(colour))
    shade = 0.35 + 0.65 * lam[:, None]
    rgb = np.clip(base[None, :] * shade, 0.0, 1.0)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)


def render_view(parts, elev, azim, title, path, subtitle=""):
    """One framed view of a prepared set of meshes."""
    fig = plt.figure(figsize=(9.5, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    tris = np.concatenate([v[f] for v, f in parts.values()])
    cols = np.concatenate([_shade(v[f], COLOUR[part], elev, azim)
                           for part, (v, f) in parts.items()])
    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=cols, edgecolors="none", linewidths=0.0))

    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    pad = 0.03 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def gallery(design, spec, out_dir, voxel, detail_voxel, tris, sweep):
    """
    A set of views committed to the repository rather than to out/.

    Meshing is the slow part and the views are cheap, so everything is built
    once and looked at from many directions. out/ is generated output that is
    never committed; these are documentation, so they live under docs/.
    """
    os.makedirs(out_dir, exist_ok=True)
    a, ch = design.assembly, design.chamber
    stamp = (f"{spec.get('name')}   {ch.thrust_sea_level / 1e3:.1f} kN sl   "
             f"Isp {ch.isp_sea_level:.0f}/{ch.isp_vacuum:.0f} s   "
             f"{a.overall_length:.0f} x {2 * a.overall_radius:.0f} mm")
    written = []

    print("meshing the closed engine")
    closed = build_parts(design, voxel, 0.0, tris)
    for az, name in ((-60.0, "01-three-quarter"), (0.0, "02-side"),
                     (-120.0, "03-rear-quarter"), (180.0, "04-head-end")):
        elev = 18.0 if name != "04-head-end" else 6.0
        written.append(render_view(
            closed, elev, az, f"{name.split('-', 1)[1].replace('-', ' ')}",
            os.path.join(out_dir, f"{name}.png"), stamp))
        print("  wrote", written[-1], flush=True)
    del closed

    print("meshing the cutaway")
    cut = build_parts(design, voxel, sweep, tris)
    for elev, az, name in ((18.0, -60.0, "05-cutaway"),
                           (-22.0, 62.0, "06-into-the-section"),
                           (60.0, -70.0, "07-from-above")):
        written.append(render_view(
            cut, elev, az, name.split("-", 1)[1].replace("-", " "),
            os.path.join(out_dir, f"{name}.png"), stamp))
        print("  wrote", written[-1], flush=True)
    del cut

    print("meshing the throat detail")
    lip = a.contour.lip_x
    detail = build_parts(design, detail_voxel, sweep, tris,
                         x_window=(lip - 26.0, lip + 6.0),
                         keep_sector=(math.radians(-14.0), math.radians(14.0)),
                         decimate_to=False)
    written.append(render_view(
        detail, 16.0, -72.0, "throat detail: cooling channels in both walls",
        os.path.join(out_dir, "08-throat-detail.png"), stamp))
    print("  wrote", written[-1], flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--voxel", type=float, default=0.45)
    ap.add_argument("--sweep", type=float, default=260.0)
    ap.add_argument("--tris", type=int, default=70000)
    ap.add_argument("--detail-voxel", type=float, default=0.12)
    ap.add_argument("--gallery", default="", help="write a set of views here")
    ap.add_argument("--name", default="engine_complete")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    d = design_engine(spec)

    if args.gallery:
        gallery(d, spec, args.gallery, args.voxel, args.detail_voxel,
                args.tris, args.sweep)
        return
    a, ch = d.assembly, d.chamber
    os.makedirs(args.out, exist_ok=True)

    print(f"meshing {spec.get('name')} at {args.voxel} mm, {args.sweep:g} deg sweep")
    parts = build_parts(d, args.voxel, args.sweep, args.tris)

    allv = np.concatenate([v for v, _ in parts.values()])
    xr = (float(allv[:, 0].min()), float(allv[:, 0].max()))
    rr = float(max(abs(allv[:, 1]).max(), abs(allv[:, 2]).max()))

    lip = a.contour.lip_x
    detail_span = (lip - 26.0, lip + 6.0)
    print(f"detail window x {detail_span[0]:.1f}..{detail_span[1]:.1f} at "
          f"{args.detail_voxel} mm")
    wedge = (math.radians(-14.0), math.radians(14.0))
    detail = build_parts(d, args.detail_voxel, args.sweep, args.tris,
                         x_window=detail_span, keep_sector=wedge,
                         decimate_to=False)

    views = ((18.0, -60.0, "cut open: the whole engine"),
             (-22.0, 62.0, "looking into the section"),
             (16.0, -72.0, "throat detail, 28 deg wedge: channels in both walls"))
    fig = plt.figure(figsize=(21, 7.6))
    for k, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        parts_here = detail if k == 2 else parts
        xr_here = detail_span if k == 2 else xr
        # One collection for everything. Separate collections are depth-sorted
        # against each other as wholes, so the spike appears to punch through
        # the cowl wherever their bounding volumes interleave; merged, every
        # triangle sorts on its own depth and the section reads correctly.
        tris = np.concatenate([v[f] for v, f in parts_here.values()])
        cols = np.concatenate([_shade(v[f], COLOUR[part], elev, azim)
                               for part, (v, f) in parts_here.items()])
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=cols, edgecolors="none", linewidths=0.0))
        # Limits from the data, not a symmetric radius. A narrow wedge lives in
        # one corner of the (y, z) square, so framing it symmetrically shrinks
        # it to a thumbnail in the middle of a lot of empty axes.
        lo = tris.reshape(-1, 3).min(axis=0)
        hi = tris.reshape(-1, 3).max(axis=0)
        pad = 0.03 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title(title, fontsize=10, y=1.02)

    n_cowl = d.circuits["cowl"].channel.n_channels if d.circuits["cowl"] else 0
    n_body = d.circuits["centrebody"].channel.n_channels if d.circuits["centrebody"] else 0
    n_inj = 2 * d.injector.n_elements if d.injector else 0
    fig.suptitle(
        f"{spec.get('name')}   {ch.thrust_sea_level / 1e3:.1f} kN sl / "
        f"{ch.thrust_vacuum / 1e3:.1f} kN vac   Isp {ch.isp_sea_level:.0f}/"
        f"{ch.isp_vacuum:.0f} s   pc {ch.chamber_pressure / 1e5:.0f} bar   "
        f"{a.overall_length:.0f} x {2 * a.overall_radius:.0f} mm   "
        f"{n_cowl}+{n_body} cooling channels, {n_inj} injector orifices",
        fontsize=13)
    out = os.path.join(args.out, f"{args.name}.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
