"""
Revolve the meridional profiles into a real triangle mesh and write STL.

Why this exists
---------------
The C# path builds voxels in PicoGK and exports from there, which is the real
pipeline. But PicoGK needs a native runtime and a GPU viewer, so on a machine
without them there is no way to check that the assembly closes into a solid at
all. This module is the fallback channel: pure Python, no native dependency,
and it produces both an STL you can slice and a PNG a code agent can read.

It is a checking tool and a deliverable, not a second source of truth. The
geometry still comes from engine_ref.py. If this file and SpikeGeometry.cs ever
disagree about the shape, both are wrong until engine_ref.py says otherwise.

The verification that matters is volume closure: the mesh volume computed by the
divergence theorem must match the analytic Pappus volume of the profile that
generated it. A mesh that is inside out, unwelded at the seam, or missing the
axis fan fails that check immediately.
"""

from __future__ import annotations

import math
import os
import struct

import numpy as np

from engine_ref import EngineAssembly, Profile

AXIS_EPS = 1e-9


# --------------------------------------------------------------------------
# profile simplification
# --------------------------------------------------------------------------

def simplify(x: np.ndarray, r: np.ndarray, tol: float = 0.005):
    """
    Douglas-Peucker decimation of a meridional profile.

    The Angelino contour clusters points hard at the shoulder, which is right
    for accuracy and wasteful for a mesh: revolving 770 vertices at 180 steps is
    a quarter of a million triangles, most of them describing a straight line.
    A 5 micron tolerance is well under any printable feature and typically cuts
    the count by an order of magnitude.
    """
    n = len(x)
    if n < 3:
        return x, r

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        px, pr = x[i], r[i]
        qx, qr = x[j], r[j]
        dx, dr = qx - px, qr - pr
        seg = math.hypot(dx, dr)
        if seg < 1e-15:
            d = np.hypot(x[i + 1:j] - px, r[i + 1:j] - pr)
        else:
            d = np.abs(dr * (x[i + 1:j] - px) - dx * (r[i + 1:j] - pr)) / seg
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))

    return x[keep], r[keep]


# --------------------------------------------------------------------------
# revolution
# --------------------------------------------------------------------------

def revolve(profile: Profile, n_theta: int = 180, tol: float = 0.005):
    """
    Revolve a closed meridional profile into a watertight triangle mesh.

    Vertices are shared between angular steps and welded at the seam, so the
    result is index-manifold rather than a soup of loose triangles. Profile
    vertices sitting on the axis collapse to a single shared vertex and their
    edges emit a triangle fan instead of a degenerate quad.
    """
    x, r = simplify(np.asarray(profile.x), np.asarray(profile.r), tol)
    n = len(x)

    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    verts: list[np.ndarray] = []
    ring: list[np.ndarray | int] = []
    for i in range(n):
        if r[i] < AXIS_EPS:
            ring.append(len(verts))                      # one shared axis vertex
            verts.append(np.array([x[i], 0.0, 0.0]))
        else:
            base = len(verts)
            ring.append(base + np.arange(n_theta))
            block = np.empty((n_theta, 3))
            block[:, 0] = x[i]
            block[:, 1] = r[i] * cos_t
            block[:, 2] = r[i] * sin_t
            verts.extend(block)

    v = np.asarray(verts, dtype=float)
    faces: list[tuple[int, int, int]] = []
    k1 = np.roll(np.arange(n_theta), -1)

    for i in range(n):
        j = (i + 1) % n
        a_axis = r[i] < AXIS_EPS
        b_axis = r[j] < AXIS_EPS
        if a_axis and b_axis:
            continue                                      # edge lies on the axis
        if a_axis:
            apex = ring[i]
            b0, b1 = ring[j], ring[j][k1]
            faces.extend(zip([apex] * n_theta, b0, b1))
        elif b_axis:
            apex = ring[j]
            a0, a1 = ring[i], ring[i][k1]
            faces.extend(zip(a0, [apex] * n_theta, a1))
        else:
            a0, a1 = ring[i], ring[i][k1]
            b0, b1 = ring[j], ring[j][k1]
            faces.extend(zip(a0, b0, b1))
            faces.extend(zip(a0, b1, a1))

    f = np.asarray(faces, dtype=np.int64)

    # orientation: revolving a CCW profile should give outward normals, but the
    # cost of being wrong is a solid that slices inside out, so check and fix
    if mesh_volume(v, f) < 0.0:
        f = f[:, ::-1].copy()
    return v, f


# --------------------------------------------------------------------------
# mesh checks
# --------------------------------------------------------------------------

def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Signed volume by the divergence theorem, mm^3. Positive if outward."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def mesh_area(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)) / 2.0)


def manifold_report(v: np.ndarray, f: np.ndarray) -> dict:
    """
    Edge and Euler bookkeeping.

    A closed surface has every edge in exactly two triangles. The Euler
    characteristic then names the topology: 2 for a sphere, 0 for a torus. A
    part whose profile touches the axis revolves into a sphere; one whose
    profile does not -- anything with a through bore -- revolves into a torus.
    Getting 1 or an odd number means the mesh has a hole in it.
    """
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    used = np.unique(f)
    n_v, n_e, n_f = len(used), len(uniq), len(f)
    return {
        "vertices": n_v,
        "edges": n_e,
        "faces": n_f,
        "euler": n_v - n_e + n_f,
        "watertight": bool(np.all(counts == 2)),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
    }


# --------------------------------------------------------------------------
# STL
# --------------------------------------------------------------------------

def write_binary_stl(path: str, v: np.ndarray, f: np.ndarray, name: str = "aerospike") -> None:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln < 1e-20] = 1.0
    nrm /= ln[:, None]

    rec = np.zeros(len(f), dtype=np.dtype([
        ("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3), ("c", "<f4", 3),
        ("attr", "<u2"),
    ]))
    rec["n"], rec["a"], rec["b"], rec["c"] = nrm, a, b, c

    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


def read_binary_stl(path: str):
    """
    Read back a binary STL as (triangles, normals).

    The writer above is hand-rolled against the format spec, so nothing proves
    it is right except reading the bytes back and recovering the same solid.
    """
    with open(path, "rb") as fh:
        fh.read(80)
        (count,) = struct.unpack("<I", fh.read(4))
        rec = np.frombuffer(fh.read(count * 50), dtype=np.dtype([
            ("n", "<f4", 3), ("a", "<f4", 3), ("b", "<f4", 3), ("c", "<f4", 3),
            ("attr", "<u2"),
        ]))
    tri = np.stack([rec["a"], rec["b"], rec["c"]], axis=1).astype(float)
    return tri, rec["n"].astype(float)


def triangle_soup_volume(tri: np.ndarray) -> float:
    """Signed volume of loose triangles, for meshes read back without indices."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def export_assembly(a: EngineAssembly, out_dir: str, name: str = "engine",
                    n_theta: int = 180, tol: float = 0.005) -> dict:
    """Write one STL per part plus a combined assembly, and report the checks."""
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    offset = 0

    for part, profile in a.profiles.items():
        v, f = revolve(profile, n_theta=n_theta, tol=tol)
        path = os.path.join(out_dir, f"{name}_{part}.stl")
        write_binary_stl(path, v, f, f"{name}_{part}")

        rep = manifold_report(v, f)
        rep["mesh_volume_mm3"] = mesh_volume(v, f)
        rep["exact_volume_mm3"] = profile.revolved_volume
        rep["volume_error"] = abs(rep["mesh_volume_mm3"] - rep["exact_volume_mm3"]) / \
            abs(rep["exact_volume_mm3"])
        rep["path"] = path
        rep["triangles"] = len(f)
        report[part] = rep

        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

    v = np.concatenate(all_v)
    f = np.concatenate(all_f)
    path = os.path.join(out_dir, f"{name}_assembly.stl")
    write_binary_stl(path, v, f, f"{name}_assembly")
    report["assembly"] = {
        "path": path,
        "triangles": len(f),
        "mesh_volume_mm3": mesh_volume(v, f),
        "exact_volume_mm3": sum(p.revolved_volume for p in a.profiles.values()),
    }
    return report


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------

PART_COLOUR = {"centrebody": "#4a7fb5", "cowl": "#c2703a", "head": "#8a8a8a"}


def render_cutaway(a: EngineAssembly, path: str, sweep_deg: float = 270.0,
                   n_theta: int = 120, tol: float = 0.02) -> str:
    """
    Render a cutaway of the assembly to PNG.

    Swept short of a full turn on purpose. A closed revolve just shows an outer
    skin, and the whole point of checking the picture is to see the things the
    skin hides: wall thickness, the internal cavity, the base cap, and whether
    the cowl actually stands off the spike at the throat.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sweep = math.radians(sweep_deg)
    theta = np.linspace(0.0, sweep, n_theta)

    fig = plt.figure(figsize=(16, 8))
    # left: looking into the removed quadrant. right: the closed outer envelope.
    ax = fig.add_subplot(121, projection="3d")
    ax_out = fig.add_subplot(122, projection="3d")

    xs_all, rs_all = [], []
    for name, profile in a.profiles.items():
        x, r = simplify(np.asarray(profile.x), np.asarray(profile.r), tol)
        x = np.append(x, x[0])
        r = np.append(r, r[0])
        xs_all.append(x)
        rs_all.append(r)

        X = np.repeat(x[:, None], n_theta, axis=1)
        Y = r[:, None] * np.cos(theta)[None, :]
        Z = r[:, None] * np.sin(theta)[None, :]
        ax.plot_surface(X, Y, Z, color=PART_COLOUR[name], shade=True,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)

        full = np.linspace(0.0, 2.0 * math.pi, n_theta)
        ax_out.plot_surface(np.repeat(x[:, None], n_theta, axis=1),
                            r[:, None] * np.cos(full)[None, :],
                            r[:, None] * np.sin(full)[None, :],
                            color=PART_COLOUR[name], shade=True,
                            linewidth=0, antialiased=True, rstride=1, cstride=1)

        # the two cut faces, so the section reads as solid rather than hollow
        for ang in (0.0, sweep):
            face = np.stack([x, r * math.cos(ang), r * math.sin(ang)], axis=1)
            pc = Poly3DCollection([face], facecolor=PART_COLOUR[name],
                                  edgecolor="#22262b", linewidths=0.4, alpha=1.0)
            ax.add_collection3d(pc)

    x_all = np.concatenate(xs_all)
    r_max = max(float(np.max(r)) for r in rs_all)

    # The sweep leaves theta in (sweep_deg, 360) open. For the default 270 that
    # is the y > 0, z < 0 quadrant, so the camera goes to positive azimuth and
    # negative elevation to look straight into the section.
    for axis, elev, azim, title in (
        (ax, -24.0, 48.0, f"{sweep_deg:g} deg cutaway, looking into the section"),
        (ax_out, 20.0, -62.0, "closed envelope"),
    ):
        axis.set_xlim(x_all.min(), x_all.max())
        axis.set_ylim(-r_max, r_max)
        axis.set_zlim(-r_max, r_max)
        axis.set_box_aspect((x_all.max() - x_all.min(), 2 * r_max, 2 * r_max))
        axis.view_init(elev=elev, azim=azim)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.set_title(title, fontsize=10, y=1.04)

    fig.suptitle(f"L={a.overall_length:.1f} mm   D={2 * a.overall_radius:.1f} mm   "
                 f"total {sum(p.revolved_volume for p in a.profiles.values()) / 1000:.1f} cm3",
                 fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    import argparse
    import json

    from engine_ref import assembly_from_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/demo.json")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--theta-steps", type=int, default=180)
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    a = assembly_from_spec(spec)
    name = spec.get("name", "engine")
    rep = export_assembly(a, args.out, name=name, n_theta=args.theta_steps)

    for part, r in rep.items():
        if part == "assembly":
            continue
        print(f"{part:12s} tris={r['triangles']:7d}  watertight={str(r['watertight']):5s}  "
              f"euler={r['euler']:+d}  vol={r['mesh_volume_mm3'] / 1000:7.3f} cm3  "
              f"err={r['volume_error']:.2e}")
    asm = rep["assembly"]
    print(f"{'assembly':12s} tris={asm['triangles']:7d}  "
          f"vol={asm['mesh_volume_mm3'] / 1000:7.3f} cm3")
    for part, r in rep.items():
        if part != "assembly":
            print(f"wrote {r['path']}")
    print(f"wrote {asm['path']}")
    print(f"wrote {render_cutaway(a, os.path.join(args.out, 'engine_3d.png'))}")


if __name__ == "__main__":
    main()
