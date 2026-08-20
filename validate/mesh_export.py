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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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

# Per-face vector maths is done a slice at a time. Gathering all three corners
# of a twenty-four-million-triangle mesh costs 1.7 GB before any arithmetic
# happens, and there are four routines here that want to do it; the print build
# was killed by the kernel for the sum of them. A couple of million faces at a
# time is large enough that the loop costs nothing measurable and small enough
# that the working set stays in the tens of megabytes.
FACE_CHUNK = 2_000_000


def _face_slices(n: int, chunk: int = FACE_CHUNK):
    for i in range(0, max(n, 1), chunk):
        yield slice(i, min(i + chunk, n))


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Signed volume by the divergence theorem, mm^3. Positive if outward."""
    total = 0.0
    for sl in _face_slices(len(f)):
        g = f[sl]
        a, b, c = v[g[:, 0]], v[g[:, 1]], v[g[:, 2]]
        total += float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))))
    return total / 6.0


def mesh_area(v: np.ndarray, f: np.ndarray) -> float:
    total = 0.0
    for sl in _face_slices(len(f)):
        g = f[sl]
        a, b, c = v[g[:, 0]], v[g[:, 1]], v[g[:, 2]]
        total += float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)))
    return total / 2.0


def face_area_stats(v: np.ndarray, f: np.ndarray):
    """
    How many triangles have no area, and how little the smallest one has.

    Both come out of one pass and neither needs the areas kept: the count is
    what the gate reads and the minimum is what tells you how close the rest
    came.
    """
    zero, smallest = 0, np.inf
    for sl in _face_slices(len(f)):
        g = f[sl]
        a, b, c = v[g[:, 0]], v[g[:, 1]], v[g[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        zero += int(np.count_nonzero(area <= 0.0))
        if len(area):
            smallest = min(smallest, float(area.min()))
    return zero, (0.0 if smallest is np.inf else smallest)


def _edge_keys(f: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Every edge of every triangle as one sorted integer, and the vertex count.

    One int64 per edge rather than a pair, because `np.unique(..., axis=0)` on
    a 72-million-row array of pairs views it as a void dtype and sorts that,
    which costs several gigabytes of copies. Packing the two indices into one
    integer -- they are both well under 2^31, so `lo * n + hi` is exact -- makes
    it an ordinary 1-D sort that can be done in place.
    """
    n = int(f.max()) + 1 if len(f) else 1
    key = np.empty(3 * len(f), dtype=np.int64)
    m = len(f)
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        u = f[:, i].astype(np.int64, copy=False)
        w = f[:, j].astype(np.int64, copy=False)
        key[k * m:(k + 1) * m] = np.minimum(u, w) * n + np.maximum(u, w)
    return key, n


def _run_lengths(sorted_key: np.ndarray) -> np.ndarray:
    """Length of each run of equal values in a sorted array."""
    if not len(sorted_key):
        return np.zeros(0, dtype=np.int64)
    edges = np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1
    bounds = np.empty(len(edges) + 2, dtype=np.int64)
    bounds[0], bounds[-1] = 0, len(sorted_key)
    bounds[1:-1] = edges
    return np.diff(bounds)


def manifold_report(v: np.ndarray, f: np.ndarray) -> dict:
    """
    Edge and Euler bookkeeping, and a count of triangles with no area.

    A closed surface has every edge in exactly two triangles. The Euler
    characteristic then names the topology: 2 for a sphere, 0 for a torus. A
    part whose profile touches the axis revolves into a sphere; one whose
    profile does not -- anything with a through bore -- revolves into a torus.
    Getting 1 or an odd number means the mesh has a hole in it.

    None of that sees a triangle with three collinear vertices: by index it is
    a perfectly ordinary face. A slicer disagrees -- it has no normal to offset
    and no side to be inside of, and a plane of them is what Cura means by
    missing or extraneous surfaces. They are counted here because the edge
    arithmetic passed a mesh that would not slice.
    """
    key, n = _edge_keys(f)
    key.sort()
    counts = _run_lengths(key)
    del key
    # bincount rather than np.unique(f): it reads the faces once into n bins
    # instead of sorting a 72-million-element copy of them.
    n_v = int(np.count_nonzero(np.bincount(f.reshape(-1), minlength=n)))
    n_e, n_f = len(counts), len(f)
    return {
        "vertices": n_v,
        "edges": n_e,
        "faces": n_f,
        "euler": n_v - n_e + n_f,
        "watertight": bool(np.all(counts == 2)),
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "degenerate_faces": face_area_stats(v, f)[0],
    }


def _face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


# --------------------------------------------------------------------------
# what a slicer needs that the edge arithmetic does not see
# --------------------------------------------------------------------------

def duplicate_faces(f: np.ndarray) -> int:
    """
    Triangles occupying the same three vertices as another triangle.

    Two coincident faces keep every edge count at two, so watertightness is
    happy. A slicer walking the surface arrives at the same place twice and
    disagrees with itself about which side is solid.
    """
    if not len(f):
        return 0
    n = int(f.max()) + 1
    a = f.astype(np.int64, copy=False)
    lo = a.min(axis=1)
    hi = a.max(axis=1)
    mid = a.sum(axis=1) - lo - hi
    # Two integers rather than a three-column sort, for the same reason the
    # edges are packed: np.unique over rows views them as a void dtype and
    # copies the lot. Three indices will not fit in one int64 at this vertex
    # count, so it is a lexsort over two.
    first = lo * n + mid
    order = np.lexsort((hi, first))
    same = (first[order][1:] == first[order][:-1]) & \
           (hi[order][1:] == hi[order][:-1])
    return int(np.count_nonzero(same))


def inconsistent_edges(f: np.ndarray) -> int:
    """
    Interior edges not traversed once in each direction.

    On a consistently wound closed surface every edge is shared by two
    triangles that walk it in opposite directions. Where two neighbours agree
    on the direction instead, one of them has its normal flipped: locally the
    mesh still closes, and the slicer reads solid and void the wrong way round
    across that patch.
    """
    if not len(f):
        return 0
    tail = f.reshape(-1)
    head = f[:, [1, 2, 0]].reshape(-1)
    n = int(f.max()) + 1
    key = np.minimum(tail, head).astype(np.int64) * n + \
        np.maximum(tail, head).astype(np.int64)
    forward = tail < head
    order = np.argsort(key, kind="stable")
    runs = _run_lengths(key[order])
    two = _run_starts(runs)[runs == 2]
    fwd = forward[order]
    return int(np.count_nonzero(
        fwd[two].astype(np.int8) + fwd[two + 1].astype(np.int8) != 1))


def _run_starts(runs: np.ndarray) -> np.ndarray:
    """Index at which each run begins."""
    starts = np.empty(len(runs), dtype=np.int64)
    if len(runs):
        starts[0] = 0
        np.cumsum(runs[:-1], out=starts[1:])
    return starts


def _next_corner(c: np.ndarray) -> np.ndarray:
    """The corner after `c` in its own triangle, by arithmetic on the index."""
    base = (c // 3) * 3
    return (base + (c - base + 1) % 3).astype(np.int32, copy=False)


def nonmanifold_vertices(f: np.ndarray) -> int:
    """
    Vertices where the surface pinches: two cones meeting at a single point.

    Every edge is still in exactly two triangles, so `manifold_report` passes
    it, and the Euler characteristic only betrays it when the number of extra
    fans is odd. A slicer that has to decide what is inside at that point has
    two answers.

    Counted by fans rather than by vertices. Around a manifold vertex the
    incident triangles form one closed ring; around a pinch they form two or
    more. The triangle corners are the nodes and each shared edge joins the two
    corners that sit on it, so the number of connected components is the number
    of fans, and anything above one apiece is a pinch.

    Written to stay inside a few gigabytes at print resolution. The corner
    indices are arithmetic rather than a stored array, the graph is int32 with
    one entry per join rather than two, and everything is freed the moment it
    stops being needed -- a twenty-four-million-triangle part took thirteen
    gigabytes the obvious way, which is what the kernel killed the print build
    for.
    """
    if not len(f):
        return 0
    n = int(f.max()) + 1
    n_he = 3 * len(f)
    tail = f.reshape(-1)
    head = f[:, [1, 2, 0]].reshape(-1)

    key = np.minimum(tail, head).astype(np.int64) * n + \
        np.maximum(tail, head).astype(np.int64)
    order = np.argsort(key, kind="stable").astype(np.int32)
    runs = _run_lengths(key[order])
    del key
    starts = _run_starts(runs)
    # Only edges shared by exactly two faces join anything. Anything else is a
    # manifold failure the edge count already reports by itself.
    two = starts[runs == 2]
    del runs, starts
    a, b = order[two], order[two + 1]
    del order, two

    same = tail[a] == tail[b]                     # do they enter from the same end
    hb = _next_corner(b)
    rows = np.concatenate([a, _next_corner(a)])
    cols = np.concatenate([np.where(same, b, hb), np.where(same, hb, b)])
    del a, b, same, hb

    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                   shape=(n_he, n_he))
    del rows, cols
    fans, _ = connected_components(g, directed=False)
    del g
    used = int(np.count_nonzero(np.bincount(f.reshape(-1), minlength=n)))
    return int(fans - used)


def slicing_report(v: np.ndarray, f: np.ndarray, deep: bool = True) -> dict:
    """
    The gate a slicer applies, which is not the gate a topologist applies.

    Everything here passes `manifold_report`. A zero-area triangle has two
    neighbours per edge; a duplicated face keeps every count at two; a flipped
    patch closes perfectly; a pinch point shares no edge with itself. All four
    produce a file that opens in a viewer, looks like the part, and slices into
    something that is not it.

    `deep` turns on the fan count, which is the only expensive one here: it
    walks a graph over every triangle corner and costs a few gigabytes on a
    twenty-million-triangle part. Left off inside a decimation ladder, where
    the genus comparison already catches a new pinch, and turned on once per
    part before anything is written.
    """
    zero_area, smallest = face_area_stats(v, f)
    return {
        "degenerate_faces": zero_area,
        "duplicate_faces": duplicate_faces(f),
        "inconsistent_edges": inconsistent_edges(f),
        "nonmanifold_vertices": nonmanifold_vertices(f) if deep else 0,
        "min_face_area_mm2": smallest,
        "finite": bool(np.isfinite(v).all()),
        "indices_in_range": bool(len(f) == 0 or int(f.max()) < len(v)),
        "outward": bool(mesh_volume(v, f) > 0.0),
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


# --------------------------------------------------------------------------
# 3MF
# --------------------------------------------------------------------------

def write_3mf(path: str, parts: dict, unit: str = "millimeter") -> None:
    """
    Write parts as a single 3MF, the format additive manufacturing actually uses.

    STL is a triangle soup with no units in it. Every slicer guesses millimetres
    and is usually right, which is not the same as being told -- and a part that
    silently arrives at a twenty-fifth of its size is a real way to waste a
    build. 3MF states its unit, carries each part as a named object, and zips,
    which matters when the geometry has several hundred cooling channels in it.

    `parts` maps a name to (vertices, faces).
    """
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>')
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>')

    # Streamed into the zip rather than assembled first. An engine with several
    # hundred cooling channels in it runs to millions of triangles, and at
    # roughly 55 bytes of markup each that is a gigabyte of XML: joining it into
    # one string costs several times that in peak memory, to produce a value
    # that is written once and thrown away.
    def emit(w) -> None:
        w.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        w.write(f'<model unit="{unit}" xml:lang="en-US" '
                'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
                '<resources>'.encode())
        for oid, (name, (verts, faces)) in enumerate(parts.items(), start=1):
            w.write(f'<object id="{oid}" type="model" name="{name}">'
                    '<mesh><vertices>'.encode())
            # 4 decimals is a tenth of a micron: far under any printer, and it
            # keeps the file from being mostly trailing digits.
            for i in range(0, len(verts), 65536):
                w.write("".join(
                    f'<vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>'
                    for v in verts[i:i + 65536]).encode())
            w.write(b'</vertices><triangles>')
            for i in range(0, len(faces), 65536):
                w.write("".join(
                    f'<triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>'
                    for f in faces[i:i + 65536]).encode())
            w.write(b'</triangles></mesh></object>')
        w.write(b'</resources><build>')
        for oid in range(1, len(parts) + 1):
            w.write(f'<item objectid="{oid}"/>'.encode())
        w.write(b'</build></model>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        # force_zip64 because the member, uncompressed, is past 4 GB: this
        # engine is 36 million triangles and 3MF spells each one out in XML.
        # Without it zipfile gets to the end of a fifteen-minute build and
        # refuses to close the archive.
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as w:
            emit(w)


def read_3mf(path: str) -> dict:
    """
    Read a 3MF back, so the writer is checked rather than trusted.

    Parsed incrementally. A whole engine at print resolution is tens of
    millions of elements and about a gigabyte of XML once decompressed; running
    a regex over that as one string needs several gigabytes to answer a
    question that only ever needs one element at a time.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    out: dict = {"unit": None, "objects": {}}
    with zipfile.ZipFile(path) as z, z.open("3D/3dmodel.model") as fh:
        name, vx, tri = None, [], []
        for event, el in ET.iterparse(fh, events=("start", "end")):
            tag = el.tag.rsplit("}", 1)[-1]
            if event == "start":
                if tag == "model":
                    out["unit"] = el.get("unit")
                elif tag == "object":
                    name, vx, tri = el.get("name"), [], []
                continue
            if tag == "vertex":
                vx.append((float(el.get("x")), float(el.get("y")), float(el.get("z"))))
            elif tag == "triangle":
                tri.append((int(el.get("v1")), int(el.get("v2")), int(el.get("v3"))))
            elif tag == "object":
                out["objects"][name] = (
                    np.asarray(vx, dtype=float).reshape(-1, 3),
                    np.asarray(tri, dtype=np.int64).reshape(-1, 3))
                name, vx, tri = None, [], []
            # Elements are dropped as they close; kept, the tree grows back into
            # the whole file and the streaming buys nothing.
            el.clear()
    return out
