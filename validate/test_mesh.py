"""
Mesh-level invariants for the exported solids.

The 2D tests prove the profiles are right. These prove the revolve turns them
into something a slicer will accept: closed, correctly oriented, of the right
topology, and enclosing the volume the maths says it should.
"""

import json
import math
import os

import numpy as np
import pytest

from engine_ref import assembly_from_spec
from mesh_export import (
    duplicate_faces,
    export_assembly,
    inconsistent_edges,
    manifold_report,
    mesh_volume,
    nonmanifold_vertices,
    read_binary_stl,
    revolve,
    simplify,
    slicing_report,
    triangle_soup_volume,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "demo.json")


@pytest.fixture(scope="module")
def engine():
    with open(SPEC, encoding="utf-8") as fh:
        return assembly_from_spec(json.load(fh))


@pytest.fixture(scope="module")
def meshes(engine):
    return {name: revolve(p, n_theta=120) for name, p in engine.profiles.items()}


# --------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------

def test_every_part_is_watertight(meshes):
    """
    No boundary edges: an open shell is not a solid and will not slice.

    Nor any triangle without area. Those pass every count here -- they have
    three vertices and three edges like any other face -- and are what a slicer
    reports as a missing surface.
    """
    for name, (v, f) in meshes.items():
        rep = manifold_report(v, f)
        assert rep["watertight"], f"{name}: {rep}"
        assert rep["boundary_edges"] == 0, name
        assert rep["nonmanifold_edges"] == 0, name
        assert rep["degenerate_faces"] == 0, f"{name}: {rep['degenerate_faces']} flat triangles"


def test_topology_is_what_the_profile_implies(meshes, engine):
    """
    A profile that touches the axis revolves into a sphere (euler 2). One with a
    through bore revolves into a torus (euler 0). Anything else means the axis
    fan or the seam weld is broken.
    """
    expected = {"centrebody": 2, "cowl": 0, "head": 0}
    for name, (v, f) in meshes.items():
        assert manifold_report(v, f)["euler"] == expected[name], name


def test_normals_point_outward(meshes):
    for name, (v, f) in meshes.items():
        assert mesh_volume(v, f) > 0.0, name


# --------------------------------------------------------------------------
# the mesh encloses the volume the maths predicts
# --------------------------------------------------------------------------

def _facet_deficit(n_theta: int) -> float:
    """
    Relative volume a regular inscribed n-gon loses against its circle.

    area_n / area_circle = (n / 2pi) * sin(2pi/n) = 1 - (2*pi^2/3)/n^2 + O(n^-4)

    This is the honest tolerance for any revolved mesh. Picking a round number
    instead would either pass a broken mesh at high n or fail a correct one at
    low n, so the bound is derived rather than guessed.
    """
    return 2.0 * math.pi ** 2 / 3.0 / (n_theta ** 2)


def test_mesh_volume_matches_the_analytic_volume(meshes, engine):
    """
    Divergence-theorem volume of the mesh against the Pappus volume of the
    profile. Different routes entirely, so agreement to the faceting bound
    means the revolve lost nothing.
    """
    bound = _facet_deficit(120)
    for name, (v, f) in meshes.items():
        exact = engine.profiles[name].revolved_volume
        got = mesh_volume(v, f)
        err = abs(got - exact) / exact
        assert err < 1.5 * bound, f"{name}: {err:.3e} vs bound {bound:.3e}"
        # inscribed, so the mesh must under-fill rather than over-fill
        assert got <= exact * (1.0 + 1e-9), name


def test_faceting_error_shrinks_quadratically(engine):
    """
    If refining the revolve does not converge on the analytic volume at the
    expected rate, the residual is a modelling bug rather than faceting.
    """
    profile = engine.profiles["cowl"]
    exact = profile.revolved_volume
    errs = []
    for n in (32, 64, 128):
        v, f = revolve(profile, n_theta=n)
        errs.append(abs(mesh_volume(v, f) - exact) / exact)
        assert errs[-1] < 1.5 * _facet_deficit(n)
    assert errs[0] > errs[1] > errs[2]
    # doubling the step count should quarter the error
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.3)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.3)


# --------------------------------------------------------------------------
# simplification must not move the surface
# --------------------------------------------------------------------------

def test_simplify_respects_its_tolerance(engine):
    p = engine.profiles["centrebody"]
    x, r = simplify(np.asarray(p.x), np.asarray(p.r), tol=0.005)
    assert len(x) < len(p.x)
    # every discarded point must lie within tolerance of the kept polyline
    from engine_ref import _distance_to_polyline
    d = _distance_to_polyline(np.asarray(p.x), np.asarray(p.r), x, r)
    assert d.max() <= 0.005 + 1e-9


def test_simplify_keeps_the_endpoints(engine):
    p = engine.profiles["cowl"]
    x, r = simplify(np.asarray(p.x), np.asarray(p.r), tol=0.02)
    assert (x[0], r[0]) == (p.x[0], p.r[0])
    assert (x[-1], r[-1]) == (p.x[-1], p.r[-1])


# --------------------------------------------------------------------------
# the file on disk
# --------------------------------------------------------------------------

def test_stl_round_trips(engine, tmp_path):
    """
    The binary STL writer is hand-rolled against the format spec. Read the bytes
    back and recover the same triangle count and the same enclosed volume.
    """
    rep = export_assembly(engine, str(tmp_path), name="t", n_theta=90)
    for part, r in rep.items():
        if part == "assembly":
            continue
        tri, nrm = read_binary_stl(r["path"])
        assert len(tri) == r["triangles"], part
        assert triangle_soup_volume(tri) == pytest.approx(
            r["mesh_volume_mm3"], rel=1e-4
        ), part
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-4), part


def test_stl_files_are_written_for_every_part(engine, tmp_path):
    rep = export_assembly(engine, str(tmp_path), name="t", n_theta=60)
    for part in ("centrebody", "cowl", "head", "assembly"):
        assert os.path.exists(rep[part]["path"]), part
        assert os.path.getsize(rep[part]["path"]) > 84


def test_assembly_volume_is_the_sum_of_its_parts(engine, tmp_path):
    rep = export_assembly(engine, str(tmp_path), name="t", n_theta=90)
    parts = sum(r["mesh_volume_mm3"] for k, r in rep.items() if k != "assembly")
    assert rep["assembly"]["mesh_volume_mm3"] == pytest.approx(parts, rel=1e-9)


# --------------------------------------------------------------------------
# 3MF, the format the print file is actually written in
# --------------------------------------------------------------------------

def _two_parts():
    """A tetrahedron and a translated copy, as two named solids."""
    v = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                  [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return {"first": (v, f), "second": (v + np.array([25.0, 0.0, 0.0]), f)}


def test_3mf_round_trips_exactly(tmp_path):
    """
    Both the writer and the reader stream, because a whole engine is about a
    gigabyte of XML. Streaming is where an off-by-one in the vertex indices or
    a dropped chunk boundary would hide, and either produces a file that opens
    cleanly and is not the part.
    """
    from mesh_export import read_3mf, write_3mf

    parts = _two_parts()
    p = tmp_path / "two.3mf"
    write_3mf(str(p), parts)
    back = read_3mf(str(p))

    assert back["unit"] == "millimeter"
    assert set(back["objects"]) == set(parts)
    for name, (v, f) in parts.items():
        v2, f2 = back["objects"][name]
        assert np.allclose(v2, v, atol=1e-4)
        assert np.array_equal(f2, f)


def test_3mf_survives_a_chunk_boundary(tmp_path):
    """
    The writer emits in 65536-element blocks. A mesh larger than one block is
    the case that exercises the seam; a tetrahedron never would.
    """
    from mesh_export import read_3mf, write_3mf

    n = 70000
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3)) * 10.0
    f = np.arange(3 * (n // 3), dtype=np.int64).reshape(-1, 3)
    p = tmp_path / "big.3mf"
    write_3mf(str(p), {"slab": (v, f)})
    v2, f2 = read_3mf(str(p))["objects"]["slab"]

    assert len(v2) == n
    assert np.array_equal(f2, f)
    assert np.allclose(v2, v, atol=1e-4)


def test_3mf_is_a_well_formed_package(tmp_path):
    """The three members a 3MF reader looks for, and the relationship between."""
    import zipfile

    from mesh_export import write_3mf

    p = tmp_path / "pkg.3mf"
    write_3mf(str(p), _two_parts())
    with zipfile.ZipFile(p) as z:
        names = set(z.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= names
        rels = z.read("_rels/.rels").decode()
        assert "/3D/3dmodel.model" in rels
        model = z.read("3D/3dmodel.model").decode()
    # every object is placed in the build, or a slicer opens an empty plate
    assert model.count("<item objectid=") == 2


def test_3mf_states_millimetres(tmp_path):
    """
    STL carries no unit and every slicer guesses. The guess is usually right,
    which is not the same as being told -- a part that silently arrives at a
    twenty-fifth of its size is a wasted build.
    """
    import zipfile

    from mesh_export import write_3mf

    p = tmp_path / "unit.3mf"
    write_3mf(str(p), _two_parts())
    with zipfile.ZipFile(p) as z:
        assert 'unit="millimeter"' in z.read("3D/3dmodel.model").decode()


# --------------------------------------------------------------------------
# the gate a slicer applies
# --------------------------------------------------------------------------

def _cube():
    """Unit cube, consistently wound outward. Every check here should pass it."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _tetrahedron(origin):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) \
        + np.asarray(origin, dtype=float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return v, f


def test_a_clean_solid_passes_every_slicing_check():
    v, f = _cube()
    rep = slicing_report(v, f)
    assert rep["degenerate_faces"] == 0
    assert rep["duplicate_faces"] == 0
    assert rep["inconsistent_edges"] == 0
    assert rep["nonmanifold_vertices"] == 0
    assert rep["finite"] and rep["indices_in_range"] and rep["outward"]


def test_a_pinched_vertex_survives_every_edge_count():
    """
    Two tetrahedra meeting at one point.

    Every edge is still in exactly two triangles, so the mesh is watertight by
    the only test that used to be applied to it. There is no edge to be
    non-manifold: the defect is at a vertex, where the surface has two sides
    and the slicer has to pick one.
    """
    v1, f1 = _tetrahedron([0, 0, 0])
    v2, f2 = _tetrahedron([-1, 0, 0])                 # its vertex 1 lands on 0, 0, 0
    v = np.concatenate([v1, np.delete(v2, 1, axis=0)])
    f = np.concatenate([f1, np.array([[4, 0, 5, 6][i] for i in f2.ravel()]).reshape(-1, 3)])

    rep = manifold_report(v, f)
    assert rep["watertight"], "the pinch does not break a single edge pairing"
    assert rep["nonmanifold_edges"] == 0
    assert nonmanifold_vertices(f) == 1
    # and the parity of the Euler characteristic is the free half of the same
    # finding: two closed surfaces should sum to 4, not 3.
    assert rep["euler"] % 2 == 1


def test_a_flipped_triangle_closes_and_still_slices_wrong():
    """
    One face wound the other way. The mesh stays closed -- the edges are all
    still shared by two triangles -- and three of them are now traversed twice
    in the same direction, which is a patch of surface with its inside out.
    """
    v, f = _cube()
    f[0] = f[0][::-1]
    assert manifold_report(v, f)["watertight"]
    assert inconsistent_edges(f) == 3


def test_a_duplicated_face_is_named_as_one():
    v, f = _cube()
    doubled = np.concatenate([f, f[:1]])
    assert duplicate_faces(doubled) == 1
    assert duplicate_faces(f) == 0


def test_a_zero_area_triangle_is_counted_not_ignored():
    v, f = _cube()
    v = np.concatenate([v, [[0.5, 0.0, 0.0]]])        # collinear with edge 0-1
    f = np.concatenate([f, [[0, 1, 8]]])
    assert slicing_report(v, f)["degenerate_faces"] == 1


def test_an_inside_out_solid_is_refused():
    v, f = _cube()
    assert slicing_report(v, f[:, ::-1].copy())["outward"] is False


def test_every_revolved_part_passes_the_slicing_gate(meshes):
    for name, (v, f) in meshes.items():
        rep = slicing_report(v, f)
        assert rep["degenerate_faces"] == 0, name
        assert rep["duplicate_faces"] == 0, name
        assert rep["inconsistent_edges"] == 0, name
        assert rep["nonmanifold_vertices"] == 0, name
        assert rep["outward"], name


def test_what_is_checked_is_what_reaches_the_file():
    """
    A sliver with area in memory and none on disk.

    3MF carries coordinates as decimal text, so writing quantises. A triangle
    whose corners are closer together than the written quantum has area right
    up until it is written and none afterwards -- and checking the mesh in
    memory and then writing it means the thing that was checked and the thing a
    slicer opens are two different meshes. They were: the first print file
    built with the level guard in place reported no zero-area triangles
    anywhere in memory and came back off disk with 416.

    Here vertex 8 sits a nanometre along the diagonal from vertex 0, which
    splits the two faces meeting there into a pair of ordinary triangles and a
    pair of slivers. Every area is positive. None of them survives six decimal
    places.
    """
    import tempfile

    from mesh_export import WRITE_DECIMALS, read_3mf, write_3mf
    from print_ready import snap_to_the_written_grid

    v, f = _cube()
    v = np.concatenate([v, [v[0] + 1e-9 * (v[2] - v[0])]])
    keep = [row for row in f.tolist() if row not in ([0, 3, 2], [0, 2, 1])]
    f = np.array(keep + [[0, 3, 8], [8, 3, 2], [0, 8, 1], [8, 2, 1]])

    assert manifold_report(v, f)["watertight"]
    assert manifold_report(v, f)["degenerate_faces"] == 0, "every area is positive here"
    assert manifold_report(np.round(v, WRITE_DECIMALS), f)["degenerate_faces"] == 2, (
        "and two of them are gone the moment the file is written")

    sv, sf = snap_to_the_written_grid(v, f)
    rep = manifold_report(sv, sf)
    assert rep["watertight"], "the weld must not open the surface"
    assert rep["boundary_edges"] == 0
    assert rep["degenerate_faces"] == 0, "the slivers are gone, not merely rounded"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapped.3mf")
        write_3mf(path, {"cube": (sv, sf)})
        got_v, got_f = read_3mf(path)["objects"]["cube"]
    assert np.array_equal(got_v, sv), "the file must carry what was checked"
    assert np.array_equal(got_f, sf)
    assert manifold_report(got_v, got_f)["degenerate_faces"] == 0


# --------------------------------------------------------------------------
# the field, asked away from the lattice
# --------------------------------------------------------------------------
#
# build_field answers on a grid, which is all the mesher ever needed. Asking a
# mesh how far it has drifted from the surface it claims to be is a question
# about points that are nowhere near a lattice, so the composition moved into
# sample_field and both callers go through it. These prove the move did not
# change the answer, and that the drift it now measures is the real distance.

def _test_part():
    """A stub with one of everything the composition can subtract."""
    from engine_ref import Profile
    from mesh_solid import HoleCut

    # A cylinder 20 mm long and 8 mm in radius, wound counter-clockwise in
    # (x, r), drilled by a ring of six 1.5 mm holes running right through it.
    prof = Profile(name="stub",
                   x=np.array([0.0, 20.0, 20.0, 0.0]),
                   r=np.array([0.0, 0.0, 8.0, 8.0]))
    holes = [HoleCut(radius_mm=5.0, diameter_mm=1.5, count=6,
                     x_start=-1.0, x_end=21.0, name="test")]
    return prof, dict(holes=holes)


def test_the_scattered_sampler_agrees_with_the_grid_builder():
    """
    Same composition, two callers, one answer.

    The grid path runs in float32 and the scattered path in float64, so they
    are allowed to differ by float32's own resolution and by nothing else. A
    real divergence here would mean the mesher and the gate that checks it had
    started describing different solids -- which is the failure the seam
    between them exists to prevent.
    """
    from mesh_solid import build_field, part_sampler

    prof, feats = _test_part()
    field, origin, voxel = build_field(prof, voxel_mm=0.6, **feats)

    # The grid is the origin plus whole voxels, which is the only
    # reconstruction that can be right: a field whose returned spacing is not
    # the spacing it was built on cannot be sampled again by anyone.
    nx, ny, nz = field.shape
    xs = origin[0] + np.arange(nx) * voxel
    ys = origin[1] + np.arange(ny) * voxel
    zs = origin[2] + np.arange(nz) * voxel
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    got = part_sampler(prof, **feats)(pts).reshape(field.shape)
    assert np.abs(got - field).max() < 1e-3, "the two paths describe different solids"


def test_deviation_reads_zero_on_the_surface_and_the_offset_off_it():
    """
    The measurement has to be the distance, not a proxy for it.

    A sphere meshed from its own field sits on that field, so it reads the
    level and nothing more. Push every vertex out by a tenth of a millimetre
    and it has to read a tenth of a millimetre -- otherwise a gate built on it
    would pass a mesh that had drifted exactly that far.
    """
    from mesh_solid import SURFACE_BIAS_MM, field_deviation
    from skimage.measure import marching_cubes

    n, half = 48, 12.0
    h = 2.0 * half / (n - 1)
    g = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    v, f, _, _ = marching_cubes(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 8.0,
                                level=SURFACE_BIAS_MM, spacing=(h, h, h))
    centre = np.array([half, half, half])

    def sample(p):
        return (np.linalg.norm(np.asarray(p) - centre, axis=1) - 8.0
                + SURFACE_BIAS_MM)

    on = field_deviation(v, f, sample)
    # Marching cubes is linear along an edge and a sphere is not, so the
    # centres of the triangles sit measurably inside it. That is the mesher's
    # error and it scales with the voxel; what matters is that it is a
    # fraction of one and not a multiple.
    assert on["max_mm"] < h, f"a mesh on its own field reads {on['max_mm']:.3f} mm out"

    outward = v - centre
    outward /= np.linalg.norm(outward, axis=1)[:, None]
    off = field_deviation(v + 0.1 * outward, f, sample)
    assert off["rms_mm"] == pytest.approx(0.1, abs=0.02), \
        "a mesh pushed a tenth of a millimetre off the surface did not say so"


def test_deviation_looks_hardest_where_the_triangles_are_biggest():
    """
    The budget is spent where chordal error lives.

    Half the sample is a stride over the whole mesh and half goes to the
    largest triangles, because a collapse leaves its surviving corners exactly
    on the surface and the flat triangle between them cutting the corner off
    whatever curve they spanned. Move one big triangle's corner and the
    measurement has to notice, even though it is one face in thousands.
    """
    from mesh_solid import SURFACE_BIAS_MM, field_deviation
    from skimage.measure import marching_cubes

    n, half = 48, 12.0
    h = 2.0 * half / (n - 1)
    g = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    v, f, _, _ = marching_cubes(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - 8.0,
                                level=SURFACE_BIAS_MM, spacing=(h, h, h))
    centre = np.array([half, half, half])

    def sample(p):
        return (np.linalg.norm(np.asarray(p) - centre, axis=1) - 8.0
                + SURFACE_BIAS_MM)

    before = field_deviation(v, f, sample)["max_mm"]
    moved = v.copy()
    # Drag one vertex a millimetre off the surface, which also makes the faces
    # around it the largest in the mesh.
    moved[f[0, 0]] += (moved[f[0, 0]] - centre) / np.linalg.norm(
        moved[f[0, 0]] - centre)
    after = field_deviation(moved, f, sample)["max_mm"]
    assert after > before + 0.5, \
        f"a vertex a millimetre out went unnoticed ({before:.3f} -> {after:.3f})"
