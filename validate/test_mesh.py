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
    export_assembly,
    manifold_report,
    mesh_volume,
    read_binary_stl,
    revolve,
    simplify,
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
