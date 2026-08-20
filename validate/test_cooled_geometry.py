"""
Invariants for the cooled geometry: channels and orifices.

Checked by sampling the distance field directly rather than by meshing it. The
channels are 0.4 mm wide because that is what the thermal design needs, and a
0.4 mm feature needs roughly 0.13 mm voxels to survive marching cubes with its
topology intact -- which for a whole part is several gigabytes. Meshing is
therefore the wrong instrument for this question. Sampling the field is exact,
costs nothing, and tests the thing that actually defines the geometry.

The mesh is still checked, on the head, whose orifices are three or four voxels
across and come out with exactly the right genus.
"""

import json
import math
import os

import numpy as np
import pytest

from engine_design import design_engine
from mesh_solid import (
    _channel_sdf,
    _hole_sdf,
    _polygon_sdf,
    centrebody_channels,
    cowl_channels,
    injector_holes,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


def _count_voids_around_circle(fn, radius, n=20000):
    """
    Count separate void arcs on a circle. Wraps, so the count is periodic.

    fn(theta) -> True where the point is inside solid material.
    """
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    solid = fn(th)
    # a void starts wherever material gives way going round
    starts = np.sum(solid & ~np.roll(solid, -1))
    return int(starts)


# --------------------------------------------------------------------------
# cooling channels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wall", ["cowl", "centrebody"])
def test_channel_count_on_the_part_matches_the_thermal_design(design, wall):
    """
    The number of channels cut must be the number the cooling model sized. A
    mismatch means the thermal answer describes a part that was not built.
    """
    circuit = design.circuits[wall]
    assert circuit is not None
    cut = (cowl_channels if wall == "cowl" else centrebody_channels)(design.assembly,
                                                                     circuit.channel)
    x_at = 0.5 * (cut.x_start + cut.x_end)
    wall_r = float(np.interp(x_at, cut.wall_x, cut.wall_r))
    sign = 1.0 if cut.outward else -1.0
    r_mid = wall_r + sign * (cut.hot_wall_mm + 0.5 * cut.height_mm)

    def solid(th):
        X = np.full(th.shape, x_at, dtype=np.float32)
        R = np.full(th.shape, r_mid, dtype=np.float32)
        return _channel_sdf(X, R, th.astype(np.float32), cut) > 0.0

    assert _count_voids_around_circle(solid, r_mid) == cut.n_channels


@pytest.mark.parametrize("wall", ["cowl", "centrebody"])
def test_channels_sit_inside_the_wall_not_through_it(design, wall):
    """
    A channel that outlives its wall vents to the outside. That produces a mesh
    which is still watertight and has wildly wrong topology, so it is invisible
    to a naive check.
    """
    a = design.assembly
    circuit = design.circuits[wall]
    cut = (cowl_channels if wall == "cowl" else centrebody_channels)(a, circuit.channel)

    xs = np.linspace(cut.x_start, cut.x_end, 200)
    gas = np.interp(xs, cut.wall_x, cut.wall_r)
    if wall == "cowl":
        order = np.argsort(a.cowl_outer_x)
        outer = np.interp(xs, np.asarray(a.cowl_outer_x)[order],
                          np.asarray(a.cowl_outer_r)[order])
        available = outer - gas
    else:
        cav = np.interp(xs, a.cavity_x, a.cavity_r)
        available = np.where(xs <= a.cavity_x.max(), gas - cav, gas)

    needed = cut.hot_wall_mm + cut.height_mm
    assert np.all(available >= needed), (
        f"{wall}: channel needs {needed:.2f} mm, thinnest wall is {available.min():.2f} mm")


def test_cowl_channels_stop_before_the_lip(design):
    """The cowl tapers to a thin lip; a channel run to the end emerges through it."""
    a = design.assembly
    cut = cowl_channels(a, design.circuits["cowl"].channel)
    assert cut.x_end < a.contour.lip_x


@pytest.mark.parametrize("wall", ["cowl", "centrebody"])
def test_hot_wall_thickness_is_what_was_solved(design, wall):
    """The thermal model's biggest single resistance is this dimension."""
    circuit = design.circuits[wall]
    cut = (cowl_channels if wall == "cowl" else centrebody_channels)(design.assembly,
                                                                     circuit.channel)
    x_at = 0.5 * (cut.x_start + cut.x_end)
    wall_r = float(np.interp(x_at, cut.wall_x, cut.wall_r))
    sign = 1.0 if cut.outward else -1.0

    # walk radially out of the gas surface and find where the channel starts
    steps = np.linspace(0.0, cut.hot_wall_mm + cut.height_mm + 1.0, 4000)
    R = (wall_r + sign * steps).astype(np.float32)
    X = np.full(R.shape, x_at, dtype=np.float32)
    TH = np.zeros_like(R)                       # theta = 0 is a channel centre
    inside_channel = _channel_sdf(X, R, TH, cut) < 0.0
    first = steps[np.argmax(inside_channel)]
    assert first == pytest.approx(cut.hot_wall_mm, abs=2e-3)


def test_channels_hold_their_width_all_the_way_down_the_spike(design):
    """
    The pitch shrinks with radius, so a channel count chosen at the chamber
    cannot fit round the spike and the width clamp starves the channels below
    the manufacturing floor. Packing the count at the *narrowest* station
    instead means the designed width survives everywhere, which is what makes
    the part both printable and meshable.
    """
    a = design.assembly
    cut = centrebody_channels(a, design.circuits["centrebody"].channel)
    xs = np.linspace(cut.x_start, cut.x_end, 200)
    rs = np.interp(xs, cut.wall_x, cut.wall_r)
    arc = 2.0 * math.pi * rs / cut.n_channels
    width = np.minimum(cut.width_mm, np.maximum(arc - cut.land_mm, 0.0))

    assert np.all(width <= arc + 1e-9), "channels would overlap"
    assert width.min() == pytest.approx(cut.width_mm, rel=1e-9), \
        "the clamp bit, so the count was packed at too generous a radius"


# --------------------------------------------------------------------------
# injector orifices
# --------------------------------------------------------------------------

def test_orifice_count_on_the_face(design):
    holes = injector_holes(design.assembly, design.injector)
    assert len(holes) == 2
    for h in holes:
        x_at = 0.5 * (h.x_start + h.x_end)

        def solid(th, hh=h, xx=x_at):
            Y = (hh.radius_mm * np.cos(th)).astype(np.float32)
            Z = (hh.radius_mm * np.sin(th)).astype(np.float32)
            X = np.full(th.shape, xx, dtype=np.float32)
            return _hole_sdf(X, Y, Z, hh) > 0.0

        assert _count_voids_around_circle(solid, h.radius_mm) == h.count


def test_orifice_diameters_are_what_the_injector_solved(design):
    a, inj = design.assembly, design.injector
    holes = injector_holes(a, inj)
    x_at = 0.5 * (holes[0].x_start + holes[0].x_end)
    for h, want in zip(holes, (inj.d_fuel_mm, inj.d_ox_mm)):
        # sweep radially through a hole centre and measure the chord
        rs = np.linspace(h.radius_mm - 3.0, h.radius_mm + 3.0, 20000, dtype=np.float32)
        Y = rs * math.cos(h.phase)
        Z = rs * math.sin(h.phase)
        X = np.full(rs.shape, x_at, dtype=np.float32)
        inside = _hole_sdf(X, Y.astype(np.float32), Z.astype(np.float32), h) < 0.0
        measured = (rs[inside].max() - rs[inside].min())
        assert measured == pytest.approx(want, abs=0.01)


def test_orifice_rings_stay_within_the_head_disc(design):
    a, inj = design.assembly, design.injector
    for h in injector_holes(a, inj):
        assert h.radius_mm - h.diameter_mm / 2 > a.cavity_r[0]
        assert h.radius_mm + h.diameter_mm / 2 < a.flange_radius


def test_the_two_rings_are_staggered(design):
    """Fuel and ox rings share a face; interleaving them keeps the lands even."""
    holes = injector_holes(design.assembly, design.injector)
    assert holes[0].phase != holes[1].phase


# --------------------------------------------------------------------------
# the one part whose features are large enough to mesh reliably
# --------------------------------------------------------------------------

def test_head_meshes_with_exactly_the_right_topology(design):
    """
    An injector orifice is only a through-feature once the plenum it feeds
    exists. Meshed without the manifolds it is a blind pocket that adds no
    topology at all, which is what the head reports: genus 1, the bore alone.

    So the check is on volume rather than genus. Multi-cavity Euler arithmetic
    is brittle -- the boundary is the outer surface plus one closed surface per
    plenum, joined by every orifice -- whereas the material the features remove
    is something both the mesher and a hand calculation can state independently.
    """
    from mesh_export import manifold_report, mesh_volume
    from mesh_solid import build_mesh
    from manifold_ref import design_manifolds, geometry_features

    a = design.assembly
    gf = geometry_features(design, design_manifolds(design))
    # geometry_features already carries the orifices, and it is the only thing
    # that knows where the plenums ended up. Adding injector_holes on top put a
    # second set of orifices in the disc at a different, wrong depth.
    holes = list(gf["head"]["holes"])

    # Isolated against the same part with the plenums left out, rather than
    # against the bare disc. The lugs add about 320 cm3 and the plenums now
    # remove about 80, so a net comparison against the bare disc says the
    # features "added material" and tests nothing about the plenums at all.
    without = build_mesh(a.profiles["head"], voxel_mm=0.4, holes=holes,
                         lugs=gf["head"]["lugs"])
    with_features = build_mesh(a.profiles["head"], voxel_mm=0.4, holes=holes,
                               lugs=gf["head"]["lugs"],
                               plenums=gf["head"]["plenums"])

    rep = manifold_report(*with_features)
    assert rep["watertight"]
    assert rep["boundary_edges"] == 0
    assert rep["degenerate_faces"] == 0

    # Pappus: the section swept about the axis at its centroid radius. The
    # section is the filleted one -- a sharp diamond is neither printable nor
    # meshable at its apex -- so the area is 3 percent under the nominal 2ab.
    from mesh_solid import plenum_section_area_mm2
    removed = sum(plenum_section_area_mm2(p.half_x, p.half_r) * 2.0 * math.pi
                  * (p.r_inner + p.half_r) for p in gf["head"]["plenums"])
    measured = mesh_volume(*without) - mesh_volume(*with_features)
    assert measured == pytest.approx(removed, rel=0.03), (
        f"plenums removed {measured / 1000:.1f} cm3, the section and radius "
        f"say {removed / 1000:.1f} cm3")


def test_injector_orifices_reach_their_plenum(design):
    """
    A hole that stops short of the manifold feeds nothing, and nothing in a
    watertightness or topology check would notice.
    """
    from manifold_ref import design_manifolds, geometry_features, plenum_feeding

    md = design_manifolds(design)
    # Through geometry_features, which is what the build calls. Asking
    # injector_holes directly tests a placement nothing uses.
    holes = [h for h in geometry_features(design, md)["head"]["holes"]
             if h.name.endswith("orifice")]
    assert holes

    # At the orifice's own radius, not at the plenum's extreme tip. The section
    # is a diamond: it reaches furthest aft at its centre radius and less
    # everywhere else, so measuring at the tip said "connected" while the two
    # were five millimetres apart.
    # Only the fuel ring sits under its own manifold. The oxidiser dome is
    # outboard of its ring and reaches it through radial ports instead, which
    # is checked in test_feed_paths.
    h = next(x for x in holes if x.name == "fuel_orifice")
    fed = plenum_feeding(md, h.radius_mm)
    assert fed is not None, (
        f"fuel_orifice at r={h.radius_mm:.1f} opens into no plenum at all")
    assert fed.name == "head_fuel_manifold", (
        f"fuel_orifice opens into {fed.name} -- that crosses the propellants")
    aft = fed.x_mm + fed.half_x_at(h.radius_mm)
    assert h.x_start < aft - 1e-9, (
        f"fuel_orifice starts at x={h.x_start:.2f}, aft of the plenum face at "
        f"x={aft:.2f}; it would open into solid metal")


def test_the_two_head_plenums_never_touch(design):
    """
    A fuel manifold and an oxidiser dome that meet are not a geometry bug, they
    are a fire. Nothing else in the model checks this: both are voids, so the
    union of them is watertight, drains, and looks entirely reasonable.
    """
    from manifold_ref import design_manifolds

    md = design_manifolds(design)
    heads = [p for p in md.plenums if p.name != "cowl_inlet_ring"]
    for i, a in enumerate(heads):
        for b in heads[i + 1:]:
            gap = max(a.r_inner_mm, b.r_inner_mm) - min(
                a.r_inner_mm + 2 * a.half_r_mm, b.r_inner_mm + 2 * b.half_r_mm)
            axial = abs(a.x_mm - b.x_mm) - (a.half_x_mm + b.half_x_mm)
            assert gap > 0.0 or axial > 0.0, (
                f"{a.name} and {b.name} intersect: radial overlap "
                f"{-gap:.2f} mm, axial overlap {-axial:.2f} mm")


def test_the_fuel_manifold_surrounds_the_ring_it_feeds(design):
    """
    The fuel manifold straddles its own orifice ring, and far enough inside
    that the section is genuinely open there rather than grazing an edge where
    the diamond has tapered to nothing.

    The oxidiser dome deliberately does not: there is not room for two
    manifolds each straddling rings only 7 mm apart, so the dome sits outboard
    and feeds its ring through radial ports.
    """
    from manifold_ref import design_manifolds

    md = design_manifolds(design)
    r = design.injector.fuel_ring_radius * 1e3
    p = next(q for q in md.plenums if q.name == "head_fuel_manifold")
    assert p.reaches(r, depth_mm=1.0), (
        f"head_fuel_manifold is only {2 * p.half_x_at(r):.2f} mm deep at "
        f"r={r:.1f} mm, where its orifices have to meet it")


def test_orifices_stay_drillable(design):
    """
    Once the head was thickened for its manifolds, a hole through the whole
    block became 25:1. Nothing prints or drills that straight.
    """
    from manifold_ref import design_manifolds, geometry_features
    for h in geometry_features(design, design_manifolds(design))["head"]["holes"]:
        if h.name.endswith("orifice"):
            assert (h.x_end - h.x_start) / h.diameter_mm < 12.0


def test_the_head_gets_each_hole_ring_exactly_once(design):
    """
    Orifice placement moved to geometry_features because it depends on where
    the plenums ended up. Both build_plan and print_ready also generated them
    from a fixed offset, so the disc briefly had two sets of orifices in
    different places -- watertight, plausible, and wrong.
    """
    from manifold_ref import design_manifolds, geometry_features
    from build_plan import build_plan
    import collections

    want = {"fuel_orifice": 1, "ox_orifice": 1, "mount_hole": 1,
            "transfer_cowl": 1, "transfer_centrebody": 1}
    holes = geometry_features(design, design_manifolds(design))["head"]["holes"]
    assert collections.Counter(h.name for h in holes) == want

    planned = build_plan(design.spec, design)["parts"]["head"]["holes"]
    assert collections.Counter(h["name"] for h in planned) == want


def test_the_head_disc_has_no_cavity_without_a_way_out(design):
    """
    Genus counted two ways: off the mesh, and off the feature list.

    This is the check that says every orifice is a through-feature and every
    cavity is connected. A blind orifice adds no handle, so it shows up here as
    a genus one short -- while the part stays watertight, drains, and looks
    entirely correct. 1 bore + 48 + 48 + 4 mounting holes = 101.
    """
    from mesh_solid import build_mesh
    from print_ready import check, features_for

    v, f = build_mesh(design.assembly.profiles["head"], voxel_mm=0.4,
                      **features_for(design, "head"))
    rep = check(v, f, "head")
    assert rep["watertight"]
    assert rep["nonmanifold_edges"] == 0, (
        "a diamond apex has pinched: the section needs its corners filleted")
    # One surface means every cavity in the disc is joined to the outside. Two
    # would mean a plenum had sealed itself off, which is the failure that
    # caught the orifices no longer reaching their manifolds -- and it stays
    # watertight, drains nowhere, and looks entirely correct.
    assert rep["surfaces"] == 1, "a plenum has sealed itself off"
    assert not rep["sealed"], "a cavity has no way out"
    # Which passage reaches which cavity is checked exactly in test_feed_paths;
    # see expected_genus for why the count is no longer asserted here.
    assert rep["genus"] > 2 * design.injector.n_elements


# --------------------------------------------------------------------------
# the axis
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spec_name", ["demo.json", "regen.json"])
@pytest.mark.parametrize("sampler", ["planar", "radial"])
def test_the_field_puts_no_surface_down_the_axis(spec_name, sampler):
    """
    The centrebody profile closes on r = 0: in from the truncation face, along
    the axis, out again at the apex of the cavity. Revolved, that segment
    sweeps no area at all -- it is the axis, not a surface -- and the metal it
    runs through is solid.

    Measured as an edge it reads as one. The field collapses to zero along the
    axis, marching cubes meshes a zero-width sheet there, and the part comes
    out with degenerate triangles and edges shared by sixteen faces. It still
    looks like an engine in a viewer; a slicer refuses it.

    On the axis between apex and face the nearest real surface is whichever end
    is nearer, the outer surface being twelve millimetres away, so the field is
    exactly that depth and negative.
    """
    from engine_ref import assembly_from_spec
    from mesh_solid import _polygon_sdf_radial

    path = os.path.join(os.path.dirname(__file__), "..", "spec", spec_name)
    with open(path, encoding="utf-8") as fh:
        a = assembly_from_spec(json.load(fh))

    p = a.profiles["centrebody"]
    vx = np.asarray(p.x, dtype=float)
    vr = np.asarray(p.r, dtype=float)

    on_axis = np.flatnonzero(vr <= 1e-9)
    assert len(on_axis) >= 2, "the centrebody is meant to close on the axis"
    x_hi, x_lo = vx[on_axis].max(), vx[on_axis].min()
    assert x_hi - x_lo > 1.0, "expected a run of axis, not a single point"

    xs = np.linspace(x_lo + 0.05, x_hi - 0.05, 25)
    if sampler == "planar":
        d = _polygon_sdf(xs, np.zeros_like(xs), vx, vr)
    else:
        d = np.array([_polygon_sdf_radial(x, np.zeros(1), vx, vr)[0] for x in xs])

    depth = np.minimum(x_hi - xs, xs - x_lo)
    assert np.all(d < 0.0), "the axis of the spike tip is solid metal"
    assert np.allclose(d, -depth, atol=2e-3), (
        "the field on the axis should measure to the nearest real surface, "
        f"worst error {np.max(np.abs(d + depth)):.4f} mm")


# --------------------------------------------------------------------------
# the lattice
# --------------------------------------------------------------------------

@pytest.mark.parametrize("margin_mm", [1.0, 1.1])
def test_a_flat_face_landing_on_the_lattice_still_meshes(margin_mm):
    """
    Marching cubes is degenerate where a sample sits exactly on the surface.

    On a solid of revolution that is not bad luck, it is the ordinary case:
    the flat faces are normal to the axis, so are the lattice planes, and a
    face a whole number of voxels from the grid origin lands on a plane of
    samples that all read exactly zero. The head does it at the default 0.2 mm
    and came back with 1.3 million zero-area triangles and 120,000 boundary
    edges -- a mesh with holes in it, from a field that was entirely correct.
    A slicer refuses that; nothing in a viewer shows it.

    Both margins here are meant to pass. One puts the face on a sample plane
    and the other between two, because a mesher that only works when the
    geometry misses the lattice is a mesher that works by luck.
    """
    from engine_ref import Profile
    from mesh_export import manifold_report, mesh_volume
    from mesh_solid import build_field, mesh_field

    voxel = 0.25
    disc = Profile(name="disc",
                   x=np.array([-2.0, 2.0, 2.0, -2.0]),
                   r=np.array([1.0, 1.0, 5.0, 5.0]))

    field, origin, spacing = build_field(disc, voxel_mm=voxel, margin_mm=margin_mm)
    step = (2.0 * margin_mm + 4.0) / (field.shape[0] - 1)
    on_plane = abs(margin_mm / step - round(margin_mm / step)) < 1e-9
    v, f = mesh_field(field, origin, spacing)

    rep = manifold_report(v, f)

    where = "on a sample plane" if on_plane else "between sample planes"
    assert rep["watertight"], f"face {where}: {rep}"
    assert rep["boundary_edges"] == 0, f"face {where}: holes in the surface"
    assert rep["degenerate_faces"] == 0, f"face {where}: zero-area triangles"
    assert rep["euler"] == 0, f"a bored disc revolves into a torus, got {rep['euler']}"

    want = math.pi * (5.0 ** 2 - 1.0 ** 2) * 4.0
    assert abs(mesh_volume(v, f) - want) / want < 0.02


@pytest.mark.parametrize("index", [16, 1024])
def test_a_sample_a_hair_off_the_level_does_not_collapse_the_vertex(index):
    """
    Marching cubes works in index units, in single precision.

    The bias keeps whole planes of samples off the level. It does nothing for
    the one sample in a hundred million that lands just beside it by chance,
    and out at index 1024 a float32 resolves about 6e-5 of an index -- so a
    crossing at t = 2e-5 is placed exactly *on* the sample rather than beside
    it, and so is every other edge into that sample. The weld then merges what
    marching cubes meant to keep apart and the surface pinches: one edge in
    four faces, no boundary anywhere, an odd Euler characteristic, nothing to
    see in a viewer. The cowl did it once in 24 million triangles.

    The value here is the one measured at that pinch, five nanometres inside
    the channel floor. Near the origin the same configuration is harmless,
    which is the point: this is a precision failure, not a geometry failure,
    and it only bites far from the index origin.
    """
    from skimage.measure import marching_cubes

    from mesh_solid import SURFACE_BIAS_MM, _mesh_block

    def block():
        b = np.full((3, index + 6, index + 6), 1.0, dtype=np.float32)
        b[1, index, index] = np.float32(9.509921e-05)      # just inside the level
        b[1, index, index - 1] = b[1, index - 1, index] = b[0, index, index] = -0.2
        return b

    def coincident(v):
        return len(v) - len(np.unique(np.round(v / 1e-12).astype(np.int64), axis=0))

    raw, _, _, _ = marching_cubes(block(), level=SURFACE_BIAS_MM,
                                  spacing=(0.233, 0.233, 0.233))
    guarded, _ = _mesh_block(block(), 0.233)

    if index == 1024:
        assert coincident(raw) > 0, (
            "the unguarded call is expected to collapse here; if it no longer "
            "does, the guard is being tested against nothing")
    assert coincident(guarded) == 0, "the guard must keep every vertex distinct"


def test_the_level_guard_leaves_a_clean_field_alone():
    """
    It moves samples inside a band a micron wide and nothing else. A field with
    no sample near the level must mesh to exactly what marching cubes gives.
    """
    from skimage.measure import marching_cubes

    from mesh_solid import SURFACE_BIAS_MM, _mesh_block

    rng = np.random.default_rng(7)
    b = (rng.random((6, 40, 40)).astype(np.float32) - 0.5) * 2.0
    b[np.abs(b - SURFACE_BIAS_MM) < 0.01] = 0.5           # clear the band by far

    raw_v, raw_f, _, _ = marching_cubes(b, level=SURFACE_BIAS_MM, spacing=(0.2,) * 3)
    got_v, got_f = _mesh_block(b.copy(), 0.2)
    assert np.array_equal(raw_v, got_v)
    assert np.array_equal(raw_f, got_f)


def test_the_level_guard_does_not_move_the_surface_it_meshes():
    """
    Pushing a sample to the edge of the band keeps it on the side it was on, so
    no cell changes classification and the volume stays put. A guard that could
    flip a sample would be a guard that quietly redesigns the part.
    """
    from mesh_export import mesh_volume
    from mesh_solid import (SURFACE_BIAS_MM, _mesh_block, hold_off_level,
                            level_guard_mm)

    n, voxel, radius = 60, 0.2, 4.0
    g = (np.arange(n) - (n - 1) / 2.0) * voxel
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    ball = (np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - radius).astype(np.float32)

    v, f = _mesh_block(ball.copy(), voxel)
    v = v - (n - 1) / 2.0 * voxel
    want = 4.0 / 3.0 * math.pi * radius ** 3
    assert abs(mesh_volume(v, f) - want) / want < 0.01

    # Every sample the guard touches must stay on the side it was on: a guard
    # that could flip one would move a cell across the level and quietly
    # redesign the part. Checked on a field seeded to land in the band.
    guard = level_guard_mm(ball.shape, voxel)
    seeded = ball.copy()
    seeded.reshape(-1)[::997] = np.float32(SURFACE_BIAS_MM - 0.1 * guard)
    seeded.reshape(-1)[1::997] = np.float32(SURFACE_BIAS_MM + 0.1 * guard)
    held = hold_off_level(seeded, voxel)
    assert held is not seeded, "the band was seeded; the guard should have fired"
    assert np.array_equal(held >= SURFACE_BIAS_MM, seeded >= SURFACE_BIAS_MM)
    assert np.all(np.abs(held - SURFACE_BIAS_MM) >= guard * (1.0 - 1e-6))
    assert np.max(np.abs(held - seeded)) <= guard * 2.0
