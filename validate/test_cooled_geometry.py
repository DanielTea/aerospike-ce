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
    Genus = 1 for the bored disc plus one per orifice. The orifices are three or
    four voxels across at 0.4 mm, which is enough for marching cubes to keep the
    topology. The 0.4 mm cooling channels are not, which is why they are checked
    against the field instead of against a mesh.
    """
    from mesh_export import manifold_report
    from mesh_solid import build_mesh

    a, inj = design.assembly, design.injector
    v, f = build_mesh(a.profiles["head"], voxel_mm=0.4,
                      holes=injector_holes(a, inj))
    rep = manifold_report(v, f)
    assert rep["watertight"]
    genus = (2 - rep["euler"]) // 2
    assert genus == 1 + 2 * inj.n_elements
