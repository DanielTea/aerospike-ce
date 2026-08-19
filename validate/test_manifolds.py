"""
Invariants for the manifolds and the mounts.

A plenum is an internal void in a printed part, so its shape is constrained
harder by the process than by the flow. The load-bearing checks here are that
the section can hold its own roof up, that it is big enough not to be the
restriction, and that a lug fails the way lugs actually fail -- bearing and
tear-out, not tension.
"""

import json
import math
import os

import pytest

from engine_design import design_engine
from manifold_ref import (
    BEARING_ALLOWABLE_PA,
    SHEAR_ALLOWABLE_PA,
    design_manifolds,
    geometry_features,
    size_mount_lugs,
    size_plenum,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


@pytest.fixture(scope="module")
def manifolds(design):
    return design_manifolds(design)


# --------------------------------------------------------------------------
# the section has to hold itself up
# --------------------------------------------------------------------------

def test_every_plenum_is_self_supporting(manifolds):
    """
    A round or square plenum presents a flat roof its own width across, and
    nothing can reach inside a closed ring to support it. Diamond sections at or
    above the process angle are the whole reason for the shape.
    """
    for p in manifolds.plenums:
        assert p.self_supporting(), f"{p.name} roof at {p.roof_angle_deg:.0f} deg"


def test_a_flat_plenum_is_refused_not_returned():
    with pytest.raises(ValueError, match="45 degrees"):
        size_plenum("flat", 0.4, 423.0, 100.0, 0.0, feeds=100, aspect=0.6)


def test_roof_angle_follows_the_aspect_ratio():
    tall = size_plenum("tall", 0.4, 423.0, 100.0, 0.0, feeds=100, aspect=2.0)
    square = size_plenum("square", 0.4, 423.0, 100.0, 0.0, feeds=100, aspect=1.0)
    assert tall.roof_angle_deg > square.roof_angle_deg
    assert square.roof_angle_deg == pytest.approx(45.0, abs=0.5)


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------

def test_plenum_area_delivers_the_target_velocity():
    """
    Round trip: size for a velocity, then measure it back. Half the flow,
    because the ring splits both ways from a single inlet.
    """
    mdot, rho, v = 0.414, 423.0, 4.0
    p = size_plenum("t", mdot, rho, 100.0, 0.0, feeds=392, velocity=v)
    measured = 0.5 * mdot / (rho * p.area_mm2 * 1e-6)
    assert measured == pytest.approx(v, rel=1e-9)


def test_a_plenum_is_not_the_restriction(design, manifolds):
    """
    The manifold must be far larger in section than the channels it feeds, or
    the dynamic head at the inlet biases flow to the near side and the jacket
    develops a hot streak opposite it.
    """
    circuit = design.circuits["cowl"]
    channel_area = (circuit.channel.n_channels * circuit.channel.width_mm
                    * circuit.channel.height_mm)
    ring = manifolds.by_name("cowl_inlet_ring")
    assert ring.area_mm2 > 0.3 * channel_area


def test_slower_plenums_are_larger():
    fast = size_plenum("f", 0.4, 423.0, 100.0, 0.0, feeds=100, velocity=8.0)
    slow = size_plenum("s", 0.4, 423.0, 100.0, 0.0, feeds=100, velocity=2.0)
    assert slow.area_mm2 > fast.area_mm2


# --------------------------------------------------------------------------
# mounts
# --------------------------------------------------------------------------

def test_the_lugs_carry_the_thrust(manifolds, design):
    L = manifolds.lugs
    assert L.ok, (f"bearing {L.bearing_stress_pa / 1e6:.0f} MPa, "
                  f"shear {L.shear_stress_pa / 1e6:.0f} MPa")
    assert L.count * L.design_load_n >= design.chamber.thrust_vacuum


def test_lugs_are_checked_on_bearing_and_tear_out(manifolds):
    L = manifolds.lugs
    assert L.bearing_stress_pa <= BEARING_ALLOWABLE_PA
    assert L.shear_stress_pa <= SHEAR_ALLOWABLE_PA


def test_edge_distance_is_at_least_one_and_a_half_diameters(manifolds):
    """
    The floor for a pinned lug, and also what stops a printed one tearing along
    its layer lines.
    """
    L = manifolds.lugs
    assert L.edge_distance_mm >= 1.5 * L.hole_diameter_mm


def test_a_heavier_engine_needs_more_lug(design):
    light = size_mount_lugs(5_000.0, 120.0, 20.0, count=4)
    heavy = size_mount_lugs(200_000.0, 120.0, 20.0, count=4)
    assert (heavy.hole_diameter_mm * heavy.thickness_mm
            > light.hole_diameter_mm * light.thickness_mm)


def test_fewer_than_three_lugs_is_refused():
    with pytest.raises(ValueError):
        size_mount_lugs(7_000.0, 120.0, 20.0, count=2)


def test_lug_holes_clear_the_flange(manifolds, design):
    L = manifolds.lugs
    assert L.hole_radius_mm - 0.5 * L.hole_diameter_mm > design.assembly.flange_radius


# --------------------------------------------------------------------------
# the head has to contain what it carries
# --------------------------------------------------------------------------

def test_the_head_is_thick_enough_for_its_manifolds(manifolds, design):
    """
    Sizing the disc by eye and then finding the dome will not fit inside it is
    the manifold equivalent of a flange too narrow for its own bolts.
    """
    assert not [n for n in manifolds.notes if "head disc" in n], manifolds.notes


def test_a_thin_head_is_reported(design):
    import copy
    thin = copy.deepcopy(design)
    object.__setattr__(thin.assembly.structure, "head_thickness_mm", 4.0)
    md = design_manifolds(thin)
    assert any("head disc" in n for n in md.notes)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_features_land_on_the_right_parts(design, manifolds):
    gf = geometry_features(design, manifolds)
    assert gf["cowl"]["bosses"] and gf["cowl"]["plenums"]
    assert len(gf["head"]["plenums"]) == 2
    assert gf["head"]["lugs"] and gf["head"]["holes"]


def test_the_plenum_sits_inside_its_boss(design, manifolds):
    """A plenum hollowed outside its boss opens straight to atmosphere."""
    gf = geometry_features(design, manifolds)
    boss = gf["cowl"]["bosses"][0]
    plen = gf["cowl"]["plenums"][0]
    assert plen.r_inner >= boss.r_inner
    assert plen.r_inner + 2 * plen.half_r <= boss.r_outer
    assert plen.half_x <= boss.half_x


def test_lug_holes_line_up_with_the_lugs(design, manifolds):
    gf = geometry_features(design, manifolds)
    lug = gf["head"]["lugs"][0]
    hole = gf["head"]["holes"][0]
    assert hole.count == lug.count
    assert hole.phase == pytest.approx(lug.phase)
    assert lug.r_inner < hole.radius_mm < lug.r_outer
