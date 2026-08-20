"""
Every stream reaches the thing it feeds, and touches nothing it must not.

This is the check the model kept not having. Three separate bugs got as far as
a print-resolution mesh because each one produced a part that was watertight,
drained, and looked entirely correct:

- injector orifices that stopped 5 mm short of their plenums
- an oxidiser dome at r 98-117 whose own orifices sat at r 88, pointing instead
  at the radius the *fuel* manifold occupies
- a port schedule naming three inlets where the geometry cut none of them, so
  the dome had no way in and the coolant leaving both jackets had no way out

Watertightness cannot see any of that. Genus cannot distinguish a passage that
reaches the right cavity from one that reaches the wrong cavity. So the checks
here are analytic and specific: does *this* feature open into *that* plenum, and
is there metal between the fuel and the oxidiser everywhere else.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from engine_design import design_engine
from manifold_ref import design_manifolds, geometry_features

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


@pytest.fixture(scope="module")
def manifolds(design):
    return design_manifolds(design)


@pytest.fixture(scope="module")
def feats(design, manifolds):
    return geometry_features(design, manifolds)


def _plenum(md, name):
    return next(p for p in md.plenums if p.name == name)


def _hole(feats, part, name):
    return next(h for h in feats[part]["holes"] if h.name == name)


def _x_span_at(p, radius):
    """Axial extent of a plenum's diamond section at one radius."""
    half = p.half_x_at(radius)
    return (p.x_mm - half, p.x_mm + half) if half > 0.0 else None


def _r_span_at(p, x):
    """Radial extent of a plenum's diamond section at one station."""
    t = abs(x - p.x_mm) / p.half_x_mm
    if t >= 1.0:
        return None
    half = p.half_r_mm * (1.0 - t)
    return (p.r_centre_mm - half, p.r_centre_mm + half)


def _overlap(a, b):
    if a is None or b is None:
        return 0.0
    return min(a[1], b[1]) - max(a[0], b[0])


def axial_hole_into(p, hole, samples=9):
    """How far, axially, an axial hole ring runs inside a plenum."""
    best = 0.0
    lo, hi = hole.radius_mm - 0.5 * hole.diameter_mm, hole.radius_mm + 0.5 * hole.diameter_mm
    for i in range(samples):
        r = lo + (hi - lo) * i / (samples - 1)
        best = max(best, _overlap(_x_span_at(p, r), (hole.x_start, hole.x_end)))
    return best


def radial_port_into(p, port, samples=9):
    """How far, radially, a radial port ring runs inside a plenum."""
    best = 0.0
    lo, hi = port.x_at - 0.5 * port.diameter_mm, port.x_at + 0.5 * port.diameter_mm
    for i in range(samples):
        x = lo + (hi - lo) * i / (samples - 1)
        best = max(best, _overlap(_r_span_at(p, x), (port.r_lo, port.r_hi)))
    return best


# --------------------------------------------------------------------------
# the fuel path: two jackets -> head manifold -> fuel orifices
# --------------------------------------------------------------------------

def test_the_centrebody_jacket_reaches_the_fuel_manifold(design, manifolds, feats):
    fm = _plenum(manifolds, "head_fuel_manifold")
    h = _hole(feats, "head", "transfer_centrebody")
    assert axial_hole_into(fm, h) > 0.5, (
        "the spike's coolant has no path from the joint face to the injector")


def test_the_cowl_jacket_reaches_the_fuel_manifold(design, manifolds, feats):
    """
    Two legs: axially down the corridor inboard of the dome, then radially in
    through the one window where the manifold has begun and the dome has not.
    """
    fm = _plenum(manifolds, "head_fuel_manifold")
    h = _hole(feats, "head", "transfer_cowl")
    coll = next(q for q in feats["head"]["ports"] if q.phase == 0.0)

    assert coll.r_hi >= h.radius_mm - 0.5 * h.diameter_mm, "collector misses the riser"
    assert coll.x_at >= h.x_start and coll.x_at <= h.x_end, "collector is not on the riser"
    assert radial_port_into(fm, coll) > 0.5, (
        "the cowl's coolant has no path from the joint face to the injector")


def test_the_jackets_discharge_where_the_transfers_collect(design, feats):
    """A groove across each jacket's face, so no channel dead-ends against it."""
    from manifold_ref import _discharge_band

    for part, hole in (("cowl", "transfer_cowl"),
                       ("centrebody", "transfer_centrebody")):
        lo, hi = _discharge_band(design, part)
        groove = feats[part]["plenums"][-1]
        assert groove.r_inner <= lo + 1e-9
        assert groove.r_inner + 2 * groove.half_r >= hi - 1e-9
        h = _hole(feats, "head", hole)
        assert (h.radius_mm - 0.5 * h.diameter_mm) < hi
        assert (h.radius_mm + 0.5 * h.diameter_mm) > lo


# --------------------------------------------------------------------------
# the oxidiser path: rim -> dome -> radial ports -> ox orifices
# --------------------------------------------------------------------------

def test_the_oxidiser_can_get_in_at_all(design, manifolds, feats):
    """
    There was no oxidiser inlet in the geometry. The schedule named one on the
    head end face at r 101, where an axial bore misses the dome entirely.
    """
    od = _plenum(manifolds, "ox_dome")
    a = design.assembly
    inlets = [q for q in feats["head"]["ports"]
              if q.r_hi > a.flange_radius and radial_port_into(od, q) > 0.5]
    assert inlets, "the oxidiser dome has no bore reaching it from outside"


def test_the_dome_reaches_its_orifices(design, manifolds, feats):
    """
    The dome sits outboard of its own orifice ring, so the two are joined by a
    ring of radial ports. Without them the dome feeds nothing.
    """
    od = _plenum(manifolds, "ox_dome")
    ox = _hole(feats, "head", "ox_orifice")
    ports = [q for q in feats["head"]["ports"]
             if q.r_lo <= ox.radius_mm <= q.r_hi and radial_port_into(od, q) > 0.5]
    assert ports, "nothing connects the oxidiser dome to the oxidiser orifices"
    p = ports[0]
    assert p.count == ox.count
    assert p.phase == pytest.approx(ox.phase)
    assert p.x_at >= ox.x_start - 1e-9, "the orifices start aft of their feed ports"


# --------------------------------------------------------------------------
# and the one that matters most
# --------------------------------------------------------------------------

def test_no_fuel_feature_touches_the_oxidiser_dome(design, manifolds, feats):
    """
    A fuel passage crossing the oxidiser dome is not a geometry bug, it is a
    fire. Both are voids, so their union is watertight, drains, and passes
    every other check in this repository.
    """
    od = _plenum(manifolds, "ox_dome")
    for h in feats["head"]["holes"]:
        if h.name.startswith("transfer") or h.name == "fuel_orifice":
            assert axial_hole_into(od, h) <= 0.0, (
                f"{h.name} at r {h.radius_mm:.1f} crosses the oxidiser dome")
    coll = next(q for q in feats["head"]["ports"] if q.phase == 0.0)
    assert radial_port_into(od, coll) <= 0.0, "the fuel collector crosses the dome"


def test_no_oxidiser_feature_touches_the_fuel_manifold(manifolds, feats):
    fm = _plenum(manifolds, "head_fuel_manifold")
    ox = _hole(feats, "head", "ox_orifice")
    assert axial_hole_into(fm, ox) <= 0.0, "the ox orifices cross the fuel manifold"
    for q in feats["head"]["ports"]:
        if q.phase != 0.0:
            assert radial_port_into(fm, q) <= 0.0, (
                "an oxidiser port crosses the fuel manifold")


def test_the_two_plenums_keep_a_wall_between_them(design, manifolds):
    fm = _plenum(manifolds, "head_fuel_manifold")
    od = _plenum(manifolds, "ox_dome")
    gap = od.r_inner_mm - (fm.r_inner_mm + 2 * fm.half_r_mm)
    assert gap >= design.assembly.structure.wall_thickness_mm, (
        f"only {gap:.2f} mm between the fuel manifold and the oxidiser dome")


def test_nothing_opens_into_the_chamber_that_should_not(design, feats):
    """
    The injector face is chamber-exposed between the shoulder and the chamber
    wall. A transfer hole breaking through there dumps fuel into the chamber
    outside the injector, which no topology check would notice.
    """
    a = design.assembly
    floor = 0.7
    for name in ("transfer_cowl", "transfer_centrebody"):
        h = _hole(feats, "head", name)
        lo = h.radius_mm - 0.5 * h.diameter_mm
        hi = h.radius_mm + 0.5 * h.diameter_mm
        assert hi <= a.shoulder_radius - floor or lo >= a.chamber_outer_radius + floor, (
            f"{name} spans r {lo:.2f}-{hi:.2f}, which breaks into the chamber "
            f"face (r {a.shoulder_radius:.2f}-{a.chamber_outer_radius:.2f})")
        assert lo >= float(a.cavity_r[0]) + floor or hi <= float(a.cavity_r[0]) - floor, (
            f"{name} breaks into the central bore")
