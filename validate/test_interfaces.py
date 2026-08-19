"""
Invariants for the external interface schedule.

These are the checks that stop the port list being a drawing annotation. A port
has to carry its flow, sit on material that exists, miss everything else, and
the joint has to actually close on the flange the design provides.
"""

import json
import math
import os

import pytest

from engine_design import design_engine
from interfaces_ref import (
    BOLTS,
    BOLT_UTILISATION,
    BOLT_YIELD_PA,
    bore_for_flow,
    build_schedule,
    port_clashes,
    size_bolted_joint,
    velocity_in_bore,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


@pytest.fixture(scope="module")
def schedule(design):
    return build_schedule(design)


# --------------------------------------------------------------------------
# sizing round-trips
# --------------------------------------------------------------------------

def test_bore_sizing_round_trips():
    """Size a bore for a velocity, measure the velocity back. Exact."""
    for mdot, rho, v in ((0.5, 423.0, 8.0), (2.3, 1141.0, 6.0), (0.09, 810.0, 10.0)):
        bore = bore_for_flow(mdot, rho, v)
        assert velocity_in_bore(mdot, rho, bore) == pytest.approx(v, rel=1e-12)


def test_every_inlet_runs_at_the_design_velocity(schedule, design):
    prop = design.chamber.propellant
    for p in schedule.ports:
        if p.kind != "inlet":
            continue
        rho = prop.density_fuel if p.fluid == "fuel" else prop.density_ox
        assert velocity_in_bore(p.mass_flow, rho, p.bore_mm) == pytest.approx(
            p.velocity, rel=1e-9), p.name


def test_rejects_nonphysical_sizing():
    with pytest.raises(ValueError):
        bore_for_flow(1.0, 800.0, 0.0)
    with pytest.raises(ValueError):
        bore_for_flow(1.0, 0.0, 8.0)


# --------------------------------------------------------------------------
# the propellant actually gets in
# --------------------------------------------------------------------------

def test_fuel_ports_carry_the_whole_fuel_flow(schedule, design):
    """
    Both jackets are fed from outside, so the fuel ports must sum to the engine's
    fuel flow. Miss one and part of the cooling circuit is fed by nothing.
    """
    total = sum(p.mass_flow for p in schedule.fuel_ports)
    assert total == pytest.approx(design.chamber.mass_flow_fuel, rel=1e-9)


def test_ox_ports_carry_the_whole_ox_flow(schedule, design):
    total = sum(p.mass_flow for p in schedule.ox_ports)
    assert total == pytest.approx(design.chamber.mass_flow_ox, rel=1e-9)


def test_fuel_split_matches_the_cooling_split(schedule, design):
    """
    The two fuel legs feed the two jackets, so they must be divided the way the
    thermal model divided the coolant. Feeding them evenly starves whichever
    circuit carries more heat.
    """
    cowl = schedule.by_name("fuel_in_cowl").mass_flow
    total = sum(p.mass_flow for p in schedule.fuel_ports)
    assert cowl / total == pytest.approx(design.coolant_split["cowl"], rel=1e-9)


def test_there_is_no_coolant_outlet(schedule):
    """
    Architectural, not cosmetic: the fuel is the coolant and it leaves the
    jacket into the injector. A coolant outlet would be dumping propellant.
    """
    assert not [p for p in schedule.ports if "coolant" in p.name and p.kind != "instrument"]
    assert not [p for p in schedule.ports if p.kind == "outlet"]


def test_the_engine_can_be_lit(schedule):
    igniter = schedule.by_name("igniter")
    assert igniter.orientation == "radial"      # the bore is carrying fuel
    assert igniter.bore_mm >= 5.0


def test_chamber_pressure_can_be_measured(schedule):
    assert schedule.by_name("pc_tap").kind == "instrument"


# --------------------------------------------------------------------------
# nothing collides
# --------------------------------------------------------------------------

def test_no_port_clashes(schedule, design):
    assert port_clashes(schedule, design) == []


def test_axial_ports_land_on_the_flange(schedule, design):
    a = design.assembly
    for p in schedule.ports:
        if p.orientation != "axial" or p.r_mm == 0.0:
            continue
        assert p.r_mm + 0.5 * p.boss_od_mm <= a.flange_radius + 1e-9, p.name


def test_radial_ports_stand_off_the_outer_skin(schedule, design):
    a = design.assembly
    for p in schedule.ports:
        if p.orientation == "radial":
            assert p.r_mm >= float(max(a.cowl_outer_r)) - 1e-6, p.name


def test_the_clash_check_actually_catches_a_clash(schedule, design):
    """A checker that never fires is not a checker."""
    from dataclasses import replace
    inj = design.injector
    bad = replace(schedule.ports[0], name="deliberate", orientation="axial",
                  r_mm=inj.fuel_ring_radius * 1e3, bore_mm=6.0, boss_od_mm=10.0)
    from interfaces_ref import InterfaceSchedule
    hits = port_clashes(InterfaceSchedule(ports=[bad], joint=None), design)
    assert any("orifice ring" in h for h in hits)


# --------------------------------------------------------------------------
# the joint
# --------------------------------------------------------------------------

def test_preload_exceeds_the_separating_load(schedule):
    j = schedule.joint
    assert j.count * j.preload_per_bolt > j.separating_load


def test_bolts_stay_inside_their_allowable(schedule):
    j = schedule.joint
    assert 0.0 < j.utilisation <= 0.71
    assert j.stress_pa < BOLT_YIELD_PA * BOLT_UTILISATION


def test_the_joint_fits_the_flange(schedule):
    j = schedule.joint
    assert j.fits, (f"needs {j.flange_radius_required_mm:.1f} mm flange radius")
    assert j.edge_margin_mm >= 1.5


def test_bolt_heads_do_not_touch(schedule):
    j = schedule.joint
    across = BOLTS[j.size][1]
    assert math.pi * j.pitch_circle_mm / j.count >= across * 1.6


def test_a_narrow_flange_is_reported_not_hidden():
    """
    A flange too thin to take its own fasteners is a real and easy mistake. The
    model must say what width it needs rather than raise or quietly proceed.
    """
    joint = size_bolted_joint(44_000.0, 220.0, 112.0)
    assert not joint.fits
    assert joint.flange_radius_required_mm > 112.0


def test_a_heavier_load_needs_more_or_bigger_bolts():
    light = size_bolted_joint(20_000.0, 220.0, 130.0)
    heavy = size_bolted_joint(200_000.0, 220.0, 130.0)
    assert (heavy.count * BOLTS[heavy.size][0]) > (light.count * BOLTS[light.size][0])
