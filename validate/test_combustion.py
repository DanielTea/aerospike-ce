"""
Invariants for the chamber conditions and delivered performance.

The parametric model here cannot be checked against truth without CEA, so these
tests pin the things that must hold regardless of how good the propellant data
is: the identities, the monotonicities, and the refusals.
"""

import math

import pytest

from combustion_ref import (
    G0,
    PROPELLANTS,
    build_chamber,
    c_star_ideal,
    chamber_temperature,
    pressure_ratio,
    thrust_coefficient,
)

PROP = PROPELLANTS["lox_ch4"]
AT = 353.4292      # mm^2, the demo throat
EPS = 8.0
ME = 3.121903


@pytest.fixture
def chamber():
    return build_chamber(PROP, PROP.mr_peak, 20.0, AT, EPS, ME)


# --------------------------------------------------------------------------
# identities
# --------------------------------------------------------------------------

def test_isp_is_c_star_times_cf(chamber):
    """
    Isp = c* * C_F / g0 is the definition, not an approximation.

    It ties the two halves of the model together: c* is everything upstream of
    the throat, C_F everything downstream. If this drifts, one of them is being
    computed with a different mass flow than the other.
    """
    expected = (chamber.c_star * chamber.cf_vacuum
                * chamber.truncation_efficiency / G0)
    assert chamber.isp_vacuum == pytest.approx(expected, rel=1e-12)
    expected_sl = (chamber.c_star * chamber.cf_sea_level
                   * chamber.truncation_efficiency / G0)
    assert chamber.isp_sea_level == pytest.approx(expected_sl, rel=1e-12)


def test_mass_flow_is_the_choked_throat_relation(chamber):
    """mdot = pc * At / c*, which is what defines c* in the first place."""
    assert chamber.mass_flow == pytest.approx(
        chamber.chamber_pressure * chamber.throat_area / chamber.c_star, rel=1e-12)


def test_propellant_split_reconstructs_the_mixture_ratio(chamber):
    assert chamber.mass_flow_fuel + chamber.mass_flow_ox == pytest.approx(
        chamber.mass_flow, rel=1e-12)
    assert chamber.mass_flow_ox / chamber.mass_flow_fuel == pytest.approx(
        chamber.mixture_ratio, rel=1e-12)


def test_choked_pressure_ratio_matches_the_closed_form():
    """p*/p0 at Mach 1 has an exact expression; the general one must hit it."""
    for gamma in (1.15, 1.20, 1.24, 1.40):
        closed = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        assert pressure_ratio(1.0, gamma) == pytest.approx(closed, rel=1e-12)


def test_thrust_is_cf_times_pc_times_at(chamber):
    assert chamber.thrust_vacuum == pytest.approx(
        chamber.cf_vacuum * chamber.chamber_pressure * chamber.throat_area
        * chamber.truncation_efficiency, rel=1e-12)


# --------------------------------------------------------------------------
# shape of the curves
# --------------------------------------------------------------------------

def test_c_star_peaks_at_the_stated_mixture_ratio():
    peak = c_star_ideal(PROP, PROP.mr_peak)
    for mr in (0.6 * PROP.mr_peak, 0.8 * PROP.mr_peak,
               1.2 * PROP.mr_peak, 1.5 * PROP.mr_peak):
        assert c_star_ideal(PROP, mr) < peak


def test_chamber_temperature_peaks_at_the_stated_mixture_ratio():
    peak = chamber_temperature(PROP, PROP.mr_peak)
    assert chamber_temperature(PROP, 0.7 * PROP.mr_peak) < peak
    assert chamber_temperature(PROP, 1.4 * PROP.mr_peak) < peak


def test_running_fuel_rich_buys_coolant_and_costs_impulse():
    """
    The trade every regeneratively cooled engine makes. Below peak mixture ratio
    there is more fuel to absorb heat and less specific impulse to show for it.
    """
    rich = build_chamber(PROP, 2.6, 20.0, AT, EPS, ME)
    peak = build_chamber(PROP, PROP.mr_peak, 20.0, AT, EPS, ME)
    assert rich.mass_flow_fuel > peak.mass_flow_fuel
    assert rich.isp_vacuum < peak.isp_vacuum


def test_vacuum_beats_sea_level(chamber):
    assert chamber.cf_vacuum > chamber.cf_sea_level
    assert chamber.thrust_vacuum > chamber.thrust_sea_level
    assert chamber.isp_vacuum > chamber.isp_sea_level


def test_mass_flow_scales_linearly_with_pressure_and_area():
    a = build_chamber(PROP, PROP.mr_peak, 20.0, AT, EPS, ME)
    b = build_chamber(PROP, PROP.mr_peak, 40.0, AT, EPS, ME)
    c = build_chamber(PROP, PROP.mr_peak, 20.0, 2.0 * AT, EPS, ME)
    assert b.mass_flow / a.mass_flow == pytest.approx(2.0, rel=1e-12)
    assert c.mass_flow / a.mass_flow == pytest.approx(2.0, rel=1e-12)


def test_higher_expansion_ratio_raises_vacuum_thrust_coefficient():
    lo = thrust_coefficient(1.2, 4.0, 2.65, 20e5, 0.0)
    hi = thrust_coefficient(1.2, 20.0, 3.75, 20e5, 0.0)
    assert hi > lo


def test_thrust_coefficient_stays_physical(chamber):
    """C_F for a real nozzle sits between about 1 and the vacuum limit."""
    assert 0.8 < chamber.cf_sea_level < 2.3
    assert 1.0 < chamber.cf_vacuum < 2.3


def test_demo_point_is_overexpanded_at_sea_level(chamber):
    """
    Exit pressure well under ambient. On a bell this is the separation regime;
    it is the condition an aerospike exists to handle, and the reason the demo
    spec is worth building as a plug rather than a cone.
    """
    assert chamber.exit_pressure < 101325.0


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_rejects_nonpositive_pressure():
    with pytest.raises(ValueError):
        build_chamber(PROP, 3.4, 0.0, AT, EPS, ME)


def test_rejects_nonpositive_throat():
    with pytest.raises(ValueError):
        build_chamber(PROP, 3.4, 20.0, -1.0, EPS, ME)


def test_rejects_efficiency_above_unity():
    with pytest.raises(ValueError):
        build_chamber(PROP, 3.4, 20.0, AT, EPS, ME, c_star_efficiency=1.4)


def test_rejects_nonpositive_mixture_ratio():
    with pytest.raises(ValueError):
        c_star_ideal(PROP, 0.0)
