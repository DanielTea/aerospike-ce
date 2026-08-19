"""
Invariants for the regenerative cooling model.

The load-bearing one is the energy balance: whatever heat the wall gives up must
appear in the coolant. It is exact, and it is the check that catches the
counterflow bookkeeping going the wrong way -- which it did, and which showed up
as a coolant that cooled down while absorbing heat.
"""

import math

import numpy as np
import pytest

from combustion_ref import PROPELLANTS, build_chamber
from cooling_ref import (
    MATERIALS,
    ChannelSpec,
    adiabatic_wall_temperature,
    bartz_coefficient,
    channels_packed,
    dittus_boelter,
    fin_efficiency,
    friction_factor,
    gas_prandtl,
    search_cooling_design,
    size_channels_for_velocity,
    solve_cooling,
)

PROP = PROPELLANTS["lox_ch4"]
DT = 3.7968      # mm, hydraulic diameter of the annular throat


@pytest.fixture
def chamber():
    return build_chamber(PROP, PROP.mr_peak, 20.0, 353.4292, 8.0, 3.121903)


def _wall(n=120):
    x = np.linspace(-32.0, 1.75, n)
    r = np.linspace(34.55, 30.0, n)
    ar = np.linspace(3.0, 1.0, n)
    mach = np.array([0.2 + 0.8 * (i / (n - 1)) ** 2 for i in range(n)])
    return x, r, ar, mach


@pytest.fixture
def solution(chamber):
    x, r, ar, mach = _wall()
    ch = ChannelSpec(n_channels=200, width_mm=0.4, height_mm=0.6,
                     land_mm=0.4, hot_wall_mm=0.3)
    return solve_cooling(chamber, x, r, ar, mach, ch, MATERIALS["grcop42"], DT, DT)


# --------------------------------------------------------------------------
# the load-bearing check
# --------------------------------------------------------------------------

def test_energy_balances_exactly(solution):
    """
    Every watt through the wall shows up as coolant enthalpy. Exact, because
    both sides are accumulated from the same per-station heat.
    """
    absorbed = (solution.coolant_mass_flow * PROP.coolant_cp
                * solution.coolant_temperature_rise)
    assert absorbed == pytest.approx(solution.total_heat, rel=1e-12)


def test_coolant_gets_hotter_not_colder(solution):
    """
    The march runs against the array order, so reading the rise off the array
    instead of the march gives it backwards -- a coolant that cools down as it
    absorbs heat.
    """
    assert solution.coolant_temperature_rise > 0.0
    assert solution.coolant_outlet_temperature > solution.coolant_inlet_temperature


def test_coolant_enters_at_the_throat_end(solution):
    """Counterflow: the coldest coolant must meet the highest heat flux."""
    assert solution.coolant_inlet_index == len(solution.x) - 1
    assert solution.t_coolant[-1] == pytest.approx(PROP.coolant_inlet_t, rel=1e-12)


def test_pressure_falls_along_the_flow(solution):
    assert solution.pressure_drop > 0.0


# --------------------------------------------------------------------------
# heat transfer physics
# --------------------------------------------------------------------------

def test_heat_flux_peaks_at_the_throat(solution):
    """The (At/A)^0.9 term is what makes the throat the hottest place by far."""
    assert int(np.argmax(solution.heat_flux)) == len(solution.heat_flux) - 1
    assert solution.heat_flux[-1] > 2.0 * solution.heat_flux[0]


def test_bartz_falls_as_the_duct_opens(chamber):
    """Same station, larger area ratio, lower gas-side coefficient."""
    throat = bartz_coefficient(chamber, 0.0038, 0.0076, 1.0, 1.0, 800.0)
    wide = bartz_coefficient(chamber, 0.0038, 0.0076, 4.0, 0.3, 800.0)
    assert wide < throat


def test_adiabatic_wall_is_below_stagnation_and_approaches_it_at_low_mach(chamber):
    pr = gas_prandtl(PROP.gamma)
    hot = adiabatic_wall_temperature(chamber.t_chamber, 0.02, PROP.gamma, pr)
    fast = adiabatic_wall_temperature(chamber.t_chamber, 3.0, PROP.gamma, pr)
    assert hot < chamber.t_chamber
    assert hot == pytest.approx(chamber.t_chamber, rel=1e-3)
    assert fast < hot


def test_wall_sits_between_the_gas_and_the_coolant(solution):
    """A wall outside its own boundary conditions means the resistances are wrong."""
    assert np.all(solution.t_wall_gas < solution.t_adiabatic)
    assert np.all(solution.t_wall_gas > solution.t_wall_coolant)
    assert np.all(solution.t_wall_coolant > solution.t_coolant - 1e-6)


def test_faster_coolant_transfers_more_heat(chamber):
    slow = dittus_boelter(5_000.0, 3.0, 0.19, 5e-4)
    fast = dittus_boelter(50_000.0, 3.0, 0.19, 5e-4)
    assert fast > slow
    # Re^0.8: an order of magnitude in Reynolds is 10^0.8 in coefficient
    assert fast / slow == pytest.approx(10.0 ** 0.8, rel=1e-9)


# --------------------------------------------------------------------------
# fins
# --------------------------------------------------------------------------

def test_fin_efficiency_is_bounded():
    assert 0.0 < fin_efficiency(5000.0, 25.0, 4e-4, 1e-3) <= 1.0


def test_a_perfect_conductor_is_a_perfect_fin():
    assert fin_efficiency(5000.0, 1e9, 4e-4, 1e-3) == pytest.approx(1.0, abs=1e-4)


def test_copper_fins_beat_nickel_fins():
    """Why a small chamber with wide lands wants a conductor, not a superalloy."""
    inconel = fin_efficiency(5000.0, 25.0, 4e-4, 1.5e-3)
    copper = fin_efficiency(5000.0, 290.0, 4e-4, 1.5e-3)
    assert copper > inconel


def test_taller_fins_are_less_efficient():
    short = fin_efficiency(5000.0, 25.0, 4e-4, 5e-4)
    tall = fin_efficiency(5000.0, 25.0, 4e-4, 3e-3)
    assert tall < short


# --------------------------------------------------------------------------
# channel sizing
# --------------------------------------------------------------------------

def test_sizing_hits_the_requested_velocity(chamber):
    """Round trip: build for a velocity, then measure it back off the geometry."""
    for target in (8.0, 15.0, 30.0):
        ch = size_channels_for_velocity(chamber, 32.0, target, min_width_mm=0.01)
        area = ch.width_mm * ch.height_mm * 1e-6
        mdot_ch = chamber.mass_flow_fuel / ch.n_channels
        v = mdot_ch / (PROP.density_fuel * area)
        assert v == pytest.approx(target, rel=0.02)


def test_sizing_respects_the_minimum_printable_width(chamber):
    ch = size_channels_for_velocity(chamber, 32.0, 30.0, min_width_mm=0.4)
    assert ch.width_mm >= 0.4 - 1e-9


def test_sizing_holds_the_aspect_ratio(chamber):
    for ar in (1.5, 2.5, 4.0):
        ch = size_channels_for_velocity(chamber, 32.0, 15.0, aspect_ratio=ar)
        assert ch.aspect_ratio == pytest.approx(ar, rel=1e-9)


def test_channels_pack_without_overlapping():
    n = channels_packed(32.0, 0.4, 0.4)
    assert n * (0.4 + 0.4) <= 2.0 * math.pi * 32.0 + 1e-9


def test_friction_factor_is_laminar_below_transition():
    assert friction_factor(1000.0, 0.01) == pytest.approx(0.064, rel=1e-9)


def test_rougher_channels_cost_more_pressure():
    assert friction_factor(20_000.0, 0.05) > friction_factor(20_000.0, 0.001)


# --------------------------------------------------------------------------
# the design search
# --------------------------------------------------------------------------

def test_search_finds_a_survivor_when_one_exists(chamber):
    x, r, ar, mach = _wall(80)
    best, evaluated = search_cooling_design(
        chamber, x, r, ar, mach, DT, DT, reference_radius_mm=32.0)
    assert len(evaluated) > 100
    assert best is not None
    assert best.survives
    assert best.solution.wall_temperature_margin > 0.0
    assert best.solution.coolant_temperature_margin > 0.0


def test_search_reports_failure_rather_than_returning_a_melted_engine(chamber):
    """
    Starve the circuit and nothing should survive. The value is in the model
    saying so: a small engine cannot always regeneratively cool itself, and
    quietly returning the least-bad candidate would hide that.
    """
    x, r, ar, mach = _wall(60)
    best, evaluated = search_cooling_design(
        chamber, x, r, ar, mach, DT, DT,
        reference_radius_mm=32.0, coolant_fraction=0.15)
    assert best is None
    assert all(not c.survives for c in evaluated)
    assert any("hot wall" in c.reason for c in evaluated)


def test_every_surviving_candidate_really_survives(chamber):
    x, r, ar, mach = _wall(60)
    _, evaluated = search_cooling_design(
        chamber, x, r, ar, mach, DT, DT, reference_radius_mm=32.0)
    for c in evaluated:
        if c.survives:
            assert c.solution.wall_temperature_margin > 0.0
            assert c.solution.coolant_temperature_margin > 0.0
            assert min(c.solution.reynolds) >= 2300.0


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_rejects_mismatched_wall_arrays(chamber):
    with pytest.raises(ValueError):
        solve_cooling(chamber, np.zeros(5), np.zeros(4), np.ones(5), np.ones(5),
                      ChannelSpec(), MATERIALS["grcop42"], DT, DT)


def test_rejects_zero_channels(chamber):
    x, r, ar, mach = _wall(10)
    with pytest.raises(ValueError):
        solve_cooling(chamber, x, r, ar, mach, ChannelSpec(n_channels=0),
                      MATERIALS["grcop42"], DT, DT)


def test_rejects_impossible_coolant_fraction(chamber):
    x, r, ar, mach = _wall(10)
    with pytest.raises(ValueError):
        solve_cooling(chamber, x, r, ar, mach, ChannelSpec(),
                      MATERIALS["grcop42"], DT, DT, coolant_fraction=0.0)
