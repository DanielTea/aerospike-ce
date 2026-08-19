"""
Invariants for film cooling, coatings, structure, startup, stability and the
plug surface integral.

Each of these is a screening model standing on a correlation, so the tests pin
the things that must hold whatever the correlation says: limits, monotonicities,
identities, and agreement with an independent calculation where one exists.
"""

import math

import numpy as np
import pytest

import plug_flow_ref as PF
import stability_ref as ST
import structural_ref as SR
import transient_ref as TR
from combustion_ref import PROPELLANTS, build_chamber, thrust_coefficient
from contour_ref import build_contour
from cooling_ref import (
    MATERIALS,
    ChannelSpec,
    CoatingSpec,
    ChannelSpec as CS,
    FilmSpec,
    film_effectiveness,
    filmed_recovery_temperature,
    solve_cooling,
)

PROP = PROPELLANTS["lox_ch4"]


@pytest.fixture(scope="module")
def contour():
    return build_contour(8.0, 75.0, PROP.gamma, n_points=300, truncate_fraction=0.30)


@pytest.fixture(scope="module")
def chamber(contour):
    return build_chamber(PROP, 2.4, 25.0, contour.throat_area, 8.0, contour.exit_mach)


def _wall(n=100):
    x = np.linspace(-80.0, 4.0, n)
    r = np.linspace(89.0, 75.0, n)
    ar = np.linspace(3.0, 1.0, n)
    mach = np.array([0.2 + 0.8 * (i / (n - 1)) ** 2 for i in range(n)])
    return x, r, ar, mach


def _solve(chamber, **kw):
    x, r, ar, mach = _wall()
    ch = CS(n_channels=400, width_mm=0.5, height_mm=1.5, land_mm=0.6, hot_wall_mm=0.7)
    return solve_cooling(chamber, x, r, ar, mach, ch, MATERIALS["grcop42"],
                         4.75, 4.75, **kw)


# --------------------------------------------------------------------------
# film cooling
# --------------------------------------------------------------------------

def test_film_effectiveness_is_bounded_and_decays():
    x = np.linspace(0.0, 200.0, 50)
    eta = film_effectiveness(x, 0.0, 0.5, 1.0)
    assert np.all((eta >= 0.0) & (eta <= 1.0))
    assert np.all(np.diff(eta) <= 1e-12)
    assert eta[0] > eta[-1]


def test_film_cools_nothing_upstream_of_the_slot():
    """A film protects what is behind it. Ahead of the slot it does not exist."""
    x = np.array([-10.0, -1.0, 0.0, 10.0])
    eta = film_effectiveness(x, 0.0, 0.5, 1.0)
    assert eta[0] == 0.0 and eta[1] == 0.0
    assert eta[3] > 0.0


def test_filmed_recovery_temperature_interpolates_the_two_limits():
    t_aw, t_film = 3400.0, 120.0
    assert filmed_recovery_temperature(t_aw, 0.0, t_film) == pytest.approx(t_aw)
    assert filmed_recovery_temperature(t_aw, 1.0, t_film) == pytest.approx(t_film)


def test_film_reduces_flux_and_wall_temperature(chamber):
    bare = _solve(chamber)
    filmed = _solve(chamber, film=FilmSpec(fuel_fraction=0.10, inject_x_mm=-80.0))
    assert filmed.peak_heat_flux < bare.peak_heat_flux
    assert filmed.peak_wall_temperature < bare.peak_wall_temperature
    assert filmed.total_heat < bare.total_heat


def test_more_film_cools_more(chamber):
    a = _solve(chamber, film=FilmSpec(0.05, -80.0))
    b = _solve(chamber, film=FilmSpec(0.15, -80.0))
    assert b.peak_wall_temperature < a.peak_wall_temperature


def test_film_costs_specific_impulse(contour):
    """
    Film fuel does not burn at the design mixture ratio. Charging it as free
    would make film cooling look like a way of getting something for nothing.
    """
    dry = build_chamber(PROP, 2.4, 25.0, contour.throat_area, 8.0, contour.exit_mach)
    wet = build_chamber(PROP, 2.4, 25.0, contour.throat_area, 8.0, contour.exit_mach,
                        film_fraction=0.10)
    assert wet.c_star < dry.c_star
    assert wet.isp_vacuum < dry.isp_vacuum


def test_film_fraction_of_one_is_rejected(contour):
    with pytest.raises(ValueError):
        build_chamber(PROP, 2.4, 25.0, contour.throat_area, 8.0,
                      contour.exit_mach, film_fraction=1.0)


# --------------------------------------------------------------------------
# coatings
# --------------------------------------------------------------------------

def test_coating_cools_the_metal_and_heats_its_own_surface(chamber):
    """
    The whole mechanism: the coating takes the temperature drop so the metal
    does not. Which means the coating surface runs hotter than the bare wall
    ever did, and that is what limits how much of it you can use.
    """
    bare = _solve(chamber)
    coated = _solve(chamber, coating=CoatingSpec(thickness_mm=0.2))
    assert coated.peak_wall_temperature < bare.peak_wall_temperature
    assert coated.peak_coating_temperature > coated.peak_wall_temperature
    assert coated.peak_coating_temperature > bare.peak_wall_temperature


def test_thicker_coating_protects_more_metal_and_runs_hotter(chamber):
    thin = _solve(chamber, coating=CoatingSpec(thickness_mm=0.1))
    thick = _solve(chamber, coating=CoatingSpec(thickness_mm=0.3))
    assert thick.peak_wall_temperature < thin.peak_wall_temperature
    assert thick.peak_coating_temperature > thin.peak_coating_temperature


def test_no_coating_leaves_the_stack_unchanged(chamber):
    bare = _solve(chamber)
    zero = _solve(chamber, coating=CoatingSpec(thickness_mm=0.0))
    assert zero.peak_wall_temperature == pytest.approx(bare.peak_wall_temperature)
    assert math.isinf(zero.coating_temperature_margin)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

@pytest.fixture
def structure(chamber):
    sol = _solve(chamber)
    _, r, _, _ = _wall()
    return SR.analyse(sol, chamber.chamber_pressure, 1.5 * chamber.chamber_pressure,
                      r, 3.0)


def test_thermal_stress_dominates_a_regen_wall(structure):
    """
    The reason regen chambers fail by strain and not by burst: the through-wall
    gradient is worth far more stress than the chamber pressure is.
    """
    assert structure.thermal_stress.max() > structure.pressure_stress.max()
    assert structure.thermal_stress.max() > structure.hoop_stress.max()


def test_yield_derates_with_temperature():
    m = MATERIALS["grcop42"]
    cold = SR.yield_at_temperature(m, np.array([293.0]))[0]
    hot = SR.yield_at_temperature(m, np.array([900.0]))[0]
    assert cold == pytest.approx(m.yield_strength)
    assert 0.0 < hot < cold


def test_hotter_wall_gives_shorter_life(chamber):
    _, r, _, _ = _wall()
    cool = SR.analyse(_solve(chamber, film=FilmSpec(0.15, -80.0)),
                      chamber.chamber_pressure, 1.5 * chamber.chamber_pressure, r, 3.0)
    hot = SR.analyse(_solve(chamber), chamber.chamber_pressure,
                     1.5 * chamber.chamber_pressure, r, 3.0)
    assert cool.cycles_to_failure >= hot.cycles_to_failure


def test_life_not_elasticity_is_the_criterion(structure):
    """A wall may yield every cycle and still be a perfectly good chamber."""
    if not structure.stays_elastic:
        assert structure.survives == (
            structure.cycles_to_failure >= structure.required_cycles
            and structure.strain_range.max() < 0.02)


# --------------------------------------------------------------------------
# startup
# --------------------------------------------------------------------------

def test_thin_copper_wall_has_a_short_diffusion_time(chamber):
    """
    Ten milliseconds against a three hundred millisecond pressure ramp. The wall
    is not the lag; the sequencing is.
    """
    run = TR.solve_startup(_solve(chamber))
    assert run.diffusion_time < 0.05


def test_coolant_first_produces_no_overshoot(chamber):
    run = TR.solve_startup(_solve(chamber), coolant_lead=0.25)
    assert run.overshoot < 5.0
    assert run.survives


def test_gas_before_coolant_overshoots_badly(chamber):
    """
    The result that constrains the valve sequence. Copper conducts the flux
    straight into a wall that has nowhere to put it.
    """
    lead = TR.solve_startup(_solve(chamber), coolant_lead=0.25)
    lag = TR.solve_startup(_solve(chamber), coolant_lead=-0.25)
    assert lag.peak_wall_temp > lead.peak_wall_temp + 100.0


def test_startup_settles_on_the_steady_answer(chamber):
    """The transient must converge on what the steady solver already said."""
    run = TR.solve_startup(_solve(chamber), coolant_lead=0.2, duration=3.0)
    assert run.wall_gas_temp[-1] == pytest.approx(run.steady_wall_temp, rel=0.05)


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------

@pytest.fixture
def stability(chamber):
    return ST.analyse(chamber, 0.122, 0.0835, 0.030, 0.20)


def test_acoustic_modes_are_ordered(stability):
    assert stability.f_tangential_2 == pytest.approx(2.0 * stability.f_tangential_1)
    assert stability.f_radial_1 > stability.f_tangential_1


def test_speed_of_sound_is_physical(stability):
    """Hot combustion products, roughly a kilometre a second and more."""
    assert 800.0 < stability.speed_of_sound < 2000.0


def test_a_bigger_chamber_holds_propellant_longer(chamber):
    small = ST.analyse(chamber, 0.05, 0.0835, 0.030, 0.20)
    big = ST.analyse(chamber, 0.20, 0.0835, 0.030, 0.20)
    assert big.residence_time > small.residence_time
    assert big.characteristic_length > small.characteristic_length


def test_soft_injection_fails_the_stiffness_screen(chamber):
    assert not ST.analyse(chamber, 0.122, 0.0835, 0.030, 0.05).stiffness_ok
    assert ST.analyse(chamber, 0.122, 0.0835, 0.030, 0.20).stiffness_ok


def test_wider_annulus_lowers_the_radial_mode(chamber):
    narrow = ST.analyse(chamber, 0.122, 0.0835, 0.010, 0.20)
    wide = ST.analyse(chamber, 0.122, 0.0835, 0.040, 0.20)
    assert wide.f_radial_1 < narrow.f_radial_1


# --------------------------------------------------------------------------
# plug surface integral and altitude compensation
# --------------------------------------------------------------------------

def test_vacuum_thrust_matches_momentum_theory(contour, chamber):
    """
    Independent check. At vacuum the thrust must approach mdot * u_exit, which
    comes from the exit Mach and nothing else. Resolving the throat term on the
    wrong angle inflates this by nearly fifty percent, which is how the error
    was found.
    """
    g = chamber.propellant.gamma
    t_e = chamber.t_chamber / (1.0 + 0.5 * (g - 1.0) * contour.exit_mach ** 2)
    u_e = contour.exit_mach * math.sqrt(g * chamber.gas_constant * t_e)
    momentum_only = chamber.mass_flow * u_e

    vac = PF.plug_thrust(contour, chamber, 0.0)
    assert vac.total_force == pytest.approx(momentum_only, rel=0.06)


def test_thrust_rises_as_ambient_falls(contour, chamber):
    pas = [1.01325e5, 0.5e5, 0.2e5, 0.0]
    cf = [PF.plug_thrust(contour, chamber, pa).thrust_coefficient for pa in pas]
    assert all(b > a for a, b in zip(cf, cf[1:]))


def test_plug_beats_the_bell_where_the_bell_is_overexpanded(contour, chamber):
    """
    The entire point of the architecture. At sea level this nozzle is badly
    overexpanded -- a bell would be losing to its own exit pressure term -- and
    the plug is not, because its outer boundary is the ambient itself.
    """
    eps = math.pi * contour.exit_radius ** 2 / contour.throat_area
    plug = PF.plug_thrust(contour, chamber, 1.01325e5).thrust_coefficient
    bell = thrust_coefficient(chamber.propellant.gamma, eps, contour.exit_mach,
                              chamber.chamber_pressure, 1.01325e5)
    assert plug > bell


def test_the_advantage_shrinks_with_altitude(contour, chamber):
    """
    Altitude compensation is a sea-level benefit. Once the bell is no longer
    overexpanded it has nothing left to lose, and the truncated plug's base drag
    starts to tell instead.
    """
    eps = math.pi * contour.exit_radius ** 2 / contour.throat_area
    def gain(pa):
        return (PF.plug_thrust(contour, chamber, pa).thrust_coefficient
                / thrust_coefficient(chamber.propellant.gamma, eps,
                                     contour.exit_mach, chamber.chamber_pressure, pa))
    assert gain(1.01325e5) > gain(0.2e5)


def test_surface_pressure_falls_along_the_spike(contour, chamber):
    p = PF.surface_pressure(contour, chamber)
    assert p[0] > p[-1]
    assert p[0] < chamber.chamber_pressure          # already sonic at the shoulder
    assert np.all(np.diff(p) <= 1e-6)


def test_the_tip_over_expands_at_sea_level(contour, chamber):
    """
    Where surface pressure drops below ambient the plume leaves the surface.
    Integrating past that point would credit the engine with suction it does
    not get.
    """
    sl = PF.plug_thrust(contour, chamber, 1.01325e5)
    vac = PF.plug_thrust(contour, chamber, 0.0)
    assert sl.attached_fraction <= vac.attached_fraction
    assert vac.attached_fraction == pytest.approx(1.0)
