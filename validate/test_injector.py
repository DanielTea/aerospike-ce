"""
Invariants for injector element sizing and face layout.

The load-bearing one is the round trip: orifices sized for a mass flow must pass
exactly that mass flow back through the same orifice equation. After that it is
the momentum balance, which is what keeps the spray on axis.
"""

import math

import pytest

from combustion_ref import PROPELLANTS, build_chamber
from injector_ref import (
    balanced_doublet_angles,
    chug_margin,
    design_injector,
    element_positions,
    fit_injector_to_face,
    fits_on_face,
    mass_flow_through,
)

PROP = PROPELLANTS["lox_ch4"]
ANNULUS = (29.2601, 34.5493)


@pytest.fixture
def chamber():
    return build_chamber(PROP, PROP.mr_peak, 20.0, 353.4292, 8.0, 3.121903)


@pytest.fixture
def injector(chamber):
    d, _ = fit_injector_to_face(chamber, *ANNULUS)
    return d


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------

def test_sized_orifices_pass_the_mass_flow_they_were_sized_for(injector, chamber):
    """Invert the orifice equation and recover the input. Exact, not approximate."""
    assert mass_flow_through(injector.area_fuel_total, PROP.density_fuel,
                             injector.dp_fuel, injector.discharge_coefficient) \
        == pytest.approx(chamber.mass_flow_fuel, rel=1e-12)
    assert mass_flow_through(injector.area_ox_total, PROP.density_ox,
                             injector.dp_ox, injector.discharge_coefficient) \
        == pytest.approx(chamber.mass_flow_ox, rel=1e-12)


def test_orifice_areas_reconstruct_the_mixture_ratio(injector, chamber):
    f = mass_flow_through(injector.area_fuel_total, PROP.density_fuel,
                          injector.dp_fuel, injector.discharge_coefficient)
    ox = mass_flow_through(injector.area_ox_total, PROP.density_ox,
                           injector.dp_ox, injector.discharge_coefficient)
    assert ox / f == pytest.approx(chamber.mixture_ratio, rel=1e-12)


def test_per_element_areas_sum_to_the_total(injector):
    n = injector.n_elements
    a_f = n * math.pi * injector.d_fuel ** 2 / 4.0
    a_ox = n * math.pi * injector.d_ox ** 2 / 4.0
    assert a_f == pytest.approx(injector.area_fuel_total, rel=1e-12)
    assert a_ox == pytest.approx(injector.area_ox_total, rel=1e-12)


# --------------------------------------------------------------------------
# momentum balance
# --------------------------------------------------------------------------

def test_doublet_resultant_lands_on_the_axis(injector):
    """
    Zero by construction, not by tuning. A doublet whose transverse momenta do
    not cancel throws its spray sideways and drives one side of the wall hot.
    """
    assert injector.resultant_angle == pytest.approx(0.0, abs=1e-9)


def test_heavier_stream_gets_the_shallower_angle(injector):
    """Physical: the stream with more momentum needs less turning to cancel."""
    if injector.momentum_ox > injector.momentum_fuel:
        assert injector.angle_ox < injector.angle_fuel
    else:
        assert injector.angle_fuel < injector.angle_ox


def test_equal_momenta_give_the_symmetric_split():
    a_f, a_ox = balanced_doublet_angles(1.0, 1.0, math.radians(60.0))
    assert a_f == pytest.approx(math.radians(30.0), rel=1e-12)
    assert a_ox == pytest.approx(math.radians(30.0), rel=1e-12)


def test_angles_sum_to_the_included_angle(injector):
    for total in (30.0, 45.0, 60.0, 90.0):
        a_f, a_ox = balanced_doublet_angles(3.0, 7.0, math.radians(total))
        assert a_f + a_ox == pytest.approx(math.radians(total), rel=1e-12)


# --------------------------------------------------------------------------
# response to the design knobs
# --------------------------------------------------------------------------

def test_stiffer_injection_gives_smaller_orifices(chamber):
    soft = design_injector(chamber, 24, 31.9, stiffness=0.10)
    stiff = design_injector(chamber, 24, 31.9, stiffness=0.30)
    assert stiff.d_fuel < soft.d_fuel
    assert stiff.v_fuel > soft.v_fuel


def test_more_elements_give_smaller_orifices_at_the_same_total_area(chamber):
    few = design_injector(chamber, 12, 31.9)
    many = design_injector(chamber, 48, 31.9)
    assert many.d_fuel < few.d_fuel
    assert many.area_fuel_total == pytest.approx(few.area_fuel_total, rel=1e-12)


def test_injection_velocity_is_independent_of_element_count(chamber):
    """Velocity follows total area, which stiffness alone fixes."""
    few = design_injector(chamber, 12, 31.9)
    many = design_injector(chamber, 48, 31.9)
    assert many.v_fuel == pytest.approx(few.v_fuel, rel=1e-12)


# --------------------------------------------------------------------------
# fitting the face
# --------------------------------------------------------------------------

def test_the_fitted_pattern_actually_fits(injector):
    ok, why = fits_on_face(injector, *ANNULUS)
    assert ok, why


def test_orifices_stay_inside_the_annulus(injector):
    fuel, ox = element_positions(injector)
    inner, outer = ANNULUS
    for y, z in fuel:
        assert inner < math.hypot(y, z) < outer
    for y, z in ox:
        assert inner < math.hypot(y, z) < outer


def test_element_count_matches_the_positions(injector):
    fuel, ox = element_positions(injector)
    assert len(fuel) == len(ox) == injector.n_elements


def test_an_impossible_face_is_reported_not_faked(chamber):
    """A 0.2 mm wide annulus cannot take a doublet. Say so rather than return one."""
    d, why = fit_injector_to_face(chamber, 30.0, 30.2)
    assert "nothing fits" in why
    assert not fits_on_face(d, 30.0, 30.2)[0]


def test_chug_margin_tracks_stiffness(chamber):
    soft = design_injector(chamber, 24, 31.9, stiffness=0.10)
    stiff = design_injector(chamber, 24, 31.9, stiffness=0.30)
    assert chug_margin(soft) < 1.0 < chug_margin(stiff)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_rejects_zero_elements(chamber):
    with pytest.raises(ValueError):
        design_injector(chamber, 0, 31.9)


def test_rejects_absurd_stiffness(chamber):
    with pytest.raises(ValueError):
        design_injector(chamber, 24, 31.9, stiffness=1.5)


def test_rejects_impossible_discharge_coefficient(chamber):
    with pytest.raises(ValueError):
        design_injector(chamber, 24, 31.9, discharge_coefficient=1.4)
