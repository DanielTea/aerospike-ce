"""
Physical invariants for the plug contour.

These are the assertions that let Claude Code iterate on the geometry without
being able to see it. If you add physics, add a test here first.
"""

import math

import pytest

from contour_ref import (
    area_ratio,
    build_contour,
    mach_from_area_ratio,
    prandtl_meyer,
)

GAMMA = 1.20


# --------------------------------------------------------------------------
# gas dynamics
# --------------------------------------------------------------------------

def test_area_ratio_unity_at_sonic():
    assert area_ratio(1.0, GAMMA) == pytest.approx(1.0, abs=1e-12)


def test_area_ratio_is_monotonic_supersonic():
    prev = 1.0
    for m in [1.2, 1.5, 2.0, 3.0, 4.0, 5.0]:
        eps = area_ratio(m, GAMMA)
        assert eps > prev
        prev = eps


def test_mach_inversion_round_trips():
    for m in [1.5, 2.0, 3.3, 4.7]:
        assert mach_from_area_ratio(area_ratio(m, GAMMA), GAMMA) == pytest.approx(m, rel=1e-8)


def test_prandtl_meyer_zero_at_sonic():
    assert prandtl_meyer(1.0, GAMMA) == pytest.approx(0.0, abs=1e-12)


def test_prandtl_meyer_matches_known_value_for_air():
    # gamma = 1.4, M = 2.0 -> nu = 26.380 deg, the standard textbook check
    assert math.degrees(prandtl_meyer(2.0, 1.4)) == pytest.approx(26.380, abs=1e-3)


# --------------------------------------------------------------------------
# contour geometry
# --------------------------------------------------------------------------

@pytest.fixture
def contour():
    return build_contour(expansion_ratio=8.0, exit_radius_mm=30.0, gamma=GAMMA, n_points=400)


def test_spike_closes_on_the_axis(contour):
    """The ideal full-length plug must reach r = 0. This is the load-bearing check."""
    assert contour.r[-1] == pytest.approx(0.0, abs=0.05)


def test_starts_at_the_lip_radius(contour):
    """First point sits on the throat, just inboard of the lip."""
    assert contour.r[0] < contour.exit_radius
    assert contour.r[0] > 0.9 * contour.exit_radius


def test_radius_decreases_monotonically(contour):
    for a, b in zip(contour.r, contour.r[1:]):
        assert b <= a + 1e-9


def test_axial_coordinate_increases_monotonically(contour):
    for a, b in zip(contour.x, contour.x[1:]):
        assert b >= a - 1e-9


def test_throat_area_matches_expansion_ratio(contour):
    exit_area = math.pi * contour.exit_radius ** 2
    assert exit_area / contour.throat_area == pytest.approx(8.0, rel=1e-9)


def test_throat_plane_is_inclined(contour):
    """nu_e is the throat inclination. For eps = 8 it should be a large angle."""
    assert 30.0 < math.degrees(contour.throat_angle) < 80.0


# --------------------------------------------------------------------------
# throat closure: the lip station and the inclined sonic line
#
# These are the counterpart to test_spike_closes_on_the_axis. That one pins the
# downstream end of the contour; these pin the upstream end. Together they mean
# a wrong lip position cannot survive the suite.
# --------------------------------------------------------------------------

def test_lip_sits_downstream_of_the_spike_shoulder(contour):
    """
    The Prandtl-Meyer fan is centred on the lip, and the first contour point is
    upstream of it. After the shift that puts the shoulder at x = 0, the lip
    must land at positive x. Drawing it at x = 0 makes the throat look radial.
    """
    assert contour.lip_x > 0.0
    assert contour.lip_x < contour.exit_radius


def test_throat_area_closes_on_the_geometry(contour):
    """
    pi*(r_e + r_0)*l computed from the placed lip must reproduce pi*r_e^2/eps.
    This is an identity, not an approximation, so the tolerance is tight enough
    to catch a lip misplaced by a micron.
    """
    assert contour.throat_area_from_geometry == pytest.approx(contour.throat_area, rel=1e-12)


def test_throat_line_is_normal_to_the_throat_plane(contour):
    """The sonic line sits at pi/2 - nu_e from the axis. Exact by construction."""
    assert contour.throat_inclination + contour.throat_angle == pytest.approx(
        math.pi / 2.0, abs=1e-12
    )


def test_radial_annulus_would_understate_the_throat(contour):
    """
    Guard against the tempting mistake. Measuring the throat as a radial annulus
    at the lip station gives a much smaller number, which would place the choke
    point upstream of the design throat.
    """
    radial = math.pi * (contour.exit_radius ** 2 - contour.r[0] ** 2)
    assert radial < 0.5 * contour.throat_area


def test_lip_position_is_independent_of_truncation():
    """Truncation chops the tail. It must not move the throat."""
    full = build_contour(8.0, 30.0, GAMMA, n_points=400)
    trunc = build_contour(8.0, 30.0, GAMMA, n_points=400, truncate_fraction=0.30)
    assert trunc.lip_x == pytest.approx(full.lip_x, rel=1e-12)
    assert trunc.throat_area_from_geometry == pytest.approx(
        full.throat_area_from_geometry, rel=1e-12
    )


def test_length_scales_with_exit_radius():
    a = build_contour(8.0, 30.0, GAMMA)
    b = build_contour(8.0, 60.0, GAMMA)
    assert b.length / a.length == pytest.approx(2.0, rel=1e-6)


def test_higher_expansion_ratio_gives_longer_spike():
    short = build_contour(4.0, 30.0, GAMMA)
    long = build_contour(20.0, 30.0, GAMMA)
    assert long.length > short.length


def test_truncation_shortens_without_changing_the_upstream_contour():
    full = build_contour(8.0, 30.0, GAMMA, n_points=400)
    trunc = build_contour(8.0, 30.0, GAMMA, n_points=400, truncate_fraction=0.25)
    assert trunc.length < 0.30 * full.length
    for i in range(len(trunc.x)):
        assert trunc.x[i] == pytest.approx(full.x[i], rel=1e-9)
        assert trunc.r[i] == pytest.approx(full.r[i], rel=1e-9)


def test_printability_no_reentrant_surface(contour):
    """
    Local half-angle of the spike surface. A wildly negative slope would mean an
    undercut that no printer will resolve without support.
    """
    for i in range(1, len(contour.x)):
        dx = contour.x[i] - contour.x[i - 1]
        dr = contour.r[i] - contour.r[i - 1]
        if dx > 1e-6:
            assert dr <= 1e-6, "spike radius must never grow with x"
