"""
Physical and geometric invariants for the full engine assembly.

Same contract as test_contour.py: these assertions are how the geometry gets
checked by someone who cannot see it. The headline one is
test_contraction_lands_on_the_lip -- if the cowl wall and the Angelino contour
ever stop agreeing about where the throat is, that test fails first and loudest.
"""

import math

import numpy as np
import pytest

from contour_ref import build_contour
from engine_ref import (
    ChamberSpec,
    StructureSpec,
    build_assembly,
    measured_flow_area,
)

GAMMA = 1.20


@pytest.fixture
def contour():
    return build_contour(8.0, 30.0, GAMMA, n_points=300, truncate_fraction=0.30)


@pytest.fixture
def engine(contour):
    return build_assembly(contour)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _segments_self_intersect(x, r) -> bool:
    """
    True if any two non-adjacent edges of the closed profile cross.

    A profile that crosses itself revolves into a solid with inverted pockets,
    which slices into something that is not the part. Vectorised because the
    spike profile carries several hundred vertices.
    """
    p = np.stack([x, r], axis=1)
    q = np.roll(p, -1, axis=0)
    n = len(p)

    a0 = p[:, None, :]
    a1 = q[:, None, :]
    b0 = p[None, :, :]
    b1 = q[None, :, :]

    def cross(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - \
               (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])

    d1 = cross(a0, a1, b0)
    d2 = cross(a0, a1, b1)
    d3 = cross(b0, b1, a0)
    d4 = cross(b0, b1, a1)
    hit = ((d1 * d2 < 0.0) & (d3 * d4 < 0.0))

    i = np.arange(n)
    adjacent = (np.abs(i[:, None] - i[None, :]) <= 1) | \
               (np.abs(i[:, None] - i[None, :]) == n - 1)
    return bool(np.any(hit & ~adjacent))


def _min_profile_distance(pa, pb) -> float:
    d = np.hypot(pa.x[:, None] - pb.x[None, :], pa.r[:, None] - pb.r[None, :])
    return float(d.min())


# --------------------------------------------------------------------------
# the load-bearing self-check
# --------------------------------------------------------------------------

def test_contraction_lands_on_the_lip(engine, contour):
    """
    The cowl wall is solved from an area schedule that knows nothing about the
    Angelino contour beyond the throat area and nu_e. Its last station must
    nevertheless land exactly on the cowl lip.

    Nothing in the construction forces this. It holds because the area schedule
    and the plug contour agree about where the throat is. If this breaks, the
    throat has moved and every downstream number is suspect.
    """
    assert engine.outer_wall_x[-1] == pytest.approx(contour.lip_x, rel=1e-9)
    assert engine.outer_wall_r[-1] == pytest.approx(contour.exit_radius, rel=1e-12)


def test_contraction_starts_on_the_chamber_wall(engine):
    """The other end of the same schedule must close on the chamber radius."""
    assert engine.outer_wall_r[0] == pytest.approx(engine.chamber_outer_radius, rel=1e-12)


# --------------------------------------------------------------------------
# areas
# --------------------------------------------------------------------------

def test_chamber_area_matches_contraction_ratio(engine, contour):
    assert engine.chamber_area / contour.throat_area == pytest.approx(3.0, rel=1e-12)


def test_chamber_radius_reproduces_its_own_area(engine):
    """Annulus between the cylindrical centrebody and the chamber wall."""
    a = math.pi * (engine.chamber_outer_radius ** 2 - engine.shoulder_radius ** 2)
    assert a == pytest.approx(engine.chamber_area, rel=1e-12)


def test_duct_chokes_at_the_throat_and_nowhere_else(engine, contour):
    """
    Measured, not prescribed: shortest distance from the cowl wall to the
    centrebody, swept as a frustum. A straight conical contraction fails this
    badly -- it pinches to about 0.47 * A_t just aft of the shoulder.
    """
    x, areas = measured_flow_area(engine, n=600)
    a_t = contour.throat_area

    assert areas.min() == pytest.approx(a_t, rel=2e-3)
    assert x[int(np.argmin(areas))] == pytest.approx(contour.lip_x, abs=0.05)
    # nothing upstream of the lip may dip to the throat area
    assert areas[:-1].min() >= a_t * (1.0 - 1e-3)


def test_area_contracts_monotonically(engine):
    """
    Small upward steps are the nearest-point search jumping between vertices of
    a discrete polyline, not a real diffuser. Anything larger is a real one.
    """
    _, areas = measured_flow_area(engine, n=600)
    assert np.max(np.diff(areas)) < 0.005 * areas[0]


# --------------------------------------------------------------------------
# wall shape
# --------------------------------------------------------------------------

def test_cowl_wall_never_doubles_back(engine):
    """A non-monotonic x means the meridional profile folds over itself."""
    assert np.all(np.diff(engine.outer_wall_x) > -1e-9)


@pytest.mark.parametrize("cr", [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0])
def test_cowl_wall_never_doubles_back_at_any_contraction_ratio(contour, cr):
    """
    The original test only ever ran at the default contraction ratio of 3, and
    the fold does not appear until 6. Everything downstream interpolates along
    this wall, so a fold of a couple of microns is enough to hand the thermal
    model a wall radius that is a millimetre wrong.

    The fold turned out not to depend on the converging length at all -- it is
    set by how large the offset distance is when the corner fan starts rotating,
    which is why the fan start is now derived from the area schedule rather than
    fixed. One converging length across every ratio is the point of the test.
    """
    a = build_assembly(contour, ChamberSpec(contraction_ratio=cr,
                                            converging_length_mm=35.0))
    assert np.all(np.diff(a.outer_wall_x) > -1e-9)
    # and the fix must not have moved the lip while straightening the wall
    assert a.outer_wall_x[-1] == pytest.approx(contour.lip_x, rel=1e-9)
    assert a.outer_wall_r[-1] == pytest.approx(contour.exit_radius, rel=1e-12)


@pytest.mark.parametrize("cr", [2.0, 6.0, 12.0, 20.0])
def test_the_duct_still_chokes_at_the_throat_at_any_contraction_ratio(contour, cr):
    """
    Moving the fan start must not disturb what the fan was for. Whatever the
    contraction ratio, the measured minimum flow area is still the throat area
    and it is still at the lip.
    """
    a = build_assembly(contour, ChamberSpec(contraction_ratio=cr,
                                            converging_length_mm=35.0))
    x, areas = measured_flow_area(a, n=300)
    assert areas.min() == pytest.approx(contour.throat_area, rel=2e-3)
    assert x[int(np.argmin(areas))] == pytest.approx(contour.lip_x, abs=0.1)


@pytest.mark.parametrize("cr", [2.0, 6.0, 12.0, 20.0])
def test_the_cowl_outer_skin_does_not_fold_either(contour, cr):
    """
    The inner wall is not the only offset in this part. The outer skin is the
    inner wall pushed outward through the throat turn, which is a concave corner
    and folds there exactly as the cavity folds at the convex shoulder.
    """
    a = build_assembly(contour, ChamberSpec(contraction_ratio=cr,
                                            converging_length_mm=35.0))
    assert np.all(np.diff(a.cowl_outer_x) > -1e-9)
    assert not _segments_self_intersect(a.profiles["cowl"].x, a.profiles["cowl"].r)


def test_both_walls_are_printable_without_support(engine):
    """
    Printed head-down, x is the build direction. Every outward-facing surface
    must shrink in radius as x grows, or it overhangs.
    """
    assert np.all(np.diff(engine.outer_wall_r) <= 1e-9)
    assert np.all(np.diff(engine.inner_wall_r) <= 1e-9)


def test_cowl_ends_at_the_lip_not_beyond(engine, contour):
    """Downstream of the lip an aerospike has no outer wall. That is the point."""
    assert engine.outer_wall_x.max() == pytest.approx(contour.lip_x, rel=1e-9)
    assert engine.inner_wall_x.max() > contour.lip_x


# --------------------------------------------------------------------------
# solids
# --------------------------------------------------------------------------

def test_every_profile_winds_the_same_way(engine):
    """Negative volume means the revolved solid is inside out."""
    for name, p in engine.profiles.items():
        assert p.revolved_volume > 0.0, name


def test_no_profile_crosses_itself(engine):
    for name, p in engine.profiles.items():
        assert not _segments_self_intersect(p.x, p.r), name


def test_cowl_clears_the_centrebody(engine, contour):
    """The parts must not interfere. Closest approach is the throat itself."""
    gap = _min_profile_distance(engine.profiles["cowl"], engine.profiles["centrebody"])
    assert gap > 0.0
    assert gap == pytest.approx(contour.throat_slant_length, abs=0.05)


def test_centrebody_wall_thickness_holds_on_the_cylinder(engine):
    """
    On the cylindrical section the inward normal is radial, so the cavity radius
    is exactly the shoulder radius less the wall thickness.
    """
    w = engine.structure.wall_thickness_mm
    p = engine.profiles["centrebody"]
    near_head = p.r[np.abs(p.x - engine.head_x) < 1e-6]
    assert near_head.min() == pytest.approx(engine.shoulder_radius - w, abs=1e-9)


def test_wall_is_never_thinner_than_specified(engine):
    """
    The cavity is an eroded offset, so at the shoulder it cuts a chord across a
    corner it cannot follow. That makes the wall locally thicker. It must never
    make it thinner -- a thin spot is a burst pressure problem, and it would not
    be visible in any render.
    """
    from engine_ref import _distance_to_polyline

    w = engine.structure.wall_thickness_mm
    d = _distance_to_polyline(
        engine.cavity_x, engine.cavity_r,
        np.asarray(engine.inner_wall_x), np.asarray(engine.inner_wall_r),
    )
    assert d.min() >= w - 1e-6


def test_internal_cavity_stays_inside_the_part(engine):
    """Wall thickness may not eat through the surface or invert past the axis."""
    w = engine.structure.wall_thickness_mm
    p = engine.profiles["centrebody"]
    assert p.r.min() >= 0.0
    interior = p.r[p.r > 1e-9]
    assert interior.min() > 0.5 * w


def test_cavity_is_open_at_the_head_so_powder_can_escape(engine):
    """
    A sealed internal void traps powder or resin. The bore stays open through
    the head disc, which is why the head is an annulus and not a plate.
    """
    head = engine.profiles["head"]
    bore = head.r.min()
    assert bore == pytest.approx(engine.shoulder_radius - engine.structure.wall_thickness_mm,
                                 abs=1e-9)
    assert bore > 0.0


def test_base_cap_leaves_the_designed_thickness(engine, contour):
    """Solid plug behind the truncation, so the spike does not end in a hole."""
    p = engine.profiles["centrebody"]
    x_tip = engine.inner_wall_x[-1]
    on_axis = p.x[p.r < 1e-9]
    assert len(on_axis) >= 2
    cap = x_tip - on_axis.min()
    assert cap == pytest.approx(engine.structure.base_cap_thickness_mm, abs=1e-6)


def test_flange_sits_outboard_of_the_cowl(engine):
    cowl_max_r = engine.profiles["cowl"].r.max()
    assert engine.flange_radius > cowl_max_r
    assert engine.flange_radius == pytest.approx(
        cowl_max_r + engine.structure.flange_width_mm, abs=1e-9
    )


# --------------------------------------------------------------------------
# scaling and parameter response
# --------------------------------------------------------------------------

def test_assembly_scales_with_exit_radius():
    """Geometric similarity: double the nozzle, double every length."""
    small = build_assembly(build_contour(8.0, 30.0, GAMMA, n_points=300),
                           ChamberSpec(3.0, 18.0, 14.0), StructureSpec())
    big = build_assembly(build_contour(8.0, 60.0, GAMMA, n_points=300),
                         ChamberSpec(3.0, 36.0, 28.0), StructureSpec(wall_thickness_mm=4.0))
    assert big.chamber_outer_radius / small.chamber_outer_radius == pytest.approx(2.0, rel=1e-9)
    assert big.shoulder_radius / small.shoulder_radius == pytest.approx(2.0, rel=1e-9)


def test_higher_contraction_ratio_widens_the_chamber(contour):
    lo = build_assembly(contour, ChamberSpec(contraction_ratio=2.0))
    hi = build_assembly(contour, ChamberSpec(contraction_ratio=4.0))
    assert hi.chamber_outer_radius > lo.chamber_outer_radius
    # and both still land on the same lip
    assert lo.outer_wall_x[-1] == pytest.approx(hi.outer_wall_x[-1], rel=1e-9)


def test_truncation_does_not_move_the_chamber():
    """Chopping the spike is a downstream decision. It must not touch the duct."""
    full = build_assembly(build_contour(8.0, 30.0, GAMMA, n_points=300))
    trunc = build_assembly(build_contour(8.0, 30.0, GAMMA, n_points=300,
                                         truncate_fraction=0.30))
    assert full.chamber_outer_radius == pytest.approx(trunc.chamber_outer_radius, rel=1e-12)
    assert np.allclose(full.outer_wall_r, trunc.outer_wall_r)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def test_contraction_ratio_below_one_is_rejected(contour):
    with pytest.raises(ValueError):
        build_assembly(contour, ChamberSpec(contraction_ratio=0.9))


def test_wall_thicker_than_the_centrebody_is_rejected(contour):
    with pytest.raises(ValueError):
        build_assembly(contour, structure=StructureSpec(wall_thickness_mm=40.0))


def test_base_cap_longer_than_the_spike_is_rejected(contour):
    with pytest.raises(ValueError):
        build_assembly(contour, structure=StructureSpec(base_cap_thickness_mm=500.0))
