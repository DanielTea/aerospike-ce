"""
Invariants for the printability analysis.

The load-bearing one is the normal convention. Taking the wrong outward normal
inverts the whole analysis -- every upward face is reported as an overhang and
every genuine one is missed -- and the output looks entirely reasonable while it
does. So the convention is pinned against the distance field, which knows
independently which side is material.
"""

import json
import math
import os

import numpy as np
import pytest

from engine_design import design_engine
from engine_ref import Profile
from mesh_solid import _polygon_sdf
from printability_ref import (
    Finding,
    ProcessSpec,
    analyse,
    check_profile,
    choose_build_direction,
    surface_angles,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


# --------------------------------------------------------------------------
# the convention everything else rests on
# --------------------------------------------------------------------------

@pytest.mark.parametrize("part", ["centrebody", "cowl", "head"])
def test_outward_normal_agrees_with_the_distance_field(design, part):
    """
    Step off each edge along the normal the analysis uses and land outside the
    solid. The distance field decides, not the winding convention I assumed.
    """
    p = design.assembly.profiles[part]
    x = np.asarray(p.x, dtype=float)
    r = np.asarray(p.r, dtype=float)
    dx = np.roll(x, -1) - x
    dr = np.roll(r, -1) - r
    length = np.hypot(dx, dr)

    tested = 0
    for i in range(0, len(x), max(1, len(x) // 40)):
        if length[i] < 1e-6:
            continue
        # the analysis takes the outward normal as (-dr, dx)
        nx, nr = -dr[i] / length[i], dx[i] / length[i]
        mx, mr = x[i] + dx[i] / 2, r[i] + dr[i] / 2
        step = 0.4
        px, pr = mx + step * nx, mr + step * nr
        if pr < 0.0:
            continue
        d = float(_polygon_sdf(np.array([[px]], dtype=np.float32),
                               np.array([[pr]], dtype=np.float32),
                               x.astype(np.float32), r.astype(np.float32))[0, 0])
        assert d > 0.0, f"{part} edge {i}: normal points into the solid"
        tested += 1
    assert tested >= min(4, len(x))


def _profile(x, r) -> Profile:
    """
    Build a test profile through Profile, so it winds the way the real ones do.

    Profile settles the winding from the revolved volume. A raw array written by
    hand is as likely as not to wind the other way, and then the test is
    measuring a different solid from the one the analysis sees.
    """
    return Profile("test", np.asarray(x, dtype=float), np.asarray(r, dtype=float))


def test_a_wall_along_the_build_axis_is_vertical():
    p = _profile([0.0, 10.0, 10.0, 0.0], [5.0, 5.0, 1.0, 1.0])
    angle, _, _, _ = surface_angles(p.x, p.r, build_plus_x=True)
    assert 90.0 == pytest.approx(max(angle))


def test_a_flat_face_reads_as_horizontal():
    p = _profile([0.0, 10.0, 10.0, 0.0], [5.0, 5.0, 1.0, 1.0])
    angle, _, _, _ = surface_angles(p.x, p.r, build_plus_x=True)
    assert 0.0 == pytest.approx(min(angle), abs=1e-9)


# --------------------------------------------------------------------------
# the checker fires, and on the right things
# --------------------------------------------------------------------------

def test_a_deliberate_overhang_is_caught():
    """A checker that never fires is not a checker."""
    process = ProcessSpec()
    # a cone flaring outward as x grows: built +x its underside overhangs badly
    p = _profile([0.0, 10.0, 10.0, 0.0], [2.0, 30.0, 32.0, 4.0])
    found = check_profile("test", p.x, p.r, process, build_plus_x=True)
    assert any(f.kind == "overhang" for f in found)


def test_a_flat_ceiling_is_caught_as_a_bridge():
    process = ProcessSpec(max_bridge_mm=1.0)
    p = _profile([0.0, 20.0, 20.0, 10.0, 10.0, 0.0],
                 [0.0, 0.0, 30.0, 30.0, 25.0, 25.0])
    found = check_profile("test", p.x, p.r, process, build_plus_x=True)
    assert any(f.kind == "bridge" for f in found)


def test_the_plate_face_is_not_an_overhang():
    """
    Whichever face is lowest sits on the build plate and is supported by it.
    Counting it reports the head disc's own underside as a 50 mm bridge.
    """
    process = ProcessSpec()
    p = _profile([0.0, 4.0, 4.0, 0.0], [20.0, 20.0, 60.0, 60.0])
    found = check_profile("disc", p.x, p.r, process, build_plus_x=True)
    assert [f for f in found if abs(f.x_mm - 0.0) < 1e-9] == []


# --------------------------------------------------------------------------
# the real engine
# --------------------------------------------------------------------------

def test_the_centrebody_is_printable(design):
    """
    The cavity cone and the slope clamp between them clear the centrebody. Before
    those it carried a 30 mm unsupported internal ceiling and 124 overhangs.
    """
    res = analyse(design, build_plus_x=True)
    assert [f for f in res.findings if f.part == "centrebody"] == []


def test_head_down_beats_spike_down(design):
    """
    Not a matter of taste. Head down, both outer surfaces narrow as they rise
    and the channels run parallel to the build direction; the other way up
    inverts all of that at once.
    """
    best, up, down = choose_build_direction(design)
    assert best.build_direction == "+x"
    assert len(up.findings) < len(down.findings)


def test_the_engine_fits_the_machine(design):
    res = analyse(design)
    assert res.envelope_ok, f"{res.height_mm:.0f} x {res.diameter_mm:.0f} mm"


def test_channels_meet_the_process_minimum(design):
    res = analyse(design)
    assert res.of_kind("feature") == []


def test_a_too_fine_channel_is_reported(design):
    """Tighten the process until the design no longer suits it, and it says so."""
    res = analyse(design, ProcessSpec(min_feature_mm=2.0, min_wall_mm=2.0))
    assert res.of_kind("feature")


def test_a_small_machine_is_reported(design):
    res = analyse(design, ProcessSpec(envelope_mm=(80.0, 80.0, 80.0)))
    assert not res.envelope_ok
    assert res.of_kind("envelope")


def test_powder_can_get_out(design):
    res = analyse(design)
    assert res.of_kind("drainage") == []
