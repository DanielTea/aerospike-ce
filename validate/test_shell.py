"""
Invariants for the external shell: stiffening ribs and mounting legs.

The rib checks are mostly about printability, because there are several hundred
of them and a marginal flank repeated four hundred times is not a marginal
problem. The leg checks are about buckling, which is the failure a leg sized
only on stress walks straight into.
"""

import json
import math
import os

import numpy as np
import pytest

from engine_design import design_engine
from mesh_solid import cowl_channels
from shell_ref import size_legs, size_ribs, shell_features

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def design():
    with open(SPEC, encoding="utf-8") as fh:
        return design_engine(json.load(fh))


@pytest.fixture(scope="module")
def shell(design):
    return shell_features(design)


# --------------------------------------------------------------------------
# ribs
# --------------------------------------------------------------------------

def test_there_is_one_rib_per_channel(design, shell):
    """
    A rib sits on a land, and there is exactly one land per channel. Any other
    count means the ribs are not on the lands, which is the whole reason they
    cost nothing to place.
    """
    _, sized, _ = shell
    for part, ring in sized.items():
        assert ring.count == design.circuits[part].channel.n_channels


def test_rib_flanks_are_self_supporting(shell):
    _, sized, _ = shell
    for part, ring in sized.items():
        assert ring.self_supporting, f"{part} flanks at {ring.flank_deg:.0f} deg"


def test_a_shallow_flank_is_refused(design):
    """Not a warning. Several hundred marginal overhangs is a support problem."""
    a = design.assembly
    cut = cowl_channels(a, design.circuits["cowl"].channel)
    with pytest.raises(ValueError, match="process angle"):
        size_ribs(cut, a.cowl_outer_x, a.cowl_outer_r, cut.x_start, cut.x_end,
                  flank_deg=30.0)


def test_ribs_multiply_the_external_area(shell):
    """The point of them, after stiffness: a chamber that radiates wants area."""
    _, sized, _ = shell
    for ring in sized.values():
        assert ring.area_ratio > 2.0


def test_rib_root_sits_on_the_land_not_the_channel(design, shell):
    """
    A rib wider than its land overhangs the channel it is meant to close; one
    narrower wastes the material it is standing on.
    """
    _, sized, _ = shell
    for part, ring in sized.items():
        channel = design.circuits[part].channel
        assert ring.root_width_mm >= channel.land_mm - 1e-6


def test_taller_ribs_need_more_flank_run(design):
    a = design.assembly
    cut = cowl_channels(a, design.circuits["cowl"].channel)
    short = size_ribs(cut, a.cowl_outer_x, a.cowl_outer_r, cut.x_start, cut.x_end,
                      height_mm=1.0)
    tall = size_ribs(cut, a.cowl_outer_x, a.cowl_outer_r, cut.x_start, cut.x_end,
                     height_mm=4.0)
    assert tall.area_ratio > short.area_ratio


# --------------------------------------------------------------------------
# legs
# --------------------------------------------------------------------------

def test_legs_do_not_lean_further_than_the_process_holds(shell):
    _, _, leg = shell
    assert leg.self_supporting, f"{leg.splay_deg:.0f} deg splay"


def test_an_over_splayed_leg_is_refused():
    with pytest.raises(ValueError, match="process"):
        size_legs(9000.0, x_top_mm=-50.0, r_top_mm=100.0, x_foot_mm=-160.0,
                  splay_deg=60.0)


def test_legs_survive_buckling(shell):
    """
    The check a leg sized on stress alone fails. A slender strut is comfortably
    strong in compression and folds sideways well below that load.
    """
    _, _, leg = shell
    assert leg.buckling_ok
    assert leg.buckling_load_n > 2.0 * leg.design_load_n


def test_legs_carry_the_vacuum_thrust(design, shell):
    _, _, leg = shell
    assert leg.count * leg.design_load_n >= design.chamber.thrust_vacuum


def test_a_longer_leg_buckles_sooner():
    """Euler goes as one over length squared; the model must show it."""
    short = size_legs(9000.0, -120.0, 100.0, -160.0, splay_deg=30.0)
    long = size_legs(9000.0, -20.0, 100.0, -160.0, splay_deg=30.0)
    assert long.length_mm > short.length_mm
    assert long.buckling_load_n < short.buckling_load_n


def test_legs_are_specified_by_where_the_feet_land():
    """
    Footprint is what a mount is actually specified by. A splay angle is a local
    property that says nothing about it: 35 degrees over a 158 mm drop puts the
    feet at r 211 mm and doubles the engine's footprint.
    """
    leg = size_legs(9000.0, -60.0, 100.0, -160.0, foot_radius_mm=140.0)
    assert leg.r_foot_mm == pytest.approx(140.0)
    assert leg.self_supporting


def test_feet_that_cannot_be_reached_are_refused():
    """Reaching too far outboard over too short a drop is an overhang, not a leg."""
    with pytest.raises(ValueError, match="splay"):
        size_legs(9000.0, -150.0, 100.0, -160.0, foot_radius_mm=250.0)


def test_the_footprint_stays_sane(shell, design):
    """
    Feet far outboard of the flange are a mount nobody can bolt to a stand. The
    render is what caught this: legs reaching r 211 mm against a 123 mm flange.
    """
    _, _, leg = shell
    assert leg.r_foot_mm < 1.6 * design.assembly.flange_radius


def test_fewer_than_three_legs_is_refused():
    with pytest.raises(ValueError):
        size_legs(9000.0, -50.0, 100.0, -160.0, count=2)


def test_the_foot_sits_outboard_of_the_chamber(shell):
    _, _, leg = shell
    assert leg.r_foot_mm > leg.r_top_mm
    assert leg.pad_radius_mm > leg.r_foot_mm


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_shell_features_land_on_the_right_parts(shell):
    feats, _, _ = shell
    assert feats["cowl"]["ribs"] and feats["centrebody"]["ribs"]
    assert feats["cowl"]["legs"]
    assert not feats["head"]["ribs"], "the head disc has no channels to rib"


def test_rib_direction_matches_its_wall(shell):
    """
    The cowl's ribs stand outward from its outer skin; the centrebody's stand
    inward from its bore. Getting the sign wrong buries them in the part.
    """
    feats, _, _ = shell
    assert feats["cowl"]["ribs"][0].outward is True
    assert feats["centrebody"]["ribs"][0].outward is False


def test_ribs_sit_between_the_channels(shell, design):
    """
    Phase offset by half a pitch. In phase with the channels, every rib would
    stand on a void.
    """
    _, sized, _ = shell
    for part, ring in sized.items():
        expected = math.pi / design.circuits[part].channel.n_channels
        assert ring.phase == pytest.approx(expected, rel=1e-9)


def test_the_ribbed_shell_is_still_watertight(design, shell):
    """
    The one that caught the real bug: material added outside the profile was
    clipped at the field boundary, and a clipped solid still meshes.
    """
    from mesh_export import manifold_report
    from mesh_solid import build_mesh_streaming

    feats, _, _ = shell
    v, f = build_mesh_streaming(design.assembly.profiles["cowl"], voxel_mm=0.7,
                                ribs=feats["cowl"]["ribs"])
    rep = manifold_report(v, f)
    assert rep["watertight"]
    assert rep["boundary_edges"] == 0


def test_ribs_add_material_rather_than_removing_it(design, shell):
    from mesh_export import mesh_volume
    from mesh_solid import build_mesh_streaming

    feats, _, _ = shell
    plain = build_mesh_streaming(design.assembly.profiles["cowl"], voxel_mm=0.7)
    ribbed = build_mesh_streaming(design.assembly.profiles["cowl"], voxel_mm=0.7,
                                  ribs=feats["cowl"]["ribs"])
    assert mesh_volume(*ribbed) > mesh_volume(*plain)
