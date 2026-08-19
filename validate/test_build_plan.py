"""
Invariants for the build plan: the seam between Python and C#.

This file is the contract. Python owns every number and writes down the
resulting shapes; C# owns the voxel kernel and reads them. The tests that matter
are the ones that stop the contract rotting: that everything the kernel needs is
present, that nothing it does not need leaks across, and that a reader can tell
when it has been handed a plan it does not understand.
"""

import json
import math
import os

import pytest

from build_plan import PLAN_VERSION, build_plan, narrowest_feature
from engine_design import design_engine

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def spec():
    with open(SPEC, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def design(spec):
    return design_engine(spec)


@pytest.fixture(scope="module")
def plan(spec, design):
    return build_plan(spec, design)


# --------------------------------------------------------------------------
# the contract itself
# --------------------------------------------------------------------------

def test_the_plan_is_versioned(plan):
    """
    A reader that cannot tell which contract it is holding will guess, and a
    silently misread plan builds a plausible engine that is not the designed one.
    """
    assert plan["plan_version"] == PLAN_VERSION
    assert plan["units"] == "mm"


def test_the_plan_serialises(plan):
    """Anything that will not round-trip through JSON is not a contract."""
    text = json.dumps(plan)
    back = json.loads(text)
    assert back == plan


def test_no_physics_crosses_the_boundary(plan):
    """
    Shapes and dimensions only. If the kernel would need a physical property to
    use a field, that field belongs on the other side -- and its presence here
    is how a second implementation of the physics starts growing back.
    """
    forbidden = ("propellant", "mixture_ratio", "gamma", "c_star", "bartz",
                 "heat_flux", "coolant", "expansion_ratio", "contraction_ratio",
                 "material", "density", "velocity", "temperature")
    text = json.dumps(plan).lower()
    for word in forbidden:
        assert word not in text, f"'{word}' leaked into the build plan"


def test_the_summary_is_reporting_only(plan):
    """
    The summary exists so a log line can say what was built. Nothing in the
    geometry may depend on it.
    """
    assert set(plan["summary"]) == {
        "thrust_sea_level_n", "thrust_vacuum_n", "isp_sea_level_s",
        "isp_vacuum_s", "chamber_pressure_bar", "overall_length_mm",
        "overall_diameter_mm"}


# --------------------------------------------------------------------------
# everything the kernel needs is there
# --------------------------------------------------------------------------

def test_every_part_has_a_closed_profile(plan):
    for name, part in plan["parts"].items():
        x, r = part["profile"]["x"], part["profile"]["r"]
        assert len(x) == len(r) >= 3, name
        assert all(v >= 0.0 for v in r), f"{name} has a negative radius"


def test_the_parts_are_the_designed_parts(plan, design):
    assert set(plan["parts"]) == set(design.assembly.profiles)


def test_profiles_match_the_design(plan, design):
    """Rounding to a micron is well under any voxel; anything more is a bug."""
    for name, part in plan["parts"].items():
        src = design.assembly.profiles[name]
        assert len(part["profile"]["x"]) == len(src.x)
        assert part["profile"]["x"][0] == pytest.approx(float(src.x[0]), abs=1e-3)
        assert part["profile"]["r"][0] == pytest.approx(float(src.r[0]), abs=1e-3)


def test_channels_carry_everything_needed_to_place_them(plan):
    fields = {"wall", "count", "width_mm", "height_mm", "hot_wall_mm",
              "land_mm", "x_start_mm", "x_end_mm", "outward"}
    found = 0
    for part in plan["parts"].values():
        for c in part["channels"]:
            assert set(c) == fields
            assert c["count"] > 0 and c["width_mm"] > 0 and c["height_mm"] > 0
            assert c["x_end_mm"] > c["x_start_mm"]
            assert len(c["wall"]["x"]) == len(c["wall"]["r"]) >= 2
            found += 1
    assert found == 2, "both jackets should be in the plan"


def test_channel_counts_match_the_thermal_design(plan, design):
    """
    The geometry has to build the number of channels the cooling model sized. A
    mismatch means the thermal answer describes a part nobody built.
    """
    for name in ("cowl", "centrebody"):
        circuit = design.circuits[name]
        planned = plan["parts"][name]["channels"][0]["count"]
        assert planned == circuit.channel.n_channels


def test_every_channel_ring_has_a_feed(plan):
    """A channel with nothing feeding it is a slot, not a cooling circuit."""
    for name, part in plan["parts"].items():
        if part["channels"]:
            assert part["ports"], f"{name} has channels but no feed ports"
            assert part["ports"][0]["count"] == part["channels"][0]["count"]


def test_the_injector_is_in_the_plan(plan, design):
    holes = plan["parts"]["head"]["holes"]
    orifices = [h for h in holes if h["count"] == design.injector.n_elements]
    assert len(orifices) >= 2, "fuel and oxidiser rings"


def test_the_mounts_are_in_the_plan(plan):
    lugs = plan["parts"]["head"]["lugs"]
    assert lugs and lugs[0]["count"] >= 3
    lug = lugs[0]
    fixings = [h for h in plan["parts"]["head"]["holes"] if h["count"] == lug["count"]]
    assert fixings, "lugs with no fixing holes are not mounts"


def test_manifolds_are_in_the_plan(plan):
    total = sum(len(p["plenums"]) for p in plan["parts"].values())
    assert total >= 3, "cowl inlet ring, head fuel manifold, oxidiser dome"


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_narrowest_feature_finds_the_narrowest_thing(plan):
    smallest = math.inf
    for part in plan["parts"].values():
        for c in part["channels"]:
            smallest = min(smallest, c["width_mm"], c["height_mm"], c["hot_wall_mm"])
        for h in part["holes"]:
            smallest = min(smallest, h["diameter_mm"])
        for p in part["ports"]:
            smallest = min(smallest, p["diameter_mm"])
    assert narrowest_feature(plan) == pytest.approx(smallest)


def test_the_plan_states_a_voxel_size(plan):
    assert plan["voxel_size_mm"] > 0.0


def test_the_build_direction_is_stated(plan):
    """Printability chose it; the kernel should not have to guess."""
    assert plan["build_direction"] in ("+x", "-x")


# --------------------------------------------------------------------------
# the plan describes the same geometry the Python mesher builds
# --------------------------------------------------------------------------

def test_plan_features_match_the_python_features(plan, design):
    """
    Both sides build from the same feature objects, so this is really a check
    that nothing was dropped or renamed on the way into the file.
    """
    from manifold_ref import design_manifolds, geometry_features
    from mesh_solid import cowl_channels, feed_ports

    a = design.assembly
    gf = geometry_features(design, design_manifolds(design))
    cut = cowl_channels(a, design.circuits["cowl"].channel)
    port = feed_ports(a, cut)

    pc = plan["parts"]["cowl"]["channels"][0]
    assert pc["count"] == cut.n_channels
    assert pc["width_mm"] == pytest.approx(cut.width_mm, abs=1e-5)
    assert pc["x_start_mm"] == pytest.approx(cut.x_start, abs=1e-4)
    assert pc["outward"] is cut.outward

    pp = plan["parts"]["cowl"]["ports"][0]
    assert pp["diameter_mm"] == pytest.approx(port.diameter_mm, abs=1e-5)
    assert pp["x_mm"] == pytest.approx(port.x_at, abs=1e-4)

    assert len(plan["parts"]["cowl"]["bosses"]) == len(gf["cowl"]["bosses"])
    assert len(plan["parts"]["head"]["plenums"]) == len(gf["head"]["plenums"])
