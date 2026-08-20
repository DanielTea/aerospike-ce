"""
Invariants for the evolutionary optimiser.

An optimiser is only testable if it is deterministic, so everything here is
seeded. The load-bearing checks are that the constraint ranking is Deb's and not
a disguised penalty function, and that elitism actually holds -- a (mu + lambda)
strategy that can lose its best design is not one.
"""

import copy
import json
import math
import os

import numpy as np
import pytest

from optimise_ref import (
    DEFAULT_GENES,
    Evaluation,
    Gene,
    apply_genome,
    evaluate,
    evolve,
    read_genome,
)

SPEC = os.path.join(os.path.dirname(__file__), "..", "spec", "regen.json")


@pytest.fixture(scope="module")
def spec():
    with open(SPEC, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# the genome maps onto the spec
# --------------------------------------------------------------------------

def test_gene_encoding_round_trips():
    g = Gene("chamber.contraction_ratio", 2.0, 6.0)
    for value in (2.0, 3.5, 6.0):
        assert g.decode(g.encode(value)) == pytest.approx(value, rel=1e-12)


def test_gene_decoding_stays_in_range():
    g = Gene("operation.chamber_pressure_bar", 10.0, 60.0)
    assert g.decode(-5.0) == 10.0
    assert g.decode(5.0) == 60.0


def test_the_genome_is_written_into_the_spec(spec):
    unit = np.full(len(DEFAULT_GENES), 0.25)
    out = apply_genome(spec, DEFAULT_GENES, unit)
    for g, u in zip(DEFAULT_GENES, unit):
        section, key = g.path.split(".")
        assert out[section][key] == pytest.approx(g.decode(u))


def test_applying_a_genome_does_not_touch_the_original(spec):
    before = copy.deepcopy(spec)
    apply_genome(spec, DEFAULT_GENES, np.full(len(DEFAULT_GENES), 0.9))
    assert spec == before


def test_reading_a_spec_recovers_its_own_genome(spec):
    unit = read_genome(spec, DEFAULT_GENES)
    rebuilt = apply_genome(spec, DEFAULT_GENES, unit)
    for g in DEFAULT_GENES:
        section, key = g.path.split(".")
        assert rebuilt[section][key] == pytest.approx(spec[section][key], rel=1e-9)


# --------------------------------------------------------------------------
# constraint ranking is Deb's, not a penalty
# --------------------------------------------------------------------------

def _ev(objective, violation):
    return Evaluation(unit=np.zeros(3), objective=objective, violation=violation)


def test_feasible_always_beats_infeasible():
    """
    Even a terrible feasible design beats a wonderful infeasible one. That is
    the whole point: there is no honest exchange rate between a melted wall and
    a few seconds of impulse.
    """
    poor_but_real = _ev(objective=1.0, violation=0.0)
    superb_but_melted = _ev(objective=1e6, violation=0.01)
    assert poor_but_real.beats(superb_but_melted)
    assert not superb_but_melted.beats(poor_but_real)


def test_among_infeasible_the_smaller_violation_wins():
    assert _ev(0.0, 1.0).beats(_ev(1e6, 2.0))


def test_among_feasible_the_better_objective_wins():
    assert _ev(2.0, 0.0).beats(_ev(1.0, 0.0))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def test_the_shipped_spec_is_feasible(spec):
    """The design in the repository has to satisfy its own constraints."""
    ev = evaluate(spec, DEFAULT_GENES, read_genome(spec, DEFAULT_GENES))
    assert ev.feasible, ev.reasons
    assert ev.objective > 200.0


def test_a_hole_in_the_space_is_a_violation_not_a_crash(spec):
    """
    Parts of this space are not merely bad, they are undefined -- a wall thicker
    than the centrebody it is hollowing, for instance. The search has to walk
    away from those rather than stop at them, so a raised exception has to come
    back as a large violation.

    Shrink the engine until the wall cannot fit inside it. The default gene box
    deliberately cannot reach this, which is the box doing its job; the handler
    still has to work when a caller widens it.
    """
    broken = copy.deepcopy(spec)
    broken["nozzle"]["exit_radius_mm"] = 6.0
    broken["geometry"]["wall_thickness_mm"] = 5.0
    ev = evaluate(broken, DEFAULT_GENES, read_genome(broken, DEFAULT_GENES))
    assert not ev.feasible
    assert ev.reasons
    assert ev.violation > 1.0


def test_the_gene_box_contains_the_shipped_design(spec):
    """
    A box that clips its own seed makes the optimiser report an optimum it never
    searched from, and quietly compares against a design nobody specified.
    """
    for g in DEFAULT_GENES:
        section, key = g.path.split(".")
        value = spec[section][key]
        assert g.lo <= value <= g.hi, f"{g.path}={value} outside [{g.lo}, {g.hi}]"


def test_a_thrust_floor_is_enforced(spec):
    unit = read_genome(spec, DEFAULT_GENES)
    loose = evaluate(spec, DEFAULT_GENES, unit, thrust_floor_n=0.0)
    tight = evaluate(spec, DEFAULT_GENES, unit,
                     thrust_floor_n=loose.metrics["thrust_sl"] * 3.0)
    assert loose.feasible
    assert not tight.feasible
    assert any("floor" in r for r in tight.reasons)


def test_metrics_are_self_consistent(spec):
    ev = evaluate(spec, DEFAULT_GENES, read_genome(spec, DEFAULT_GENES))
    m = ev.metrics
    assert m["thrust_vac"] > m["thrust_sl"]
    assert m["isp_vac"] > m["isp_sl"]
    assert m["thrust_to_mass"] == pytest.approx(m["thrust_sl"] / m["mass_kg"], rel=1e-9)


# --------------------------------------------------------------------------
# the strategy itself
# --------------------------------------------------------------------------

def test_the_search_is_deterministic(spec):
    """Same seed, same answer. Without this none of the rest is testable."""
    a = evolve(spec, mu=3, lam=4, generations=2, seed=11, workers=1, verbose=False)
    b = evolve(spec, mu=3, lam=4, generations=2, seed=11, workers=1, verbose=False)
    assert np.allclose(a.best.unit, b.best.unit)
    assert a.best.objective == pytest.approx(b.best.objective)


def test_a_different_seed_explores_differently(spec):
    a = evolve(spec, mu=3, lam=4, generations=2, seed=1, workers=1, verbose=False)
    b = evolve(spec, mu=3, lam=4, generations=2, seed=2, workers=1, verbose=False)
    assert not np.allclose(a.best.unit, b.best.unit)


def test_elitism_never_loses_ground(spec):
    """
    (mu + lambda) keeps parents, so the best can never get worse from one
    generation to the next. If it can, the survival step is wrong.
    """
    res = evolve(spec, mu=4, lam=6, generations=4, seed=5, workers=1, verbose=False)
    for earlier, later in zip(res.history, res.history[1:]):
        assert later.beats(earlier) or not earlier.beats(later)


def test_the_search_starts_from_the_given_spec(spec):
    """
    Seeding from the incoming design means the optimiser can only improve on it,
    and gives it a feasible foothold to walk from.
    """
    res = evolve(spec, mu=2, lam=2, generations=1, seed=3, workers=1,
                 verbose=False, seed_from_spec=True)
    assert res.best.feasible


def test_the_result_rebuilds_a_usable_spec(spec):
    res = evolve(spec, mu=2, lam=2, generations=1, seed=4, workers=1, verbose=False)
    out = res.best_spec(spec)
    from engine_design import design_engine
    d = design_engine(out)
    assert d.chamber.thrust_sea_level > 0.0


# --------------------------------------------------------------------------
# is the answer the physics, or the box?
# --------------------------------------------------------------------------

def test_bounds_report_finds_a_pinned_variable():
    """
    The check that matters most. A search ending hard against a limit has found
    the edge of someone's box, not an optimum, and the number will keep moving
    if the box is widened.
    """
    from optimise_ref import bounds_report
    unit = np.full(len(DEFAULT_GENES), 0.5)
    unit[0] = 1.0
    unit[3] = 0.0
    found = {b.gene: b.at for b in bounds_report(unit, DEFAULT_GENES)}
    assert found[DEFAULT_GENES[0].path] == "upper"
    assert found[DEFAULT_GENES[3].path] == "lower"


def test_a_design_in_the_interior_reports_nothing():
    from optimise_ref import bounds_report
    assert bounds_report(np.full(len(DEFAULT_GENES), 0.5), DEFAULT_GENES) == []


# --------------------------------------------------------------------------
# multi-objective
# --------------------------------------------------------------------------

def _m(**kw):
    ev = Evaluation(unit=np.zeros(3), objective=0.0, violation=0.0)
    ev.metrics = kw
    return ev


def test_feasibility_outranks_dominance():
    """
    A feasible design dominates an infeasible one whatever the objectives say.
    Pareto sorting without that produces a front full of melted engines.
    """
    from optimise_ref import dominates
    good = _m(isp_sl=200.0, mass_kg=20.0)
    melted = _m(isp_sl=400.0, mass_kg=1.0)
    melted.violation = 5.0
    assert dominates(good, melted, ["isp_sl", "mass_kg"])
    assert not dominates(melted, good, ["isp_sl", "mass_kg"])


def test_dominance_needs_no_worse_everywhere():
    from optimise_ref import dominates
    a = _m(isp_sl=300.0, mass_kg=10.0)
    b = _m(isp_sl=280.0, mass_kg=12.0)
    trade = _m(isp_sl=320.0, mass_kg=14.0)
    assert dominates(a, b, ["isp_sl", "mass_kg"])
    assert not dominates(a, trade, ["isp_sl", "mass_kg"])
    assert not dominates(trade, a, ["isp_sl", "mass_kg"])


def test_a_front_contains_no_dominated_member():
    from optimise_ref import dominates, nondominated_fronts
    pop = [_m(isp_sl=300.0, mass_kg=10.0), _m(isp_sl=280.0, mass_kg=8.0),
           _m(isp_sl=320.0, mass_kg=14.0), _m(isp_sl=250.0, mass_kg=20.0)]
    front = nondominated_fronts(pop, ["isp_sl", "mass_kg"])[0]
    for p in front:
        assert not any(dominates(q, p, ["isp_sl", "mass_kg"]) for q in pop if q is not p)


def test_every_member_lands_in_exactly_one_front():
    from optimise_ref import nondominated_fronts
    pop = [_m(isp_sl=float(i), mass_kg=float(20 - i)) for i in range(6)]
    pop += [_m(isp_sl=float(i), mass_kg=float(30 - i)) for i in range(6)]
    fronts = nondominated_fronts(pop, ["isp_sl", "mass_kg"])
    assert sum(len(f) for f in fronts) == len(pop)
    seen = [id(p) for f in fronts for p in f]
    assert len(set(seen)) == len(pop)


def test_crowding_keeps_the_extremes():
    """
    The ends of a front are what define its range; dropping them collapses the
    trade-off towards whichever corner was reached first.
    """
    from optimise_ref import crowding_distance
    front = [_m(isp_sl=200.0, mass_kg=5.0), _m(isp_sl=250.0, mass_kg=10.0),
             _m(isp_sl=300.0, mass_kg=20.0)]
    dist = crowding_distance(front, ["isp_sl", "mass_kg"])
    assert dist[id(front[0])] == math.inf
    assert dist[id(front[2])] == math.inf
    assert dist[id(front[1])] < math.inf


def test_an_unknown_objective_is_refused(spec):
    from optimise_ref import evolve_pareto
    with pytest.raises(ValueError, match="unknown objective"):
        evolve_pareto(spec, objectives=("cost_per_kilo",), generations=1,
                      mu=2, lam=2, workers=1, verbose=False)


def test_the_pareto_search_returns_a_real_front(spec):
    from optimise_ref import dominates, evolve_pareto
    front = evolve_pareto(spec, objectives=("isp_sl", "mass_kg"), mu=4, lam=4,
                          generations=2, seed=3, workers=1, verbose=False)
    assert front
    for p in front:
        assert not any(dominates(q, p, ["isp_sl", "mass_kg"]) for q in front if q is not p)
