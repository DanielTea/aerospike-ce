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
