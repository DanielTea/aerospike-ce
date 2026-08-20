"""
Evolutionary optimisation of the whole engine.

    python optimise_ref.py --spec ../spec/regen.json --generations 30

The design space here is not smooth, not convex, and not everywhere defined. Ask
for a slightly higher chamber pressure and the cooling search may find nothing
at all; ask for a slightly thinner wall and the printability check may reject
the cavity. Gradients do not exist across those cliffs, and a great deal of the
space is simply infeasible. That is what evolutionary search is for: it needs
only a ranking, and it is happy to walk along the edge of a feasible region it
cannot differentiate.

Algorithm
---------
A (mu + lambda) evolution strategy with self-adaptive step sizes. Each
individual carries its own mutation width per gene; those widths mutate
log-normally alongside the genes, so the search tightens as it converges without
being told to. Parents survive into the next generation, so a good design is
never lost to an unlucky mutation.

Constraints are handled by Deb's rules rather than by penalty weights:

    a feasible design always beats an infeasible one
    between two infeasible designs, the smaller total violation wins
    between two feasible designs, the better objective wins

That needs no weighting between "melted by 40 K" and "three seconds of impulse",
which is a comparison nobody can make honestly anyway, and it means the search
can start from an infeasible seed and walk itself into the feasible region.

What is optimised
-----------------
Nine variables spanning the nozzle, the chamber, the operating point and the
structure. The objective defaults to sea-level specific impulse subject to a
thrust floor, because that is the trade an upper stage actually cares about;
thrust-to-mass is available for a booster. Everything else -- cooling closing,
the wall surviving, the part being printable, the joint bolting up -- enters as
a constraint, not as a term to be traded away.

Determinism
-----------
Seeded throughout. The same seed and the same spec give the same answer, which
is what makes an optimiser testable at all.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from engine_design import design_engine
from interfaces_ref import build_schedule, port_clashes
from printability_ref import ProcessSpec, analyse as printability


# --------------------------------------------------------------------------
# design variables
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Gene:
    """One design variable, as a dotted path into the spec and a range."""
    path: str
    lo: float
    hi: float

    def decode(self, unit: float) -> float:
        return self.lo + float(np.clip(unit, 0.0, 1.0)) * (self.hi - self.lo)

    def encode(self, value: float) -> float:
        return (value - self.lo) / (self.hi - self.lo)


# The box has to contain the design it is asked to improve on, or the search
# starts by clipping its own seed and reports an optimum it never explored from.
# Ranges are set wide enough to hold spec/regen.json with room either side.
DEFAULT_GENES: tuple = (
    Gene("nozzle.expansion_ratio", 4.0, 22.0),
    Gene("nozzle.truncate_fraction", 0.15, 0.60),
    Gene("chamber.contraction_ratio", 2.0, 9.0),
    Gene("chamber.chamber_length_mm", 40.0, 220.0),
    Gene("chamber.converging_length_mm", 15.0, 60.0),
    Gene("operation.chamber_pressure_bar", 10.0, 60.0),
    Gene("operation.mixture_ratio", 2.0, 3.6),
    Gene("geometry.wall_thickness_mm", 1.5, 5.0),
    Gene("geometry.flange_width_mm", 10.0, 34.0),
)


def apply_genome(spec: dict, genes, unit: np.ndarray) -> dict:
    """A copy of the spec with the genome written into it."""
    out = copy.deepcopy(spec)
    for gene, u in zip(genes, unit):
        section, key = gene.path.split(".")
        out.setdefault(section, {})[key] = gene.decode(u)
    return out


def read_genome(spec: dict, genes) -> np.ndarray:
    """Where an existing spec sits in the search space, clipped to the box."""
    return np.array([
        np.clip(g.encode(spec.get(g.path.split(".")[0], {}).get(
            g.path.split(".")[1], 0.5 * (g.lo + g.hi))), 0.0, 1.0)
        for g in genes])


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@dataclass
class Evaluation:
    unit: np.ndarray
    objective: float = -math.inf     # maximised
    violation: float = math.inf      # zero when feasible
    metrics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.violation <= 0.0

    def beats(self, other: "Evaluation") -> bool:
        """
        Deb's feasibility rules. No weighting between a melted wall and a few
        seconds of impulse, because there is no honest exchange rate.
        """
        if self.feasible != other.feasible:
            return self.feasible
        if not self.feasible:
            return self.violation < other.violation
        return self.objective > other.objective


def _mass_kg(design) -> float:
    volume_mm3 = sum(p.revolved_volume for p in design.assembly.profiles.values())
    rho = 8190.0
    for c in design.circuits.values():
        if c is not None:
            rho = c.material.density
            break
    return volume_mm3 * 1e-9 * rho


def evaluate(spec: dict, genes=DEFAULT_GENES, unit=None,
             objective: str = "isp_sl", thrust_floor_n: float = 0.0,
             process: ProcessSpec | None = None) -> Evaluation:
    """
    Score one genome.

    Anything that throws counts as a large violation rather than an error: the
    space genuinely has holes in it -- a contraction ratio that leaves no room
    for an injector, a wall thicker than the centrebody -- and the search should
    walk away from them rather than stop.
    """
    unit = np.full(len(genes), 0.5) if unit is None else np.asarray(unit, dtype=float)
    ev = Evaluation(unit=unit.copy())
    trial = apply_genome(spec, genes, unit)

    try:
        d = design_engine(trial)
    except Exception as exc:                       # noqa: BLE001 - holes are data
        ev.violation = 50.0
        ev.reasons.append(f"design failed: {type(exc).__name__}: {exc}")
        return ev

    violation = 0.0

    # ---- cooling has to close on both circuits ----
    for name, circuit in d.circuits.items():
        if circuit is None:
            violation += 10.0
            ev.reasons.append(f"{name}: no cooling circuit survives")
            continue
        s = circuit.solution
        if s.wall_temperature_margin < 0.0:
            violation += -s.wall_temperature_margin / 100.0
            ev.reasons.append(f"{name}: wall over by {-s.wall_temperature_margin:.0f} K")
        if s.coolant_temperature_margin < 0.0:
            violation += -s.coolant_temperature_margin / 100.0
            ev.reasons.append(f"{name}: coolant over by {-s.coolant_temperature_margin:.0f} K")

    # ---- the wall has to survive being hot ----
    # Life, not yield. A regeneratively cooled hot wall is expected to go
    # plastic every firing -- the through-wall gradient is large and the jacket
    # restrains it -- so constraining the yield margin to stay positive would
    # reject every chamber ever flown. What matters is cycles and strain range.
    for name, st in (d.structure or {}).items():
        if getattr(st, "survives", True):
            continue
        shortfall = (st.required_cycles - st.cycles_to_failure) / max(st.required_cycles, 1.0)
        violation += max(0.0, shortfall)
        strain = float(np.max(st.strain_range))
        violation += 50.0 * max(0.0, strain - 0.02)
        ev.reasons.append(
            f"{name}: {st.cycles_to_failure:.0f} cycles against "
            f"{st.required_cycles:.0f} required, strain range {strain:.4f}")

    # ---- it has to be printable ----
    # Unsupportable geometry is a defect; a supportable overhang is a finishing
    # operation. Treating them alike either rejects every printable engine or
    # accepts one with supports welded inside a sealed cavity for ever.
    process = process or ProcessSpec()
    pr = printability(d, process, build_plus_x=True)
    if pr.unsupportable:
        violation += 0.1 * len(pr.unsupportable)
        ev.reasons.append(f"{len(pr.unsupportable)} unsupportable overhangs")
    hard = [f for f in pr.findings if f.kind in ("feature", "drainage", "envelope")]
    if hard:
        violation += 0.5 * len(hard)
        ev.reasons.append("; ".join(f.detail for f in hard[:2]))

    # ---- and it has to be connectable ----
    try:
        sched = build_schedule(d)
        clashes = port_clashes(sched, d)
        if clashes:
            violation += 0.2 * len(clashes)
            ev.reasons.append(f"{len(clashes)} port clashes")
        if sched.joint is not None and not sched.joint.fits:
            violation += 1.0
            ev.reasons.append("head joint does not close")
    except Exception as exc:                       # noqa: BLE001
        violation += 5.0
        ev.reasons.append(f"interfaces failed: {exc}")

    thrust = d.chamber.thrust_sea_level
    if thrust_floor_n > 0.0 and thrust < thrust_floor_n:
        violation += (thrust_floor_n - thrust) / thrust_floor_n
        ev.reasons.append(f"thrust {thrust:.0f} N under the {thrust_floor_n:.0f} N floor")

    mass = _mass_kg(d)
    ev.metrics = {
        "isp_sl": d.chamber.isp_sea_level,
        "isp_vac": d.chamber.isp_vacuum,
        "thrust_sl": thrust,
        "thrust_vac": d.chamber.thrust_vacuum,
        "mass_kg": mass,
        "thrust_to_mass": thrust / max(mass, 1e-9),
        "pc_bar": d.chamber.chamber_pressure / 1e5,
        "mixture_ratio": d.chamber.mixture_ratio,
        "expansion_ratio": d.chamber.expansion_ratio,
        "heat_kw": d.total_heat / 1e3,
        "length_mm": d.assembly.overall_length,
        "diameter_mm": 2.0 * d.assembly.overall_radius,
        "printability_findings": len(pr.findings),
        "unsupportable": len(pr.unsupportable),
        "needs_support": len(pr.needs_support),
    }
    ev.objective = ev.metrics.get(objective, -math.inf)
    ev.violation = violation
    return ev


# --------------------------------------------------------------------------
# the evolution strategy
# --------------------------------------------------------------------------

@dataclass
class Result:
    best: Evaluation
    history: list = field(default_factory=list)     # best per generation
    evaluations: int = 0
    seconds: float = 0.0
    genes: tuple = DEFAULT_GENES

    def best_spec(self, spec: dict) -> dict:
        return apply_genome(spec, self.genes, self.best.unit)


def _select(pool: list, mu: int) -> list:
    """
    The best `mu` of a pool, ordered, under Deb's rules.

    Not a plain sort: `beats` is not a total order that Python can use as a key,
    because it compares feasibility, violation and objective in that priority.
    Repeated selection of the best remaining is the honest way to express it.
    """
    chosen: list = []
    remaining = list(pool)
    while remaining and len(chosen) < mu:
        best_i = 0
        for i in range(1, len(remaining)):
            if remaining[i].beats(remaining[best_i]):
                best_i = i
        chosen.append(remaining.pop(best_i))
    return chosen


def _worker(args):
    spec, genes, unit, objective, floor = args
    return evaluate(spec, genes, unit, objective, floor)


def evolve(
    spec: dict,
    genes=DEFAULT_GENES,
    mu: int = 8,
    lam: int = 24,
    generations: int = 30,
    objective: str = "isp_sl",
    thrust_floor_n: float = 0.0,
    seed: int = 0,
    workers: int | None = None,
    seed_from_spec: bool = True,
    sigma0: float = 0.22,
    verbose: bool = True,
) -> Result:
    """
    Run a (mu + lambda) evolution strategy over the design space.

    Parents survive alongside their offspring, so the best design found can
    never be lost to an unlucky mutation. Step sizes are carried per gene and
    mutate log-normally with the genes, which lets the search open up while it
    is far from anything feasible and tighten once it is not.
    """
    rng = np.random.default_rng(seed)
    n = len(genes)
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
    tau0 = 1.0 / math.sqrt(2.0 * n)

    # initial population: the incoming spec plus a spread around it
    pop_units = []
    if seed_from_spec:
        pop_units.append(read_genome(spec, genes))
    while len(pop_units) < mu:
        pop_units.append(rng.random(n))
    sigmas = [np.full(n, sigma0) for _ in pop_units]

    from multiprocessing import Pool
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    n_eval = 0

    def run(units):
        nonlocal n_eval
        n_eval += len(units)
        payload = [(spec, genes, u, objective, thrust_floor_n) for u in units]
        if workers == 1:
            return [_worker(p) for p in payload]
        with Pool(workers) as pool:
            return pool.map(_worker, payload)

    parents = run(pop_units)
    for p, s in zip(parents, sigmas):
        p.sigma = s
    parents = _select(parents, mu)
    history = []

    for gen in range(generations):
        # ---- produce offspring ----
        kids_units, kids_sigma = [], []
        for _ in range(lam):
            i = rng.integers(0, len(parents))
            s = parents[i].sigma * np.exp(tau0 * rng.normal() + tau * rng.normal(size=n))
            s = np.clip(s, 1e-3, 0.5)
            u = np.clip(parents[i].unit + s * rng.normal(size=n), 0.0, 1.0)
            kids_units.append(u)
            kids_sigma.append(s)

        kids = run(kids_units)
        for k, s in zip(kids, kids_sigma):
            k.sigma = s

        # ---- (mu + lambda) survival ----
        parents = _select(parents + kids, mu)

        best = parents[0]
        history.append(best)
        if verbose:
            tag = (f"obj {best.objective:8.3f}" if best.feasible
                   else f"infeasible, violation {best.violation:7.3f}")
            print(f"  gen {gen + 1:3d}/{generations}  {tag}  "
                  f"evals {n_eval:5d}  {time.time() - t0:6.1f}s", flush=True)

    return Result(best=parents[0], history=history, evaluations=n_eval,
                  seconds=time.time() - t0, genes=genes)


def describe(result: Result, spec: dict) -> str:
    b = result.best
    lines = [
        f"{result.evaluations} evaluations in {result.seconds:.0f} s"
        f"   {'FEASIBLE' if b.feasible else 'INFEASIBLE'}",
    ]
    if not b.feasible:
        lines.append(f"  violation {b.violation:.3f}")
        for r in b.reasons:
            lines.append(f"    - {r}")
    lines.append("  design variables:")
    for g, u in zip(result.genes, b.unit):
        lines.append(f"    {g.path:36s} {g.decode(u):10.3f}   "
                     f"[{g.lo:g} .. {g.hi:g}]")
    lines.append("  performance:")
    for k in ("thrust_sl", "thrust_vac", "isp_sl", "isp_vac", "mass_kg",
              "thrust_to_mass", "heat_kw", "length_mm", "diameter_mm"):
        if k in b.metrics:
            lines.append(f"    {k:36s} {b.metrics[k]:10.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--mu", type=int, default=8)
    ap.add_argument("--lam", type=int, default=24)
    ap.add_argument("--objective", default="isp_sl")
    ap.add_argument("--thrust-floor", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pareto", default="",
                    help="comma-separated objectives for a multi-objective run")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)

    if args.pareto:
        names = tuple(n.strip() for n in args.pareto.split(","))
        print(f"optimising {spec.get('name', 'engine')} for {' vs '.join(names)}"
              + (f", thrust floor {args.thrust_floor:.0f} N" if args.thrust_floor else ""))
        front = evolve_pareto(spec, objectives=names, generations=args.generations,
                              mu=max(args.mu, 12), lam=args.lam,
                              thrust_floor_n=args.thrust_floor, seed=args.seed,
                              workers=args.workers or None)
        print()
        print(describe_front(front, names))
        return

    print(f"optimising {spec.get('name', 'engine')} for {args.objective}"
          + (f", thrust floor {args.thrust_floor:.0f} N" if args.thrust_floor else ""))
    res = evolve(spec, generations=args.generations, mu=args.mu, lam=args.lam,
                 objective=args.objective, thrust_floor_n=args.thrust_floor,
                 seed=args.seed, workers=args.workers or None)
    print()
    print(describe(res, spec))

    if args.out:
        best = res.best_spec(spec)
        best["name"] = spec.get("name", "engine") + "-optimised"
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(best, fh, indent=2)
        print(f"\nwrote {args.out}")




# --------------------------------------------------------------------------
# is the answer the physics, or the box?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundFinding:
    gene: str
    value: float
    limit: float
    at: str                     # "lower" or "upper"


def bounds_report(unit: np.ndarray, genes=DEFAULT_GENES,
                  tol: float = 0.01) -> list:
    """
    Which variables finished pinned against their limits.

    This matters more than the objective value. A search that ends hard against
    a bound has not found an optimum, it has found the edge of the box someone
    drew -- the answer is a statement about the bounds, not about the engine,
    and it will keep moving if they are widened. Reporting it is the difference
    between an optimiser and an oracle.
    """
    out: list[BoundFinding] = []
    for gene, u in zip(genes, np.atleast_1d(unit)):
        if u <= tol:
            out.append(BoundFinding(gene.path, gene.decode(u), gene.lo, "lower"))
        elif u >= 1.0 - tol:
            out.append(BoundFinding(gene.path, gene.decode(u), gene.hi, "upper"))
    return out


# --------------------------------------------------------------------------
# more than one objective
# --------------------------------------------------------------------------

OBJECTIVES = {
    "isp_sl": "max", "isp_vac": "max",
    "thrust_sl": "max", "thrust_vac": "max",
    "thrust_to_mass": "max",
    "mass_kg": "min", "heat_kw": "min",
    "length_mm": "min", "diameter_mm": "min",
}


def dominates(a: Evaluation, b: Evaluation, objectives: list) -> bool:
    """
    Pareto dominance, with feasibility taking precedence over everything.

    Deb's rules first -- a feasible design dominates an infeasible one whatever
    its objectives, and between two infeasible ones the smaller violation wins.
    Only among feasible designs does dominance mean what it usually means: no
    worse anywhere, better somewhere.
    """
    if a.feasible != b.feasible:
        return a.feasible
    if not a.feasible:
        return a.violation < b.violation

    better_somewhere = False
    for name in objectives:
        sign = 1.0 if OBJECTIVES[name] == "max" else -1.0
        av, bv = sign * a.metrics.get(name, -math.inf), sign * b.metrics.get(name, -math.inf)
        if av < bv:
            return False
        if av > bv:
            better_somewhere = True
    return better_somewhere


def nondominated_fronts(pop: list, objectives: list) -> list:
    """Sort a population into successive Pareto fronts."""
    # Identity, not equality. Evaluation is a dataclass holding a numpy array,
    # so `p in front` compares arrays elementwise and raises on the ambiguous
    # truth value -- and if the array were scalar it would silently drop distinct
    # individuals that happened to match.
    fronts: list[list] = []
    remaining = list(pop)
    while remaining:
        front = [p for p in remaining
                 if not any(dominates(q, p, objectives) for q in remaining if q is not p)]
        if not front:                       # cycles cannot happen, but do not hang
            front = list(remaining)
        fronts.append(front)
        chosen = {id(p) for p in front}
        remaining = [p for p in remaining if id(p) not in chosen]
    return fronts


def crowding_distance(front: list, objectives: list) -> dict:
    """
    Spacing measure, so a front stays spread instead of clustering.

    Without it a multi-objective search converges onto whichever corner of the
    front it happens to reach first, and reports a "trade-off" containing three
    nearly identical designs.
    """
    dist = {id(p): 0.0 for p in front}
    if len(front) <= 2:
        return {id(p): math.inf for p in front}

    for name in objectives:
        ordered = sorted(front, key=lambda p: p.metrics.get(name, 0.0))
        lo = ordered[0].metrics.get(name, 0.0)
        hi = ordered[-1].metrics.get(name, 0.0)
        span = hi - lo
        dist[id(ordered[0])] = math.inf
        dist[id(ordered[-1])] = math.inf
        if span <= 0.0:
            continue
        for i in range(1, len(ordered) - 1):
            step = (ordered[i + 1].metrics.get(name, 0.0)
                    - ordered[i - 1].metrics.get(name, 0.0))
            dist[id(ordered[i])] += step / span
    return dist


def evolve_pareto(
    spec: dict,
    objectives: tuple = ("isp_sl", "mass_kg"),
    genes=DEFAULT_GENES,
    mu: int = 16,
    lam: int = 32,
    generations: int = 20,
    thrust_floor_n: float = 0.0,
    seed: int = 0,
    workers: int | None = None,
    sigma0: float = 0.22,
    verbose: bool = True,
) -> list:
    """
    Multi-objective search, returning the non-dominated front.

    A single objective forces a trade nobody agreed to. Maximising specific
    impulse alone drove chamber pressure to its ceiling and left mass to fall
    where it may; the honest output is the set of designs where you cannot
    improve one thing without giving up another, and a person picks from it.

    Selection is NSGA-II's: sort into fronts, fill by front, and break the last
    front by crowding distance so the answer stays spread rather than clustering
    in one corner.
    """
    for name in objectives:
        if name not in OBJECTIVES:
            raise ValueError(f"unknown objective '{name}'; know {sorted(OBJECTIVES)}")

    rng = np.random.default_rng(seed)
    n = len(genes)
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
    tau0 = 1.0 / math.sqrt(2.0 * n)
    objectives = list(objectives)

    from multiprocessing import Pool
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))

    def run(units):
        payload = [(spec, genes, u, objectives[0], thrust_floor_n) for u in units]
        if workers == 1:
            return [_worker(p) for p in payload]
        with Pool(workers) as pool:
            return pool.map(_worker, payload)

    units = [read_genome(spec, genes)] + [rng.random(n) for _ in range(mu - 1)]
    parents = run(units)
    for p, u in zip(parents, units):
        p.sigma = np.full(n, sigma0)

    for gen in range(generations):
        kids_units, kids_sigma = [], []
        for _ in range(lam):
            i = rng.integers(0, len(parents))
            s = parents[i].sigma * np.exp(tau0 * rng.normal() + tau * rng.normal(size=n))
            s = np.clip(s, 1e-3, 0.5)
            kids_units.append(np.clip(parents[i].unit + s * rng.normal(size=n), 0.0, 1.0))
            kids_sigma.append(s)
        kids = run(kids_units)
        for k, s in zip(kids, kids_sigma):
            k.sigma = s

        pool_all = parents + kids
        chosen: list = []
        for front in nondominated_fronts(pool_all, objectives):
            if len(chosen) + len(front) <= mu:
                chosen.extend(front)
                continue
            dist = crowding_distance(front, objectives)
            front.sort(key=lambda p: dist[id(p)], reverse=True)
            chosen.extend(front[: mu - len(chosen)])
            break
        parents = chosen

        if verbose:
            feasible = [p for p in parents if p.feasible]
            best = nondominated_fronts(parents, objectives)[0]
            print(f"  gen {gen + 1:3d}/{generations}  front {len(best):3d}  "
                  f"feasible {len(feasible):3d}/{len(parents)}", flush=True)

    return nondominated_fronts(parents, objectives)[0]


def describe_front(front: list, objectives: tuple, genes=DEFAULT_GENES) -> str:
    """The trade-off, plus a note on any variable that finished on its bound."""
    names = list(objectives)
    rows = [f"{len(front)} non-dominated designs",
            "  " + "  ".join(f"{n:>14s}" for n in names)]
    ordered = sorted(front, key=lambda p: p.metrics.get(names[0], 0.0),
                     reverse=OBJECTIVES[names[0]] == "max")
    for p in ordered:
        rows.append("  " + "  ".join(f"{p.metrics.get(n, float('nan')):14.3f}"
                                     for n in names))

    pinned: dict = {}
    for p in front:
        for b in bounds_report(p.unit, genes):
            pinned.setdefault(b.gene, set()).add(b.at)
    if pinned:
        rows.append("")
        rows.append("  variables finishing on their bounds -- the box is answering, "
                    "not the physics:")
        for gene, ends in sorted(pinned.items()):
            rows.append(f"    {gene:36s} at its {', '.join(sorted(ends))} limit")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
