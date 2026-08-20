"""
The verification pipeline. One entry point, five stages, nothing optional.

    python pipeline.py                                  # every stage, screening depth
    python pipeline.py --depth print                    # at the voxel the part ships at
    python pipeline.py --stage physics,geometry         # a subset, in order
    python pipeline.py --file ../docs/print/engine.3mf  # re-derive the last stage from disk
    python pipeline.py --json ../out/pipeline.json      # machine-readable, for CI
    python pipeline.py --list                           # what each stage checks, and why

Why a pipeline and not a test run
---------------------------------
`pytest` proves the maths. It does not prove the part. The defects that have
actually shipped from this repo were all downstream of a green suite: a cooling
channel closed by decimation, a 27 cm3 ring of copper attached to nothing, 6334
triangles with no area in a published file. Each was invisible to the stage
above it and obvious to the stage below, and there was no stage below.

So the gates are ordered the way the failure propagates, and each stage is only
meaningful if the one before it passed:

    physics       the numbers. A wrong throat area is a wrong engine, and no
                  amount of mesh checking will say so.
    geometry      the shapes those numbers imply, and the seam the C# reads. A
                  channel that leaves its wall is a geometry failure the
                  physics cannot see and the mesh reports as a strange genus.
    printability  whether a machine can build those shapes at all. A support
                  inside a sealed cavity stays there for ever, and no amount of
                  mesh checking has an opinion about that either.
    watertight    the solid those shapes close into. Boundary edges, pinches,
                  loose fragments, cavities with no way out.
    slicing       what a slicer needs and a topologist does not: an area on
                  every triangle, one triangle per place, a consistent winding,
                  a declared unit. This stage is the one that was missing.

Depth
-----
`screen` meshes at a voxel chosen to run in minutes; it proves closure,
orientation and connectivity, and it is what belongs in CI. `print` meshes at
the voxel the narrowest feature demands and takes about an hour; it is the gate
the release asset has to pass. Both apply the same checks -- except the ones
that are only true once the features are resolved, which are marked and skipped
with a reason rather than quietly asserted at a resolution that cannot support
them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from build_plan import build_plan, narrowest_feature
from engine_design import design_engine
from mesh_export import (manifold_report, mesh_area, mesh_volume, read_3mf,
                         slicing_report)
from mesh_solid import build_mesh_streaming
from printability_ref import choose_build_direction
from print_ready import features_for, surfaces

HERE = os.path.dirname(os.path.abspath(__file__))

# Every test module in the tree belongs to exactly one stage. The pipeline
# refuses to run if that stops being true, because a test nobody runs is a
# test nobody has, and the way a suite rots is one unclaimed file at a time.
TEST_MODULES = {
    "physics": [
        "test_contour.py",          # the Angelino construction and the throat
        "test_combustion.py",       # chamber conditions and delivered performance
        "test_cooling.py",          # Bartz, Dittus-Boelter, fins, film, the search
        "test_injector.py",         # element sizing and momentum balance
        "test_engine_systems.py",   # structure, transient, stability, plug flow
        "test_optimise.py",         # the search over the spec, seeded
    ],
    "geometry": [
        "test_engine.py",           # the assembly the contour implies
        "test_shell.py",            # ribs, legs, the outer shell
        "test_manifolds.py",        # plenums, collectors, the head joint
        "test_interfaces.py",       # ports, bosses, what connects where
        "test_feed_paths.py",       # every circuit reaching what feeds it
        "test_cooled_geometry.py",  # the features on the part, and the lattice
        "test_build_plan.py",       # the seam: shapes only, no physics
    ],
    "printability": [
        "test_printability.py",     # overhangs, bridges, drainage, build direction
    ],
    "watertight": [
        "test_mesh.py",             # closure, topology, volume, the slicing gate
    ],
    "slicing": [
        "test_print_ready.py",      # cavities, fragments, the decimation ladder
    ],
}

STAGES = tuple(TEST_MODULES)

# Meshing every part at the print voxel takes about an hour. The screening
# voxel is coarse enough to run in a few minutes and fine enough to close.
DEPTHS = {"screen": 0.6, "print": None}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    def line(self) -> str:
        mark = "skip" if self.skipped else ("ok" if self.ok else "FAIL")
        return f"    {mark:>4s}  {self.name:34s} {self.detail}"


@dataclass
class Stage:
    name: str
    checks: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def add(self, name, ok, detail="", skipped=False):
        self.checks.append(Check(name, bool(ok), detail, skipped))
        return self.checks[-1]


# --------------------------------------------------------------------------
# the suite, grouped by stage
# --------------------------------------------------------------------------

def _unclaimed_modules() -> list[str]:
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(HERE, "test_*.py"))}
    claimed = {m for mods in TEST_MODULES.values() for m in mods}
    return sorted((on_disk - claimed) | (claimed - on_disk))


def run_tests(stage: Stage, modules: list[str]) -> None:
    """
    Run this stage's share of the suite in-process.

    In-process rather than by subprocess so a collection error surfaces as a
    stage failure with its traceback, instead of an exit code with nothing
    attached to it.
    """
    import pytest

    paths = [os.path.join(HERE, m) for m in modules]
    missing = [m for m, p in zip(modules, paths) if not os.path.exists(p)]
    if missing:
        stage.add("suite", False, f"missing: {', '.join(missing)}")
        return

    class Tally:
        def __init__(self):
            self.passed = self.failed = self.skipped = 0
            self.failures: list[str] = []

        def pytest_runtest_logreport(self, report):
            # One tally per test: the call phase for tests that ran, and the
            # setup phase for the ones that never got that far. A collection
            # error or a broken fixture reports only at setup, and counting
            # just the call phase is how a stage comes back green having run
            # nothing at all.
            if report.when == "call":
                pass
            elif report.when == "setup" and (report.failed or report.skipped):
                pass
            else:
                return
            if report.failed:
                self.failed += 1
                if len(self.failures) < 8:
                    self.failures.append(report.nodeid)
            elif report.skipped:
                self.skipped += 1
            else:
                self.passed += 1

    tally = Tally()
    code = pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", *paths],
                       plugins=[tally])
    detail = (f"{len(modules)} modules, {tally.passed} passed"
              + (f", {tally.skipped} skipped" if tally.skipped else "")
              + (f", {tally.failed} FAILED: {', '.join(tally.failures)}"
                 if tally.failed else ""))
    stage.add("suite", code == 0 and tally.failed == 0, detail)


# --------------------------------------------------------------------------
# stage 1: physics
# --------------------------------------------------------------------------

def stage_physics(design, spec) -> Stage:
    """
    The numbers close, and the model says so itself rather than being told.

    `design_engine` returns no cooling circuit at all when it cannot make one
    work, which is the honest failure and the one worth gating on: a quietly
    returned melted engine passes every mesh check ever written.
    """
    st = Stage("physics")
    run_tests(st, TEST_MODULES["physics"])

    ch = design.chamber
    st.add("chamber conditions",
           ch.chamber_pressure > 0.0 and ch.t_chamber > 0.0 and ch.mass_flow > 0.0,
           f"{ch.chamber_pressure / 1e5:.1f} bar, {ch.t_chamber:.0f} K, "
           f"mdot {ch.mass_flow:.3f} kg/s, MR {ch.mixture_ratio:.2f}")
    st.add("delivered performance",
           ch.thrust_sea_level > 0.0 and ch.isp_vacuum > ch.isp_sea_level,
           f"{ch.thrust_sea_level / 1e3:.1f} kN sea level, "
           f"Isp {ch.isp_sea_level:.0f} s / {ch.isp_vacuum:.0f} s vacuum, "
           f"eps {ch.expansion_ratio:.1f}")

    st.add("injector sized", design.injector is not None,
           design.injector_note or "no injector returned")

    st.add("every wall has a circuit", design.cooled,
           ", ".join(f"{k}: {'closed' if v else 'NO CIRCUIT'}"
                     for k, v in design.circuits.items()))

    for name, cand in design.circuits.items():
        if cand is None:
            st.add(f"{name} wall margin", False, "no circuit was found")
            continue
        sol = cand.solution
        st.add(f"{name} wall margin", sol.survives,
               f"peak {sol.peak_wall_temperature:.0f} K, "
               f"{sol.wall_temperature_margin:+.0f} K to the limit, "
               f"coolant {sol.coolant_temperature_margin:+.0f} K")

    st.add("coolant can carry the heat",
           design.total_heat < design.coolant_capacity,
           f"{design.total_heat / 1e3:.1f} kW into a "
           f"{design.coolant_capacity / 1e3:.1f} kW capacity")

    for name, res in design.structure.items():
        st.add(f"{name} cycle life", res.cycles_to_failure >= res.required_cycles,
               f"{res.cycles_to_failure:.0f} cycles against "
               f"{res.required_cycles:.0f} required, worst stress margin "
               f"{res.worst_margin:+.2f}")

    for name, res in design.startup.items():
        st.add(f"{name} startup", res.survives,
               f"peak {res.peak_wall_temp:.0f} K at {res.peak_time:.2f} s, "
               f"{res.overshoot:+.0f} K over steady")

    return st


# --------------------------------------------------------------------------
# stage 2: geometry
# --------------------------------------------------------------------------

def stage_geometry(design, spec) -> Stage:
    """
    The shapes, and the plan the C# reads.

    The plan is the seam. It carries profiles as point lists and features as
    their parameters, in millimetres, and nothing that would let the reader
    derive a number of its own -- which is the property `test_build_plan`
    exists to hold and this stage re-checks on the plan actually written.
    """
    st = Stage("geometry")
    run_tests(st, TEST_MODULES["geometry"])

    a = design.assembly
    st.add("three parts", sorted(a.profiles) == ["centrebody", "cowl", "head"],
           ", ".join(sorted(a.profiles)))

    for name, p in a.profiles.items():
        x, r = np.asarray(p.x), np.asarray(p.r)
        closed = len(x) >= 3 and np.isfinite(x).all() and np.isfinite(r).all()
        st.add(f"{name} profile", closed and r.min() >= 0.0,
               f"{len(x)} points, x {x.min():.1f}..{x.max():.1f} mm, "
               f"r {r.min():.1f}..{r.max():.1f} mm, "
               f"{p.revolved_volume / 1000:.1f} cm3 revolved")

    plan = build_plan(spec, design)
    st.add("plan is versioned", plan.get("plan_version") is not None,
           f"plan_version {plan.get('plan_version')}, units {plan.get('units')}, "
           f"build direction {plan.get('build_direction')}")
    st.add("plan is in millimetres", plan.get("units") == "mm", str(plan.get("units")))
    st.add("plan covers every part",
           sorted(plan["parts"]) == sorted(a.profiles),
           ", ".join(sorted(plan["parts"])))

    narrow = narrowest_feature(plan)
    st.add("narrowest feature is real", narrow > 0.0, f"{narrow:.3f} mm")

    # the plan must survive a round trip through JSON exactly as the C# reads it
    try:
        again = json.loads(json.dumps(plan))
        st.add("plan round-trips", again == plan, "json in, json out")
    except (TypeError, ValueError) as exc:
        st.add("plan round-trips", False, f"{type(exc).__name__}: {exc}")

    check_the_reader_compiles(st)
    return st


def check_the_reader_compiles(st: Stage) -> None:
    """
    The other side of the seam still builds.

    The C# derives no geometry -- that is the whole arrangement -- but it does
    have to compile against the plan it reads, and a field renamed on this side
    breaks it silently as far as pytest is concerned. Skipped, with the reason
    said out loud, when the SDK or the vendored PicoGK is not on the machine:
    a gate that quietly disappears is worse than one that is not there.
    """
    exe = shutil.which("dotnet")
    proj = os.path.join(HERE, "..", "model", "AerospikeCE.csproj")
    picogk = os.path.join(HERE, "..", "vendor", "PicoGK", "PicoGK.csproj")
    if exe is None:
        st.add("the C# reader compiles", True,
               "no dotnet SDK here; not compiled", skipped=True)
        return
    if not os.path.exists(picogk):
        st.add("the C# reader compiles", True,
               "vendor/PicoGK is not checked out; git submodule update --init "
               "--recursive", skipped=True)
        return
    try:
        r = subprocess.run([exe, "build", proj, "--nologo", "-v", "q"],
                           capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        st.add("the C# reader compiles", False, f"{type(exc).__name__}: {exc}")
        return
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    st.add("the C# reader compiles", r.returncode == 0,
           "dotnet build clean" if r.returncode == 0 else " | ".join(tail))


# --------------------------------------------------------------------------
# stage 3: can a machine build it
# --------------------------------------------------------------------------

def stage_printability(design) -> Stage:
    """
    Buildable, in the orientation the model derives rather than one chosen.

    Supportable is not the same as printable, and the difference is the whole
    stage. An overhang on an outer surface takes a support that gets broken off
    afterwards; an overhang inside a sealed cavity takes one that stays there
    for ever, because nothing can reach in to remove it. `printability_ref`
    casts a ray down the build direction and asks whether a support column
    could actually get to the facet, so the two are counted separately: the
    unsupportable ones are defects and the rest are a finishing operation.
    """
    st = Stage("printability")
    run_tests(st, TEST_MODULES["printability"])

    best, up, down = choose_build_direction(design)
    other = down if best is up else up
    st.add("build direction is derived", best.build_direction in ("+x", "-x"),
           f"{best.build_direction}: {len(best.findings)} findings against "
           f"{len(other.findings)} the other way up")
    st.add("fits the build envelope", best.envelope_ok,
           f"{best.height_mm:.0f} mm tall, {best.diameter_mm:.0f} mm across")
    st.add("nothing unsupportable", not best.unsupportable,
           f"{len(best.unsupportable)} facet(s) a support column cannot reach"
           if best.unsupportable else
           f"{len(best.needs_support)} overhang(s) need removable support, all "
           f"reachable from the plate")
    for kind in ("feature", "drainage", "envelope"):
        found = best.of_kind(kind)
        st.add(f"no {kind} finding", not found,
               "clear" if not found else "; ".join(f.detail for f in found[:3]))
    st.add("printable as it stands", best.printable,
           best.notes[-1] if best.notes else "no notes")
    return st


# --------------------------------------------------------------------------
# stages 4 and 5: the solid, and what a slicer makes of it
# --------------------------------------------------------------------------

def mesh_parts(design, voxel_mm: float, verbose: bool = True):
    """
    Mesh every part, and integrate the field's own volume on the way past.

    The second number is what makes the volume check worth having: it comes
    from the same field by a different route -- occupancy summed cell by cell,
    against the divergence theorem over the triangles -- so agreement means the
    mesher lost nothing, and it costs one sum per plane.
    """
    parts, fields = {}, {}
    for name in design.assembly.profiles:
        t0 = time.time()
        stats: dict = {}
        v, f = build_mesh_streaming(design.assembly.profiles[name],
                                    voxel_mm=voxel_mm, stats=stats,
                                    **features_for(design, name))
        parts[name] = (v, f)
        fields[name] = stats
        if verbose:
            print(f"    meshed {name:11s} {len(f):9d} triangles  "
                  f"{time.time() - t0:6.0f}s", flush=True)
    return parts, fields


def stage_watertight(design, parts, fields, voxel_mm, narrow, resolved: bool,
                     from_file: str | None = None) -> Stage:
    """
    Does it close, and is it one solid with a way out of every cavity.

    `resolved` says whether the voxel is fine enough for the features to exist
    at all. Below that, closure and connectivity still mean something and the
    genus does not: at 0.6 mm the cowl reports 709 handles and at 0.233 mm it
    reports 392, because at 0.6 mm most of what it is counting is aliasing.

    Given a written file, the same questions are asked of what is on disk. The
    resolution gate goes with the mesher and is skipped: a file does not carry
    the voxel it was built at, and asserting one from the spec would be
    checking the spec against itself.
    """
    st = Stage("watertight")
    run_tests(st, TEST_MODULES["watertight"])
    if from_file:
        st.add("resolution", True,
               f"not asked of a written file; {os.path.basename(from_file)} "
               f"carries no voxel size", skipped=True)
    else:
        st.add("resolution", voxel_mm <= narrow / 3.0 + 1e-9,
               f"voxel {voxel_mm:.3f} mm against {narrow:.2f} mm narrowest "
               f"({narrow / voxel_mm:.1f} samples across)"
               + ("" if resolved else "  -- screening only, genus not asserted"),
               skipped=not resolved)

    for name, (v, f) in parts.items():
        rep = manifold_report(v, f)
        surf = surfaces(v, f)
        sealed = [vol for _, vol in surf if vol < 0.0]
        solids = sorted((vol for _, vol in surf if vol > 0.0), reverse=True)
        loose = solids[1:]
        genus = (2 * len(surf) - rep["euler"]) // 2

        st.add(f"{name} closes", rep["watertight"],
               f"{len(f)} triangles, {rep['boundary_edges']} boundary, "
               f"{rep['nonmanifold_edges']} non-manifold edges")
        # Free, exact, and the half of the pinch test that costs nothing: two
        # closed orientable surfaces sum to an even Euler characteristic, so an
        # odd one proves the surface meets itself somewhere.
        st.add(f"{name} euler is even", rep["euler"] % 2 == 0,
               f"euler {rep['euler']}, genus {genus}, {len(surf)} surfaces")
        st.add(f"{name} is one solid", not loose,
               "one piece" if not loose else
               f"{len(loose) + 1} pieces, {sum(loose) / 1000:.1f} cm3 adrift")
        st.add(f"{name} cavities drain", not sealed,
               "no sealed void" if not sealed else
               f"{len(sealed)} sealed, {sum(abs(s) for s in sealed) / 1000:.1f} cm3 "
               f"of powder with no way out")

        got = mesh_volume(v, f)
        want = fields.get(name, {}).get("field_volume_mm3")
        if want:
            # Same field, two different routes: occupancy summed cell by cell
            # against the divergence theorem over the triangles. Read the
            # difference as a length rather than a percentage -- it is the
            # uniform offset that would account for it, spread over the whole
            # surface -- because that is the quantity that has to be small
            # against the voxel, and a percentage is not comparable between a
            # solid disc and a part with 392 channels in it. A genuine loss --
            # a dropped slab, a region meshed inside out -- is a whole feature,
            # orders above this.
            area = mesh_area(v, f)
            offset = (got - want) / area
            st.add(f"{name} volume", abs(offset) < 0.15 * voxel_mm,
                   f"{got / 1000:.1f} cm3 meshed against {want / 1000:.1f} cm3 "
                   f"from the field: {offset:+.4f} mm of surface offset, "
                   f"{abs(offset) / voxel_mm:.3f} of a voxel")
        else:
            st.add(f"{name} volume", got > 0.0, f"{got / 1000:.1f} cm3", skipped=True)
    return st


def stage_slicing(parts, path: str | None = None, unit: str | None = None) -> Stage:
    """
    Everything a slicer reads that the edge arithmetic never looks at.

    When a written file is given, the checks are re-derived from what is on
    disk rather than from what was in memory, because the writer is the part a
    slicer actually sees.
    """
    st = Stage("slicing")
    run_tests(st, TEST_MODULES["slicing"])

    if path:
        st.add("file states its unit", unit == "millimeter",
               f"{unit}, {os.path.getsize(path) / 1e6:.1f} MB")
        st.add("file carries every solid", bool(parts),
               f"{len(parts)} objects: {', '.join(parts)}")
        where = os.path.basename(path)
    else:
        where = "in memory"

    for name, (v, f) in parts.items():
        rep = slicing_report(v, f)
        st.add(f"{name} triangles have area", rep["degenerate_faces"] == 0,
               f"{rep['degenerate_faces']} zero-area, smallest "
               f"{rep['min_face_area_mm2']:.2e} mm2 ({where})")
        st.add(f"{name} one triangle per place", rep["duplicate_faces"] == 0,
               f"{rep['duplicate_faces']} duplicated")
        st.add(f"{name} winding is consistent", rep["inconsistent_edges"] == 0,
               f"{rep['inconsistent_edges']} edges wound the same way twice")
        st.add(f"{name} surface never pinches", rep["nonmanifold_vertices"] == 0,
               f"{rep['nonmanifold_vertices']} vertices where the surface "
               f"meets itself")
        st.add(f"{name} solid is outward", rep["outward"] and rep["finite"]
               and rep["indices_in_range"],
               "normals out, coordinates finite, indices in range")

    if parts:
        lo = np.min([v.min(axis=0) for v, _ in parts.values()], axis=0)
        hi = np.max([v.max(axis=0) for v, _ in parts.values()], axis=0)
        st.add("assembled envelope", True,
               f"{hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} x {hi[2] - lo[2]:.1f} mm")
    else:
        st.add("assembled envelope", False,
               "nothing to measure: no meshes and no file")
    return st


# --------------------------------------------------------------------------

def describe() -> str:
    out = [__doc__.strip(), "", "Stages and the suite each one owns:", ""]
    for name, mods in TEST_MODULES.items():
        out.append(f"  {name}")
        for m in mods:
            out.append(f"      {m}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="the verification pipeline")
    ap.add_argument("--spec", default=os.path.join(HERE, "..", "spec", "regen.json"))
    ap.add_argument("--stage", default=",".join(STAGES),
                    help=f"comma-separated subset of {', '.join(STAGES)}")
    ap.add_argument("--depth", choices=sorted(DEPTHS), default="screen")
    ap.add_argument("--voxel", type=float, default=0.0, help="override the depth")
    ap.add_argument("--file", default=None,
                    help="a written 3MF; the mesh stages read it instead of meshing")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    args = ap.parse_args()

    if args.list:
        print(describe())
        return 0

    unclaimed = _unclaimed_modules()
    if unclaimed:
        print("refusing to run: these test modules belong to no stage:")
        for m in unclaimed:
            print(f"  {m}")
        print("add each to TEST_MODULES in pipeline.py, next to the stage it checks")
        return 2

    wanted = [s.strip() for s in args.stage.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}")
        return 2

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    design = design_engine(spec)
    plan = build_plan(spec, design)
    narrow = narrowest_feature(plan)
    print_voxel = round(narrow / 3.0, 3)
    voxel = args.voxel or DEPTHS[args.depth] or print_voxel
    resolved = voxel <= print_voxel + 1e-9

    print(f"{spec.get('name', 'engine')}  spec {os.path.relpath(args.spec, HERE)}  "
          f"depth {args.depth} (voxel {voxel:.3f} mm, print voxel {print_voxel:.3f} mm)")
    print()

    # A written file replaces the mesher for both mesh stages: the gates are
    # then asked of exactly what a slicer would open, which is the point of
    # having a file to check at all.
    parts, fields, unit = None, {}, None
    if args.file:
        doc = read_3mf(args.file)
        parts, unit = doc["objects"], doc["unit"]
        shown = os.path.relpath(args.file, HERE)
        if len(shown) > len(args.file):
            shown = args.file
        print(f"  reading {shown}: {len(parts)} objects, unit {unit}\n")

    stages: list[Stage] = []
    for name in STAGES:
        if name not in wanted:
            continue
        t0 = time.time()
        print(f"  {name}", flush=True)
        if name == "physics":
            st = stage_physics(design, spec)
        elif name == "geometry":
            st = stage_geometry(design, spec)
        elif name == "printability":
            st = stage_printability(design)
        elif name == "watertight":
            if parts is None:
                parts, fields = mesh_parts(design, voxel)
            st = stage_watertight(design, parts, fields, voxel, narrow, resolved,
                                  from_file=args.file)
        else:
            if parts is None:
                parts, fields = mesh_parts(design, voxel)
            st = stage_slicing(parts, args.file, unit)
        st.seconds = time.time() - t0
        stages.append(st)
        for c in st.checks:
            print(c.line())
        print(f"    -- {name}: {'PASS' if st.ok else 'FAIL'} "
              f"({len(st.checks)} checks, {st.seconds:.0f}s)\n", flush=True)

    ok = all(s.ok for s in stages)
    print("PASS" if ok else "FAIL", "  ".join(
        f"{s.name}:{'ok' if s.ok else 'FAIL'}" for s in stages))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "spec": os.path.abspath(args.spec),
                "name": spec.get("name"),
                "depth": args.depth,
                "voxel_mm": voxel,
                "print_voxel_mm": print_voxel,
                "file": args.file,
                "ok": ok,
                "stages": [{
                    "name": s.name, "ok": s.ok, "seconds": round(s.seconds, 2),
                    "checks": [{"name": c.name, "ok": c.ok, "skipped": c.skipped,
                                "detail": c.detail} for c in s.checks],
                } for s in stages],
            }, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
