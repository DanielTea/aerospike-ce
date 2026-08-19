"""
Whether the engine can actually be printed, and which way up.

A watertight solid is not a printable one. Powder-bed fusion builds one layer at
a time onto the layer beneath, so any surface that faces downward has to be
shallow enough to hold itself up, every flat ceiling is a bridge that will droop
across its span, and every internal void has to have a way for the powder to get
out. None of that is visible in a mesh check.

Build direction dominates all of it. This engine is a body of revolution, so
there are only two sensible orientations -- head down with the spike pointing up,
or the reverse -- and they are not equivalent. Built head down, the cowl and
centrebody outer surfaces both narrow as they rise, which is a cone point-up and
self-supporting everywhere; the cooling channels run parallel to the build
direction, so they are vertical tunnels with no roof at all. Built the other way
up, every one of those becomes an overhang.

What this checks
----------------
- overhang angle on every downward-facing surface, against the process limit
- flat downward-facing ceilings, which bridge rather than overhang, against the
  span the process can cross unsupported
- powder escape from every internal void
- the narrowest feature against what the process can hold
- whether the part fits the machine

What it does not
----------------
No thermal distortion, no residual stress, no recoater-collision check, no
support generation, and no opinion on scan strategy. Those decide whether a
build succeeds as much as geometry does, and none of them are here.

All lengths mm, all angles degrees at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProcessSpec:
    """
    Capability of the process the part will be built on.

    Defaults are ordinary laser powder-bed fusion in a nickel superalloy. The
    self-supporting angle is the usual 45 degrees from the build plate; the
    bridge span is what a flat ceiling can cross before it needs support under
    it, which is much less than people expect.
    """
    name: str = "LPBF, nickel superalloy"
    min_wall_mm: float = 0.4
    min_feature_mm: float = 0.4
    self_supporting_deg: float = 45.0
    max_bridge_mm: float = 1.0
    layer_mm: float = 0.05
    envelope_mm: tuple = (250.0, 250.0, 325.0)   # width, depth, height


@dataclass(frozen=True)
class Finding:
    part: str
    kind: str               # overhang | bridge | feature | drainage | envelope
    x_mm: float
    r_mm: float
    value: float            # angle in degrees, or span/size in mm
    limit: float
    detail: str
    supportable: bool = True
    """
    Whether a support column could reach this facet from the build plate.

    The distinction decides whether a finding is a nuisance or a defect. An
    overhang on an outer surface, or on the wall of a duct that is open at both
    ends, takes a support that gets broken off afterwards -- ugly, and routine.
    An overhang inside a sealed cavity takes a support that stays there for
    ever, because nothing can reach in to remove it. Treating the two alike
    either rejects every printable engine or accepts an unbuildable one.
    """

    @property
    def severity(self) -> float:
        """How far past the limit, as a fraction. Positive is a violation."""
        if self.kind == "overhang":
            return (self.limit - self.value) / max(self.limit, 1e-9)
        return (self.value - self.limit) / max(self.limit, 1e-9)


@dataclass
class PrintabilityResult:
    build_direction: str            # "+x" is head down, spike up
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    envelope_ok: bool = True
    height_mm: float = 0.0
    diameter_mm: float = 0.0

    @property
    def unsupportable(self) -> list:
        """Findings a support column cannot reach. These are the real defects."""
        return [f for f in self.findings if not f.supportable]

    @property
    def needs_support(self) -> list:
        """Findings that print fine with support that can be removed afterwards."""
        return [f for f in self.findings if f.supportable
                and f.kind in ("overhang", "bridge")]

    @property
    def printable(self) -> bool:
        """
        Buildable as it stands. Supportable overhangs do not disqualify a part;
        they cost support material and a finishing operation.
        """
        return not self.unsupportable and self.envelope_ok and not [
            f for f in self.findings if f.kind in ("feature", "drainage", "envelope")]

    @property
    def worst(self):
        return max(self.findings, key=lambda f: f.severity, default=None)

    def of_kind(self, kind: str) -> list:
        return [f for f in self.findings if f.kind == kind]


# --------------------------------------------------------------------------
# surface analysis
# --------------------------------------------------------------------------

def surface_angles(x: np.ndarray, r: np.ndarray, build_plus_x: bool):
    """
    Angle from the build plate and downward-facing flag for each profile edge.

    The outward normal of an edge running (dx, dr) is (-dr, dx). Note the sign:
    Profile enforces a positive *revolved* volume, and under the Pappus form
    that means the meridional polygon winds clockwise in (x, r), not
    counter-clockwise. Taking the other normal inverts the whole analysis --
    every upward face is reported as an overhang and every real overhang is
    missed -- and it looks entirely plausible while doing so. Verified against
    the distance field in test_printability.

    A face is downward-facing when that normal opposes the build direction. Its
    angle from the plate is atan2(|dx|, |dr|): ninety degrees for a wall
    parallel to the build axis, zero for a flat ceiling.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    dx = np.roll(x, -1) - x
    dr = np.roll(r, -1) - r

    nx = -dr                                  # x component of the outward normal
    facing_down = nx < 0.0 if build_plus_x else nx > 0.0
    angle = np.degrees(np.arctan2(np.abs(dx), np.abs(dr)))
    return angle, facing_down, dx, dr


def reachable_from_plate(x_mid: float, r_mid: float,
                         px: np.ndarray, pr: np.ndarray,
                         build_plus_x: bool) -> bool:
    """
    Whether a support column can rise from the plate to this facet.

    Casts a ray straight down the build direction at constant radius and counts
    how many times it crosses the meridional profile. No crossings means clear
    air all the way to the plate, so a support can stand there. Any crossing
    means the column would have to pass through solid metal to get here, which
    is the signature of an enclosed void.

    Valid because the part is a body of revolution: a ray at constant radius in
    the meridional plane is a cylinder in three dimensions, and the profile
    tells the whole story.
    """
    ax, ar = px, pr
    bx, br = np.roll(px, -1), np.roll(pr, -1)

    # segments straddling this radius
    straddles = ((ar > r_mid) != (br > r_mid))
    if not straddles.any():
        return True
    t = (r_mid - ar[straddles]) / (br[straddles] - ar[straddles])
    cross_x = ax[straddles] + t * (bx[straddles] - ax[straddles])

    below = cross_x < x_mid - 1e-9 if build_plus_x else cross_x > x_mid + 1e-9
    return int(np.count_nonzero(below)) == 0


def check_profile(part: str, x, r, process: ProcessSpec,
                  build_plus_x: bool = True, axis_tol: float = 1e-9) -> list:
    """
    Overhangs and unsupported bridges on one meridional profile.

    Each part is printed on its own, so whichever of its faces is lowest sits on
    the build plate and is supported by definition. Counting the plate face as
    an overhang reports the head disc's own underside as a fifty-millimetre
    bridge, which is both alarming and meaningless.
    """
    angle, down, dx, dr = surface_angles(x, r, build_plus_x)
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    out: list[Finding] = []

    plate = x.min() if build_plus_x else x.max()
    on_plate = 2.0 * process.layer_mm

    for i in range(len(x)):
        if not down[i]:
            continue
        # sitting on the plate, or within a layer or two of it
        j = (i + 1) % len(x)
        if abs(x[i] - plate) <= on_plate and abs(x[j] - plate) <= on_plate:
            continue
        length = math.hypot(dx[i], dr[i])
        if length < 1e-9:
            continue
        # an edge lying on the axis encloses nothing and prints nothing
        if r[i] < axis_tol and r[(i + 1) % len(r)] < axis_tol:
            continue

        mx = 0.5 * (x[i] + x[j])
        mr = 0.5 * (r[i] + r[j])
        supportable = reachable_from_plate(mx, mr, x, r, build_plus_x)
        where = "" if supportable else " inside a sealed cavity, so it cannot be supported"

        if angle[i] < 5.0:
            # effectively a flat ceiling: it bridges rather than overhangs
            span = abs(dr[i])
            if span > process.max_bridge_mm:
                out.append(Finding(
                    part, "bridge", float(x[i]), float(r[i]), span,
                    process.max_bridge_mm,
                    f"flat downward face spanning {span:.1f} mm; it will sag "
                    f"across anything past {process.max_bridge_mm:.1f} mm{where}",
                    supportable))
        elif angle[i] < process.self_supporting_deg - 1e-6:
            out.append(Finding(
                part, "overhang", float(x[i]), float(r[i]), float(angle[i]),
                process.self_supporting_deg,
                f"downward face at {angle[i]:.1f} deg from the plate, under the "
                f"{process.self_supporting_deg:.0f} deg the process holds{where}",
                supportable))
    return out


# --------------------------------------------------------------------------
# whole-engine analysis
# --------------------------------------------------------------------------

def analyse(design, process: ProcessSpec | None = None,
            build_plus_x: bool = True) -> PrintabilityResult:
    """Printability of every part, in one build orientation."""
    process = process or ProcessSpec()
    a = design.assembly
    res = PrintabilityResult(build_direction="+x" if build_plus_x else "-x")

    for part, profile in a.profiles.items():
        res.findings.extend(
            check_profile(part, profile.x, profile.r, process, build_plus_x))

    # ---- envelope ----
    x_all = np.concatenate([np.asarray(p.x) for p in a.profiles.values()])
    r_all = np.concatenate([np.asarray(p.r) for p in a.profiles.values()])
    res.height_mm = float(x_all.max() - x_all.min())
    res.diameter_mm = float(2.0 * r_all.max())
    w, dpt, h = process.envelope_mm
    res.envelope_ok = (res.height_mm <= h and res.diameter_mm <= min(w, dpt))
    if not res.envelope_ok:
        res.findings.append(Finding(
            "assembly", "envelope", 0.0, 0.0,
            max(res.height_mm / h, res.diameter_mm / min(w, dpt)), 1.0,
            f"{res.height_mm:.0f} x {res.diameter_mm:.0f} mm does not fit a "
            f"{h:.0f} mm tall, {min(w, dpt):.0f} mm wide envelope"))

    # ---- features ----
    for name, circuit in design.circuits.items():
        if circuit is None:
            continue
        ch = circuit.channel
        for label, value in (("width", ch.width_mm), ("hot wall", ch.hot_wall_mm)):
            limit = process.min_feature_mm if label == "width" else process.min_wall_mm
            if value < limit:
                res.findings.append(Finding(
                    name, "feature", 0.0, 0.0, value, limit,
                    f"channel {label} {value:.2f} mm is under the {limit:.2f} mm "
                    f"the process can hold"))

    # ---- channels want to run along the build direction ----
    if any(c is not None for c in design.circuits.values()):
        res.notes.append(
            "cooling channels run axially, so building along the axis makes them "
            "vertical tunnels with no roof to support; any other orientation turns "
            "several hundred of them into overhangs at once")

    # ---- drainage ----
    res.findings.extend(drainage(design, process))
    return res


def drainage(design, process: ProcessSpec) -> list:
    """
    Whether loose powder can get out of every internal void.

    A sealed cavity in a powder-bed part stays full of powder for ever. It adds
    mass nobody accounted for, it can come loose later, and on a cooling jacket
    it blocks the thing the jacket is for. Each void needs an opening at least
    as large as the process minimum, and preferably at the low end of the build.
    """
    a = design.assembly
    out: list[Finding] = []

    # the centrebody cavity drains through the head bore
    bore = 2.0 * float(a.cavity_r[0])
    if bore < 4.0:
        out.append(Finding(
            "centrebody", "drainage", float(a.head_x), float(a.cavity_r[0]),
            bore, 4.0,
            f"centre bore is only {bore:.1f} mm across; powder will not clear a "
            f"cavity this long through an opening that small"))

    # each cooling circuit drains through its feed ports and its channel exits
    for name, circuit in design.circuits.items():
        if circuit is None:
            continue
        ch = circuit.channel
        area = ch.width_mm * ch.height_mm
        if area < process.min_feature_mm ** 2:
            out.append(Finding(
                name, "drainage", 0.0, 0.0, area, process.min_feature_mm ** 2,
                f"channel section {area:.2f} mm2 is too fine to clear powder "
                f"reliably over the length of the jacket"))
    return out


def choose_build_direction(design, process: ProcessSpec | None = None):
    """
    Try both orientations and report which is better, with the evidence.

    Not a free choice and not a matter of taste: on a body of revolution with
    axial channels one orientation is self-supporting nearly everywhere and the
    other is not.
    """
    process = process or ProcessSpec()
    up = analyse(design, process, build_plus_x=True)
    down = analyse(design, process, build_plus_x=False)

    def score(r: PrintabilityResult) -> tuple:
        return (len(r.findings), sum(max(f.severity, 0.0) for f in r.findings))

    best, other = (up, down) if score(up) <= score(down) else (down, up)
    best.notes.append(
        f"building {best.build_direction} gives {len(best.findings)} findings "
        f"against {len(other.findings)} the other way up"
        + (" (head down, spike up)" if best.build_direction == "+x"
           else " (spike down, head up)"))
    return best, up, down


def report(result: PrintabilityResult) -> str:
    lines = [f"build direction {result.build_direction}"
             f"   {result.height_mm:.0f} mm tall, {result.diameter_mm:.0f} mm across"
             f"   envelope {'ok' if result.envelope_ok else 'EXCEEDED'}"]
    lines.append(f"  {len(result.unsupportable)} unsupportable, "
                 f"{len(result.needs_support)} need removable support")
    if not result.findings:
        lines.append("  no printability findings")
    for f in sorted(result.findings, key=lambda f: -f.severity):
        lines.append(f"  [{f.kind:9s}] {f.part:11s} x={f.x_mm:+8.2f} r={f.r_mm:7.2f}  {f.detail}")
    for n in result.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)
