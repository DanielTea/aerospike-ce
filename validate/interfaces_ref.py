"""
External interfaces: every port, boss and fastener the engine needs to be used.

The flowpath is not an engine. Somebody has to connect propellant to it, ignite
it, instrument it and bolt it to something, and each of those is a hole in a
pressure vessel that has to be sized, placed where there is material, and kept
clear of everything else. This module derives that schedule from the design
rather than leaving it to a drawing.

Fluid routing for a regeneratively cooled aerospike
---------------------------------------------------
Fuel does two jobs and does them in series, which is what "regenerative" means:

    fuel inlet -> cowl jacket        -.
                                       >- head fuel manifold -> fuel orifices
    fuel inlet -> bore -> spike jacket -'

    oxidiser inlet -> ox dome -> oxidiser orifices

So there is no external coolant outlet. The coolant *is* the fuel, and it leaves
the jacket into the injector. A port that took coolant out of the engine would
be throwing away both the heat and the propellant.

The centrebody jacket is fed down the middle: the spike's channels start at the
throat end, so fuel runs aft along the central bore, turns at the feed ports and
comes back forward through the wall. That is why the bore is a bore and not a
solid plug, and why it must not be used for the igniter.

Sizing
------
Line bores come from a target flow velocity, not from a catalogue. Liquid feed
lines run at five to ten metres per second: slower wastes envelope, faster costs
pressure and invites water-hammer on valve closure.

Bolts are sized against the separating load, which is chamber pressure acting
over the sealed area plus the thrust reaction, with preload enough that the
joint never gaps. A gapped joint fretts its seal and then leaks.

All lengths mm, pressures Pa internally, mass flows kg/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Coarse metric bolt table: thread, tensile stress area mm^2, head across-flats.
BOLTS: dict[str, tuple[float, float]] = {
    "M3": (5.03, 5.5),
    "M4": (8.78, 7.0),
    "M5": (14.2, 8.0),
    "M6": (20.1, 10.0),
    "M8": (36.6, 13.0),
    "M10": (58.0, 16.0),
}

# A286 bolts, a normal choice next to a hot engine. Allowable is kept to 60% of
# yield so the joint stays elastic through thermal cycling.
BOLT_YIELD_PA = 690e6
BOLT_UTILISATION = 0.60


@dataclass(frozen=True)
class Port:
    """One external connection."""
    name: str
    kind: str               # inlet | igniter | instrument | drain
    fluid: str              # fuel | oxidiser | gas | none
    x_mm: float             # axial station of the boss face
    r_mm: float             # radius the boss sits at
    theta_deg: float        # angular position
    orientation: str        # radial | axial
    bore_mm: float          # through-bore diameter
    boss_od_mm: float       # outside diameter of the raised boss
    mass_flow: float        # kg/s, zero for instruments
    velocity: float         # m/s in the bore, zero for instruments
    pressure_bar: float     # design pressure at the port
    temperature_k: float
    connect_to: str         # plain-language: what goes here

    @property
    def fitting(self) -> str:
        """Nearest sensible AN tube size for the bore."""
        for an, od in (("AN-3", 4.8), ("AN-4", 6.4), ("AN-6", 9.5),
                       ("AN-8", 12.7), ("AN-10", 15.9), ("AN-12", 19.1)):
            if od >= self.bore_mm:
                return an
        return "AN-16"


@dataclass(frozen=True)
class BoltedJoint:
    size: str
    count: int
    pitch_circle_mm: float
    separating_load: float      # N
    preload_per_bolt: float     # N
    stress_pa: float
    utilisation: float          # of allowable
    edge_margin_mm: float
    flange_radius_required_mm: float = 0.0
    fits: bool = True
    reason: str = "ok"


@dataclass(frozen=True)
class InterfaceSchedule:
    ports: list = field(default_factory=list)
    joint: BoltedJoint | None = None
    notes: list = field(default_factory=list)

    def by_name(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def fuel_ports(self) -> list:
        return [p for p in self.ports if p.fluid == "fuel" and p.kind == "inlet"]

    @property
    def ox_ports(self) -> list:
        return [p for p in self.ports if p.fluid == "oxidiser" and p.kind == "inlet"]


def bore_for_flow(mass_flow: float, density: float, velocity: float) -> float:
    """Bore diameter in mm that carries a mass flow at a target velocity."""
    if velocity <= 0.0 or density <= 0.0:
        raise ValueError("density and velocity must be positive")
    area = mass_flow / (density * velocity)
    return 2.0 * math.sqrt(area / math.pi) * 1e3


def velocity_in_bore(mass_flow: float, density: float, bore_mm: float) -> float:
    """Inverse of bore_for_flow, so the sizing can be shown to round-trip."""
    area = math.pi * (bore_mm * 1e-3) ** 2 / 4.0
    return mass_flow / (density * area)


def size_bolted_joint(
    separating_load: float,
    pitch_circle_diameter_mm: float,
    flange_outer_radius_mm: float,
    safety_factor: float = 2.0,
    preload_fraction: float = 1.4,
    min_count: int = 8,
    max_count: int = 36,
    max_utilisation: float = 0.70,
) -> BoltedJoint:
    """
    Smallest bolt that closes the head joint, and how many of them.

    Preload has to exceed the separating load or the joint gaps, and a gapped
    joint works its seal until it leaks. `preload_fraction` is the margin over
    the applied load; 1.4 is conventional for a static face seal.

    Prefers more small bolts to fewer large ones: it spreads the line load on
    the seal, and small bolts fit the narrow flange an annular engine has.
    """
    required = separating_load * safety_factor * preload_fraction
    circumference = math.pi * pitch_circle_diameter_mm
    best: BoltedJoint | None = None

    # Prefer a joint of a dozen to three dozen fasteners. Taking the smallest
    # bolt that closes the load gives fifty M4s, which is a correct answer and a
    # miserable one to build, torque evenly or inspect.
    for size in ("M8", "M6", "M10", "M5", "M4"):
        area, across_flats = BOLTS[size]
        # Work to a fraction of allowable rather than right up to it. A joint
        # sized to 95 percent has nothing left for preload scatter, and preload
        # scatter on a hand-torqued fastener is tens of percent.
        per_bolt_cap = BOLT_YIELD_PA * BOLT_UTILISATION * max_utilisation * area * 1e-6
        count = max(min_count, math.ceil(required / per_bolt_cap))
        count += count % 2                                            # keep it even

        edge = flange_outer_radius_mm - 0.5 * pitch_circle_diameter_mm - across_flats / 2.0
        needed = 0.5 * pitch_circle_diameter_mm + across_flats / 2.0 + 1.5
        per_bolt = required / count

        reasons = []
        if count > max_count:
            reasons.append(f"{count} fasteners is past the {max_count} this "
                           f"joint should need; use a larger bolt circle")
        if circumference / count < across_flats * 1.6:
            reasons.append(f"{count} x {size} heads do not fit on a "
                           f"{pitch_circle_diameter_mm:.0f} mm bolt circle")
        if edge < 1.5:
            reasons.append(f"needs {needed:.1f} mm flange radius, "
                           f"design gives {flange_outer_radius_mm:.1f} mm")

        joint = BoltedJoint(
            size=size, count=count, pitch_circle_mm=pitch_circle_diameter_mm,
            separating_load=separating_load,
            preload_per_bolt=per_bolt,
            stress_pa=per_bolt / (area * 1e-6),
            utilisation=(per_bolt / (area * 1e-6)) / (BOLT_YIELD_PA * BOLT_UTILISATION),
            edge_margin_mm=edge,
            flange_radius_required_mm=needed,
            fits=not reasons,
            reason="ok" if not reasons else "; ".join(reasons),
        )
        if joint.fits:
            return joint
        # Never raise. A joint that cannot be closed is a design result, and the
        # caller can only act on it if it comes back with the numbers attached.
        if best is None or count * area > best.count * BOLTS[best.size][0]:
            best = joint

    return best


def _split_inlet(design, a, mass_flow: float, density: float, velocity: float,
                 margin_mm: float = 3.0, max_ports: int = 6):
    """
    How many feed ports it takes to get a flow onto the head face, and where.

    A single port carrying the whole oxidiser flow at eight metres per second
    wants an eighteen millimetre bore, and on an annular head there is nowhere
    to put one: outboard of the oxidiser ring it runs off the flange, inboard of
    the fuel ring it runs into the bore. Splitting the flow keeps the total area
    and shrinks each hole until it fits between the rings, which is what a real
    manifold does anyway.

    Returns (count, bore_mm, radius_mm) or raises if even the split does not fit.
    """
    for n in range(1, max_ports + 1):
        bore = bore_for_flow(mass_flow / n, density, velocity)
        r = _clear_radius(design, a, bore, margin_mm)
        if r is None:
            continue
        # do n bosses of this size fit round that circle?
        if 2.0 * math.pi * r / n >= (bore + 2.0 * margin_mm) * 1.4:
            return n, bore, r
    raise ValueError(
        f"cannot fit {mass_flow * 1e3:.0f} g/s onto the head face in {max_ports} "
        f"ports; the face is too congested or the flange too small")


def _clear_radius(design, a, bore_mm: float, margin_mm: float = 3.0):
    """
    A radius on the head face where a boss of this bore hits nothing.

    Tries just outboard of the oxidiser orifice ring first, then just inboard of
    the fuel ring, and finally gives up and returns the mid-flange radius so the
    caller's collision check can report it rather than silently overlapping.
    """
    if design.injector is None:
        return 0.7 * a.flange_radius
    r_ox = design.injector.ox_ring_radius * 1e3
    r_fuel = design.injector.fuel_ring_radius * 1e3
    dd_ox = design.injector.d_ox_mm
    dd_fuel = design.injector.d_fuel_mm
    half = 0.5 * bore_mm

    outboard = r_ox + 0.5 * dd_ox + half + margin_mm
    if outboard + half + margin_mm < a.flange_radius:
        return outboard
    between = 0.5 * (r_fuel + r_ox)
    if (between - half - margin_mm > r_fuel + 0.5 * dd_fuel and
            between + half + margin_mm < r_ox - 0.5 * dd_ox):
        return between
    inboard = r_fuel - 0.5 * dd_fuel - half - margin_mm
    if inboard - half - margin_mm > a.cavity_r[0]:
        return inboard
    return None


def port_clashes(schedule: InterfaceSchedule, design) -> list:
    """
    Every place a port overlaps something it must not.

    Checked rather than assumed: a boss placed on a drawing looks fine and a
    boss placed on an orifice ring is a hole through the injector. Compares
    axial ports against both orifice rings and against each other, and radial
    ports against each other.
    """
    a = design.assembly
    out: list[str] = []
    inj = design.injector

    for p in schedule.ports:
        if p.orientation == "axial" and inj is not None and p.r_mm > 0.0:
            lo, hi = p.r_mm - 0.5 * p.bore_mm, p.r_mm + 0.5 * p.bore_mm
            for ring, rr, dd in (("fuel", inj.fuel_ring_radius * 1e3, inj.d_fuel_mm),
                                 ("oxidiser", inj.ox_ring_radius * 1e3, inj.d_ox_mm)):
                if lo < rr + 0.5 * dd and hi > rr - 0.5 * dd:
                    out.append(f"{p.name} (r {lo:.1f}-{hi:.1f} mm) crosses the "
                               f"{ring} orifice ring at r {rr:.1f} mm")
        if p.orientation == "axial" and p.r_mm + 0.5 * p.boss_od_mm > a.flange_radius:
            out.append(f"{p.name} boss overhangs the flange rim")

    for i, p in enumerate(schedule.ports):
        for q in schedule.ports[i + 1:]:
            if p.orientation != q.orientation:
                continue
            dtheta = abs((p.theta_deg - q.theta_deg + 180.0) % 360.0 - 180.0)
            if p.orientation == "radial":
                if abs(p.x_mm - q.x_mm) < 0.5 * (p.boss_od_mm + q.boss_od_mm):
                    arc = math.radians(dtheta) * max(p.r_mm, q.r_mm)
                    if arc < 0.5 * (p.boss_od_mm + q.boss_od_mm):
                        out.append(f"{p.name} and {q.name} bosses overlap")
            else:
                dr = abs(p.r_mm - q.r_mm)
                arc = math.radians(dtheta) * max(p.r_mm, q.r_mm, 1.0)
                if math.hypot(dr, arc) < 0.5 * (p.boss_od_mm + q.boss_od_mm):
                    out.append(f"{p.name} and {q.name} bosses overlap")
    return out


def build_schedule(design) -> InterfaceSchedule:
    """
    Derive every external interface from a solved design.

    Ports are placed on material that exists and kept clear of the channels and
    the injector rings. Nothing here is a free choice except the angular slots,
    which are spread to keep the plumbing from fouling itself.
    """
    a = design.assembly
    ch = design.chamber
    prop = ch.propellant
    notes: list[str] = []
    ports: list[Port] = []

    pc_bar = ch.chamber_pressure / 1e5
    feed_bar = pc_bar * 1.5          # what the feed system must deliver

    # ---- fuel: split between the two jackets in the ratio they are cooled ----
    split = design.coolant_split
    v_line = 8.0

    # cowl jacket, fed radially through the outer skin near the throat.
    #
    # Taken from the inlet ring itself. Composed instead from the aft end of
    # the wall and the largest radius anywhere on it -- two unrelated stations
    # -- this reported a fitting at x -1.6, r 101.8: twenty-five millimetres
    # downstream of the manifold it feeds, and nineteen millimetres out in mid
    # air, because the cowl has tapered to r 82 by then.
    from manifold_ref import design_manifolds
    _md = design_manifolds(design)
    _ring = next((q for q in _md.plenums if q.name == "cowl_inlet_ring"), None)
    if _ring is not None:
        x_cowl = _ring.x_mm
        r_cowl = (_ring.r_inner_mm + 2.0 * _ring.half_r_mm
                  + a.structure.wall_thickness_mm)
    else:
        x_cowl = float(a.outer_wall_x[-1]) - 6.0
        r_cowl = float(max(a.cowl_outer_r)) + 1.0
    mdot_cowl = ch.mass_flow_fuel * split.get("cowl", 0.5)
    ports.append(Port(
        name="fuel_in_cowl", kind="inlet", fluid="fuel",
        x_mm=x_cowl, r_mm=r_cowl, theta_deg=0.0, orientation="radial",
        bore_mm=bore_for_flow(mdot_cowl, prop.density_fuel, v_line),
        boss_od_mm=bore_for_flow(mdot_cowl, prop.density_fuel, v_line) + 6.0,
        mass_flow=mdot_cowl, velocity=v_line,
        pressure_bar=feed_bar, temperature_k=prop.coolant_inlet_t,
        connect_to=(f"{prop.fuel} supply. Feeds the cowl jacket manifold; the fuel "
                    f"then runs forward through the cowl channels to the injector."),
    ))

    # centrebody jacket, fed axially down the central bore from the head end
    mdot_body = ch.mass_flow_fuel * split.get("centrebody", 0.5)
    x_head_face = a.head_x - a.structure.head_thickness_mm
    ports.append(Port(
        name="fuel_in_spike", kind="inlet", fluid="fuel",
        x_mm=x_head_face, r_mm=0.0, theta_deg=0.0, orientation="axial",
        bore_mm=min(bore_for_flow(mdot_body, prop.density_fuel, v_line),
                    2.0 * a.cavity_r[0] - 4.0),
        boss_od_mm=2.0 * a.cavity_r[0],
        mass_flow=mdot_body, velocity=v_line,
        pressure_bar=feed_bar, temperature_k=prop.coolant_inlet_t,
        connect_to=(f"{prop.fuel} supply, second leg. Runs aft down the centre bore to "
                    f"the spike feed ports, then forward through the spike channels."),
    ))

    # ---- oxidiser: straight into the dome, it does no cooling ----
    # Split across as many bosses as the face has room for. One port carrying
    # the whole flow needs an 18 mm bore and there is nowhere on an annular head
    # to put one without driving it through an orifice ring.
    # Radially into the dome through the rim, not axially into the end face.
    # The dome sits outboard, so an axial bore on the end face at a radius
    # clear of the orifice rings misses it entirely and carries on into the
    # injector; there was no oxidiser inlet in the geometry at all.
    _dome = next((q for q in _md.plenums if q.name == "ox_dome"), None)
    t_ox = 90.0 if prop.oxidiser == "LOX" else 290.0
    if _dome is not None:
        # Six bores rather than one, because a radial hole is a bridge over its
        # own width and an 18 mm span is past what the process will hold up.
        n_ox, bore_ox = 6, 8.0
        x_ox, r_ox_boss, orient_ox = _dome.x_mm, a.flange_radius, "radial"
        v_ox = velocity_in_bore(ch.mass_flow_ox / n_ox, prop.density_ox, bore_ox)
    else:
        n_ox, bore_ox, r_ox_boss = _split_inlet(
            design, a, ch.mass_flow_ox, prop.density_ox, v_line)
        x_ox, orient_ox, v_ox = x_head_face, "axial", v_line
    for i in range(n_ox):
        ports.append(Port(
            name="ox_in" if n_ox == 1 else f"ox_in_{i + 1}",
            kind="inlet", fluid="oxidiser",
            x_mm=x_ox, r_mm=r_ox_boss,
            theta_deg=(15.0 + i * 360.0 / n_ox) % 360.0,
            orientation=orient_ox,
            bore_mm=bore_ox, boss_od_mm=bore_ox + 6.0,
            mass_flow=ch.mass_flow_ox / n_ox, velocity=v_ox,
            pressure_bar=feed_bar, temperature_k=t_ox,
            connect_to=(f"{prop.oxidiser} supply"
                        + (f", leg {i + 1} of {n_ox}" if n_ox > 1 else "")
                        + ". Into the dome and out through the oxidiser orifices. "
                          "It does no cooling, so it stays cold."),
        ))

    # ---- igniter: through the cowl into the chamber, clear of the bore ----
    x_ign = 0.5 * (a.head_x + a.converging_start_x)
    ports.append(Port(
        name="igniter", kind="igniter", fluid="gas",
        x_mm=x_ign, r_mm=float(max(a.cowl_outer_r)) + 1.0, theta_deg=90.0,
        orientation="radial", bore_mm=8.0, boss_od_mm=18.0,
        mass_flow=0.0, velocity=0.0,
        pressure_bar=pc_bar, temperature_k=ch.t_chamber,
        connect_to=("Torch igniter, firing into the chamber annulus. Radial rather than "
                    "axial because the centre bore is carrying fuel."),
    ))

    # ---- instrumentation ----
    ports.append(Port(
        name="pc_tap", kind="instrument", fluid="none",
        x_mm=x_ign, r_mm=float(max(a.cowl_outer_r)) + 1.0, theta_deg=270.0,
        orientation="radial", bore_mm=1.5, boss_od_mm=10.0,
        mass_flow=0.0, velocity=0.0,
        pressure_bar=pc_bar, temperature_k=ch.t_chamber,
        connect_to="Chamber pressure transducer. Opposite the igniter so neither sees the other.",
    ))
    # Instrument bosses tucked inside the flange rim. Sitting them 4 mm in from
    # the rim overhangs it once the boss has any diameter at all.
    r_inst = a.flange_radius - 0.5 * 9.0 - 2.0
    for name, theta, what in (("t_coolant_out", 45.0, "coolant outlet temperature"),
                              ("p_coolant_out", 135.0, "coolant outlet pressure")):
        ports.append(Port(
            name=name, kind="instrument", fluid="none",
            x_mm=a.head_x - 0.5 * a.structure.head_thickness_mm,
            r_mm=r_inst, theta_deg=theta, orientation="axial",
            bore_mm=1.5, boss_od_mm=9.0, mass_flow=0.0, velocity=0.0,
            pressure_bar=feed_bar,
            temperature_k=max((c.solution.coolant_outlet_temperature
                               for c in design.circuits.values() if c), default=300.0),
            connect_to=f"Instrumentation, {what} in the head manifold.",
        ))

    # ---- the head joint ----
    # Chamber pressure over the sealed annulus, plus the thrust the head reacts.
    r_bore = float(a.cavity_r[0])
    sealed_area = math.pi * ((a.chamber_outer_radius * 1e-3) ** 2 - (r_bore * 1e-3) ** 2)
    separating = ch.chamber_pressure * sealed_area + ch.thrust_sea_level

    # Bolt circle sits midway between the cowl wall and the flange rim, and is
    # carried as a DIAMETER throughout. Mixing radius and diameter here is worth
    # two on the head count and reads as a comfortable joint that is not one.
    pcd = 2.0 * (0.5 * (float(max(a.cowl_outer_r)) + a.flange_radius))
    joint = size_bolted_joint(separating, pcd, a.flange_radius)
    if not joint.fits:
        notes.append(
            f"the head joint does not close: {joint.reason}. A flange sized by eye "
            f"rather than by its fasteners is the usual reason a head joint cannot "
            f"be assembled; geometry.flange_width_mm is the knob.")

    notes.append(
        "There is no coolant outlet port. The fuel is the coolant: it leaves the "
        "jackets into the injector manifold, so taking it out of the engine would "
        "throw away the propellant along with the heat.")
    if design.injector is not None:
        notes.append(
            f"Both fuel legs converge in the head manifold and feed "
            f"{design.injector.n_elements} fuel orifices; the dome feeds "
            f"{design.injector.n_elements} oxidiser orifices.")
    return InterfaceSchedule(ports=ports, joint=joint, notes=notes)


def port_schedule_table(schedule: InterfaceSchedule) -> str:
    """The schedule as a table, which is how a person actually uses it."""
    rows = [
        f"{'port':16s} {'fluid':9s} {'where':28s} {'bore':>6s} {'fitting':8s} "
        f"{'flow':>9s} {'P bar':>6s} {'T K':>6s}",
        "-" * 104,
    ]
    for p in schedule.ports:
        where = (f"x={p.x_mm:+.1f} r={p.r_mm:.1f} th={p.theta_deg:.0f} {p.orientation}")
        flow = f"{p.mass_flow * 1e3:.0f} g/s" if p.mass_flow > 0 else "-"
        rows.append(
            f"{p.name:16s} {p.fluid:9s} {where:28s} {p.bore_mm:6.2f} {p.fitting:8s} "
            f"{flow:>9s} {p.pressure_bar:6.1f} {p.temperature_k:6.0f}")
    if schedule.joint:
        j = schedule.joint
        rows.append("")
        verdict = "ok" if j.fits else f"DOES NOT CLOSE: {j.reason}"
        rows.append(f"head joint: {j.count} x {j.size} on a {j.pitch_circle_mm:.1f} mm "
                    f"bolt circle, {j.utilisation * 100:.0f}% of allowable, "
                    f"{j.edge_margin_mm:+.1f} mm edge margin -- {verdict}")
        rows.append(f"  separating load {j.separating_load / 1e3:.1f} kN, "
                    f"preload {j.preload_per_bolt / 1e3:.1f} kN per bolt")
    return "\n".join(rows)
