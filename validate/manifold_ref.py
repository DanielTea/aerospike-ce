"""
Manifolds and mounts: the parts that make the engine connectable and holdable.

interfaces_ref.py says where the propellant goes in. This says what it goes into.
A feed port on its own opens onto a single cooling channel; four hundred of them
need a plenum that distributes evenly around the circumference, and that plenum
is an internal void in a printed part, which constrains its shape far more than
its flow does.

Shape before size
-----------------
A manifold ring is a torus of fluid inside metal. Built head down, its roof is
unsupported and nothing can reach inside to take a support out afterwards, so
the cross-section has to hold itself up. A round or square section will not: it
presents a horizontal ceiling the full width of the plenum. A diamond does, if
its upper faces stand at or above the process angle, which for a symmetric
diamond means it is at least as tall as it is wide.

That is why the sections here are diamonds and not circles. It costs a little
flow area for the same envelope and buys a plenum that can actually be printed
closed.

Size
----
Plenums run slow -- three to five metres per second -- because a manifold that
runs fast does not distribute. The dynamic head at the inlet then biases flow
towards the channels nearest it, and the jacket develops a hot streak on the
opposite side. Peak flow is half the total, since it splits both ways round the
ring from a single inlet.

Mounts
------
Lugs at the head end, which is where the build plate is, so they are supported
by definition. Sized against thrust with a factor, checked for bearing on the
hole and for tear-out to the edge, both of which are the failures that actually
happen to a lug rather than the tensile one people size for.

All lengths mm, loads N, pressures Pa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# 316L / IN718-ish printed allowables, conservative
BEARING_ALLOWABLE_PA = 300e6
SHEAR_ALLOWABLE_PA = 180e6
PIN_SIZES_MM = (6.0, 8.0, 10.0, 12.0, 16.0)


@dataclass(frozen=True)
class Plenum:
    """An annular distribution manifold with a self-supporting section."""
    name: str
    x_mm: float                 # axial centre
    r_inner_mm: float           # inboard face of the void
    half_x_mm: float            # half height, along the axis
    half_r_mm: float            # half width, radially
    mass_flow: float            # kg/s total through the ring
    velocity: float             # m/s at the busiest section
    density: float
    feeds: int                  # how many channels it serves

    @property
    def area_mm2(self) -> float:
        """Diamond section: half the bounding rectangle."""
        return 2.0 * self.half_x_mm * self.half_r_mm

    @property
    def r_centre_mm(self) -> float:
        return self.r_inner_mm + self.half_r_mm

    @property
    def roof_angle_deg(self) -> float:
        """
        Angle the upper faces make with the build plate.

        A symmetric diamond gives atan(half_x / half_r); at or above the process
        angle it holds itself up, below it the roof sags into the plenum.
        """
        return math.degrees(math.atan2(self.half_x_mm, self.half_r_mm))

    def self_supporting(self, limit_deg: float = 45.0) -> bool:
        return self.roof_angle_deg >= limit_deg - 1e-9

    def half_x_at(self, radius_mm: float) -> float:
        """
        How deep the section is at one radius, axially. Zero outside.

        The section is a diamond, so its axial extent is greatest at the centre
        radius and tapers to nothing at either edge. Reading the extreme tip as
        though it applied across the whole width is how an orifice ends up
        stopping five millimetres short of the plenum it is supposed to enter,
        while every dimension involved still looks right.
        """
        t = abs(radius_mm - self.r_centre_mm) / max(self.half_r_mm, 1e-9)
        return self.half_x_mm * max(1.0 - t, 0.0)

    def reaches(self, radius_mm: float, depth_mm: float = 1.0) -> bool:
        """Whether the section is genuinely open at a radius, not just grazing."""
        return self.half_x_at(radius_mm) >= depth_mm

    def x_admitting(self, radius_mm: float, bore_radius_mm: float) -> float | None:
        """
        Aft-most x at which a bore of this radius is *wholly* inside the section.

        Not the same as the aft face. The section narrows to a rounded point at
        its tip, so a hole that merely crosses the tip meets a cavity a few
        tenths of a millimetre across and seals at any real voxel. Starting the
        hole here instead means it opens into full section.
        """
        t = (abs(radius_mm - self.r_centre_mm) + bore_radius_mm) / max(self.half_r_mm, 1e-9)
        if t >= 1.0:
            return None
        return self.x_mm + self.half_x_mm * (1.0 - t)


@dataclass(frozen=True)
class MountLug:
    """One mounting lug: a radial pad at the head end with a through hole."""
    count: int
    r_inner_mm: float
    r_outer_mm: float
    thickness_mm: float         # axial
    width_deg: float
    hole_diameter_mm: float
    hole_radius_mm: float       # where the hole centre sits
    design_load_n: float        # per lug
    bearing_stress_pa: float
    shear_stress_pa: float
    edge_distance_mm: float

    @property
    def bearing_ok(self) -> bool:
        return self.bearing_stress_pa <= BEARING_ALLOWABLE_PA

    @property
    def tear_out_ok(self) -> bool:
        return (self.shear_stress_pa <= SHEAR_ALLOWABLE_PA
                and self.edge_distance_mm >= 1.5 * self.hole_diameter_mm)

    @property
    def ok(self) -> bool:
        return self.bearing_ok and self.tear_out_ok


@dataclass
class ManifoldDesign:
    plenums: list = field(default_factory=list)
    lugs: MountLug | None = None
    notes: list = field(default_factory=list)

    def by_name(self, name: str) -> Plenum:
        for p in self.plenums:
            if p.name == name:
                return p
        raise KeyError(name)


def size_plenum(
    name: str,
    mass_flow: float,
    density: float,
    r_inner_mm: float,
    x_mm: float,
    feeds: int,
    velocity: float = 4.0,
    aspect: float = 1.0,
    min_half_mm: float = 2.0,
) -> Plenum:
    """
    Size an annular plenum for a flow, with a section that can be printed.

    `aspect` is half-height over half-width. One gives a 45 degree roof, which
    is the shallowest that holds; above one the roof is steeper and safer and
    the ring grows taller for the same area. Below one it will not print closed,
    so it is refused rather than quietly returned.
    """
    if aspect < 1.0:
        raise ValueError(
            f"aspect {aspect} gives a {math.degrees(math.atan(aspect)):.0f} deg roof; "
            f"a plenum flatter than 45 degrees cannot hold itself up and nothing "
            f"can reach inside to support it")
    if velocity <= 0.0 or density <= 0.0:
        raise ValueError("velocity and density must be positive")

    # the ring splits both ways from the inlet, so the busiest section sees half
    area_m2 = 0.5 * mass_flow / (density * velocity)
    area_mm2 = area_m2 * 1e6

    # diamond area = 2 a b, with a = aspect * b
    half_r = math.sqrt(area_mm2 / (2.0 * aspect))
    half_x = aspect * half_r
    half_r = max(half_r, min_half_mm)
    half_x = max(half_x, min_half_mm)

    return Plenum(name=name, x_mm=x_mm, r_inner_mm=r_inner_mm,
                  half_x_mm=half_x, half_r_mm=half_r,
                  mass_flow=mass_flow, velocity=velocity,
                  density=density, feeds=feeds)


def size_mount_lugs(
    thrust_n: float,
    flange_radius_mm: float,
    head_thickness_mm: float,
    count: int = 4,
    safety: float = 2.0,
    width_deg: float = 26.0,
) -> MountLug:
    """
    Size mounting lugs against thrust, on bearing and tear-out.

    A lug rarely fails in tension across its section. It fails by the pin
    crushing the hole, or by the material between hole and edge shearing out, so
    those are the two checks. Edge distance of one and a half diameters is the
    usual floor and is also what keeps a printed lug from tearing along its
    layer lines.

    Placed at the head end, which sits on the build plate, so the lugs are
    supported by definition and need no overhang treatment at all.
    """
    if count < 3:
        raise ValueError("fewer than three lugs cannot constrain the engine")

    load = thrust_n * safety / count
    best = None

    for d in PIN_SIZES_MM:
        # Derive the lug from the pin, not the other way round. A fixed lug
        # length with the hole at a fixed fraction of it means a larger pin
        # leaves *less* edge, so the search gets worse as it tries harder and
        # every size fails tear-out. Edge distance has to scale with diameter.
        r_hole = flange_radius_mm + 2.2 * d
        r_outer = r_hole + 2.2 * d
        edge = r_outer - r_hole - 0.5 * d          # 1.7 d, comfortably over 1.5

        thickness = max(head_thickness_mm, 6.0,
                        load / (d * BEARING_ALLOWABLE_PA) * 1e6)
        bearing = load / (d * thickness * 1e-6)
        shear = load / (2.0 * edge * thickness * 1e-6)

        lug = MountLug(
            count=count, r_inner_mm=flange_radius_mm, r_outer_mm=r_outer,
            thickness_mm=thickness, width_deg=width_deg,
            hole_diameter_mm=d, hole_radius_mm=r_hole,
            design_load_n=load, bearing_stress_pa=bearing,
            shear_stress_pa=shear, edge_distance_mm=edge)
        if lug.ok:
            return lug
        best = lug

    return best         # the largest tried; caller reports it as not closing


def design_manifolds(design, velocity: float = 4.0, aspect: float = 1.2,
                     lug_count: int = 4, head_velocity: float = 8.0) -> ManifoldDesign:
    """
    Every plenum and mount the engine needs, derived from the solved design.

    `head_velocity` is separate from `velocity` because the head plenums are
    depth-limited by the disc rather than area-limited by anything physical: at
    4 m/s the fuel manifold wants to be 44 mm deep in a 46 mm disc, leaving
    nowhere for the passages that feed it. 8 m/s is still slow against the
    36 m/s the orifices run at, which is what governs distribution.
    """
    from mesh_solid import centrebody_channels, cowl_channels, feed_ports

    a = design.assembly
    ch = design.chamber
    prop = ch.propellant
    out = ManifoldDesign()

    split = design.coolant_split

    # ---- cowl inlet ring, over the radial feed ports ----
    if design.circuits.get("cowl"):
        cut = cowl_channels(a, design.circuits["cowl"].channel)
        port = feed_ports(a, cut)
        # The skin the ring sits on, read off the wall at that station. Taken
        # from the feed port's outer reach instead -- which is where the port
        # ends, not where the metal is -- the ring lands 2.1 mm proud of a
        # surface that has already begun to taper, its boss touches the cowl
        # only at the forward edge, and hollowing the plenum inside it cuts
        # that last connection: 27 cm3 of copper attached to nothing.
        r_skin = float(np.interp(port.x_at, np.asarray(a.cowl_outer_x),
                                 np.asarray(a.cowl_outer_r)))
        out.plenums.append(size_plenum(
            "cowl_inlet_ring",
            mass_flow=ch.mass_flow_fuel * split.get("cowl", 0.5),
            density=prop.density_fuel,
            r_inner_mm=r_skin,
            x_mm=port.x_at,
            feeds=cut.n_channels,
            velocity=velocity, aspect=aspect))

    # ---- head collector, where both jackets discharge into the injector ----
    n_feeds = sum(c.channel.n_channels for c in design.circuits.values() if c)
    out.plenums.append(size_plenum(
        "head_fuel_manifold",
        mass_flow=ch.mass_flow_fuel,
        density=prop.density_fuel,
        r_inner_mm=a.shoulder_radius - 6.0,
        x_mm=a.head_x - 0.5 * a.structure.head_thickness_mm,
        feeds=n_feeds,
        velocity=velocity, aspect=aspect))

    # ---- oxidiser dome ----
    out.plenums.append(size_plenum(
        "ox_dome",
        mass_flow=ch.mass_flow_ox,
        density=prop.density_ox,
        r_inner_mm=a.chamber_outer_radius - 4.0,
        x_mm=a.head_x - 0.5 * a.structure.head_thickness_mm,
        feeds=design.injector.n_elements if design.injector else 0,
        velocity=velocity, aspect=aspect))

    # ---- mounts ----
    out.lugs = size_mount_lugs(
        thrust_n=ch.thrust_vacuum,
        flange_radius_mm=a.flange_radius,
        head_thickness_mm=a.structure.head_thickness_mm,
        count=lug_count)
    if not out.lugs.ok:
        out.notes.append(
            f"mount lugs do not close: bearing "
            f"{out.lugs.bearing_stress_pa / 1e6:.0f} MPa, shear "
            f"{out.lugs.shear_stress_pa / 1e6:.0f} MPa, edge "
            f"{out.lugs.edge_distance_mm:.1f} mm. Lengthen the lug or add more.")

    # ---- lay the head out so every stream reaches the plenum it feeds ----
    #
    # Three constraints at once, and satisfying any two of them is what went
    # wrong here repeatedly. A plenum has to fit inside the metal that carries
    # it; it has to be reachable by the orifices it feeds; and it has to be
    # reachable by whatever feeds *it*.
    #
    # The arrangement that satisfies all three puts the fuel manifold inboard,
    # wide, straddling its own orifice ring, and the oxidiser dome outboard and
    # wide -- too far out to straddle its ring, so the ox orifices are fed by a
    # short ring of radial ports instead. That is what a coaxial post does,
    # done with the primitives this model already has.
    #
    # Sizing them on flow alone put the two of them across r 67 to 135 mm on a
    # disc that runs 70 to 123, so they severed it. Packing them outward from
    # the bore to fix that left the dome at r 98-117 with its orifices at r 88:
    # the dome fed nothing and the orifices pointed at the radius the *fuel*
    # manifold occupies. Both are watertight either way. Only the sealed-void
    # check noticed, as 325 cm3 of powder with no way out.
    r_bore = float(a.cavity_r[0])
    wall = a.structure.wall_thickness_mm
    x_face = a.head_x
    head_plenums = [p for p in out.plenums if p.name != "cowl_inlet_ring"]

    if head_plenums and design.injector is not None:
        r_fuel = design.injector.fuel_ring_radius * 1e3
        r_ox = design.injector.ox_ring_radius * 1e3
        d_ox = design.injector.d_ox_mm

        def _fit(p, r_lo, r_hi, x_aft, velocity):
            """Re-place one plenum in a radial band, sized on flow, aspect >= 1."""
            half_r = 0.5 * (r_hi - r_lo)
            area = 0.5 * p.mass_flow / (p.density * velocity) * 1e6
            half_r = max(half_r, 1.0)
            half_x = max(area / (2.0 * half_r), half_r)     # aspect >= 1: it must hold its roof up
            # x_aft is the aft face of the cavity, and it must stay a wall
            # short of the injector face. Nudged past it, the manifold opens
            # into the chamber -- which is watertight, drains, and is an engine
            # that dumps its fuel manifold into the combustion zone.
            x_c = x_aft - half_x
            from mesh_solid import plenum_section_area_mm2
            v = 0.5 * p.mass_flow / (
                p.density * plenum_section_area_mm2(half_x, half_r) * 1e-6)
            return Plenum(name=p.name, x_mm=x_c, r_inner_mm=r_lo,
                          half_x_mm=half_x, half_r_mm=half_r,
                          mass_flow=p.mass_flow, velocity=v,
                          density=p.density, feeds=p.feeds)

        for p in list(head_plenums):
            if p.name == "head_fuel_manifold":
                # Inboard, straddling the fuel ring, and reaching far enough in
                # that the centrebody's coolant can enter it through the narrow
                # joint the spike shoulder leaves.
                fixed = _fit(p, r_bore + 1.1, r_ox - 0.5 * d_ox - 3.0,
                             x_face - wall, head_velocity)
            elif p.name == "ox_dome":
                # Outboard, clear of the fuel manifold and of the ox orifice
                # ring, which it feeds through radial ports rather than by
                # sitting on top of.
                #
                # Its inner radius is set by the cowl's coolant, not by the
                # oxidiser. The cowl discharges at r 99 and has to run axially
                # down to the fuel manifold; the dome is a continuous ring, so
                # anything inside its radial band crosses it, and fuel crossing
                # the oxidiser dome is the one failure this whole layout exists
                # to prevent. The dome starts outboard of that corridor.
                fixed = _fit(p, _cowl_discharge(design)[1] + 3.0,
                             a.flange_radius - wall, x_face - wall, head_velocity)
            else:
                continue
            out.plenums[out.plenums.index(p)] = fixed

        fm = next(q for q in out.plenums if q.name == "head_fuel_manifold")
        od = next(q for q in out.plenums if q.name == "ox_dome")
        gap = (od.r_inner_mm) - (fm.r_inner_mm + 2.0 * fm.half_r_mm)
        if gap < wall:
            out.notes.append(
                f"only {gap:.1f} mm of metal between the fuel manifold and the "
                f"oxidiser dome. They must not meet.")
        need = 2.0 * max(fm.half_x_mm, od.half_x_mm) + 2.0 * wall
        if need > a.structure.head_thickness_mm:
            out.notes.append(
                f"head disc is {a.structure.head_thickness_mm:.0f} mm thick and its "
                f"manifolds need {need:.0f} mm. Raise geometry.head_thickness_mm.")

    for p in out.plenums:
        if not p.self_supporting():
            out.notes.append(
                f"{p.name} roof at {p.roof_angle_deg:.0f} deg will sag: "
                f"raise the aspect ratio")
    return out


def report(md: ManifoldDesign) -> str:
    rows = [f"{'plenum':20s} {'x':>8s} {'r_in':>7s} {'section':>13s} "
            f"{'area':>8s} {'roof':>6s} {'flow':>9s} {'feeds':>6s}",
            "-" * 88]
    for p in md.plenums:
        rows.append(
            f"{p.name:20s} {p.x_mm:8.1f} {p.r_inner_mm:7.1f} "
            f"{2 * p.half_x_mm:5.1f}x{2 * p.half_r_mm:<6.1f} "
            f"{p.area_mm2:8.1f} {p.roof_angle_deg:5.0f}d "
            f"{p.mass_flow * 1e3:7.0f} g/s {p.feeds:6d}")
    if md.lugs:
        L = md.lugs
        rows.append("")
        rows.append(f"mounts: {L.count} lugs, {L.hole_diameter_mm:.0f} mm holes on a "
                    f"{2 * L.hole_radius_mm:.0f} mm circle, {L.thickness_mm:.0f} mm thick")
        rows.append(f"  {L.design_load_n / 1e3:.1f} kN per lug   "
                    f"bearing {L.bearing_stress_pa / 1e6:.0f} MPa "
                    f"({'ok' if L.bearing_ok else 'OVER'})   "
                    f"tear-out {L.shear_stress_pa / 1e6:.0f} MPa, edge "
                    f"{L.edge_distance_mm:.1f} mm "
                    f"({'ok' if L.tear_out_ok else 'OVER'})")
    for n in md.notes:
        rows.append(f"  note: {n}")
    return "\n".join(rows)


import numpy as np  # noqa: E402  (used by design_manifolds)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _discharge_band(design, part: str):
    """
    Radial band where a jacket's channels open onto the joint face.

    This is where the coolant physically arrives, and every transfer feature
    has to be built around it rather than around a nominal wall radius.
    """
    import numpy as np
    from mesh_solid import centrebody_channels, cowl_channels
    maker = cowl_channels if part == "cowl" else centrebody_channels
    c = maker(design.assembly, design.circuits[part].channel)
    w = float(np.interp(c.x_start, c.wall_x, c.wall_r))
    sign = 1.0 if c.outward else -1.0
    lo = w + sign * c.hot_wall_mm
    hi = lo + sign * c.height_mm
    return (min(lo, hi), max(lo, hi))


def _cowl_discharge(design):
    return _discharge_band(design, "cowl")


def plenum_feeding(md: ManifoldDesign, radius_mm: float, depth_mm: float = 1.0):
    """
    The head plenum a ring of orifices at this radius actually opens into.

    By geometry, not by name. An orifice does not care which manifold was meant
    to feed it -- it connects to whatever void lies at its radius, and getting
    that wrong crosses the propellants rather than merely failing to plumb them.
    """
    found = [p for p in md.plenums
             if p.name != "cowl_inlet_ring" and p.reaches(radius_mm, depth_mm)]
    return found[0] if len(found) == 1 else None


def orifice_start_x(md: ManifoldDesign, radius_mm: float, bore_mm: float,
                    fallback: float) -> float:
    """
    Where an orifice must begin so that it opens into the plenum feeding it.

    Deep enough that the whole bore is inside the section, not merely touching
    its tip. Touching is not connecting: the diamond tapers to a rounded point,
    so a hole that just crosses the extreme face meets a few tenths of a
    millimetre of cavity and rounds shut at any real voxel size, leaving the
    plenum sealed and full of powder.

    Measured at the orifice's own radius, because the section reaches furthest
    aft at its centre radius and less everywhere else.
    """
    p = plenum_feeding(md, radius_mm, depth_mm=bore_mm)
    if p is None:
        return fallback
    x = p.x_admitting(radius_mm, 0.5 * bore_mm)
    return fallback if x is None else x


def geometry_features(design, md: ManifoldDesign, wall_mm: float | None = None):
    """
    Turn the sized manifolds and mounts into distance-field features, per part.

    Returns {part: {"bosses": [...], "plenums": [...], "lugs": [...],
    "holes": [...]}} ready to hand to the mesher. Bosses wrap each plenum in
    metal; the plenum is then hollowed inside the boss, which is why material is
    added before anything is subtracted.
    """
    from mesh_solid import HoleCut, LugAdd, PlenumCut, PortCut, RingBoss

    a = design.assembly
    wall = wall_mm if wall_mm is not None else a.structure.wall_thickness_mm
    out = {part: {"bosses": [], "plenums": [], "lugs": [], "holes": [],
                  "ports": []}
           for part in a.profiles}

    for p in md.plenums:
        if p.name == "cowl_inlet_ring":
            # a raised ring round the cowl, with the plenum hollowed inside it
            r_in = p.r_inner_mm
            out["cowl"]["bosses"].append(RingBoss(
                x_at=p.x_mm, r_inner=r_in - 1.0,
                r_outer=r_in + 2.0 * p.half_r_mm + wall,
                half_x=p.half_x_mm + wall))
            out["cowl"]["plenums"].append(PlenumCut(
                x_at=p.x_mm, r_inner=r_in + 0.5 * wall,
                half_x=p.half_x_mm, half_r=p.half_r_mm))
        else:
            # inside the head disc, which was thickened to hold them
            out["head"]["plenums"].append(PlenumCut(
                x_at=p.x_mm, r_inner=p.r_inner_mm,
                half_x=p.half_x_mm, half_r=p.half_r_mm))

    # ---- the feed paths ----
    #
    # Everything below exists because a plenum that nothing reaches is not a
    # manifold, it is a void full of powder. The port schedule used to name
    # three inlets and the geometry cut none of them: the oxidiser dome had no
    # way in at all, and the coolant leaving both jackets had no way from the
    # joint face to the injector.
    if design.injector is not None:
        from mesh_solid import injector_holes
        fallback = a.head_x - (wall + 3.0)
        inj = design.injector
        fm = next((q for q in md.plenums if q.name == "head_fuel_manifold"), None)
        od = next((q for q in md.plenums if q.name == "ox_dome"), None)
        half_pitch = math.pi / inj.n_elements

        # The fuel orifices sit under their manifold and start inside it. The
        # oxidiser orifices do not: their dome is outboard, so they start at
        # the ring of radial ports that feeds them.
        x_ox = od.x_mm if od is not None else fallback
        out["head"]["holes"].extend(injector_holes(
            a, inj,
            x_start_fuel=orifice_start_x(
                md, inj.fuel_ring_radius * 1e3, inj.d_fuel_mm, fallback),
            x_start_ox=x_ox))

        if od is not None:
            # The oxidiser inlet itself: radial bores through the rim into the
            # dome. There was no oxidiser inlet in the geometry at all -- the
            # schedule named one on the head end face at r 101, where an axial
            # hole misses the dome entirely and runs on into the injector.
            # Split six ways so no single bore is a bridge wider than the
            # process will span, and clocked off the bolt circle.
            out["head"]["ports"].append(PortCut(
                x_at=od.x_mm, diameter_mm=8.0, count=6,
                r_lo=od.r_inner_mm + 2.0, r_hi=a.flange_radius + 2.0,
                phase=math.pi / 12.0))

            # Radial ports from the dome in to the head of every oxidiser
            # orifice -- a coaxial post, built from the primitives to hand.
            out["head"]["ports"].append(PortCut(
                x_at=od.x_mm, diameter_mm=max(1.5, inj.d_ox_mm + 0.6),
                count=inj.n_elements,
                r_lo=inj.ox_ring_radius * 1e3 - 1.0,
                r_hi=od.r_inner_mm + 2.0, phase=half_pitch))

        if fm is not None and design.circuits:
            # Both jackets discharge onto the joint face; from there the fuel
            # has to reach the manifold under the injector.
            cowl_lo, cowl_hi = _discharge_band(design, "cowl")
            body_lo, body_hi = _discharge_band(design, "centrebody")
            n_x = inj.n_elements
            x_face = a.head_x

            # A groove across each jacket's face collects all of its channels,
            # so the transfer holes opposite need not line up with any of them.
            # Centred on the face, so it is open across its full width there
            # and tapers to a self-supporting point inside the part.
            out["cowl"]["plenums"].append(PlenumCut(
                x_at=x_face, r_inner=cowl_lo,
                half_x=max(2.5, 0.5 * (cowl_hi - cowl_lo)),
                half_r=0.5 * (cowl_hi - cowl_lo)))
            out["centrebody"]["plenums"].append(PlenumCut(
                x_at=x_face, r_inner=body_lo,
                half_x=max(2.5, 0.5 * (body_hi - body_lo)),
                half_r=0.5 * (body_hi - body_lo)))

            # The centrebody's coolant comes in close to the bore, where the
            # manifold already reaches, so it goes straight in.
            # Sized by what the joint leaves, not by the flow: the spike
            # shoulder and the bore are 3 mm apart, so the hole is as wide as
            # the discharge and no wider, and the count carries the flow.
            out["head"]["holes"].append(HoleCut(
                radius_mm=0.5 * (body_lo + body_hi),
                diameter_mm=body_hi - body_lo,
                count=max(n_x, 128),
                x_start=fm.x_mm - 0.4 * fm.half_x_mm, x_end=x_face + 1.0,
                phase=0.0, name="transfer_centrebody"))

            # The cowl's comes in at r 99 and has to cross to r 78. It cannot
            # do that anywhere the dome exists, because the dome is a
            # continuous ring: it runs axially down the corridor inboard of
            # the dome, then turns in through the one window where the
            # manifold has already begun and the dome has not.
            # Set outward from the discharge, so the web left between the hole
            # and the chamber stays above the process floor. Centred on the
            # discharge instead, a wider hole leaves 0.4 mm of metal between
            # the fuel and the combustion chamber.
            d_cowl = 1.9
            r_cowl = max(0.5 * (cowl_lo + cowl_hi),
                         a.chamber_outer_radius + 0.7 + 0.5 * d_cowl)
            x_turn = (0.5 * (fm.x_mm - fm.half_x_mm + od.x_mm - od.half_x_mm)
                      if od is not None else fm.x_mm - 0.8 * fm.half_x_mm)
            out["head"]["holes"].append(HoleCut(
                radius_mm=r_cowl, diameter_mm=d_cowl,
                count=n_x, x_start=x_turn - 1.5, x_end=x_face + 1.0,
                phase=0.0, name="transfer_cowl"))
            out["head"]["ports"].append(PortCut(
                x_at=x_turn, diameter_mm=2.0, count=n_x,
                r_lo=fm.r_inner_mm + 0.4 * fm.half_r_mm,
                r_hi=cowl_hi + 1.0, phase=0.0))

    if md.lugs is not None:
        L = md.lugs
        x_face = a.head_x - a.structure.head_thickness_mm
        x_mid = x_face + 0.5 * L.thickness_mm
        out["head"]["lugs"].append(LugAdd(
            count=L.count, x_at=x_mid, half_x=0.5 * L.thickness_mm,
            r_inner=L.r_inner_mm - 2.0, r_outer=L.r_outer_mm,
            half_width_deg=0.5 * L.width_deg, phase=math.radians(45.0)))
        out["head"]["holes"].append(HoleCut(
            radius_mm=L.hole_radius_mm, diameter_mm=L.hole_diameter_mm,
            count=L.count, x_start=x_mid - L.thickness_mm,
            x_end=x_mid + L.thickness_mm, phase=math.radians(45.0),
            name="mount_hole"))
    return out
