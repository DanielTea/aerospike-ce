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
                     lug_count: int = 4) -> ManifoldDesign:
    """
    Every plenum and mount the engine needs, derived from the solved design.
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
        r_skin = float(np.interp(port.x_at, np.asarray(a.cowl_outer_x),
                                 np.asarray(a.cowl_outer_r))) if False else port.r_hi - 5.0
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

    # A plenum has to fit inside the metal that carries it, and it has to sit
    # around the ring of orifices it exists to feed. Those are two constraints,
    # and satisfying only the first is what went wrong here: sized on flow alone
    # the fuel manifold and the oxidiser dome span r 67 to 135 mm between them,
    # across a head disc that only runs 70 to 123, so they cut it clean in half.
    # Packing them outward from the bore to fix that left the oxidiser dome at
    # r 98 to 117 while its own orifices sat at r 88 -- the dome fed nothing,
    # and the orifices pointed instead at the radius the *fuel* manifold
    # occupies. Nothing downstream would have caught that: the part is
    # watertight either way, and it took the sealed-void check noticing 325 cm3
    # of powder with no way out.
    #
    # So each band is built around its own ring, and the wall between two bands
    # falls halfway between the two rings.
    r_bore = float(a.cavity_r[0])
    wall = a.structure.wall_thickness_mm
    r_lo, r_hi = r_bore + wall, a.flange_radius - wall
    head_plenums = [p for p in out.plenums if p.name != "cowl_inlet_ring"]

    rings: dict[str, float] = {}
    if design.injector is not None:
        rings["head_fuel_manifold"] = design.injector.fuel_ring_radius * 1e3
        rings["ox_dome"] = design.injector.ox_ring_radius * 1e3

    if head_plenums and all(p.name in rings for p in head_plenums):
        ordered = sorted(head_plenums, key=lambda p: rings[p.name])
        radii = [rings[p.name] for p in ordered]

        edges = [r_lo]
        for lo, hi in zip(radii, radii[1:]):
            if hi - lo <= 2.0 * wall:
                out.notes.append(
                    f"orifice rings at r {lo:.1f} and {hi:.1f} mm are "
                    f"{hi - lo:.1f} mm apart, which leaves no room for a "
                    f"{wall:.1f} mm wall between the manifolds feeding them. "
                    f"Spread the injector rings or thin the wall.")
            edges.append(0.5 * (lo + hi))
        edges.append(r_hi)

        # Depth available inside the disc, wall either side.
        max_half_x = max(0.5 * (a.structure.head_thickness_mm - 2.0 * wall), 1.0)

        for i, p in enumerate(ordered):
            inner = edges[i] + (0.5 * wall if i else 0.0)
            outer = edges[i + 1] - (0.5 * wall if i < len(ordered) - 1 else 0.0)
            # Centred on the ring, so the section is deepest exactly where the
            # orifices meet it rather than tapering to nothing there.
            half_r = max(min(radii[i] - inner, outer - radii[i]), 1.0)
            half_x = min(max(p.area_mm2 / (2.0 * half_r), half_r), max_half_x)

            # The area is now whatever fits, so the velocity is whatever that
            # area implies. Reporting the velocity that was asked for would be
            # reporting a number the geometry does not have -- and the area is
            # the filleted section's, not the sharp diamond's, for the same
            # reason.
            from mesh_solid import plenum_section_area_mm2
            area_m2 = plenum_section_area_mm2(half_x, half_r) * 1e-6
            velocity = 0.5 * p.mass_flow / (p.density * area_m2)

            fixed = Plenum(
                name=p.name, x_mm=p.x_mm, r_inner_mm=radii[i] - half_r,
                half_x_mm=half_x, half_r_mm=half_r,
                mass_flow=p.mass_flow, velocity=velocity,
                density=p.density, feeds=p.feeds)
            out.plenums[out.plenums.index(p)] = fixed

            if velocity > p.velocity * 1.05:
                out.notes.append(
                    f"{p.name} runs at {velocity:.1f} m/s, not the "
                    f"{p.velocity:.1f} asked for: the ring is {2 * half_r:.1f} mm "
                    f"wide because that is the space between its orifice ring and "
                    f"its neighbour's, and {2 * half_x:.0f} mm deep because that "
                    f"is the disc. A manifold this fast distributes less evenly.")
            if not fixed.reaches(radii[i], depth_mm=1.0):
                out.notes.append(
                    f"{p.name} is only {2 * fixed.half_x_at(radii[i]):.2f} mm deep "
                    f"at r {radii[i]:.1f} mm where its orifices meet it. They will "
                    f"not connect.")

    # The head has to be thick enough to contain the plenums it carries, plus a
    # wall either side. Sizing the disc by eye and then discovering the dome
    # does not fit inside it is the manifold equivalent of the flange being too
    # narrow for its own bolts.
    in_head = [p for p in out.plenums if abs(p.x_mm - a.head_x) < a.structure.head_thickness_mm + 40.0
               and p.name != "cowl_inlet_ring"]
    if in_head:
        need = 2.0 * max(p.half_x_mm for p in in_head) + 2.0 * a.structure.wall_thickness_mm
        if need > a.structure.head_thickness_mm:
            out.notes.append(
                f"head disc is {a.structure.head_thickness_mm:.0f} mm thick and the "
                f"manifolds inside it need {need:.0f} mm. Raise "
                f"geometry.head_thickness_mm, or run the plenums slower so they "
                f"shrink.")

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
    from mesh_solid import HoleCut, LugAdd, PlenumCut, RingBoss

    a = design.assembly
    wall = wall_mm if wall_mm is not None else a.structure.wall_thickness_mm
    out = {part: {"bosses": [], "plenums": [], "lugs": [], "holes": []}
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

    # The orifices belong with the plenums they feed, so they are built here
    # where both are known rather than from an assumed offset elsewhere.
    if design.injector is not None:
        from mesh_solid import injector_holes
        fallback = a.head_x - (wall + 3.0)
        out["head"]["holes"].extend(injector_holes(
            a, design.injector,
            x_start_fuel=orifice_start_x(
                md, design.injector.fuel_ring_radius * 1e3,
                design.injector.d_fuel_mm, fallback),
            x_start_ox=orifice_start_x(
                md, design.injector.ox_ring_radius * 1e3,
                design.injector.d_ox_mm, fallback)))

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
