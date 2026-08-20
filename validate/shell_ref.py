"""
External shell features: channel ribs and mounting legs.

Two things that change how the engine looks, and both earn their place.

Ribs
----
A regeneratively cooled wall can close its channels with a smooth cylindrical
jacket, which is what this model did, or it can let the lands stand proud as
ribs and close each channel with a thin skin between them. The second is what
LEAP 71's printed chambers look like, and it is not styling:

- the rib is where the land already is, so it costs nothing to place
- it stiffens a thin shell against buckling in exactly the hoop direction the
  pressure loads it
- it triples the external area, which matters for a chamber that radiates
- the skin over the channel can then be thinner than a jacket carrying hoop
  load on its own, so the wall gets lighter, not heavier

The constraint is printability: the rib flanks have to stand at or above the
process angle or the shell grows overhangs once per channel, several hundred
times over.

Legs
----
Flat pads on the head flange are the simplest mount and they put the fixing
holes at the widest part of the engine. Legs carry the load down from higher up
the chamber and let the holes sit outboard on a wider circle, which is what
takes the overturning moment on a gimballed or side-loaded engine.

They splay, and the splay is bounded by the same process angle: a leg leaning
further than that from the build direction is an overhang along its whole
length.

All lengths mm, angles degrees at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RibRing:
    """Lands standing proud of the channel closure, once per channel."""
    count: int
    height_mm: float            # how far the rib stands above the skin
    skin_mm: float              # closure over the channel itself
    root_width_mm: float        # rib width where it leaves the skin
    flank_deg: float            # flank angle from the shell surface
    base_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    base_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hot_wall_mm: float = 0.0
    channel_height_mm: float = 0.0
    x_start: float = 0.0
    x_end: float = 0.0
    outward: bool = True
    phase: float = 0.0

    @property
    def self_supporting(self) -> bool:
        return self.flank_deg >= 45.0 - 1e-9

    @property
    def area_ratio(self) -> float:
        """
        External area with ribs over the area of a plain cylinder.

        Each pitch gains two flanks and a crown in place of a flat arc.
        """
        pitch = self.root_width_mm + max(self.skin_mm, 0.1)
        flank = self.height_mm / max(math.sin(math.radians(self.flank_deg)), 1e-6)
        return (2.0 * flank + self.root_width_mm + self.skin_mm) / max(pitch, 1e-6)


@dataclass(frozen=True)
class Leg:
    """One splayed mounting leg, from the chamber down to a footed pad."""
    count: int
    x_top_mm: float
    r_top_mm: float
    x_foot_mm: float
    r_foot_mm: float
    width_deg: float
    thickness_mm: float
    hole_diameter_mm: float
    pad_radius_mm: float
    design_load_n: float
    bearing_stress_pa: float
    buckling_load_n: float
    phase: float = 0.0

    @property
    def splay_deg(self) -> float:
        """Lean from the build direction. Past the process angle it overhangs."""
        return math.degrees(math.atan2(abs(self.r_foot_mm - self.r_top_mm),
                                       abs(self.x_foot_mm - self.x_top_mm)))

    @property
    def length_mm(self) -> float:
        return math.hypot(self.x_foot_mm - self.x_top_mm,
                          self.r_foot_mm - self.r_top_mm)

    @property
    def self_supporting(self) -> bool:
        return self.splay_deg <= 45.0 + 1e-9

    @property
    def buckling_ok(self) -> bool:
        return self.buckling_load_n > self.design_load_n


def size_ribs(channel, base_x, base_r, x_start, x_end, outward=True,
              height_mm: float | None = None, skin_mm: float | None = None,
              flank_deg: float = 55.0) -> RibRing:
    """
    Turn the lands of a channel ring into ribs.

    The rib sits on the land, so its root width is the land width and its count
    is the channel count -- neither is a free choice. What is free is how far it
    stands proud and how thin the skin over the channel can get, and those trade
    against each other: a taller rib carries more of the hoop load, which is
    what lets the skin thin down.

    Flank angle defaults above the process minimum rather than at it. At exactly
    45 degrees every rib is marginal, and there are several hundred of them.

    These ribs are added to the existing jacket, not carved out of a thinned
    one. That is the conservative half of the idea: it buys the stiffness and
    the area without touching the shell the structural model has already
    signed off. The mass-saving half -- thinning the skin because the ribs now
    carry the hoop load -- needs structural_ref to credit them for it, and
    until it does, thinning the wall would be optimism rather than design.
    """
    if flank_deg < 45.0:
        raise ValueError(
            f"{flank_deg:.0f} degree flanks are under the process angle; with "
            f"{channel.n_channels} ribs that is an overhang several hundred "
            f"times over")

    pitch_at = 2.0 * math.pi * float(np.max(base_r)) / channel.n_channels
    land = max(pitch_at - channel.width_mm, channel.land_mm)
    skin = skin_mm if skin_mm is not None else max(0.4, 0.5 * channel.hot_wall_mm + 0.25)
    height = height_mm if height_mm is not None else max(1.0, 1.5 * channel.height_mm)

    return RibRing(
        count=channel.n_channels, height_mm=height, skin_mm=skin,
        root_width_mm=land, flank_deg=flank_deg,
        base_x=np.asarray(base_x, dtype=float), base_r=np.asarray(base_r, dtype=float),
        hot_wall_mm=channel.hot_wall_mm, channel_height_mm=channel.height_mm,
        x_start=x_start, x_end=x_end, outward=outward,
        phase=math.pi / channel.n_channels)      # ribs sit between channels


def size_legs(
    thrust_n: float,
    x_top_mm: float,
    r_top_mm: float,
    x_foot_mm: float,
    count: int = 4,
    safety: float = 2.0,
    foot_radius_mm: float | None = None,
    splay_deg: float | None = None,
    width_deg: float = 14.0,
    material_e_pa: float = 190e9,
    bearing_allowable_pa: float = 300e6,
) -> Leg:
    """
    Splayed legs carrying thrust from the chamber to a footed pad.

    Checked on three things: bearing at the pin, Euler buckling of the leg as a
    strut, and the splay staying inside the process angle. Buckling is the one
    that decides a slender leg and the one people forget -- a leg sized only on
    stress is comfortably strong and folds sideways.
    """
    if count < 3:
        raise ValueError("fewer than three legs cannot constrain the engine")

    drop = abs(x_foot_mm - x_top_mm)

    # Parameterise by where the feet land, not by how far the leg leans. A splay
    # angle is a local property and says nothing about footprint: 35 degrees
    # over a 158 mm drop puts the feet at r 211 mm, doubling the engine's
    # footprint and hanging them in space well outboard of the flange. What a
    # mount is actually specified by is its bolt circle.
    if foot_radius_mm is not None:
        r_foot = foot_radius_mm
        splay = math.degrees(math.atan2(r_foot - r_top_mm, drop))
        if splay > 45.0:
            raise ValueError(
                f"reaching r {r_foot:.0f} mm over a {drop:.0f} mm drop needs "
                f"{splay:.0f} degrees of splay, past what the process holds. "
                f"Start the leg further forward or bring the feet in.")
        if splay < 0.0:
            raise ValueError("the feet are inboard of where the legs start")
    else:
        splay = 35.0 if splay_deg is None else splay_deg
        if splay > 45.0:
            raise ValueError(
                f"{splay:.0f} degree splay leans further than the process holds; "
                f"the leg overhangs along its whole length")
        r_foot = r_top_mm + drop * math.tan(math.radians(splay))
    load = thrust_n * safety / count
    length = math.hypot(drop, r_foot - r_top_mm)

    for d in (6.0, 8.0, 10.0, 12.0, 16.0):
        thickness = max(6.0, load / (d * bearing_allowable_pa) * 1e6)
        bearing = load / (d * thickness * 1e-6)

        # Euler, pinned-pinned, bending about the weak (circumferential) axis
        width = math.radians(width_deg) * r_foot
        i_weak = width * thickness ** 3 / 12.0 * 1e-12          # m^4
        crit = math.pi ** 2 * material_e_pa * i_weak / (length * 1e-3) ** 2

        leg = Leg(
            count=count, x_top_mm=x_top_mm, r_top_mm=r_top_mm,
            x_foot_mm=x_foot_mm, r_foot_mm=r_foot, width_deg=width_deg,
            thickness_mm=thickness, hole_diameter_mm=d,
            pad_radius_mm=r_foot + 2.2 * d,
            design_load_n=load, bearing_stress_pa=bearing, buckling_load_n=crit,
            phase=math.radians(45.0))
        if leg.bearing_stress_pa <= bearing_allowable_pa and leg.buckling_ok:
            return leg
    return leg


def report(ribs: dict, legs: Leg | None) -> str:
    rows = [f"{'ribs on':12s} {'count':>6s} {'height':>7s} {'skin':>6s} "
            f"{'root':>6s} {'flank':>6s} {'area x':>7s}", "-" * 58]
    for name, r in ribs.items():
        rows.append(f"{name:12s} {r.count:6d} {r.height_mm:6.2f}m {r.skin_mm:5.2f}m "
                    f"{r.root_width_mm:5.2f}m {r.flank_deg:5.0f}d {r.area_ratio:6.2f}x")
    if legs is not None:
        rows.append("")
        rows.append(f"legs: {legs.count} at {legs.splay_deg:.0f} deg splay, "
                    f"{legs.length_mm:.0f} mm long, {legs.thickness_mm:.0f} mm thick, "
                    f"{legs.hole_diameter_mm:.0f} mm holes at r {legs.r_foot_mm:.0f} mm")
        rows.append(f"  {legs.design_load_n / 1e3:.1f} kN each   "
                    f"bearing {legs.bearing_stress_pa / 1e6:.0f} MPa   "
                    f"buckling {legs.buckling_load_n / 1e3:.1f} kN "
                    f"({'ok' if legs.buckling_ok else 'FOLDS'})   "
                    f"{'self-supporting' if legs.self_supporting else 'OVERHANGS'}")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def shell_features(design, rib_flank_deg: float = 55.0, leg_count: int = 4,
                   splay_deg: float = 35.0):
    """
    Ribs and legs as distance-field features, per part.

    Ribs ride each part's own outer surface -- the cowl's outer skin, the
    centrebody's gas-side wall taken inward -- so they sit on the lands rather
    than anywhere convenient.
    """
    from mesh_solid import LegAdd, RibAdd, centrebody_channels, cowl_channels

    a = design.assembly
    out = {part: {"ribs": [], "legs": []} for part in a.profiles}
    sized = {}

    for part, maker, base in (
            ("cowl", cowl_channels, (a.cowl_outer_x, a.cowl_outer_r)),
            ("centrebody", centrebody_channels, (a.cavity_x, a.cavity_r))):
        circuit = design.circuits.get(part)
        if circuit is None:
            continue
        cut = maker(a, circuit.channel)
        outward = part == "cowl"
        ring = size_ribs(cut, base[0], base[1], cut.x_start, cut.x_end,
                         outward=outward, flank_deg=rib_flank_deg)
        sized[part] = ring
        out[part]["ribs"].append(RibAdd(
            base_x=ring.base_x, base_r=ring.base_r, count=ring.count,
            height_mm=ring.height_mm, root_width_mm=ring.root_width_mm,
            flank_deg=ring.flank_deg, x_start=ring.x_start, x_end=ring.x_end,
            outward=outward, phase=ring.phase))

    # Feet a little outboard of the flange -- enough to take an overturning
    # moment, not so far that the engine needs its own postcode.
    leg = size_legs(
        design.chamber.thrust_vacuum,
        x_top_mm=0.5 * (a.head_x + a.converging_start_x),
        r_top_mm=float(max(a.cowl_outer_r)),
        x_foot_mm=a.head_x - a.structure.head_thickness_mm,
        count=leg_count,
        foot_radius_mm=1.18 * a.flange_radius)
    out["cowl"]["legs"].append(LegAdd(
        count=leg.count, x_top=leg.x_top_mm, r_top=leg.r_top_mm,
        x_foot=leg.x_foot_mm, r_foot=leg.r_foot_mm,
        thickness_mm=leg.thickness_mm, half_width_deg=0.5 * leg.width_deg,
        pad_radius_mm=leg.pad_radius_mm, phase=leg.phase))
    return out, sized, leg
