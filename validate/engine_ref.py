"""
Full engine assembly geometry, derived from the plug contour.

Authoritative alongside contour_ref.py. The C# in ../model/SpikeGeometry.cs is a
port of this file; if the two disagree, this one is right.

What this file adds to the bare spike
-------------------------------------
contour_ref.py stops at the supersonic plug surface. An engine also needs the
annular combustion chamber that feeds it, the cowl that forms the outer side of
the throat, and enough structure to bolt the thing down. All of it is derived
from the same descriptive input; nothing here is drawn.

The one non-obvious construction is the cowl inner wall through the contraction.
A straight cone from the chamber radius to the lip looks reasonable and is
wrong: it pinches the flow to roughly 0.6 * A_t about half a millimetre upstream
of the design throat, so the engine would choke in the wrong place and the
expansion ratio the spike was built for would never be reached.

So the wall is not drawn, it is solved. At each station the offset direction
rotates from radial -- where the inner body is a cylinder and the normal is
radial -- to the sonic-line direction at the throat. Given that direction theta
and a target area A, the offset distance d is the positive root of

    pi * (2*r0 + d*cos(theta)) * d = A            (frustum between the walls)

    =>  d = (-r0 + sqrt(r0^2 + cos(theta)*A/pi)) / cos(theta)

Feed it a monotonically decreasing area schedule and the contraction cannot
choke early, by construction rather than by luck.

The payoff is a self-check as strong as spike closure: at the last station
theta = nu_e and A = A_t, and the solved point lands on the cowl lip at
(lip_x, r_e) to machine precision. Nothing forces that. It falls out of the
Angelino contour and the area schedule agreeing about where the throat is. That
is test_contraction_lands_on_the_lip, and it is the assertion to trust when you
change anything in here.

Coordinates: x along the axis, increasing downstream. x = 0 is the spike
shoulder, so the chamber and cowl live at negative x and the lip at small
positive x. r is radius from the axis. All lengths mm, all angles radians.

Out of scope, deliberately, per CLAUDE.md: no combustion modelling, no injector
orifice pattern, no cooling channels, no thrust numbers. The head disc is a
blank closure, not an injector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from contour_ref import PlugContour, build_contour


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _smoothstep(t: np.ndarray | float):
    """C1 blend, zero slope at both ends. Keeps the wall tangent-continuous."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vertex_normals(x: np.ndarray, r: np.ndarray, outward: bool):
    """
    Unit normals at each vertex of a meridional polyline.

    Averaged from the two adjacent segment normals so that a corner -- the spike
    shoulder is the one that matters -- produces a single mitred direction
    instead of two overlapping offset segments.
    """
    dx = np.diff(x)
    dr = np.diff(r)
    seg = np.hypot(dx, dr)
    seg[seg < 1e-15] = 1e-15

    # segment normal: (-dr, dx) points away from the axis for a left-to-right walk
    sign = 1.0 if outward else -1.0
    nx = sign * (-dr / seg)
    nr = sign * (dx / seg)

    vx = np.empty_like(x)
    vr = np.empty_like(r)
    vx[0], vr[0] = nx[0], nr[0]
    vx[-1], vr[-1] = nx[-1], nr[-1]
    vx[1:-1] = nx[:-1] + nx[1:]
    vr[1:-1] = nr[:-1] + nr[1:]

    n = np.hypot(vx, vr)
    n[n < 1e-15] = 1e-15
    return vx / n, vr / n


def _offset_polyline(x: np.ndarray, r: np.ndarray, distance: float, outward: bool):
    """Offset a meridional polyline by a constant distance along its normal."""
    nx, nr = _vertex_normals(x, r, outward)
    return x + distance * nx, r + distance * nr


def _distance_to_polyline(px: np.ndarray, pr: np.ndarray,
                          x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Shortest distance from each query point to a polyline, segment-wise.

    Vertex-only distance would overstate the clearance in the middle of a long
    segment, which is exactly where the cylindrical section lives, so the wall
    thickness check would pass on a wall that is too thin.
    """
    ax, ar = x[:-1], r[:-1]
    bx, br = x[1:], r[1:]
    dx, dr = bx - ax, br - ar
    ll = dx * dx + dr * dr
    ll[ll < 1e-30] = 1e-30

    t = ((px[:, None] - ax[None, :]) * dx[None, :] +
         (pr[:, None] - ar[None, :]) * dr[None, :]) / ll[None, :]
    t = np.clip(t, 0.0, 1.0)

    cx = ax[None, :] + t * dx[None, :]
    cr = ar[None, :] + t * dr[None, :]
    return np.min(np.hypot(px[:, None] - cx, pr[:, None] - cr), axis=1)


def _erode_polyline(x: np.ndarray, r: np.ndarray, distance: float):
    """
    Inward offset of a surface, with the folded-over part removed.

    A plain normal offset is only valid where the offset distance is smaller
    than the local radius of curvature. The spike shoulder is sharper than any
    sensible wall thickness, so the naive offset there turns itself inside out
    and the profile self-intersects.

    Rather than special-case the shoulder, drop every offset point that ended up
    closer to the original surface than it was supposed to be -- a folded point
    always is -- and then enforce monotone x. What survives is a chord across
    the corner, so the wall there is slightly thicker than asked for, never
    thinner. Erring thick is the safe direction for a pressure part.
    """
    ox, orr = _offset_polyline(x, r, distance, outward=False)

    valid = _distance_to_polyline(ox, orr, x, r) >= distance - 1e-6
    valid &= orr > 0.0
    if not valid.any():
        raise ValueError("wall_thickness_mm consumes the whole centrebody")
    ox, orr = ox[valid], orr[valid]

    # the surviving points must still march forward, or the profile folds again
    keep = np.ones(len(ox), dtype=bool)
    run = -np.inf
    for i, xv in enumerate(ox):
        if xv <= run:
            keep[i] = False
        else:
            run = xv
    return ox[keep], orr[keep]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChamberSpec:
    """Descriptive chamber input. Geometric ratios only -- no combustion."""
    contraction_ratio: float = 3.0      # A_chamber / A_throat
    chamber_length_mm: float = 18.0     # constant-area annular section
    converging_length_mm: float = 14.0  # axial length of the contraction


@dataclass(frozen=True)
class StructureSpec:
    """Wall and mounting dimensions."""
    wall_thickness_mm: float = 2.0
    lip_thickness_mm: float = 0.8       # blunt face at the cowl lip
    lip_taper_fraction: float = 0.35    # fraction of the cowl tapered to the lip
    base_cap_thickness_mm: float = 3.0  # solid plug behind the spike truncation
    head_thickness_mm: float = 4.0      # closure disc / mounting face
    flange_width_mm: float = 8.0        # radial width beyond the cowl outer wall


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """
    A closed meridional polygon. Revolved about the x axis it is a solid.

    Vertices run counter-clockwise in the (x, r) half-plane. r may touch zero,
    which the mesher turns into a fan rather than a degenerate quad strip.
    """
    name: str
    x: np.ndarray
    r: np.ndarray

    def __post_init__(self):
        """
        Drop repeated vertices, then force counter-clockwise winding.

        Zero-length edges have no direction, so every downstream consumer --
        normals, self-intersection, the mesher -- has to special-case them.
        Cheaper to make them impossible here.

        The mesher derives facet normals from vertex order, so a clockwise
        profile revolves into a solid that is inside out. Slicers and STL
        viewers disagree about how to repair that, so it is fixed here once.
        """
        x = np.asarray(self.x, dtype=float)
        r = np.asarray(self.r, dtype=float)
        keep = np.ones(len(x), dtype=bool)
        keep[1:] = (np.abs(np.diff(x)) > 1e-12) | (np.abs(np.diff(r)) > 1e-12)
        if len(x) > 1 and abs(x[-1] - x[0]) < 1e-12 and abs(r[-1] - r[0]) < 1e-12:
            keep[-1] = False          # the closing edge is implicit
        object.__setattr__(self, "x", x[keep])
        object.__setattr__(self, "r", r[keep])

        if self.revolved_volume < 0.0:
            object.__setattr__(self, "x", self.x[::-1].copy())
            object.__setattr__(self, "r", self.r[::-1].copy())

    @property
    def closed_area(self) -> float:
        """Shoelace area of the meridional section, mm^2. Positive if CCW."""
        return 0.5 * float(np.sum(self.x * np.roll(self.r, -1) - np.roll(self.x, -1) * self.r))

    @property
    def revolved_volume(self) -> float:
        """
        Pappus / divergence-theorem volume of the solid of revolution, mm^3.

        V = 2*pi * integral(r * dA) over the meridional section, evaluated on the
        boundary as pi/3 * sum over edges of (r0^2 + r0*r1 + r1^2) * (x1 - x0).
        """
        x0, r0 = self.x, self.r
        x1, r1 = np.roll(self.x, -1), np.roll(self.r, -1)
        return float(math.pi / 3.0 * np.sum((r0 * r0 + r0 * r1 + r1 * r1) * (x1 - x0)))


@dataclass(frozen=True)
class EngineAssembly:
    contour: PlugContour
    chamber: ChamberSpec
    structure: StructureSpec

    # flow surfaces
    inner_wall_x: np.ndarray            # centrebody side, head to spike tip
    inner_wall_r: np.ndarray
    outer_wall_x: np.ndarray            # cowl side, head to lip
    outer_wall_r: np.ndarray
    cavity_x: np.ndarray                # centrebody internal cavity, head to base cap
    cavity_r: np.ndarray
    cowl_outer_x: np.ndarray            # cowl outer skin, head to lip
    cowl_outer_r: np.ndarray

    # solids
    profiles: dict = field(default_factory=dict)

    # derived stations and areas
    head_x: float = 0.0
    converging_start_x: float = 0.0
    shoulder_radius: float = 0.0
    chamber_outer_radius: float = 0.0
    chamber_area: float = 0.0
    flange_radius: float = 0.0

    @property
    def overall_length(self) -> float:
        return float(self.inner_wall_x[-1]) - (self.head_x - self.structure.head_thickness_mm)

    @property
    def overall_radius(self) -> float:
        return self.flange_radius


def _contraction_wall(
    c: PlugContour,
    chamber_area: float,
    converging_length_mm: float,
    n: int = 200,
    corner_fan_start: float = 0.65,
):
    """
    Cowl inner surface through the contraction, solved from an area schedule.

    Two phases, because the inner body has a convex corner at the shoulder and a
    plain normal offset cannot turn a corner:

      A. s <= corner_fan_start -- the anchor slides along the cylinder and the
         offset is radial. This is an ordinary parallel offset.
      B. s >  corner_fan_start -- the anchor is pinned at the shoulder (0, r0)
         and the offset direction fans from radial round to the sonic line. This
         is the standard corner fan of an offset curve, and it is what keeps the
         wall from cutting back across the spike.

    Anchoring phase B on the cylinder instead -- letting x_anchor run past the
    shoulder -- looks equivalent and is not. It pinches the duct to roughly
    0.47 * A_t just aft of the shoulder, which is a worse choke than the naive
    straight cone it was meant to replace.

    Returns (x, r, area), area being the prescribed frustum area at each station.
    """
    if not 0.0 < corner_fan_start < 1.0:
        raise ValueError("corner_fan_start must be in (0, 1)")

    r0 = c.r[0]
    nu_e = c.throat_angle
    a_t = c.throat_area

    s = np.linspace(0.0, 1.0, n)
    area = chamber_area + (a_t - chamber_area) * _smoothstep(s)

    # phase A: anchor walks the cylinder to the shoulder and stops there
    x_anchor = -converging_length_mm * (1.0 - np.minimum(s / corner_fan_start, 1.0))

    # phase B: direction fans about the shoulder, zero until the fan begins
    fan = np.clip((s - corner_fan_start) / (1.0 - corner_fan_start), 0.0, 1.0)
    theta = nu_e * _smoothstep(fan)

    cos_t = np.cos(theta)
    # positive root of  cos(theta)*d^2 + 2*r0*d - area/pi = 0
    d = (-r0 + np.sqrt(r0 * r0 + cos_t * area / math.pi)) / cos_t

    x = x_anchor + d * np.sin(theta)
    r = r0 + d * cos_t
    return x, r, area


def build_assembly(
    contour: PlugContour,
    chamber: ChamberSpec | None = None,
    structure: StructureSpec | None = None,
    n_contraction: int = 200,
) -> EngineAssembly:
    """Assemble the full engine from a plug contour plus descriptive input."""
    chamber = chamber or ChamberSpec()
    structure = structure or StructureSpec()

    if chamber.contraction_ratio <= 1.0:
        raise ValueError("contraction_ratio must exceed 1")
    if chamber.converging_length_mm <= 0.0 or chamber.chamber_length_mm <= 0.0:
        raise ValueError("chamber and converging lengths must be positive")

    c = contour
    w = structure.wall_thickness_mm
    r0 = c.r[0]
    a_t = c.throat_area
    a_ch = chamber.contraction_ratio * a_t

    # chamber outer radius from the annular area, inner body being a cylinder
    r_ch = math.sqrt(r0 * r0 + a_ch / math.pi)

    x_conv0 = -chamber.converging_length_mm
    x_head = x_conv0 - chamber.chamber_length_mm

    if w >= r0:
        raise ValueError("wall_thickness_mm must be smaller than the shoulder radius")

    # ---------------- flow surfaces ----------------

    # inner: cylinder from the head to the shoulder, then the Angelino spike
    n_cyl = max(2, int(abs(x_head) * 4))
    xi_cyl = np.linspace(x_head, 0.0, n_cyl, endpoint=False)
    inner_x = np.concatenate([xi_cyl, np.asarray(c.x, dtype=float)])
    inner_r = np.concatenate([np.full(n_cyl, r0), np.asarray(c.r, dtype=float)])

    # outer: constant-area chamber, then the solved contraction to the lip
    conv_x, conv_r, conv_area = _contraction_wall(
        c, a_ch, chamber.converging_length_mm, n_contraction
    )
    n_ch = max(2, int(chamber.chamber_length_mm * 4))
    xo_ch = np.linspace(x_head, x_conv0, n_ch, endpoint=False)
    outer_x = np.concatenate([xo_ch, conv_x])
    outer_r = np.concatenate([np.full(n_ch, r_ch), conv_r])

    # ---------------- centrebody solid ----------------

    x_tip = float(inner_x[-1])
    r_tip = float(inner_r[-1])
    x_cav_end = x_tip - structure.base_cap_thickness_mm

    cav_x, cav_r = _erode_polyline(inner_x, inner_r, w)
    keep = cav_x <= x_cav_end
    if not keep.any():
        raise ValueError("base_cap_thickness_mm leaves no room for the internal cavity")
    cav_x, cav_r = cav_x[keep], cav_r[keep]
    r_bore = float(cav_r[0])

    centrebody = Profile(
        "centrebody",
        np.concatenate([
            inner_x,                                    # outer surface, head -> tip
            [x_tip, x_cav_end],                         # base face to the axis, then forward
            cav_x[::-1],                                # cavity wall, aft -> forward
        ]),                                             # head face closes implicitly
        np.concatenate([
            inner_r,
            [0.0, 0.0],
            cav_r[::-1],
        ]),
    )

    # ---------------- cowl solid ----------------

    # wall tapers to a thinner blunt lip over the aft part of the cowl
    n_out = len(outer_x)
    taper_n = max(2, int(structure.lip_taper_fraction * n_out))
    thick = np.full(n_out, w)
    tail = _smoothstep(np.linspace(0.0, 1.0, taper_n))
    thick[-taper_n:] = w + (structure.lip_thickness_mm - w) * tail

    onx, onr = _vertex_normals(outer_x, outer_r, outward=True)
    cowl_out_x = outer_x + thick * onx
    cowl_out_r = outer_r + thick * onr

    cowl = Profile(
        "cowl",
        np.concatenate([outer_x, cowl_out_x[::-1]]),
        np.concatenate([outer_r, cowl_out_r[::-1]]),
    )

    # ---------------- head closure and flange ----------------

    r_flange = float(cowl_out_r[0]) + structure.flange_width_mm
    x_h1 = x_head
    x_h0 = x_head - structure.head_thickness_mm

    head = Profile(
        "head",
        np.array([x_h0, x_h1, x_h1, x_h0]),
        np.array([r_bore, r_bore, r_flange, r_flange]),
    )

    return EngineAssembly(
        contour=c,
        chamber=chamber,
        structure=structure,
        inner_wall_x=inner_x,
        inner_wall_r=inner_r,
        outer_wall_x=outer_x,
        outer_wall_r=outer_r,
        cavity_x=cav_x,
        cavity_r=cav_r,
        cowl_outer_x=cowl_out_x,
        cowl_outer_r=cowl_out_r,
        profiles={"centrebody": centrebody, "cowl": cowl, "head": head},
        head_x=x_head,
        converging_start_x=x_conv0,
        shoulder_radius=r0,
        chamber_outer_radius=r_ch,
        chamber_area=a_ch,
        flange_radius=r_flange,
    )


# --------------------------------------------------------------------------
# flow area, measured rather than prescribed
# --------------------------------------------------------------------------

def measured_flow_area(a: EngineAssembly, n: int = 400):
    """
    Flow area along the duct, measured from the built geometry.

    For each point on the cowl inner wall, take the shortest distance to the
    centrebody surface and sweep it as a frustum. This deliberately ignores the
    area schedule that produced the wall: it is the independent check that the
    duct really does contract monotonically to A_t at the throat and not before.
    """
    ix = np.asarray(a.inner_wall_x)
    ir = np.asarray(a.inner_wall_r)

    idx = np.linspace(0, len(a.outer_wall_x) - 1, n).astype(int)
    ox = np.asarray(a.outer_wall_x)[idx]
    orr = np.asarray(a.outer_wall_r)[idx]

    areas = np.empty(n)
    for i in range(n):
        d = np.hypot(ix - ox[i], ir - orr[i])
        k = int(np.argmin(d))
        areas[i] = math.pi * (orr[i] + ir[k]) * d[k]
    return ox, areas


def duct_state(a: EngineAssembly, n: int = 240):
    """
    Local flow state along each cooled wall, for the thermal model.

    Returns two dicts, "cowl" and "centrebody", each carrying x, r, area_ratio
    and mach. Area ratio is measured off the built geometry by the same
    shortest-gap frustum used to prove the duct chokes at the throat, so the
    thermal model and the geometry check cannot drift apart.

    Mach comes from the area ratio on the subsonic branch upstream of the
    throat. Downstream, on the spike, the contour already carries its own Mach
    distribution from the Angelino construction and that is used directly --
    inverting the area ratio there would throw away the better answer.
    """
    from contour_ref import mach_from_area_ratio_subsonic

    c = a.contour
    gamma_hint = None  # gamma is a gas property; the caller supplies it
    ox, areas = measured_flow_area(a, n=n)
    ar = areas / c.throat_area
    orr = np.interp(ox, np.asarray(a.outer_wall_x), np.asarray(a.outer_wall_r))
    return {
        "cowl": {"x": ox, "r": orr, "area_ratio": ar},
        "centrebody": {
            "x": np.asarray(a.inner_wall_x, dtype=float),
            "r": np.asarray(a.inner_wall_r, dtype=float),
        },
    }


def wall_flow_state(a: EngineAssembly, gamma: float, n: int = 240):
    """
    x, r, area ratio and Mach along the cowl and the centrebody.

    The centrebody spans both regimes: subsonic from the head to the shoulder,
    then the supersonic plug surface, where the contour's own Mach values are
    authoritative.
    """
    from contour_ref import mach_from_area_ratio_subsonic

    c = a.contour
    d = duct_state(a, n=n)

    cowl = d["cowl"]
    cowl_mach = np.array([mach_from_area_ratio_subsonic(max(e, 1.0), gamma)
                          for e in cowl["area_ratio"]])

    ix = np.asarray(a.inner_wall_x, dtype=float)
    ir = np.asarray(a.inner_wall_r, dtype=float)
    n_spike = len(c.x)
    n_up = len(ix) - n_spike

    # upstream of the shoulder the centrebody sees the same duct as the cowl
    up_ar = np.interp(ix[:n_up], cowl["x"], cowl["area_ratio"])
    up_mach = np.array([mach_from_area_ratio_subsonic(max(e, 1.0), gamma) for e in up_ar])

    spike_mach = np.asarray(c.mach, dtype=float)
    spike_ar = np.array([area_ratio_for(m, gamma) for m in spike_mach])

    return {
        "cowl": {"x": cowl["x"], "r": cowl["r"],
                 "area_ratio": cowl["area_ratio"], "mach": cowl_mach},
        "centrebody": {
            "x": ix, "r": ir,
            "area_ratio": np.concatenate([up_ar, spike_ar]),
            "mach": np.concatenate([up_mach, spike_mach]),
        },
    }


def area_ratio_for(mach: float, gamma: float) -> float:
    from contour_ref import area_ratio
    return area_ratio(mach, gamma)


def assembly_from_spec(spec: dict) -> EngineAssembly:
    """Build the whole engine straight from a parsed spec/*.json."""
    n = spec["nozzle"]
    contour = build_contour(
        expansion_ratio=n["expansion_ratio"],
        exit_radius_mm=n["exit_radius_mm"],
        gamma=spec["gas"]["gamma"],
        n_points=n.get("contour_points", 300),
        truncate_fraction=n.get("truncate_fraction", 1.0),
    )
    ch = spec.get("chamber", {})
    g = spec.get("geometry", {})
    return build_assembly(
        contour,
        ChamberSpec(
            contraction_ratio=ch.get("contraction_ratio", 3.0),
            chamber_length_mm=ch.get("chamber_length_mm", 18.0),
            converging_length_mm=ch.get("converging_length_mm", 14.0),
        ),
        StructureSpec(
            wall_thickness_mm=g.get("wall_thickness_mm", 2.0),
            lip_thickness_mm=g.get("lip_thickness_mm", 0.8),
            base_cap_thickness_mm=g.get("base_cap_thickness_mm", 3.0),
            head_thickness_mm=g.get("head_thickness_mm", 4.0),
            flange_width_mm=g.get("flange_width_mm", 8.0),
        ),
    )
