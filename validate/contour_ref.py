"""
Reference implementation of the Angelino approximate plug-nozzle contour.

This is the authoritative statement of the maths. The C# model in ../model must
reproduce these numbers. If the two disagree, this file is right and the C# is
wrong, unless you deliberately change the physics here first.

Derivation (full external-expansion annular plug, ideal spike reaching the axis)
-------------------------------------------------------------------------------
Throat is an annular gap at the outer lip, radius r_e, at x = 0. Flow expands
through a Prandtl-Meyer fan centred on the lip. Along the characteristic at
Mach M the flow is uniform, turned by (nu_e - nu(M)) from the axis, and the
characteristic itself sits at

    alpha(M) = nu_e - nu(M) + mu(M),      mu = asin(1/M)

measured from the axis. The velocity component normal to a Mach line is exactly
the local speed of sound, so mass conservation across the conical surface swept
by the characteristic segment (slant length l, radii r_e and r) gives

    pi * (r_e + r) * l = A_t * M * eps(M)

With r = r_e - l*sin(alpha), xi = l/r_e and A_t = pi*r_e^2/eps_e this reduces to

    xi^2 * sin(alpha) - 2*xi + M*eps(M)/eps_e = 0

    =>  xi = (1 - sqrt(1 - sin(alpha) * M * eps(M) / eps_e)) / sin(alpha)

    x = r_e * xi * cos(alpha)
    r = r_e * (1 - xi * sin(alpha))

Self-check: at M = M_e, alpha = mu_e, so the radicand vanishes, xi = M_e and
r = 0. The spike closes on the axis. At M = 1 the point lies upstream of the
lip, which is correct: the throat plane is inclined at nu_e.

All angles in radians, all lengths in mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------
# gas dynamics
# --------------------------------------------------------------------------

def area_ratio(mach: float, gamma: float) -> float:
    """Isentropic area ratio A/A* for a given Mach number."""
    if mach <= 0.0:
        raise ValueError("mach must be positive")
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    return (1.0 / mach) * ((2.0 / gp1) * (1.0 + 0.5 * gm1 * mach * mach)) ** (gp1 / (2.0 * gm1))


def prandtl_meyer(mach: float, gamma: float) -> float:
    """Prandtl-Meyer function nu(M) in radians. Zero at M = 1."""
    if mach < 1.0:
        raise ValueError("Prandtl-Meyer is only defined for supersonic flow")
    if mach == 1.0:
        return 0.0
    gp1 = gamma + 1.0
    gm1 = gamma - 1.0
    m2m1 = mach * mach - 1.0
    return (math.sqrt(gp1 / gm1) * math.atan(math.sqrt(gm1 / gp1 * m2m1))
            - math.atan(math.sqrt(m2m1)))


def mach_from_area_ratio(eps: float, gamma: float, tol: float = 1e-12) -> float:
    """Invert the area-Mach relation on the supersonic branch by bisection."""
    if eps < 1.0:
        raise ValueError("expansion ratio must be >= 1")
    lo, hi = 1.0, 2.0
    while area_ratio(hi, gamma) < eps:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("expansion ratio out of range")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid, gamma) < eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# contour
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlugContour:
    x: list[float]              # mm, monotonically increasing, starts at 0
    r: list[float]              # mm, monotonically decreasing
    mach: list[float]
    exit_mach: float
    throat_angle: float         # rad, nu_e: inclination of the throat plane
    exit_radius: float          # mm
    throat_area: float          # mm^2
    length: float               # mm, throat station to spike tip
    lip_x: float                # mm, axial station of the cowl lip

    @property
    def truncation_index(self) -> int:
        return len(self.x)

    # ----------------------------------------------------------------------
    # The throat is NOT a radial annulus. It is the sonic line running from the
    # cowl lip at (lip_x, exit_radius) to the first contour point at (0, r[0]),
    # inclined at nu_e from the radial direction. Measuring it radially instead
    # understates the area by a factor of about 2.5 at eps = 8, which is enough
    # to put the real choke point in the wrong place.
    # ----------------------------------------------------------------------

    @property
    def throat_slant_length(self) -> float:
        """Length of the sonic line from the lip to the spike, in mm."""
        return math.hypot(self.lip_x - self.x[0], self.exit_radius - self.r[0])

    @property
    def throat_area_from_geometry(self) -> float:
        """
        Annular throat area swept by the sonic line, pi*(r_e + r_0)*l.

        Must agree with `throat_area` (= pi*r_e^2/eps) to machine precision.
        That agreement is what proves `lip_x` is placed correctly, and it is the
        second load-bearing self-check in this file after spike closure.
        """
        return math.pi * (self.exit_radius + self.r[0]) * self.throat_slant_length

    @property
    def throat_inclination(self) -> float:
        """Angle of the sonic line from the axis, in radians. Equals pi/2 - nu_e."""
        return math.atan2(self.exit_radius - self.r[0], self.lip_x - self.x[0])


def build_contour(
    expansion_ratio: float,
    exit_radius_mm: float,
    gamma: float = 1.20,
    n_points: int = 200,
    truncate_fraction: float = 1.0,
) -> PlugContour:
    """
    Generate the plug contour.

    truncate_fraction < 1.0 chops the spike at that fraction of its ideal
    length, which is what every real engine does. A 20 percent spike recovers
    most of the performance at a fraction of the length and mass.
    """
    if not 0.0 < truncate_fraction <= 1.0:
        raise ValueError("truncate_fraction must be in (0, 1]")

    exit_mach = mach_from_area_ratio(expansion_ratio, gamma)
    nu_e = prandtl_meyer(exit_mach, gamma)

    xs: list[float] = []
    rs: list[float] = []
    ms: list[float] = []

    for i in range(n_points):
        # cluster points near the throat where curvature is highest
        t = i / (n_points - 1)
        mach = 1.0 + (exit_mach - 1.0) * (t ** 1.5)
        mach = min(mach, exit_mach)

        nu = prandtl_meyer(mach, gamma)
        mu = math.asin(min(1.0, 1.0 / mach))
        alpha = nu_e - nu + mu
        sin_a, cos_a = math.sin(alpha), math.cos(alpha)

        eps_m = area_ratio(mach, gamma)
        radicand = max(0.0, 1.0 - sin_a * mach * eps_m / expansion_ratio)
        xi = (1.0 - math.sqrt(radicand)) / sin_a

        xs.append(exit_radius_mm * xi * cos_a)
        rs.append(exit_radius_mm * (1.0 - xi * sin_a))
        ms.append(mach)

    # Shift so the first contour point sits at x = 0. The Prandtl-Meyer fan is
    # centred on the cowl lip, which was the origin before the shift, so the lip
    # lands at -x0 -- downstream of the spike shoulder, not level with it.
    x0 = xs[0]
    xs = [x - x0 for x in xs]
    lip_x = -x0

    full_length = xs[-1]
    if truncate_fraction < 1.0:
        cut = full_length * truncate_fraction
        keep = [i for i, x in enumerate(xs) if x <= cut] or [0]
        last = keep[-1] + 1
        xs, rs, ms = xs[:last], rs[:last], ms[:last]

    throat_area = math.pi * exit_radius_mm ** 2 / expansion_ratio

    return PlugContour(
        x=xs, r=rs, mach=ms,
        exit_mach=exit_mach,
        throat_angle=nu_e,
        exit_radius=exit_radius_mm,
        throat_area=throat_area,
        length=xs[-1],
        lip_x=lip_x,
    )


def to_csv(contour: PlugContour, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_mm,r_mm,mach\n")
        for x, r, m in zip(contour.x, contour.r, contour.mach):
            f.write(f"{x:.6f},{r:.6f},{m:.6f}\n")
