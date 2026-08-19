"""
Thrust from the plug surface pressure, and altitude compensation.

The thrust coefficient in combustion_ref.py is the ideal bell value: momentum
term plus a pressure term fixed by the exit area. It is the right number for a
bell and the wrong one for a plug, and it is wrong in the direction that matters
-- it understates the aerospike everywhere below the design altitude, which is
the entire reason to build one.

A plug has no fixed exit area. Its outer boundary is the ambient air, so as the
ambient pressure falls the plume simply expands outward and the effective area
ratio rises with it. Below the design altitude the flow instead compresses onto
the spike, and the surface sees pressures above ambient over a shorter length.

What this does
--------------
Integrates the pressure the Angelino construction already gives on the spike
surface. Every contour point carries a Mach number, so it carries a static
pressure, and the axial force is that pressure acting on the annular projection:

    F_plug = integral of (p(r) - p_ambient) * 2 pi r dr,  from tip to lip

At high ambient the integrand goes negative near the tip, where the flow has
over-expanded below ambient. That region is where a real plug's plume detaches
from the surface instead, so the integration stops at the crossing rather than
crediting the engine with the loss. That truncation *is* the altitude
compensation mechanism, expressed geometrically.

What this is not
----------------
Not a method of characteristics solution. The contour was built by the Angelino
approximate method, which assumes a centred Prandtl-Meyer fan and uniform flow
along each characteristic; the base region behind a truncated spike is not
modelled at all, and base pressure on a truncated plug is worth several percent
of thrust. A real answer needs a full MOC solve with a base flow model, or CFD.

SI unless a name says mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from combustion_ref import ChamberConditions, pressure_ratio
from contour_ref import PlugContour


@dataclass(frozen=True)
class PlugThrust:
    ambient_pressure: float
    surface_pressure: np.ndarray    # Pa along the contour
    attached_fraction: float        # of the spike length still pushing
    plug_force: float               # N, axial, from the surface integral
    throat_force: float             # N, momentum and pressure through the throat
    total_force: float              # N
    thrust_coefficient: float
    isp: float                      # s

    @property
    def separated(self) -> bool:
        return self.attached_fraction < 0.999


def surface_pressure(contour: PlugContour, chamber: ChamberConditions) -> np.ndarray:
    """Static pressure along the plug surface, from the contour's own Mach numbers."""
    gamma = chamber.propellant.gamma
    return np.array([chamber.chamber_pressure * pressure_ratio(m, gamma)
                     for m in contour.mach])


def plug_thrust(contour: PlugContour, chamber: ChamberConditions,
                ambient_pressure: float) -> PlugThrust:
    """
    Axial thrust with the plug surface integrated against the ambient.

    The throat term is the momentum and pressure crossing the sonic line,
    resolved along the axis. The sonic line sits at pi/2 - nu_e from the axis,
    so its axial projection carries the cosine of that.
    """
    p_surf = surface_pressure(contour, chamber)
    r = np.asarray(contour.r, dtype=float) * 1e-3

    # walk from the lip inward; stop where the surface falls below ambient
    over = p_surf > ambient_pressure
    if not over[0]:
        attached = 0
    else:
        drop = np.argmin(over) if not over.all() else len(over)
        attached = int(drop)

    if attached < 2:
        plug_force = 0.0
    else:
        pr = p_surf[:attached] - ambient_pressure
        rr = r[:attached]
        # F = integral (p - pa) 2 pi r dr, r decreasing along the contour
        plug_force = float(-np.trapezoid(pr * 2.0 * math.pi * rr, rr))

    gamma = chamber.propellant.gamma
    p_throat = chamber.chamber_pressure * pressure_ratio(1.0, gamma)
    r_gas = chamber.gas_constant
    t_throat = chamber.t_chamber / (1.0 + 0.5 * (gamma - 1.0))
    v_throat = math.sqrt(gamma * r_gas * t_throat)

    # The flow crosses the sonic line inclined at nu_e to the axis -- strongly
    # turned outward at the throat, straightening as it expands along the spike.
    # Only the axial component pushes. Resolving on pi/2 - nu_e instead, which is
    # the angle the sonic *line* makes with the axis rather than the angle the
    # flow makes, inflates the throat term by a factor of 2.4 and hands the plug
    # a 48 percent gain over an ideal bell that no aerospike has ever shown.
    axial = math.cos(contour.throat_angle)
    throat_force = (chamber.mass_flow * v_throat
                    + (p_throat - ambient_pressure) * chamber.throat_area) * axial

    total = (plug_force + throat_force) * chamber.truncation_efficiency
    cf = total / (chamber.chamber_pressure * chamber.throat_area)
    return PlugThrust(
        ambient_pressure=ambient_pressure,
        surface_pressure=p_surf,
        attached_fraction=attached / max(len(p_surf), 1),
        plug_force=plug_force,
        throat_force=throat_force,
        total_force=total,
        thrust_coefficient=cf,
        isp=total / (chamber.mass_flow * 9.80665),
    )


def altitude_sweep(contour: PlugContour, chamber: ChamberConditions,
                   ambient_pressures: np.ndarray):
    """Thrust coefficient against ambient pressure, for the compensation curve."""
    return [plug_thrust(contour, chamber, float(pa)) for pa in ambient_pressures]
