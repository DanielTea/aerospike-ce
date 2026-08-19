"""
Combustion stability screening.

Stability cannot be predicted from geometry. What can be done, and what every
preliminary design does, is compute the frequencies the chamber will support and
the classic screening parameters, then keep the injector away from them and test.

Three families matter:

- **Chug**, a low-frequency coupling between the chamber and the feed system.
  Guarded against with injection stiffness; the floor is around 15 to 20 percent
  of chamber pressure.
- **Acoustic modes** of the chamber cavity. The first tangential is the one that
  destroys hardware, because it sweeps the injector face and drives heat flux
  into the wall an order of magnitude above design. For an annular chamber its
  wavelength is set by the mean circumference, not by the gap.
- **Intermediate frequency** coupling to propellant residence and vaporisation.
  Screened with the characteristic length L*.

What this does not do
---------------------
No Rayleigh criterion evaluation, no combustion response function, no linear
stability eigenvalue problem, no nonlinear triggering. Those need a combustion
response measured on the actual injector. This computes where the chamber wants
to ring and tells you whether the standard screens pass; it cannot tell you the
engine is stable, and nothing short of hot fire can.

SI throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from combustion_ref import R_UNIVERSAL, ChamberConditions


@dataclass(frozen=True)
class StabilityResult:
    speed_of_sound: float           # m/s in the chamber
    chamber_volume: float           # m^3
    characteristic_length: float    # L* = V_c / A_t, m
    residence_time: float           # s
    f_longitudinal_1: float         # Hz
    f_tangential_1: float
    f_tangential_2: float
    f_radial_1: float
    f_chug_estimate: float
    injection_stiffness: float
    l_star_ok: bool
    stiffness_ok: bool
    separation_ok: bool             # chug well clear of the first acoustic mode

    @property
    def screens_pass(self) -> bool:
        return self.l_star_ok and self.stiffness_ok and self.separation_ok


def analyse(
    chamber: ChamberConditions,
    chamber_length_m: float,
    mean_radius_m: float,
    annulus_gap_m: float,
    injection_stiffness: float,
    l_star_min: float = 0.6,
    l_star_max: float = 1.5,
) -> StabilityResult:
    """
    Chamber acoustics and the standard screening parameters.

    The annular chamber's transverse modes follow the mean circumference:

        f_mT = m * c / (2 pi R_mean)

    A cylindrical chamber of the same volume would put its first tangential
    much higher, which is one of the quiet penalties of an annular architecture:
    the mode sits lower, closer to where injector response is strong.
    """
    p = chamber.propellant
    r_gas = R_UNIVERSAL / p.molar_mass
    c_sound = math.sqrt(p.gamma * r_gas * chamber.t_chamber)

    volume = 2.0 * math.pi * mean_radius_m * annulus_gap_m * chamber_length_m
    l_star = volume / chamber.throat_area

    rho_c = chamber.chamber_pressure / (r_gas * chamber.t_chamber)
    residence = rho_c * volume / chamber.mass_flow

    f_1l = c_sound / (2.0 * chamber_length_m)
    f_1t = c_sound / (2.0 * math.pi * mean_radius_m)
    f_2t = 2.0 * f_1t
    f_1r = c_sound / (2.0 * annulus_gap_m)

    # chug sits at the chamber-filling timescale, not at an acoustic one
    f_chug = 1.0 / (2.0 * math.pi * residence)

    return StabilityResult(
        speed_of_sound=c_sound,
        chamber_volume=volume,
        characteristic_length=l_star,
        residence_time=residence,
        f_longitudinal_1=f_1l,
        f_tangential_1=f_1t,
        f_tangential_2=f_2t,
        f_radial_1=f_1r,
        f_chug_estimate=f_chug,
        injection_stiffness=injection_stiffness,
        l_star_ok=l_star_min <= l_star <= l_star_max,
        stiffness_ok=injection_stiffness >= 0.15,
        separation_ok=f_chug < 0.2 * f_1t,
    )
