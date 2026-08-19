"""
Injector element sizing and face layout.

Unlike-impinging doublets on an annular face: each element is one oxidiser
orifice and one fuel orifice, angled so the two jets collide at a fixed distance
downstream of the face. Impingement does the atomisation and the mixing, which
is why the momentum ratio of the two streams matters as much as the areas.

Sizing is the incompressible orifice equation

    mdot = Cd * A * sqrt(2 * rho * dP)

solved for area at a chosen injection stiffness dP/pc. Stiffness is the design
knob: too soft and the chamber talks back to the feed system and the engine
chugs, too stiff and the pumps or the tank pressure pay for it. Twenty percent
is the usual starting point for a pressure-fed engine.

What this does not do
---------------------
No spray modelling, no droplet size, no vaporisation length, no combustion
stability analysis. The stability rule applied here is the pressure-drop
criterion and a check on the Hewitt number; both are correlations, not physics,
and neither will tell you the chamber is about to enter a tangential mode. A
real injector gets hot-fire tested, and this file cannot substitute for that.

Momentum ratio and mixing quality follow the standard impinging-doublet
correlations. They are design guides, not guarantees.

SI internally: Pa, m, kg, s. Diameters are reported in mm at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from combustion_ref import ChamberConditions


@dataclass(frozen=True)
class InjectorDesign:
    n_elements: int
    element_circle_radius: float        # m, where the element axes sit

    d_fuel: float                       # m, per-orifice
    d_ox: float
    area_fuel_total: float              # m^2
    area_ox_total: float

    dp_fuel: float                      # Pa
    dp_ox: float
    stiffness_fuel: float               # dP / pc
    stiffness_ox: float

    v_fuel: float                       # m/s injection velocity
    v_ox: float
    momentum_fuel: float                # N, stream momentum flux
    momentum_ox: float

    angle_fuel: float                   # rad from the element axis
    angle_ox: float
    impingement_distance: float         # m, from the face

    element_pitch: float                # m, arc spacing between element axes
    discharge_coefficient: float

    @property
    def d_fuel_mm(self) -> float:
        return self.d_fuel * 1e3

    @property
    def d_ox_mm(self) -> float:
        return self.d_ox * 1e3

    @property
    def momentum_ratio(self) -> float:
        return self.momentum_ox / self.momentum_fuel

    @property
    def included_angle(self) -> float:
        return self.angle_fuel + self.angle_ox

    @property
    def resultant_angle(self) -> float:
        """
        Angle of the combined jet from the element axis, rad.

        The two streams carry different momentum, so splitting the included
        angle evenly throws the resultant off axis -- 11 degrees for a typical
        LOX/methane split, which streaks the chamber and drives one side of the
        wall hot. The angles are solved to cancel the transverse components
        instead, so this is zero by construction rather than by tuning.
        """
        transverse = (self.momentum_ox * math.sin(self.angle_ox)
                      - self.momentum_fuel * math.sin(self.angle_fuel))
        axial = (self.momentum_ox * math.cos(self.angle_ox)
                 + self.momentum_fuel * math.cos(self.angle_fuel))
        return math.atan2(transverse, axial)

    @property
    def fuel_ring_radius(self) -> float:
        return self.element_circle_radius - self.impingement_distance * math.tan(self.angle_fuel)

    @property
    def ox_ring_radius(self) -> float:
        return self.element_circle_radius + self.impingement_distance * math.tan(self.angle_ox)


def _orifice_area(mdot: float, rho: float, dp: float, cd: float) -> float:
    """Total area from the incompressible orifice equation."""
    if dp <= 0.0:
        raise ValueError("injection pressure drop must be positive")
    return mdot / (cd * math.sqrt(2.0 * rho * dp))


def mass_flow_through(area: float, rho: float, dp: float, cd: float) -> float:
    """Inverse of _orifice_area. Used to prove the sizing round-trips."""
    return cd * area * math.sqrt(2.0 * rho * dp)


def balanced_doublet_angles(momentum_fuel: float, momentum_ox: float,
                            included_angle: float) -> tuple[float, float]:
    """
    Split an included angle so the doublet's resultant lands on the axis.

    Wanting  m_f sin(a_f) = m_ox sin(a_ox)  with  a_f + a_ox = total  gives

        a_f = atan2( m_ox sin(total),  m_f + m_ox cos(total) )

    Equal momenta collapse it to the symmetric half-angle, as they should. The
    heavier stream gets the shallower angle, which is the physical result: it
    needs less turning to be cancelled.
    """
    a_f = math.atan2(momentum_ox * math.sin(included_angle),
                     momentum_fuel + momentum_ox * math.cos(included_angle))
    return a_f, included_angle - a_f


def design_injector(
    chamber: ChamberConditions,
    n_elements: int,
    element_circle_radius_mm: float,
    stiffness: float = 0.20,
    discharge_coefficient: float = 0.75,
    included_angle_deg: float = 60.0,
    impingement_distance_mm: float = 4.0,
) -> InjectorDesign:
    """
    Size a momentum-balanced unlike-doublet ring for the given chamber.

    Both circuits get the same stiffness, so the pressure drops match and the
    areas fall out of the density and mass-flow split alone. The included angle
    is then divided between the two streams to cancel transverse momentum.
    """
    if n_elements < 1:
        raise ValueError("need at least one element")
    if not 0.0 < stiffness < 1.0:
        raise ValueError("stiffness must be a fraction of chamber pressure in (0, 1)")
    if not 0.0 < discharge_coefficient <= 1.0:
        raise ValueError("discharge coefficient must be in (0, 1]")
    if not 0.0 < included_angle_deg < 180.0:
        raise ValueError("included angle must be in (0, 180) degrees")

    prop = chamber.propellant
    dp = stiffness * chamber.chamber_pressure

    a_f = _orifice_area(chamber.mass_flow_fuel, prop.density_fuel, dp, discharge_coefficient)
    a_ox = _orifice_area(chamber.mass_flow_ox, prop.density_ox, dp, discharge_coefficient)

    d_f = math.sqrt(4.0 * a_f / (n_elements * math.pi))
    d_ox = math.sqrt(4.0 * a_ox / (n_elements * math.pi))

    v_f = chamber.mass_flow_fuel / (prop.density_fuel * a_f)
    v_ox = chamber.mass_flow_ox / (prop.density_ox * a_ox)

    mom_f = prop.density_fuel * v_f * v_f * a_f
    mom_ox = prop.density_ox * v_ox * v_ox * a_ox

    ang_f, ang_ox = balanced_doublet_angles(mom_f, mom_ox, math.radians(included_angle_deg))
    r_circle = element_circle_radius_mm * 1e-3

    return InjectorDesign(
        n_elements=n_elements,
        element_circle_radius=r_circle,
        d_fuel=d_f, d_ox=d_ox,
        area_fuel_total=a_f, area_ox_total=a_ox,
        dp_fuel=dp, dp_ox=dp,
        stiffness_fuel=stiffness, stiffness_ox=stiffness,
        v_fuel=v_f, v_ox=v_ox,
        momentum_fuel=mom_f, momentum_ox=mom_ox,
        angle_fuel=ang_f, angle_ox=ang_ox,
        impingement_distance=impingement_distance_mm * 1e-3,
        element_pitch=2.0 * math.pi * r_circle / n_elements,
        discharge_coefficient=discharge_coefficient,
    )


def element_positions(design: InjectorDesign):
    """
    Orifice centres on the injector face, in mm, as (fuel_yz, ox_yz).

    The pair straddles the element circle so the jets converge on the element
    axis. The model x axis is normal to the face.
    """
    r_f = design.fuel_ring_radius * 1e3
    r_ox = design.ox_ring_radius * 1e3
    fuel, ox = [], []
    for i in range(design.n_elements):
        a = 2.0 * math.pi * i / design.n_elements
        ca, sa = math.cos(a), math.sin(a)
        fuel.append((r_f * ca, r_f * sa))
        ox.append((r_ox * ca, r_ox * sa))
    return fuel, ox


def fits_on_face(design: InjectorDesign, annulus_inner_mm: float,
                 annulus_outer_mm: float, min_land_mm: float = 0.8) -> tuple[bool, str]:
    """
    Whether the orifice pattern physically fits the annular face.

    Four ways it does not: either ring runs off an edge, neighbouring elements
    collide, or the land between the two rings disappears. On a narrow annulus
    the last one binds first, and on a printed part the land is also the
    thinnest feature the process has to hold.
    """
    r_f = design.fuel_ring_radius * 1e3
    r_ox = design.ox_ring_radius * 1e3

    if r_f - design.d_fuel_mm / 2.0 < annulus_inner_mm + min_land_mm:
        return False, "fuel orifice ring runs off the inner edge of the face"
    if r_ox + design.d_ox_mm / 2.0 > annulus_outer_mm - min_land_mm:
        return False, "oxidiser orifice ring runs off the outer edge of the face"

    pitch_f = 2.0 * math.pi * r_f / design.n_elements
    pitch_ox = 2.0 * math.pi * r_ox / design.n_elements
    if pitch_f - design.d_fuel_mm < min_land_mm:
        return False, "fuel orifices collide with their neighbours"
    if pitch_ox - design.d_ox_mm < min_land_mm:
        return False, "oxidiser orifices collide with their neighbours"

    gap = (r_ox - design.d_ox_mm / 2.0) - (r_f + design.d_fuel_mm / 2.0)
    if gap < min_land_mm:
        return False, "no land left between the fuel and oxidiser rings"
    return True, "ok"


def fit_injector_to_face(
    chamber: ChamberConditions,
    annulus_inner_mm: float,
    annulus_outer_mm: float,
    stiffness: float = 0.20,
    discharge_coefficient: float = 0.75,
    min_land_mm: float = 0.8,
    max_elements: int = 48,
) -> tuple[InjectorDesign, str]:
    """
    Search for the densest doublet ring that fits the annular face.

    An annular chamber gives an injector far less room than a cylindrical one --
    here about 5 mm of radial width -- so the pattern is usually constrained by
    the face rather than by the flow. Rather than let the caller guess, walk the
    element count down from `max_elements` and the impingement distance down
    with it, and return the first design that fits along with why.

    More elements mix better, so the search prefers them. If nothing fits, the
    last failure reason comes back with the design, because the fix is
    upstream -- usually a wider annulus, meaning a higher contraction ratio.
    """
    r_mid = 0.5 * (annulus_inner_mm + annulus_outer_mm)
    last = "no candidate evaluated"
    best: InjectorDesign | None = None

    for n in range(max_elements, 0, -1):
        for d_imp in (6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5):
            for angle in (60.0, 50.0, 40.0, 30.0):
                d = design_injector(
                    chamber, n, r_mid,
                    stiffness=stiffness,
                    discharge_coefficient=discharge_coefficient,
                    included_angle_deg=angle,
                    impingement_distance_mm=d_imp,
                )
                ok, why = fits_on_face(d, annulus_inner_mm, annulus_outer_mm, min_land_mm)
                if ok:
                    return d, (f"{n} elements, {angle:g} deg included, "
                               f"impinging {d_imp:g} mm from the face")
                last = why
                if best is None:
                    best = d
    return best, f"nothing fits: {last}"


def chug_margin(design: InjectorDesign) -> float:
    """
    Ratio of injection stiffness to the usual low-frequency stability floor.

    The classical rule wants dP/pc above roughly 0.15 for a pressure-fed
    chamber. Below 1.0 here means the design sits in the band where chug has
    been seen on real hardware. It is a correlation, not a stability analysis:
    passing proves nothing, failing is worth taking seriously.
    """
    return design.stiffness_fuel / 0.15
