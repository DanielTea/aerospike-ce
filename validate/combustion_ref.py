"""
Chamber conditions and delivered performance.

This is the layer the rest of the engine sizes from: the injector needs mass
flow and densities, the cooling jacket needs chamber temperature and mass flow,
and the whole thing needs a chamber pressure before any of it has a number.

What this is
------------
A parametric performance model of the kind used for preliminary sizing: stated
propellant properties, isentropic nozzle relations, and standard efficiency
factors. c* and T_c follow a fitted parabolic falloff either side of their peak
mixture ratio, which reproduces the shape of a real c* curve well enough to size
hardware and is wrong in the wings.

What this is NOT
----------------
An equilibrium chemistry solver. There is no Gibbs minimisation here and no
thermochemical database. The propellant table below is tabulated textbook data
at one reference condition, not computed. For anything that will be built and
fired, run CEA or RPA at your actual chamber pressure and mixture ratio and put
those numbers into the spec; every routine here accepts them as input for
exactly that reason.

Frozen flow is assumed throughout the nozzle. Real expansions sit between frozen
and shifting equilibrium, so delivered Isp will land a few percent above these
numbers, which is the conservative direction.

All SI internally: Pa, K, m, kg, s. The geometry modules are in mm; conversions
happen at the boundary and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665            # m/s^2, standard gravity, by definition
R_UNIVERSAL = 8314.462  # J/(kmol K)


# --------------------------------------------------------------------------
# propellants
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Propellant:
    """
    Tabulated combustion properties at one reference condition.

    c_star_peak and t_chamber_peak are the values at mr_peak. The falloff
    coefficients shape the curve either side of it; 0.30 and 0.25 reproduce the
    characteristic asymmetry-free parabola that textbook c* curves show near
    their peak. They are a fit, not a derivation.

    Coolant properties are for the fuel, since that is what a regenerative
    jacket almost always carries.
    """
    name: str
    fuel: str
    oxidiser: str
    mr_peak: float              # O/F by mass at peak c*
    c_star_peak: float          # m/s
    t_chamber_peak: float       # K
    gamma: float
    molar_mass: float           # kg/kmol
    density_fuel: float         # kg/m^3
    density_ox: float           # kg/m^3
    coolant_cp: float           # J/(kg K)
    coolant_k: float            # W/(m K)
    coolant_mu: float           # Pa s
    coolant_inlet_t: float      # K
    coolant_max_t: float        # K, coking or critical limit
    c_star_falloff: float = 0.30
    t_chamber_falloff: float = 0.25


PROPELLANTS: dict[str, Propellant] = {
    "lox_rp1": Propellant(
        name="lox_rp1", fuel="RP-1", oxidiser="LOX",
        mr_peak=2.56, c_star_peak=1800.0, t_chamber_peak=3670.0,
        gamma=1.24, molar_mass=23.3,
        density_fuel=810.0, density_ox=1141.0,
        coolant_cp=2000.0, coolant_k=0.14, coolant_mu=1.6e-3,
        coolant_inlet_t=290.0, coolant_max_t=560.0,   # RP-1 cokes above ~560 K
    ),
    "lox_ch4": Propellant(
        name="lox_ch4", fuel="CH4", oxidiser="LOX",
        mr_peak=3.40, c_star_peak=1857.0, t_chamber_peak=3550.0,
        gamma=1.20, molar_mass=21.5,
        density_fuel=423.0, density_ox=1141.0,
        coolant_cp=3480.0, coolant_k=0.19, coolant_mu=1.2e-4,
        coolant_inlet_t=111.0, coolant_max_t=700.0,
    ),
    "n2o_ethanol": Propellant(
        name="n2o_ethanol", fuel="ethanol", oxidiser="N2O",
        mr_peak=4.00, c_star_peak=1580.0, t_chamber_peak=3100.0,
        gamma=1.23, molar_mass=25.0,
        density_fuel=789.0, density_ox=1220.0,
        coolant_cp=2440.0, coolant_k=0.17, coolant_mu=1.2e-3,
        coolant_inlet_t=290.0, coolant_max_t=460.0,
    ),
}


def c_star_ideal(prop: Propellant, mixture_ratio: float) -> float:
    """Characteristic velocity, m/s. Parabolic falloff about the peak."""
    if mixture_ratio <= 0.0:
        raise ValueError("mixture_ratio must be positive")
    d = (mixture_ratio - prop.mr_peak) / prop.mr_peak
    return prop.c_star_peak * max(0.0, 1.0 - prop.c_star_falloff * d * d)


def chamber_temperature(prop: Propellant, mixture_ratio: float) -> float:
    """Stagnation combustion temperature, K. Same parabolic treatment."""
    d = (mixture_ratio - prop.mr_peak) / prop.mr_peak
    return prop.t_chamber_peak * max(0.0, 1.0 - prop.t_chamber_falloff * d * d)


# --------------------------------------------------------------------------
# nozzle performance
# --------------------------------------------------------------------------

def pressure_ratio(mach: float, gamma: float) -> float:
    """p/p0 for isentropic flow at a given Mach number."""
    return (1.0 + 0.5 * (gamma - 1.0) * mach * mach) ** (-gamma / (gamma - 1.0))


def thrust_coefficient(
    gamma: float,
    expansion_ratio: float,
    exit_mach: float,
    chamber_pressure: float,
    ambient_pressure: float,
) -> float:
    """
    Nozzle thrust coefficient C_F, dimensionless.

    Momentum term plus the pressure term. The pressure term is what makes an
    aerospike interesting: on a bell it is fixed by the exit area, whereas the
    plug's outer boundary is the ambient itself, so the spike trims its own
    effective expansion as the ambient falls. This function returns the ideal
    bell value; `altitude_compensation_note` in the tests records that the real
    plug sits above it below the design altitude and that modelling that
    properly needs a method-of-characteristics solution this repo does not have.
    """
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    pe_pc = pressure_ratio(exit_mach, gamma)

    momentum = math.sqrt(
        (2.0 * gamma * gamma / gm1)
        * (2.0 / gp1) ** (gp1 / gm1)
        * (1.0 - pe_pc ** (gm1 / gamma))
    )
    pressure = (pe_pc - ambient_pressure / chamber_pressure) * expansion_ratio
    return momentum + pressure


@dataclass(frozen=True)
class ChamberConditions:
    propellant: Propellant
    mixture_ratio: float
    chamber_pressure: float       # Pa
    throat_area: float            # m^2
    expansion_ratio: float
    exit_mach: float

    c_star_efficiency: float
    truncation_efficiency: float

    c_star: float                 # m/s, delivered
    t_chamber: float              # K
    mass_flow: float              # kg/s
    mass_flow_fuel: float
    mass_flow_ox: float

    cf_sea_level: float
    cf_vacuum: float
    thrust_sea_level: float       # N
    thrust_vacuum: float          # N
    isp_sea_level: float          # s
    isp_vacuum: float             # s

    @property
    def exit_pressure(self) -> float:
        return self.chamber_pressure * pressure_ratio(self.exit_mach, self.propellant.gamma)

    @property
    def gas_constant(self) -> float:
        """Specific gas constant of the products, J/(kg K)."""
        return R_UNIVERSAL / self.propellant.molar_mass


def build_chamber(
    propellant: Propellant,
    mixture_ratio: float,
    chamber_pressure_bar: float,
    throat_area_mm2: float,
    expansion_ratio: float,
    exit_mach: float,
    ambient_pressure_pa: float = 101325.0,
    c_star_efficiency: float = 0.94,
    truncation_efficiency: float = 0.99,
    film_fraction: float = 0.0,
    film_combustion_efficiency: float = 0.5,
) -> ChamberConditions:
    """
    Chamber conditions and delivered performance.

    c_star_efficiency covers mixing and vaporisation losses; 0.92-0.96 is the
    usual band for a sane injector, and it is an input rather than a prediction
    because predicting it requires modelling the spray.

    truncation_efficiency is the thrust lost by cutting the spike short. A 20-30
    percent spike gives up on the order of 1 percent, which is the entire reason
    nobody flies a full-length one. Also an input; the base flow that sets it is
    not modelled here.

    film_fraction is fuel diverted to cool the wall rather than to burn at the
    design mixture ratio. It is not free: the film runs along the wall fuel-rich
    and only partly reacts, so it carries mass through the nozzle without having
    contributed its full share of energy. Charging it at
    `film_combustion_efficiency` -- half, by default, which is the usual band for
    a wall film -- is crude, and it is far better than pretending a tenth of the
    fuel can be spent on cooling for nothing.
    """
    if chamber_pressure_bar <= 0.0:
        raise ValueError("chamber pressure must be positive")
    if throat_area_mm2 <= 0.0:
        raise ValueError("throat area must be positive")
    if not 0.0 < c_star_efficiency <= 1.0:
        raise ValueError("c_star_efficiency must be in (0, 1]")

    pc = chamber_pressure_bar * 1e5
    at = throat_area_mm2 * 1e-6
    gamma = propellant.gamma

    if not 0.0 <= film_fraction < 1.0:
        raise ValueError("film_fraction must be in [0, 1)")

    # fuel spent on the film is charged at a reduced combustion efficiency
    film_penalty = 1.0 - film_fraction * (1.0 - film_combustion_efficiency) \
        / (1.0 + mixture_ratio)
    c_star = c_star_ideal(propellant, mixture_ratio) * c_star_efficiency * film_penalty
    if c_star <= 0.0:
        raise ValueError("mixture ratio is far enough off peak that c* collapses")

    tc = chamber_temperature(propellant, mixture_ratio)

    # the defining relation of a choked throat
    mdot = pc * at / c_star
    mdot_ox = mdot * mixture_ratio / (1.0 + mixture_ratio)
    mdot_f = mdot / (1.0 + mixture_ratio)

    cf_sl = thrust_coefficient(gamma, expansion_ratio, exit_mach, pc, ambient_pressure_pa)
    cf_vac = thrust_coefficient(gamma, expansion_ratio, exit_mach, pc, 0.0)

    f_sl = cf_sl * pc * at * truncation_efficiency
    f_vac = cf_vac * pc * at * truncation_efficiency

    return ChamberConditions(
        propellant=propellant,
        mixture_ratio=mixture_ratio,
        chamber_pressure=pc,
        throat_area=at,
        expansion_ratio=expansion_ratio,
        exit_mach=exit_mach,
        c_star_efficiency=c_star_efficiency,
        truncation_efficiency=truncation_efficiency,
        c_star=c_star,
        t_chamber=tc,
        mass_flow=mdot,
        mass_flow_fuel=mdot_f,
        mass_flow_ox=mdot_ox,
        cf_sea_level=cf_sl,
        cf_vacuum=cf_vac,
        thrust_sea_level=f_sl,
        thrust_vacuum=f_vac,
        isp_sea_level=f_sl / (mdot * G0),
        isp_vacuum=f_vac / (mdot * G0),
    )


def chamber_from_spec(spec: dict, throat_area_mm2: float,
                      expansion_ratio: float, exit_mach: float) -> ChamberConditions:
    """Build chamber conditions from a parsed spec/*.json."""
    p = spec.get("propellant", {})
    o = spec.get("operation", {})

    name = p.get("combination", "lox_ch4")
    if name not in PROPELLANTS:
        raise ValueError(f"unknown propellant combination: {name}")
    base = PROPELLANTS[name]

    # any tabulated value may be overridden from the spec, which is how CEA or
    # RPA output gets into this model
    prop = Propellant(
        name=base.name, fuel=base.fuel, oxidiser=base.oxidiser,
        mr_peak=p.get("mr_peak", base.mr_peak),
        c_star_peak=p.get("c_star_peak_ms", base.c_star_peak),
        t_chamber_peak=p.get("t_chamber_peak_k", base.t_chamber_peak),
        gamma=p.get("gamma", base.gamma),
        molar_mass=p.get("molar_mass", base.molar_mass),
        density_fuel=p.get("density_fuel", base.density_fuel),
        density_ox=p.get("density_ox", base.density_ox),
        coolant_cp=p.get("coolant_cp", base.coolant_cp),
        coolant_k=p.get("coolant_k", base.coolant_k),
        coolant_mu=p.get("coolant_mu", base.coolant_mu),
        coolant_inlet_t=p.get("coolant_inlet_t_k", base.coolant_inlet_t),
        coolant_max_t=p.get("coolant_max_t_k", base.coolant_max_t),
    )

    return build_chamber(
        prop,
        mixture_ratio=o.get("mixture_ratio", prop.mr_peak),
        chamber_pressure_bar=o.get("chamber_pressure_bar", 20.0),
        throat_area_mm2=throat_area_mm2,
        expansion_ratio=expansion_ratio,
        exit_mach=exit_mach,
        ambient_pressure_pa=o.get("ambient_pressure_pa", 101325.0),
        c_star_efficiency=o.get("c_star_efficiency", 0.94),
        truncation_efficiency=o.get("truncation_efficiency", 0.99),
        film_fraction=spec.get("film", {}).get("fuel_fraction", 0.0),
        film_combustion_efficiency=spec.get("film", {}).get(
            "combustion_efficiency", 0.5),
    )
