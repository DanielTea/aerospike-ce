"""
Thermostructural check on the cooled wall, and a cycle life estimate.

A regeneratively cooled chamber does not fail by bursting. It fails by thermal
strain: the hot wall wants to expand and the cold jacket behind it will not let
it, so the hot wall goes into compression, yields, and comes back into tension
when the engine shuts down. Repeat that and the wall bulges into the channel,
thins, and splits along the land. The failure is called doghouse deformation and
it is the life-limiting mechanism for every regen chamber ever flown.

What is checked here
--------------------
- thermal stress from the through-wall gradient, biaxially constrained
- pressure stress on the hot wall treated as a plate spanning between lands
- hoop stress from chamber pressure on the shell
- combined stress against yield derated for temperature
- low-cycle fatigue life by Coffin-Manson

What is not
-----------
No finite elements, no creep, no ratchetting accumulation, no weld or
build-direction anisotropy, no residual stress from the print, and no
consideration of the land as a three-dimensional structure. Printed material
properties vary with build direction and heat treatment by more than the margins
this reports. This is a screening calculation: it will tell you a wall is
obviously too thin, and it will not tell you one is safe.

SI throughout: Pa, m, K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cooling_ref import CoolingSolution, Material


@dataclass(frozen=True)
class StructuralResult:
    x: np.ndarray
    thermal_stress: np.ndarray      # Pa
    pressure_stress: np.ndarray     # Pa
    hoop_stress: np.ndarray         # Pa
    combined_stress: np.ndarray     # Pa
    yield_at_temp: np.ndarray       # Pa
    margin: np.ndarray              # yield/combined - 1
    strain_range: np.ndarray
    cycles_to_failure: float
    material: Material
    required_cycles: float = 100.0

    @property
    def worst_margin(self) -> float:
        return float(self.margin.min())

    @property
    def worst_station(self) -> float:
        return float(self.x[int(np.argmin(self.margin))])

    @property
    def stays_elastic(self) -> bool:
        return self.worst_margin > 0.0

    @property
    def survives(self) -> bool:
        """
        Life, not yield, is the criterion.

        A regeneratively cooled hot wall is expected to exceed yield: the
        through-wall gradient is large, the jacket restrains it, and it cycles
        plastically every firing. Requiring it to stay elastic would reject
        every chamber ever flown. What matters is whether the plastic strain per
        cycle is small enough to survive the intended number of cycles, and
        whether the wall is nowhere near tearing on the first one.
        """
        return (self.cycles_to_failure >= self.required_cycles
                and float(self.strain_range.max()) < 0.02)


def yield_at_temperature(material: Material, temperature: np.ndarray) -> np.ndarray:
    """
    Yield strength derated linearly to zero at the melting point.

    Crude, and the direction of the crudeness is what matters: real alloys lose
    strength faster than linearly near the melt, so this is optimistic at the
    hot end. Do not spend the last ten percent of the margin it reports.
    """
    f = np.clip(1.0 - (temperature - 293.0) / (material.melting_point - 293.0), 0.02, 1.0)
    return material.yield_strength * f


def analyse(
    solution: CoolingSolution,
    chamber_pressure: float,
    coolant_pressure: float,
    wall_radius_mm: np.ndarray,
    shell_thickness_mm: float,
    required_cycles: float = 100.0,
) -> StructuralResult:
    """
    Screen the cooled wall for stress and cycle life.

    Thermal stress uses the constrained-plate result

        sigma = E alpha dT / (2 (1 - nu))

    with dT the drop across the hot wall. The factor of two is the standard
    treatment for a linear gradient in a plate restrained in its own plane.

    Pressure stress treats the hot wall as a plate spanning between lands under
    the coolant-to-chamber differential, which is the load that pushes it into
    the channel:

        sigma = dp (w / t)^2 / 2
    """
    m = solution.material
    ch = solution.channel

    t_hot = ch.hot_wall_mm * 1e-3
    w_ch = ch.width_mm * 1e-3

    dt_wall = np.maximum(solution.t_wall_gas - solution.t_wall_coolant, 0.0)
    sigma_th = m.youngs_modulus * m.expansion * dt_wall / (2.0 * (1.0 - m.poisson))

    dp = abs(coolant_pressure - chamber_pressure)
    sigma_p = np.full_like(sigma_th, dp * (w_ch / t_hot) ** 2 / 2.0)

    r = np.asarray(wall_radius_mm, dtype=float) * 1e-3
    sigma_hoop = chamber_pressure * r / (shell_thickness_mm * 1e-3)

    # von Mises-ish combination of a biaxial membrane and a bending term
    sigma_total = np.sqrt(sigma_th ** 2 + sigma_p ** 2 + sigma_hoop ** 2
                          - sigma_th * sigma_hoop)

    sy = yield_at_temperature(m, solution.t_wall_gas)
    margin = sy / np.maximum(sigma_total, 1.0) - 1.0

    # Coffin-Manson on the plastic part of the strain range
    strain_total = sigma_total / m.youngs_modulus
    strain_elastic = sy / m.youngs_modulus
    strain_plastic = np.maximum(strain_total - strain_elastic, 0.0)

    worst_plastic = float(strain_plastic.max())
    if worst_plastic <= 1e-9:
        cycles = float("inf")           # stays elastic: no low-cycle mechanism
    else:
        cycles = 0.5 * (worst_plastic / m.fatigue_ductility) ** (1.0 / m.fatigue_exponent)

    return StructuralResult(
        x=np.asarray(solution.x, dtype=float),
        thermal_stress=sigma_th,
        pressure_stress=sigma_p,
        hoop_stress=sigma_hoop,
        combined_stress=sigma_total,
        yield_at_temp=sy,
        margin=margin,
        strain_range=strain_total,
        cycles_to_failure=cycles,
        material=m,
        required_cycles=required_cycles,
    )
