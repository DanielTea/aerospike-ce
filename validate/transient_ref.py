"""
Startup transient through the cooled wall.

The steady solution says what the wall settles at. It says nothing about the
first half second, which is when regen chambers are actually lost.

The hot wall is thin and, in copper, an extremely good conductor: its diffusion
time is around ten milliseconds, far shorter than any chamber pressure ramp. So
the wall itself has no meaningful thermal lag and tracks the flux quasi
statically. That is a reassuring result and it is not the risk.

The risk is sequencing. If the propellant valves open before the coolant circuit
is primed, the wall sees a rising heat flux with nothing behind it to take the
heat away, and copper's advantage becomes a liability -- it conducts the heat
straight into a wall that has no sink. This module marches an explicit
one-dimensional conduction solution through the wall thickness with both the gas
side flux and the coolant side coefficient ramping on their own schedules, so
the cost of a coolant lead or lag can be read off directly.

Not modelled: axial conduction (the wall is thin against its length), the
jacket's own heat capacity, ignition overpressure, or two-phase coolant during
priming. Any of those can matter.

SI throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cooling_ref import CoolingSolution, Material


@dataclass(frozen=True)
class StartupResult:
    time: np.ndarray                # s
    wall_gas_temp: np.ndarray       # K, gas-side face
    wall_cool_temp: np.ndarray      # K, coolant-side face
    heat_flux: np.ndarray           # W/m^2
    peak_wall_temp: float
    peak_time: float
    steady_wall_temp: float
    overshoot: float                # K above the steady value
    material: Material
    survives: bool

    @property
    def diffusion_time(self) -> float:
        return self._diffusion

    _diffusion: float = 0.0


def solve_startup(
    solution: CoolingSolution,
    station: int | None = None,
    ramp_time: float = 0.30,
    coolant_lead: float = 0.0,
    duration: float = 2.0,
    nodes: int = 41,
    safety: float = 0.25,
) -> StartupResult:
    """
    March the hot wall through startup at one station.

    `coolant_lead` is how far ahead of the gas the coolant is established, in
    seconds. Positive means coolant first, which is what a sane start sequence
    does. Negative means the gas arrives first, and is the case worth looking at
    before writing a valve sequence.

    Explicit conduction, so the step is bounded by the stability limit
    dt < dx^2 / (2 alpha); `safety` is the fraction of that used.
    """
    m = solution.material
    if station is None:
        station = int(np.argmax(solution.heat_flux))

    t_hot = solution.channel.hot_wall_mm * 1e-3
    dx = t_hot / (nodes - 1)
    alpha = m.conductivity / (m.density * 385.0)      # cp of the alloy, J/(kg K)
    dt = safety * dx * dx / (2.0 * alpha)
    steps = max(2, int(duration / dt))

    q_steady = float(solution.heat_flux[station])
    t_aw = float(solution.t_adiabatic[station])
    t_cool = float(solution.t_coolant[station])
    # coolant-side coefficient implied by the steady solution at this station
    h_cool = q_steady / max(solution.t_wall_coolant[station] - t_cool, 1e-6)
    h_gas = q_steady / max(t_aw - solution.t_wall_gas[station], 1e-6)

    temp = np.full(nodes, t_cool)
    times = np.empty(steps)
    tg = np.empty(steps)
    tc = np.empty(steps)
    qq = np.empty(steps)

    for k in range(steps):
        t = k * dt
        gas_frac = min(max(t / ramp_time, 0.0), 1.0)
        cool_frac = min(max((t + coolant_lead) / ramp_time, 0.0), 1.0)

        hg = h_gas * gas_frac
        hc = h_cool * cool_frac
        q_in = hg * (t_aw - temp[0])
        q_out = hc * (temp[-1] - t_cool)

        new = temp.copy()
        new[1:-1] = temp[1:-1] + alpha * dt / (dx * dx) * (
            temp[2:] - 2.0 * temp[1:-1] + temp[:-2])
        # faces: half-cell energy balance
        new[0] = temp[0] + 2.0 * alpha * dt / (dx * dx) * (
            temp[1] - temp[0] + q_in * dx / m.conductivity)
        new[-1] = temp[-1] + 2.0 * alpha * dt / (dx * dx) * (
            temp[-2] - temp[-1] - q_out * dx / m.conductivity)
        temp = new

        times[k] = t
        tg[k] = temp[0]
        tc[k] = temp[-1]
        qq[k] = q_in

    peak = float(tg.max())
    steady = float(solution.t_wall_gas[station])
    return StartupResult(
        time=times, wall_gas_temp=tg, wall_cool_temp=tc, heat_flux=qq,
        peak_wall_temp=peak, peak_time=float(times[int(np.argmax(tg))]),
        steady_wall_temp=steady, overshoot=peak - steady,
        material=m, survives=peak < m.max_wall_temp,
        _diffusion=t_hot * t_hot / alpha,
    )
