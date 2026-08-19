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
    steps_target: int = 4000,
    explicit: bool = False,
) -> StartupResult:
    """
    March the hot wall through startup at one station.

    `coolant_lead` is how far ahead of the gas the coolant is established, in
    seconds. Positive means coolant first, which is what a sane start sequence
    does. Negative means the gas arrives first, and is the case worth looking at
    before writing a valve sequence.

    Implicit conduction (backward Euler), solved on the tridiagonal system each
    step. The scheme matters more than it looks: explicit marching is bounded by
    dt < dx^2 / (2 alpha), and on a 0.3 mm wall at forty nodes that is 80
    nanoseconds -- twenty-four million steps for a two-second transient, and
    around a minute of wall clock for a result that is wanted hundreds of times
    inside an optimiser. Backward Euler is unconditionally stable, so the step
    is set by how finely the ramp needs resolving rather than by stability, and
    the same answer arrives about a thousand times sooner.

    Pass `explicit=True` for the old marching scheme. It is kept because the two
    agreeing is what shows the implicit solve is right, not because it is
    useful; `safety` applies only to it.
    """
    m = solution.material
    if station is None:
        station = int(np.argmax(solution.heat_flux))

    t_hot = solution.channel.hot_wall_mm * 1e-3
    dx = t_hot / (nodes - 1)
    alpha = m.conductivity / (m.density * 385.0)      # cp of the alloy, J/(kg K)
    dt_stable = safety * dx * dx / (2.0 * alpha)
    if explicit:
        dt = dt_stable
    else:
        # resolve the ramp finely, and never take a step coarser than that
        dt = min(duration / steps_target, ramp_time / 200.0)
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

    lam = alpha * dt / (dx * dx)
    k_w = m.conductivity

    if explicit:
        for k in range(steps):
            t = k * dt
            hg = h_gas * min(max(t / ramp_time, 0.0), 1.0)
            hc = h_cool * min(max((t + coolant_lead) / ramp_time, 0.0), 1.0)
            q_in = hg * (t_aw - temp[0])
            q_out = hc * (temp[-1] - t_cool)

            new = temp.copy()
            new[1:-1] = temp[1:-1] + lam * (temp[2:] - 2.0 * temp[1:-1] + temp[:-2])
            new[0] = temp[0] + 2.0 * lam * (temp[1] - temp[0] + q_in * dx / k_w)
            new[-1] = temp[-1] + 2.0 * lam * (temp[-2] - temp[-1] - q_out * dx / k_w)
            temp = new

            times[k] = t
            tg[k] = temp[0]
            tc[k] = temp[-1]
            qq[k] = q_in
    else:
        from scipy.linalg import solve_banded

        # Tridiagonal system in banded form: rows are super, main, sub.
        # Interior rows never change; only the two faces move with the ramp.
        ab = np.zeros((3, nodes))
        ab[0, 2:] = -lam                      # super-diagonal, rows 1..n-2
        ab[1, 1:-1] = 1.0 + 2.0 * lam
        ab[2, :-2] = -lam                     # sub-diagonal, rows 1..n-2
        rhs = np.empty(nodes)

        for k in range(steps):
            t = k * dt
            hg = h_gas * min(max(t / ramp_time, 0.0), 1.0)
            hc = h_cool * min(max((t + coolant_lead) / ramp_time, 0.0), 1.0)

            # gas face: T0 (1 + 2L + 2L hg dx/k) - 2L T1 = T0^n + 2L hg Taw dx/k
            bg = 2.0 * lam * hg * dx / k_w
            ab[1, 0] = 1.0 + 2.0 * lam + bg
            ab[0, 1] = -2.0 * lam
            rhs[0] = temp[0] + bg * t_aw

            rhs[1:-1] = temp[1:-1]

            bc = 2.0 * lam * hc * dx / k_w
            ab[1, -1] = 1.0 + 2.0 * lam + bc
            ab[2, -2] = -2.0 * lam
            rhs[-1] = temp[-1] + bc * t_cool

            temp = solve_banded((1, 1), ab, rhs)

            times[k] = t
            tg[k] = temp[0]
            tc[k] = temp[-1]
            qq[k] = hg * (t_aw - temp[0])

    peak = float(tg.max())
    steady = float(solution.t_wall_gas[station])
    return StartupResult(
        time=times, wall_gas_temp=tg, wall_cool_temp=tc, heat_flux=qq,
        peak_wall_temp=peak, peak_time=float(times[int(np.argmax(tg))]),
        steady_wall_temp=steady, overshoot=peak - steady,
        material=m, survives=peak < m.max_wall_temp,
        _diffusion=t_hot * t_hot / alpha,
    )
