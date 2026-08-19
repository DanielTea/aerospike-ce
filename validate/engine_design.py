"""
The whole engine, end to end, from one spec file.

    python engine_design.py --spec ../spec/regen.json

Order matters and is not arbitrary:

    contour -> assembly -> chamber -> heat load -> coolant split -> cooling
                                                                 -> injector

The coolant split is the interesting link. The cowl and the centrebody do not
carry the same heat -- the centrebody has the chamber wall *and* the whole plug
surface, so it runs roughly 60/40 against the cowl -- and splitting the fuel
evenly starves it. Rather than accept a fraction as input, the load of each
circuit is measured first with a reference channel and the fuel is divided in
that ratio. A guessed split is how the centrebody ends up melting while the cowl
sits comfortably.

Everything this reports is derived. The only inputs are in spec/*.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

from combustion_ref import ChamberConditions, chamber_from_spec
from cooling_ref import (
    MATERIALS,
    ChannelSpec,
    CoolingCandidate,
    search_cooling_design,
    solve_cooling,
)
from engine_ref import EngineAssembly, assembly_from_spec, wall_flow_state
from injector_ref import InjectorDesign, chug_margin, fit_injector_to_face, fits_on_face

# used only to weigh the circuits against each other, never to size them
REFERENCE_CHANNEL = ChannelSpec(n_channels=200, width_mm=0.4, height_mm=0.6,
                                land_mm=0.4, hot_wall_mm=0.3)


@dataclass
class EngineDesign:
    spec: dict
    assembly: EngineAssembly
    chamber: ChamberConditions
    injector: InjectorDesign | None
    injector_note: str
    circuits: dict                      # name -> CoolingCandidate or None
    coolant_split: dict                 # name -> fraction
    heat_load: dict                     # name -> W, at reference channel
    notes: list

    @property
    def cooled(self) -> bool:
        return all(c is not None for c in self.circuits.values())

    @property
    def total_heat(self) -> float:
        return sum(c.solution.total_heat for c in self.circuits.values()
                   if c is not None)

    @property
    def coolant_capacity(self) -> float:
        """Heat the fuel could absorb before hitting its temperature limit, W."""
        p = self.chamber.propellant
        return (self.chamber.mass_flow_fuel * p.coolant_cp
                * (p.coolant_max_t - p.coolant_inlet_t))


def _circuit_walls(a: EngineAssembly, gamma: float):
    w = wall_flow_state(a, gamma)
    return {
        "cowl": (w["cowl"], a.chamber_outer_radius),
        "centrebody": (w["centrebody"], a.shoulder_radius),
    }


def design_engine(spec: dict) -> EngineDesign:
    notes: list[str] = []

    a = assembly_from_spec(spec)
    c = a.contour
    chamber = chamber_from_spec(spec, c.throat_area,
                                spec["nozzle"]["expansion_ratio"], c.exit_mach)

    if abs(spec["gas"]["gamma"] - chamber.propellant.gamma) > 1e-9:
        notes.append(
            f"gas.gamma {spec['gas']['gamma']} disagrees with propellant gamma "
            f"{chamber.propellant.gamma}: the contour and the performance model "
            f"are describing different gases")

    cool = spec.get("cooling", {})
    materials = tuple(cool.get("materials", ("grcop42", "inconel718")))
    dt = 2.0 * c.throat_slant_length     # hydraulic diameter of the slot throat

    walls = _circuit_walls(a, chamber.propellant.gamma)

    # --- measure each circuit's load before dividing the coolant -------------
    heat_load: dict[str, float] = {}
    for name, (d, r_ref) in walls.items():
        sol = solve_cooling(chamber, d["x"], d["r"], d["area_ratio"], d["mach"],
                            REFERENCE_CHANNEL, MATERIALS[materials[0]], dt, dt,
                            coolant_fraction=1.0)
        heat_load[name] = sol.total_heat

    total_load = sum(heat_load.values())
    split = {k: v / total_load for k, v in heat_load.items()}

    # --- solve each circuit on its share ------------------------------------
    circuits: dict[str, CoolingCandidate | None] = {}
    for name, (d, r_ref) in walls.items():
        best, evaluated = search_cooling_design(
            chamber, d["x"], d["r"], d["area_ratio"], d["mach"], dt, dt,
            reference_radius_mm=r_ref,
            materials=materials,
            coolant_fraction=split[name],
            max_pressure_drop_fraction=cool.get("max_pressure_drop_fraction", 0.5),
            min_channel_width_mm=cool.get("min_channel_width_mm", 0.4),
        )
        circuits[name] = best
        if best is None:
            worst = max(evaluated, key=lambda cc: cc.score)
            notes.append(f"{name}: no channel geometry survives. Closest was "
                         f"{worst.material.name} -- {worst.reason}")

    capacity = (chamber.mass_flow_fuel * chamber.propellant.coolant_cp
                * (chamber.propellant.coolant_max_t - chamber.propellant.coolant_inlet_t))
    if total_load > capacity:
        notes.append(
            f"heat load {total_load / 1e3:.0f} kW exceeds what the fuel can absorb "
            f"({capacity / 1e3:.0f} kW). No channel geometry fixes that: the engine "
            f"needs more mass flow, a higher chamber pressure, film cooling, or to "
            f"be larger")

    # --- injector -----------------------------------------------------------
    inj_spec = spec.get("injector", {})
    injector, why = fit_injector_to_face(
        chamber, a.shoulder_radius, a.chamber_outer_radius,
        stiffness=inj_spec.get("stiffness", 0.20),
        discharge_coefficient=inj_spec.get("discharge_coefficient", 0.75),
        min_land_mm=inj_spec.get("min_land_mm", 0.8),
    )
    if injector is not None and not fits_on_face(
            injector, a.shoulder_radius, a.chamber_outer_radius,
            inj_spec.get("min_land_mm", 0.8))[0]:
        notes.append(f"injector does not fit the face: {why}")
        injector = None
    if injector is not None and chug_margin(injector) < 1.0:
        notes.append(f"injection stiffness {injector.stiffness_fuel:.2f} is below the "
                     f"0.15 chug floor")

    return EngineDesign(spec, a, chamber, injector, why,
                        circuits, split, heat_load, notes)


def report(d: EngineDesign) -> str:
    a, c, ch = d.assembly, d.assembly.contour, d.chamber
    L = []
    L.append(f"=== {d.spec.get('name', 'engine')} ===")
    L.append("")
    L.append("geometry")
    L.append(f"  throat area        {c.throat_area:10.3f} mm2   (inclined, slant {c.throat_slant_length:.3f} mm)")
    L.append(f"  exit Mach          {c.exit_mach:10.4f}")
    L.append(f"  contraction ratio  {a.chamber_area / c.throat_area:10.4f}")
    L.append(f"  overall            {a.overall_length:10.2f} mm long, {2 * a.overall_radius:.2f} mm dia")
    L.append(f"  mass (parts)       {sum(p.revolved_volume for p in a.profiles.values()) / 1000:10.2f} cm3")
    L.append("")
    L.append(f"performance  ({ch.propellant.oxidiser}/{ch.propellant.fuel} at MR {ch.mixture_ratio:.2f}, pc {ch.chamber_pressure / 1e5:.1f} bar)")
    L.append(f"  mass flow          {ch.mass_flow * 1e3:10.1f} g/s  (fuel {ch.mass_flow_fuel * 1e3:.1f}, ox {ch.mass_flow_ox * 1e3:.1f})")
    L.append(f"  chamber temp       {ch.t_chamber:10.0f} K")
    L.append(f"  c* (delivered)     {ch.c_star:10.1f} m/s")
    L.append(f"  thrust             {ch.thrust_sea_level:10.0f} N sea level, {ch.thrust_vacuum:.0f} N vacuum")
    L.append(f"  Isp                {ch.isp_sea_level:10.1f} s sea level, {ch.isp_vacuum:.1f} s vacuum")
    L.append(f"  exit pressure      {ch.exit_pressure / 1e5:10.4f} bar"
             f"   ({'overexpanded' if ch.exit_pressure < 101325 else 'underexpanded'} at sea level)")
    L.append("")
    L.append("injector")
    if d.injector is None:
        L.append(f"  NONE FOUND: {d.injector_note}")
    else:
        i = d.injector
        L.append(f"  layout             {d.injector_note}")
        L.append(f"  fuel orifice       {i.d_fuel_mm:10.3f} mm x {i.n_elements}  at r {i.fuel_ring_radius * 1e3:.2f} mm, {math.degrees(i.angle_fuel):.1f} deg")
        L.append(f"  ox orifice         {i.d_ox_mm:10.3f} mm x {i.n_elements}  at r {i.ox_ring_radius * 1e3:.2f} mm, {math.degrees(i.angle_ox):.1f} deg")
        L.append(f"  injection dP       {i.dp_fuel / 1e5:10.2f} bar  ({i.stiffness_fuel * 100:.0f}% of pc, chug margin {chug_margin(i):.2f})")
        L.append(f"  velocities         {i.v_fuel:10.1f} m/s fuel, {i.v_ox:.1f} m/s ox")
        L.append(f"  momentum ratio     {i.momentum_ratio:10.3f}   resultant {math.degrees(i.resultant_angle):+.2e} deg off axis")
    L.append("")
    L.append(f"cooling   (fuel split by measured heat load, capacity {d.coolant_capacity / 1e3:.0f} kW)")
    for name, cand in d.circuits.items():
        share = d.coolant_split[name]
        if cand is None:
            L.append(f"  {name:12s} {share * 100:4.0f}% of fuel   NO SURVIVING DESIGN")
            continue
        s, cc = cand.solution, cand.channel
        L.append(f"  {name:12s} {share * 100:4.0f}% of fuel   {cand.material.name}, "
                 f"{cc.n_channels} x {cc.width_mm:.2f}x{cc.height_mm:.2f} mm, "
                 f"{cc.hot_wall_mm:.1f} mm hot wall")
        L.append(f"  {'':12s} peak flux {s.peak_heat_flux / 1e6:5.1f} MW/m2   "
                 f"wall {s.peak_wall_temperature:4.0f} K (margin {s.wall_temperature_margin:+.0f})   "
                 f"coolant out {s.coolant_outlet_temperature:.0f} K (margin {s.coolant_temperature_margin:+.0f})")
        L.append(f"  {'':12s} heat {s.total_heat / 1e3:5.0f} kW   drop {s.pressure_drop / 1e5:.2f} bar   "
                 f"Re {s.reynolds.min():.0f}-{s.reynolds.max():.0f}   "
                 f"land/channel {cc.land_to_channel_ratio(a.chamber_outer_radius):.2f}")
    L.append("")
    L.append(f"  total heat         {d.total_heat / 1e3:10.0f} kW"
             f"   ({d.total_heat / d.coolant_capacity * 100:.0f}% of coolant capacity)")
    if d.notes:
        L.append("")
        L.append("NOTES")
        for n in d.notes:
            L.append(f"  - {n}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="../spec/regen.json")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    print(report(design_engine(spec)))


if __name__ == "__main__":
    main()
