"""
Regenerative cooling: channel sizing, wall temperatures, coolant rise and drop.

Axial channels milled into the hot wall and closed by a jacket. Coolant is the
fuel, entering at the throat end where the flux is worst and leaving at the
injector, so the coldest coolant meets the hottest station.

Method
------
At each station the three resistances sit in series:

    q = (T_aw - T_coolant) / (1/h_g + t_wall/k_wall + 1/h_coolant)

h_g from Bartz, h_coolant from Dittus-Boelter, and the wall by conduction. The
Bartz correction factor sigma depends on the hot-wall temperature it is being
used to find, so each station iterates a handful of times; it converges in three
or four because sigma is a weak function of its own argument.

Marching forward, the coolant picks up q * dA at each station, its temperature
rises, and the friction drop accumulates by Darcy-Weisbach.

Standing on
-----------
Bartz (1957) for the gas-side coefficient. It is a correlation fitted to
conventional bell nozzles, it is commonly quoted as +/-30 percent, and it is
being applied here to an annular throat with a hydraulic diameter substituted
for the throat diameter. Treat the absolute numbers as an order of magnitude
and the trends as real. A design that only just passes on Bartz has not passed.

Dittus-Boelter assumes fully developed turbulent flow in a smooth round tube.
Channels are neither round nor smooth, especially printed ones, so the hydraulic
diameter and a roughness multiplier are doing real work here.

Not modelled: film cooling, thermal barrier coatings, curvature enhancement in
the coolant channels, nucleate boiling or supercritical property variation,
transient startup, or the structural problem of a hot wall in compression
against a cold jacket. Any of those can decide whether a real chamber survives.

SI throughout: W, m, K, Pa, kg, s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from combustion_ref import R_UNIVERSAL, ChamberConditions


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """
    Thermal and mechanical properties.

    Yield and modulus are quoted at room temperature and derated linearly to
    zero at the melting point, which is crude but captures the effect that
    matters: a hot wall is much weaker than a cold one, and the hot wall is
    exactly where the stress is.

    Ductility and the fatigue exponent feed a Coffin-Manson life estimate.
    Regeneratively cooled chambers do not fail by bursting; they fail by
    thermal-strain ratchetting over tens to hundreds of cycles, and the
    canonical failure is the hot wall bulging into the channel and thinning
    until it splits.
    """
    name: str
    conductivity: float         # W/(m K)
    max_wall_temp: float        # K, sustained
    density: float              # kg/m^3
    youngs_modulus: float = 120e9       # Pa at 293 K
    poisson: float = 0.33
    yield_strength: float = 300e6       # Pa at 293 K
    expansion: float = 17e-6            # 1/K
    melting_point: float = 1600.0       # K
    fatigue_ductility: float = 0.35     # Coffin-Manson eps_f'
    fatigue_exponent: float = -0.6      # Coffin-Manson c


MATERIALS: dict[str, Material] = {
    # printed nickel superalloy, the usual choice for a small regen chamber
    "inconel718": Material("inconel718", 25.0, 1150.0, 8190.0,
                           youngs_modulus=200e9, poisson=0.29,
                           yield_strength=1030e6, expansion=13e-6,
                           melting_point=1610.0, fatigue_ductility=0.25),
    # printed copper alloys: far better conductors, much lower temperature limit
    "cucrzr": Material("cucrzr", 320.0, 800.0, 8890.0,
                       youngs_modulus=128e9, poisson=0.34,
                       yield_strength=280e6, expansion=17e-6,
                       melting_point=1350.0, fatigue_ductility=0.40),
    "grcop42": Material("grcop42", 290.0, 1000.0, 8756.0,
                        youngs_modulus=110e9, poisson=0.34,
                        yield_strength=190e6, expansion=17.5e-6,
                        melting_point=1350.0, fatigue_ductility=0.45),
}


@dataclass(frozen=True)
class ChannelSpec:
    """
    Channel cross-section.

    The defaults suit a sub-kilonewton engine. Scaling them up naively is how
    you end up with coolant crawling through at 1 m/s: the flow area grows with
    the channel while the mass flow does not, the coolant-side coefficient
    collapses, and the wall cooks. Use `size_channels_for_velocity` instead of
    guessing.
    """
    n_channels: int = 80
    width_mm: float = 0.6           # circumferential width of the channel
    height_mm: float = 1.5          # radial depth of the channel
    land_mm: float = 0.8            # rib between adjacent channels
    hot_wall_mm: float = 0.5        # gas side wall, the thermal bottleneck
    roughness_um: float = 20.0      # as-printed, unpolished

    @property
    def aspect_ratio(self) -> float:
        return self.height_mm / self.width_mm

    def land_to_channel_ratio(self, radius_mm: float) -> float:
        """
        Land width over channel width at a given radius.

        Above about 2 the lands carry more of the hot wall than the channels do,
        and the fin efficiency rather than the coolant velocity is what sets the
        wall temperature. Small engines land here easily, because the coolant
        flow is too small to justify many channels.
        """
        pitch = 2.0 * math.pi * radius_mm / self.n_channels
        return (pitch - self.width_mm) / self.width_mm


def size_channels_for_velocity(
    chamber: ChamberConditions,
    reference_radius_mm: float,      # the NARROWEST station, not the widest
    target_velocity: float,
    aspect_ratio: float = 2.5,
    land_mm: float = 0.8,
    hot_wall_mm: float = 0.5,
    roughness_um: float = 20.0,
    coolant_fraction: float = 1.0,
    min_width_mm: float = 0.4,
) -> ChannelSpec:
    """
    Solve the channel count and cross-section for a target coolant velocity.

    Velocity is the design variable that matters: the coolant-side coefficient
    goes as v^0.8 while the pressure drop goes as v^2, so the whole design sits
    on that trade. Ten to sixty m/s is the usual band.

    Fixing the channel count first and solving only the depth is what produces
    a 0.08 mm deep channel three millimetres wide: the pitch is whatever the
    circumference gives, so the channel goes wide and flat, the hydraulic
    diameter collapses, and the pressure drop explodes. Real channels are taller
    than they are wide.

    So the aspect ratio is held instead and the count is solved. For n channels
    each carrying mdot/n at velocity v the area is fixed, giving

        w(n) = sqrt(mdot / (n * rho * v * AR))

    and packing them round the circumference wants  n * (w(n) + land) = 2*pi*r.
    That is monotonic in n, so bisection finds it without fuss.

    Packing is not the only limit, and on a small engine it is not the binding
    one. Width falls as the count rises, so on a sub-kilonewton chamber the
    circumference will happily accept two hundred channels a sixth of a
    millimetre wide, which no powder-bed machine will print. `min_width_mm`
    caps the count at what the process can actually hold, and the caller finds
    out from `land_to_channel_ratio` what that cost them: fewer channels on a
    fixed circumference means wider lands, and wider lands mean the fin
    efficiency starts deciding the wall temperature.
    """
    if target_velocity <= 0.0:
        raise ValueError("target velocity must be positive")
    if aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be positive")

    prop = chamber.propellant
    mdot = chamber.mass_flow_fuel * coolant_fraction
    circumference = 2.0 * math.pi * reference_radius_mm * 1e-3
    land = land_mm * 1e-3

    def width_for(n: float) -> float:
        return math.sqrt(mdot / (n * prop.density_fuel * target_velocity * aspect_ratio))

    def packing(n: float) -> float:
        return n * (width_for(n) + land) - circumference

    lo, hi = 1.0, 4.0
    while packing(hi) < 0.0:
        hi *= 2.0
        if hi > 100000.0:
            raise ValueError("cannot pack channels at this velocity and land width")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if packing(mid) < 0.0:
            lo = mid
        else:
            hi = mid

    n_packing = 0.5 * (lo + hi)

    # width(n) = min_width solved for n
    w_min = min_width_mm * 1e-3
    n_manufacturable = mdot / (prop.density_fuel * target_velocity * aspect_ratio * w_min * w_min)

    n_channels = max(1, int(math.floor(min(n_packing, n_manufacturable))))
    width = width_for(n_channels)
    height = aspect_ratio * width

    return ChannelSpec(
        n_channels=n_channels,
        width_mm=width * 1e3,
        height_mm=height * 1e3,
        land_mm=land_mm,
        hot_wall_mm=hot_wall_mm,
        roughness_um=roughness_um,
    )


def fin_efficiency(h_coolant: float, conductivity: float,
                   land_thickness: float, channel_height: float) -> float:
    """
    Efficiency of the rib between two channels treated as a straight fin.

    The lands are not passive. Heat entering the hot wall above a land has to
    run sideways into the rib and down into the coolant, and the rib is only as
    good at that as its conductivity and thickness allow. Ignoring it credits
    the design with cooled area it does not have -- the error grows with the
    land-to-channel ratio and is worst exactly where the channels are widest
    apart.

    Adiabatic-tip straight fin: eta = tanh(mL)/(mL), m = sqrt(2h/(k*t)).
    """
    m = math.sqrt(2.0 * h_coolant / (conductivity * land_thickness))
    ml = m * channel_height
    return math.tanh(ml) / ml if ml > 1e-9 else 1.0


# --------------------------------------------------------------------------
# gas properties
# --------------------------------------------------------------------------

def gas_cp(gamma: float, molar_mass: float) -> float:
    """Specific heat at constant pressure of the products, J/(kg K)."""
    return gamma * (R_UNIVERSAL / molar_mass) / (gamma - 1.0)


def gas_prandtl(gamma: float) -> float:
    """Prandtl number of combustion products, the standard 4g/(9g-5) estimate."""
    return 4.0 * gamma / (9.0 * gamma - 5.0)


def gas_viscosity(molar_mass: float, temperature: float) -> float:
    """Bartz's own viscosity correlation, Pa s."""
    return 1.184e-7 * math.sqrt(molar_mass) * temperature ** 0.6


def adiabatic_wall_temperature(t_chamber: float, mach: float,
                               gamma: float, prandtl: float) -> float:
    """
    Recovery temperature seen by the wall, K.

    Lower than the stagnation temperature because recovery is imperfect; the
    turbulent recovery factor is Pr^(1/3).
    """
    r = prandtl ** (1.0 / 3.0)
    m2 = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    return t_chamber * (1.0 + r * 0.5 * (gamma - 1.0) * mach * mach) / m2


def bartz_coefficient(
    chamber: ChamberConditions,
    throat_diameter: float,
    curvature_radius: float,
    area_ratio: float,
    mach: float,
    wall_temperature: float,
) -> float:
    """
    Gas-side film coefficient from the Bartz correlation, W/(m^2 K).

    area_ratio is A/A_t at the station, so it is 1 at the throat and larger
    everywhere else, and the (A_t/A)^0.9 term is what makes the throat the
    hottest place in the engine by a wide margin.
    """
    prop = chamber.propellant
    gamma = prop.gamma
    cp = gas_cp(gamma, prop.molar_mass)
    pr = gas_prandtl(gamma)
    mu = gas_viscosity(prop.molar_mass, chamber.t_chamber)

    m2 = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
    tw_ratio = wall_temperature / chamber.t_chamber
    sigma = 1.0 / ((0.5 * tw_ratio * m2 + 0.5) ** 0.68 * m2 ** 0.12)

    return (
        (0.026 / throat_diameter ** 0.2)
        * (mu ** 0.2 * cp / pr ** 0.6)
        * (chamber.chamber_pressure / chamber.c_star) ** 0.8
        * (throat_diameter / curvature_radius) ** 0.1
        * (1.0 / area_ratio) ** 0.9
        * sigma
    )


# --------------------------------------------------------------------------
# coolant side
# --------------------------------------------------------------------------

def dittus_boelter(reynolds: float, prandtl: float, conductivity: float,
                   hydraulic_diameter: float, roughness_factor: float = 1.0) -> float:
    """
    Coolant-side film coefficient, W/(m^2 K).

    Heating correlation, so the Prandtl exponent is 0.4. Below Re 2300 the
    correlation does not apply at all; the caller checks for that rather than
    letting it return a confident wrong number.
    """
    return roughness_factor * 0.023 * reynolds ** 0.8 * prandtl ** 0.4 \
        * conductivity / hydraulic_diameter


def friction_factor(reynolds: float, relative_roughness: float) -> float:
    """
    Darcy friction factor, Haaland's explicit approximation to Colebrook.

    Printed channels are rough enough that the smooth-tube Blasius result
    understates the drop substantially, which is why roughness is an input.
    """
    if reynolds < 2300.0:
        return 64.0 / max(reynolds, 1e-6)
    inv = -1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds)
    return 1.0 / (inv * inv)


# --------------------------------------------------------------------------
# the marching solution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoolingSolution:
    x: np.ndarray                   # mm, station positions in model coordinates
    r: np.ndarray                   # mm, wall radius
    mach: np.ndarray
    area_ratio: np.ndarray
    heat_flux: np.ndarray           # W/m^2
    t_adiabatic: np.ndarray         # K
    t_wall_gas: np.ndarray          # K
    t_wall_coolant: np.ndarray      # K
    t_coolant: np.ndarray           # K
    h_gas: np.ndarray               # W/(m^2 K)
    h_coolant: np.ndarray
    channel_width: np.ndarray       # mm
    hydraulic_diameter: np.ndarray  # mm
    velocity: np.ndarray            # m/s
    reynolds: np.ndarray
    pressure: np.ndarray            # Pa, coolant static pressure

    material: Material
    channel: ChannelSpec
    coolant_mass_flow: float        # kg/s, total
    total_heat: float               # W

    # Arrays are indexed in *gas* order, injector to throat. The coolant runs
    # the other way, so its inlet is the last index and its outlet the first.
    # Reading the rise off the array order instead gives it backwards, which
    # looks like the coolant cooling down as it absorbs heat.
    coolant_inlet_index: int
    coolant_outlet_index: int
    coolant_outlet_temp: float = 0.0
    coolant_inlet_temp: float = 0.0
    coolant_pressure_out: float = 0.0
    coolant_pressure_in: float = 0.0
    t_coating_surface: np.ndarray | None = None
    film_effectiveness_profile: np.ndarray | None = None
    coating: "CoatingSpec | None" = None

    @property
    def peak_coating_temperature(self) -> float:
        """Hottest point on the coating, which is not the hottest metal."""
        return float(self.t_coating_surface.max()) if self.t_coating_surface is not None \
            else self.peak_wall_temperature

    @property
    def coating_temperature_margin(self) -> float:
        if self.coating is None or self.coating.thickness_mm <= 0.0:
            return float("inf")
        return self.coating.max_surface_temp - self.peak_coating_temperature

    @property
    def peak_heat_flux(self) -> float:
        return float(self.heat_flux.max())

    @property
    def peak_wall_temperature(self) -> float:
        return float(self.t_wall_gas.max())

    @property
    def coolant_inlet_temperature(self) -> float:
        return self.coolant_inlet_temp

    @property
    def coolant_outlet_temperature(self) -> float:
        """
        True outlet temperature.

        Taken from the end of the march, not from t_coolant[0]. The array holds
        the temperature *entering* each station, so its first entry is missing
        the last station's heat and the energy balance would miss with it.
        """
        return self.coolant_outlet_temp

    @property
    def coolant_temperature_rise(self) -> float:
        return self.coolant_outlet_temperature - self.coolant_inlet_temperature

    @property
    def pressure_drop(self) -> float:
        return self.coolant_pressure_in - self.coolant_pressure_out

    @property
    def wall_temperature_margin(self) -> float:
        """How far the hot wall sits below the material limit, K."""
        return self.material.max_wall_temp - self.peak_wall_temperature

    coolant_max_temp: float = 0.0

    @property
    def coolant_temperature_margin(self) -> float:
        """How far the coolant stays below its coking or critical limit, K."""
        return self.coolant_max_temp - self.coolant_outlet_temperature

    @property
    def survives(self) -> bool:
        return self.wall_temperature_margin > 0.0


def solve_cooling(
    chamber: ChamberConditions,
    wall_x_mm: np.ndarray,
    wall_r_mm: np.ndarray,
    area_ratio: np.ndarray,
    mach: np.ndarray,
    channel: ChannelSpec,
    material: Material,
    throat_hydraulic_diameter_mm: float,
    curvature_radius_mm: float,
    coolant_fraction: float = 1.0,
    inlet_pressure_bar: float | None = None,
    film: "FilmSpec | None" = None,
    coating: "CoatingSpec | None" = None,
    channel_x_range: tuple[float, float] | None = None,
) -> CoolingSolution:
    """
    March the coolant from the last station to the first.

    The arrays are given in flow order for the hot gas -- injector to throat --
    and the coolant runs against it, so the march is reversed. That is the point
    of counterflow: the coolant is coldest exactly where the flux is highest.

    coolant_fraction splits the fuel between circuits when more than one surface
    is cooled; the cowl and the centrebody each take a share.
    """
    n = len(wall_x_mm)
    if not (len(wall_r_mm) == len(area_ratio) == len(mach) == n):
        raise ValueError("wall arrays must be the same length")
    if n < 2:
        raise ValueError("need at least two stations")
    if not 0.0 < coolant_fraction <= 1.0:
        raise ValueError("coolant_fraction must be in (0, 1]")
    if channel.n_channels < 1:
        raise ValueError("need at least one channel")

    prop = chamber.propellant
    mdot_c = chamber.mass_flow_fuel * coolant_fraction
    mdot_per_channel = mdot_c / channel.n_channels

    pr_c = prop.coolant_mu * prop.coolant_cp / prop.coolant_k
    dt = float(throat_hydraulic_diameter_mm) * 1e-3
    rc = float(curvature_radius_mm) * 1e-3

    # film effectiveness along the wall, zero upstream of the slot
    if film is not None and film.fuel_fraction > 0.0:
        mdot_film = chamber.mass_flow_fuel * film.fuel_fraction
        slot_r = float(np.interp(film.inject_x_mm, wall_x_mm, wall_r_mm))
        slot_area = 2.0 * math.pi * slot_r * 1e-3 * film.slot_height_mm * 1e-3
        v_film = mdot_film / (prop.density_fuel * slot_area)
        # blowing ratio against the core flow at the injector end
        core_flux = chamber.mass_flow / max(math.pi * (slot_r * 1e-3) ** 2, 1e-9)
        blowing = (prop.density_fuel * v_film) / max(core_flux, 1e-9)
        eta_film = film_effectiveness(np.asarray(wall_x_mm, dtype=float),
                                      film.inject_x_mm, film.slot_height_mm,
                                      max(blowing, 0.05))
        t_film = film.coolant_temp_k if film.coolant_temp_k > 0.0 else prop.coolant_inlet_t
    else:
        eta_film = np.zeros(n)
        t_film = prop.coolant_inlet_t

    r_coating = (coating.thickness_mm * 1e-3 / coating.conductivity
                 if coating is not None and coating.thickness_mm > 0.0 else 0.0)

    q = np.zeros(n)
    t_aw = np.zeros(n)
    t_coat = np.zeros(n)
    t_wg = np.zeros(n)
    t_wc = np.zeros(n)
    t_c = np.zeros(n)
    hg = np.zeros(n)
    hc = np.zeros(n)
    w_ch = np.zeros(n)
    dh = np.zeros(n)
    vel = np.zeros(n)
    re = np.zeros(n)
    press = np.zeros(n)

    gamma = prop.gamma
    pr_g = gas_prandtl(gamma)
    t_wall_guess = 0.6 * chamber.t_chamber

    p = (inlet_pressure_bar * 1e5 if inlet_pressure_bar is not None
         else 1.5 * chamber.chamber_pressure)
    t_cool = prop.coolant_inlet_t
    total_q = 0.0

    order = list(range(n - 1, -1, -1))   # counterflow: aft station first
    for k, i in enumerate(order):
        r_m = wall_r_mm[i] * 1e-3
        pitch = 2.0 * math.pi * r_m / channel.n_channels
        # the channel keeps its designed width; the land takes up the slack as
        # the radius changes, which is what a straight-milled channel does
        width = min(channel.width_mm * 1e-3, max(pitch - channel.land_mm * 1e-3, 1e-4))
        height = channel.height_mm * 1e-3
        land = max(pitch - width, 1e-5)

        area_c = width * height
        d_h = 2.0 * area_c / (width + height)
        v = mdot_per_channel / (prop.density_fuel * area_c)
        re_i = prop.density_fuel * v * d_h / prop.coolant_mu

        rough = channel.roughness_um * 1e-6
        # printed channels run rough; the standard uplift on the heat transfer
        # side is modest compared with what it does to the pressure drop
        rough_factor = 1.0 + 0.3 * min(rough / d_h / 0.01, 1.0)
        hc_bare = dittus_boelter(re_i, pr_c, prop.coolant_k, d_h, rough_factor)

        # Referred to the hot-gas area of one pitch. The channel floor is fully
        # effective, the two rib flanks only as far as the fin carries.
        eta = fin_efficiency(hc_bare, material.conductivity, land, height)
        hc_i = hc_bare * (width + 2.0 * eta * height) / pitch

        t_aw_i = adiabatic_wall_temperature(chamber.t_chamber, mach[i], gamma, pr_g)
        t_aw_i = filmed_recovery_temperature(t_aw_i, eta_film[i], t_film)

        # a station with no channel under it gets no coolant-side path at all
        has_channel = (channel_x_range is None
                       or channel_x_range[0] <= wall_x_mm[i] <= channel_x_range[1])

        # sigma depends on the wall temperature it is used to compute
        tw = t_wall_guess
        r_wall = channel.hot_wall_mm * 1e-3 / material.conductivity
        r_cool = 1.0 / hc_i if has_channel else 1e9   # uncooled: no path out
        for _ in range(12):
            hg_i = bartz_coefficient(chamber, dt, rc, area_ratio[i], mach[i], tw)
            r_total = 1.0 / hg_i + r_coating + r_wall + r_cool
            q_i = (t_aw_i - t_cool) / r_total
            tw_new = t_aw_i - q_i / hg_i
            if abs(tw_new - tw) < 0.01:
                tw = tw_new
                break
            tw = 0.5 * (tw + tw_new)
        t_wall_guess = tw

        hg_i = bartz_coefficient(chamber, dt, rc, area_ratio[i], mach[i], tw)
        r_total = 1.0 / hg_i + r_coating + r_wall + r_cool
        q_i = (t_aw_i - t_cool) / r_total

        # axial length this station is responsible for
        if k == 0:
            dx = abs(wall_x_mm[i] - wall_x_mm[i - 1]) * 1e-3
        else:
            j = order[k - 1]
            dx = abs(wall_x_mm[j] - wall_x_mm[i]) * 1e-3
        seg = math.hypot(dx, 0.0) if dx > 0 else 0.0

        # hot-gas-side area belonging to one channel at this station
        area_hot = pitch * seg
        heat = q_i * area_hot * channel.n_channels
        total_q += heat

        q[i] = q_i
        t_aw[i] = t_aw_i
        t_coat[i] = tw                       # coating surface, or bare metal
        t_wg[i] = tw - q_i * r_coating       # metal under the coating
        t_wc[i] = t_cool + q_i / hc_i
        t_c[i] = t_cool
        hg[i] = hg_i
        hc[i] = hc_i
        w_ch[i] = width * 1e3
        dh[i] = d_h * 1e3
        vel[i] = v
        re[i] = re_i
        press[i] = p

        # advance the coolant. t_c[i] holds the temperature entering this
        # station, so the outlet is only known once the march finishes.
        t_cool += heat / (mdot_c * prop.coolant_cp)
        f = friction_factor(re_i, rough / d_h)
        p -= f * (seg / d_h) * 0.5 * prop.density_fuel * v * v

    return CoolingSolution(
        x=np.asarray(wall_x_mm, dtype=float),
        r=np.asarray(wall_r_mm, dtype=float),
        mach=np.asarray(mach, dtype=float),
        area_ratio=np.asarray(area_ratio, dtype=float),
        heat_flux=q, t_adiabatic=t_aw, t_wall_gas=t_wg, t_wall_coolant=t_wc,
        t_coolant=t_c, h_gas=hg, h_coolant=hc,
        channel_width=w_ch, hydraulic_diameter=dh, velocity=vel, reynolds=re,
        pressure=press,
        material=material, channel=channel,
        coolant_mass_flow=mdot_c, total_heat=total_q,
        coolant_inlet_index=n - 1, coolant_outlet_index=0,
        coolant_max_temp=prop.coolant_max_t,
        t_coating_surface=t_coat,
        film_effectiveness_profile=eta_film,
        coating=coating,
        coolant_outlet_temp=t_cool,
        coolant_inlet_temp=prop.coolant_inlet_t,
        coolant_pressure_out=p,
        coolant_pressure_in=(inlet_pressure_bar * 1e5 if inlet_pressure_bar is not None
                             else 1.5 * chamber.chamber_pressure),
    )


# --------------------------------------------------------------------------
# design search
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoolingCandidate:
    channel: ChannelSpec
    material: Material
    solution: CoolingSolution
    survives: bool
    reason: str
    score: float


def channels_packed(reference_radius_mm: float, width_mm: float, land_mm: float) -> int:
    """How many channels of a given width and land fit round a circumference."""
    pitch = width_mm + land_mm
    return max(1, int(2.0 * math.pi * reference_radius_mm / pitch))


def _downsample(a: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return np.asarray(a)[idx]


def search_cooling_design(
    chamber: ChamberConditions,
    wall_x_mm: np.ndarray,
    wall_r_mm: np.ndarray,
    area_ratio: np.ndarray,
    mach: np.ndarray,
    throat_hydraulic_diameter_mm: float,
    curvature_radius_mm: float,
    reference_radius_mm: float,
    materials: tuple[str, ...] = ("grcop42", "inconel718"),
    coolant_fraction: float = 1.0,
    max_pressure_drop_fraction: float = 0.5,
    search_stations: int = 80,
    min_channel_width_mm: float = 0.4,
    min_hot_wall_mm: float = 0.3,
    max_depth_mm: float = float("inf"),
    film: "FilmSpec | None" = None,
    coating: "CoatingSpec | None" = None,
) -> tuple[CoolingCandidate | None, list[CoolingCandidate]]:
    """
    Search channel geometry and material for a jacket that survives.

    Sizing a cooling circuit by picking a coolant velocity works on a big engine
    and fails on a small one. The binding constraint here is not velocity, it is
    how much cooled area fits on the circumference once the channels are no
    narrower than the process can print. Chasing velocity drives the channel
    count down, the lands then carry most of the hot wall, and the fin
    efficiency decides the wall temperature no matter how fast the coolant goes.

    So this searches instead of solving in one shot. It packs the circumference
    at each candidate width and land, sweeps depth, hot wall and material, and
    keeps anything whose wall and coolant both stay inside their limits and
    whose pressure drop stays under a stated fraction of chamber pressure.

    `min_hot_wall_mm` is the other process floor, and the one that usually
    matters more. The search will happily pick the thinnest wall on offer,
    because it is the largest single thermal resistance in the stack. A third of
    a millimetre of metal between 3400 K gas and cryogenic methane is not
    something a powder-bed machine holds reliably, and it is also too thin to
    survive being meshed or CT inspected afterwards.

    `reference_radius_mm` must be the smallest radius the channels have to run
    round, not the largest. Packing at the widest station puts more channels on
    the circumference than the narrowest one can hold, and they then have to
    narrow to fit -- on this engine down to 0.27 mm from a designed 0.40 mm,
    which is thinner than the process floor the search was given and too thin to
    mesh or inspect. A real design bifurcates channels rather than starving
    them; this one simply carries fewer, which is the conservative answer.

    `max_depth_mm` is what the structure will actually give up. Hot wall plus
    channel depth has to fit inside the wall with something left behind it, and
    the search has no other reason to care how deep it goes -- left unbounded it
    will take a 1.0 mm hot wall and a 1.6 mm channel out of a 3.0 mm wall and
    leave 0.4 mm of backing, or ask for more than the wall has. Geometry
    constrains the thermal design here, not the other way round.

    `min_channel_width_mm` is the process floor. It is worth setting well above
    the absolute minimum a machine can hold: the narrowest surviving design is
    rarely the best one to build, and a channel only a voxel or two wide also
    cannot be meshed or inspected reliably afterwards.

    Returns the best candidate and every candidate evaluated, so a design that
    fails everywhere can be diagnosed rather than merely reported.
    """
    xs = _downsample(wall_x_mm, search_stations)
    rs = _downsample(wall_r_mm, search_stations)
    ars = _downsample(area_ratio, search_stations)
    ms = _downsample(mach, search_stations)

    evaluated: list[CoolingCandidate] = []
    dp_limit = max_pressure_drop_fraction * chamber.chamber_pressure

    for mat_name in materials:
        material = MATERIALS[mat_name]
        for width in [w for w in (0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
                      if w >= min_channel_width_mm - 1e-9]:
            for land in (0.4, 0.6, 0.8, 1.2):
                n = channels_packed(reference_radius_mm, width, land)
                for aspect in (1.5, 2.0, 3.0, 4.0):
                    for hot_wall in [h for h in (0.3, 0.4, 0.5, 0.7, 1.0)
                                     if h >= min_hot_wall_mm - 1e-9]:
                        if hot_wall + aspect * width > max_depth_mm + 1e-9:
                            continue
                        ch = ChannelSpec(
                            n_channels=n,
                            width_mm=width,
                            height_mm=aspect * width,
                            land_mm=land,
                            hot_wall_mm=hot_wall,
                        )
                        try:
                            sol = solve_cooling(
                                chamber, xs, rs, ars, ms, ch, material,
                                throat_hydraulic_diameter_mm, curvature_radius_mm,
                                coolant_fraction=coolant_fraction,
                                film=film, coating=coating,
                            )
                        except (ValueError, ZeroDivisionError, OverflowError) as e:
                            evaluated.append(CoolingCandidate(
                                ch, material, None, False, f"solver: {e}", -1e18))
                            continue

                        reasons = []
                        if sol.wall_temperature_margin <= 0.0:
                            reasons.append(
                                f"hot wall {sol.peak_wall_temperature:.0f} K over "
                                f"{material.max_wall_temp:.0f} K")
                        if sol.coolant_temperature_margin <= 0.0:
                            reasons.append(
                                f"coolant {sol.coolant_outlet_temperature:.0f} K over "
                                f"{chamber.propellant.coolant_max_t:.0f} K")
                        if sol.pressure_drop > dp_limit:
                            reasons.append(
                                f"drop {sol.pressure_drop / 1e5:.1f} bar over "
                                f"{dp_limit / 1e5:.1f} bar")
                        if min(sol.reynolds) < 2300.0:
                            reasons.append(
                                f"coolant laminar at Re {min(sol.reynolds):.0f}, "
                                "Dittus-Boelter does not apply")
                        if sol.coating_temperature_margin <= 0.0:
                            reasons.append(
                                f"coating surface {sol.peak_coating_temperature:.0f} K over "
                                f"{coating.max_surface_temp:.0f} K and will spall")

                        survives = not reasons
                        # prefer margin, then a quiet circuit
                        score = (min(sol.wall_temperature_margin,
                                     sol.coolant_temperature_margin)
                                 - 30.0 * sol.pressure_drop / chamber.chamber_pressure)
                        evaluated.append(CoolingCandidate(
                            ch, material, sol, survives,
                            "ok" if survives else "; ".join(reasons), score))

    survivors = [c for c in evaluated if c.survives]
    best = max(survivors, key=lambda c: c.score) if survivors else None
    return best, evaluated


# --------------------------------------------------------------------------
# film cooling and thermal barrier coatings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FilmSpec:
    """
    A fuel film injected along the wall.

    `fuel_fraction` is diverted from the injector, so it is fuel that does not
    burn at the design mixture ratio. That costs specific impulse roughly in
    proportion, which is the price of the wall surviving.
    """
    fuel_fraction: float = 0.0          # of total fuel
    inject_x_mm: float = 0.0            # station where the film enters
    slot_height_mm: float = 0.3
    coolant_temp_k: float = 0.0         # 0 -> use the propellant inlet temperature


@dataclass(frozen=True)
class CoatingSpec:
    """
    Thermal barrier coating on the gas side.

    Yttria-stabilised zirconia is about 1.0 W/(m K) against 290 for GRCop-42, so
    a tenth of a millimetre of it is worth three millimetres of copper. It buys
    hot-wall temperature at the cost of a coating surface that runs far hotter
    and spalls if it is asked for too much.
    """
    thickness_mm: float = 0.0
    conductivity: float = 1.0           # W/(m K), YSZ
    max_surface_temp: float = 1600.0    # K


def film_effectiveness(x_mm: np.ndarray, inject_x_mm: float, slot_height_mm: float,
                       blowing_ratio: float, gas_velocity_ratio: float = 1.0):
    """
    Adiabatic film effectiveness downstream of a slot.

    Correlation of the classic slot-film form: effectiveness holds near unity
    for a short potential core and then decays with the turbulent mixing
    parameter x/(M s). Upstream of the slot it is zero -- film cools what is
    behind it and nothing in front.

    eta = 1 / (1 + 0.0329 * xi^0.8),  xi = x / (M * s)

    This is a correlation, and film cooling in a real chamber is worse than any
    correlation because the film is also burning. Treat it as an upper bound.
    """
    xi = np.maximum(x_mm - inject_x_mm, 0.0) / max(blowing_ratio * slot_height_mm, 1e-6)
    eta = 1.0 / (1.0 + 0.0329 * np.power(np.maximum(xi, 0.0), 0.8))
    return np.where(x_mm >= inject_x_mm, np.clip(eta, 0.0, 1.0), 0.0)


def filmed_recovery_temperature(t_aw: np.ndarray, eta: np.ndarray,
                                t_film: float) -> np.ndarray:
    """
    Driving temperature the wall actually sees under a film.

        T_aw,eff = T_aw - eta * (T_aw - T_film)

    At eta = 1 the wall sees coolant temperature; at eta = 0 it sees the full
    recovery temperature and the film has been consumed.
    """
    return t_aw - eta * (t_aw - t_film)
