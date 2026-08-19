//
// Regenerative cooling: channel sizing, wall temperatures, coolant rise and drop.
//
// Port of validate/cooling_ref.py. That file is authoritative.
//
// Axial channels in the hot wall, closed by a jacket. Coolant is the fuel,
// entering at the throat end where the flux is worst and leaving at the
// injector, so the coldest coolant meets the hottest station.
//
// At each station three resistances sit in series:
//
//     q = (T_aw - T_coolant) / (1/h_g + t_wall/k_wall + 1/h_coolant)
//
// h_g from Bartz, h_coolant from Dittus-Boelter with a fin correction for the
// lands, and the wall by conduction. Bartz's sigma depends on the hot-wall
// temperature it is used to find, so each station iterates a few times.
//
// Bartz is a correlation fitted to bell nozzles, commonly quoted at +/-30
// percent, and it is applied here to an annular throat with a hydraulic
// diameter substituted for the throat diameter. Treat absolute numbers as an
// order of magnitude and trends as real. Not modelled: film cooling, coatings,
// curvature enhancement, boiling or supercritical property variation, transient
// startup, or the structural problem of a hot wall against a cold jacket.
//
// SI throughout: W, m, K, Pa, kg, s.
//

namespace AerospikeCE
{
    /// <summary>
    /// Thermal and mechanical properties.
    ///
    /// Yield and modulus are quoted at room temperature and derated linearly to
    /// zero at the melting point -- crude, but it captures what matters: a hot
    /// wall is much weaker than a cold one, and the hot wall is where the
    /// stress is. Ductility and the fatigue exponent feed a Coffin-Manson life
    /// estimate, because regen chambers fail by thermal-strain ratchetting over
    /// tens to hundreds of cycles rather than by bursting.
    /// </summary>
    public sealed class Material
    {
        public string Name = "";
        public double Conductivity;             // W/(m K)
        public double MaxWallTemp;              // K
        public double Density;                  // kg/m^3
        public double YoungsModulus = 120e9;    // Pa at 293 K
        public double Poisson = 0.33;
        public double YieldStrength = 300e6;    // Pa at 293 K
        public double Expansion = 17e-6;        // 1/K
        public double MeltingPoint = 1600.0;    // K
        public double FatigueDuctility = 0.35;  // Coffin-Manson eps_f'
        public double FatigueExponent = -0.6;   // Coffin-Manson c
    }

    public sealed class ChannelSpec
    {
        public int NChannels = 80;
        public double WidthMm = 0.6;
        public double HeightMm = 1.5;
        public double LandMm = 0.8;
        public double HotWallMm = 0.5;
        public double RoughnessUm = 20.0;

        public double AspectRatio => HeightMm / WidthMm;

        /// <summary>
        /// Land width over channel width at a radius.
        ///
        /// Above about 2 the lands carry more of the hot wall than the channels
        /// do, and fin efficiency rather than coolant velocity sets the wall
        /// temperature. Small engines land here easily.
        /// </summary>
        public double LandToChannelRatio(double radiusMm)
            => (2.0 * Math.PI * radiusMm / NChannels - WidthMm) / WidthMm;
    }

    /// <summary>
    /// A fuel film injected along the wall. Port of FilmSpec in cooling_ref.py.
    ///
    /// FuelFraction is diverted from the injector, so it is fuel that does not
    /// burn at the design mixture ratio. Combustion.cs charges it for that.
    /// </summary>
    public sealed class FilmSpec
    {
        public double FuelFraction;          // of total fuel
        public double InjectXMm;
        public double SlotHeightMm = 0.3;
        public double CoolantTempK;          // 0 -> propellant inlet temperature
    }

    /// <summary>
    /// Thermal barrier coating on the gas side.
    ///
    /// YSZ is about 1.0 W/(m K) against 290 for GRCop-42, so a tenth of a
    /// millimetre is worth three millimetres of copper. It buys hot-wall
    /// temperature at the cost of a coating surface that runs far hotter and
    /// spalls if asked for too much.
    /// </summary>
    public sealed class CoatingSpec
    {
        public double ThicknessMm;
        public double Conductivity = 1.0;
        public double MaxSurfaceTemp = 1600.0;
    }

    public sealed class CoolingSolution
    {
        public double[] X = Array.Empty<double>();
        public double[] HeatFlux = Array.Empty<double>();
        public double[] TWallGas = Array.Empty<double>();
        public double[] TWallCoolant = Array.Empty<double>();
        public double[] TCoolant = Array.Empty<double>();
        public double[] Reynolds = Array.Empty<double>();
        public double[] TCoatingSurface = Array.Empty<double>();
        public double[] TAdiabatic = Array.Empty<double>();
        public double[] FilmEffectiveness = Array.Empty<double>();
        public double[] WallRadiusMm = Array.Empty<double>();
        public CoatingSpec? Coating;

        public Material Material = null!;
        public ChannelSpec Channel = null!;
        public double CoolantMassFlow;
        public double TotalHeat;              // W
        public double CoolantInletTemp;
        public double CoolantOutletTemp;      // taken from the end of the march
        public double CoolantMaxTemp;
        public double PressureIn;
        public double PressureOut;

        public double PeakHeatFlux => HeatFlux.Max();
        public double PeakWallTemperature => TWallGas.Max();
        public double CoolantTemperatureRise => CoolantOutletTemp - CoolantInletTemp;
        public double PressureDrop => PressureIn - PressureOut;
        public double WallTemperatureMargin => Material.MaxWallTemp - PeakWallTemperature;

        /// <summary>Hottest point on the coating, which is not the hottest metal.</summary>
        public double PeakCoatingTemperature =>
            TCoatingSurface.Length > 0 ? TCoatingSurface.Max() : PeakWallTemperature;

        public double CoatingTemperatureMargin =>
            Coating is null || Coating.ThicknessMm <= 0.0
                ? double.PositiveInfinity
                : Coating.MaxSurfaceTemp - PeakCoatingTemperature;
        public double CoolantTemperatureMargin => CoolantMaxTemp - CoolantOutletTemp;
        public double MinReynolds => Reynolds.Min();
        public double MaxReynolds => Reynolds.Max();
    }

    public sealed class CoolingCandidate
    {
        public ChannelSpec Channel = null!;
        public Material Material = null!;
        public CoolingSolution? Solution;
        public bool Survives;
        public string Reason = "";
        public double Score;
    }

    public static class Cooling
    {
        public static readonly Dictionary<string, Material> Materials = new()
        {
            ["inconel718"] = new Material
            {
                Name = "inconel718", Conductivity = 25.0, MaxWallTemp = 1150.0, Density = 8190.0,
                YoungsModulus = 200e9, Poisson = 0.29, YieldStrength = 1030e6,
                Expansion = 13e-6, MeltingPoint = 1610.0, FatigueDuctility = 0.25,
            },
            ["cucrzr"] = new Material
            {
                Name = "cucrzr", Conductivity = 320.0, MaxWallTemp = 800.0, Density = 8890.0,
                YoungsModulus = 128e9, Poisson = 0.34, YieldStrength = 280e6,
                Expansion = 17e-6, MeltingPoint = 1350.0, FatigueDuctility = 0.40,
            },
            ["grcop42"] = new Material
            {
                Name = "grcop42", Conductivity = 290.0, MaxWallTemp = 1000.0, Density = 8756.0,
                YoungsModulus = 110e9, Poisson = 0.34, YieldStrength = 190e6,
                Expansion = 17.5e-6, MeltingPoint = 1350.0, FatigueDuctility = 0.45,
            },
        };

        public static double GasCp(double gamma, double molarMass)
            => gamma * (Combustion.RUniversal / molarMass) / (gamma - 1.0);

        public static double GasPrandtl(double gamma) => 4.0 * gamma / (9.0 * gamma - 5.0);

        public static double GasViscosity(double molarMass, double temperature)
            => 1.184e-7 * Math.Sqrt(molarMass) * Math.Pow(temperature, 0.6);

        /// <summary>Recovery temperature seen by the wall, K. Recovery factor Pr^(1/3).</summary>
        public static double AdiabaticWallTemperature(double tChamber, double mach,
                                                      double gamma, double prandtl)
        {
            double r = Math.Pow(prandtl, 1.0 / 3.0);
            double m2 = 1.0 + 0.5 * (gamma - 1.0) * mach * mach;
            return tChamber * (1.0 + r * 0.5 * (gamma - 1.0) * mach * mach) / m2;
        }

        /// <summary>
        /// Gas-side film coefficient from Bartz, W/(m^2 K).
        ///
        /// areaRatio is A/A_t, so it is 1 at the throat, and the (A_t/A)^0.9
        /// term is what makes the throat the hottest place by a wide margin.
        /// </summary>
        public static double BartzCoefficient(ChamberConditions c, double throatDiameter,
                                              double curvatureRadius, double areaRatio,
                                              double mach, double wallTemperature)
        {
            Propellant p = c.Propellant;
            double gamma = p.Gamma;
            double cp = GasCp(gamma, p.MolarMass);
            double pr = GasPrandtl(gamma);
            double mu = GasViscosity(p.MolarMass, c.TChamber);

            double m2 = 1.0 + 0.5 * (gamma - 1.0) * mach * mach;
            double twRatio = wallTemperature / c.TChamber;
            double sigma = 1.0 / (Math.Pow(0.5 * twRatio * m2 + 0.5, 0.68) * Math.Pow(m2, 0.12));

            return (0.026 / Math.Pow(throatDiameter, 0.2))
                 * (Math.Pow(mu, 0.2) * cp / Math.Pow(pr, 0.6))
                 * Math.Pow(c.ChamberPressure / c.CStar, 0.8)
                 * Math.Pow(throatDiameter / curvatureRadius, 0.1)
                 * Math.Pow(1.0 / areaRatio, 0.9)
                 * sigma;
        }

        /// <summary>Coolant-side film coefficient, W/(m^2 K). Heating exponent 0.4.</summary>
        public static double DittusBoelter(double reynolds, double prandtl, double conductivity,
                                           double hydraulicDiameter, double roughnessFactor = 1.0)
            => roughnessFactor * 0.023 * Math.Pow(reynolds, 0.8) * Math.Pow(prandtl, 0.4)
               * conductivity / hydraulicDiameter;

        /// <summary>Darcy friction factor, Haaland's approximation to Colebrook.</summary>
        public static double FrictionFactor(double reynolds, double relativeRoughness)
        {
            if (reynolds < 2300.0) return 64.0 / Math.Max(reynolds, 1e-6);
            double inv = -1.8 * Math.Log10(Math.Pow(relativeRoughness / 3.7, 1.11) + 6.9 / reynolds);
            return 1.0 / (inv * inv);
        }

        /// <summary>
        /// Efficiency of the rib between channels, as a straight fin.
        ///
        /// The lands are not passive: heat above a land runs sideways into the
        /// rib and down into the coolant, and the rib is only as good at that as
        /// its conductivity and thickness allow. Ignoring it credits the design
        /// with cooled area it does not have.
        /// </summary>
        public static double FinEfficiency(double hCoolant, double conductivity,
                                           double landThickness, double channelHeight)
        {
            double m = Math.Sqrt(2.0 * hCoolant / (conductivity * landThickness));
            double ml = m * channelHeight;
            return ml > 1e-9 ? Math.Tanh(ml) / ml : 1.0;
        }

        /// <summary>
        /// Adiabatic film effectiveness downstream of a slot.
        ///
        ///     eta = 1 / (1 + 0.0329 * xi^0.8),   xi = x / (M * s)
        ///
        /// Zero upstream of the slot: a film cools what is behind it and nothing
        /// in front. This is a correlation, and a real film is worse than any
        /// correlation because it is also burning. Treat it as an upper bound.
        /// </summary>
        public static double FilmEffectiveness(double xMm, double injectXMm,
                                               double slotHeightMm, double blowingRatio)
        {
            if (xMm < injectXMm) return 0.0;
            double xi = Math.Max(xMm - injectXMm, 0.0)
                        / Math.Max(blowingRatio * slotHeightMm, 1e-6);
            return Math.Clamp(1.0 / (1.0 + 0.0329 * Math.Pow(Math.Max(xi, 0.0), 0.8)), 0.0, 1.0);
        }

        /// <summary>T_aw,eff = T_aw - eta (T_aw - T_film).</summary>
        public static double FilmedRecoveryTemperature(double tAw, double eta, double tFilm)
            => tAw - eta * (tAw - tFilm);

        public static int ChannelsPacked(double referenceRadiusMm, double widthMm, double landMm)
            => Math.Max(1, (int)(2.0 * Math.PI * referenceRadiusMm / (widthMm + landMm)));

        /// <summary>
        /// March the coolant from the last station to the first.
        ///
        /// Arrays arrive in gas order, injector to throat; the coolant runs
        /// against it. That is the point of counterflow, and it is also why the
        /// outlet temperature is taken from the end of the march rather than
        /// from the first array entry, which holds the temperature *entering*
        /// that station and would miss the last station's heat.
        /// </summary>
        public static CoolingSolution Solve(
            ChamberConditions chamber, double[] wallXMm, double[] wallRMm,
            double[] areaRatio, double[] mach, ChannelSpec channel, Material material,
            double throatHydraulicDiameterMm, double curvatureRadiusMm,
            double coolantFraction = 1.0, double? inletPressureBar = null,
            FilmSpec? film = null, CoatingSpec? coating = null,
            (double lo, double hi)? channelXRange = null)
        {
            int n = wallXMm.Length;
            if (wallRMm.Length != n || areaRatio.Length != n || mach.Length != n)
                throw new ArgumentException("wall arrays must be the same length");
            if (n < 2)
                throw new ArgumentException("need at least two stations");
            if (coolantFraction <= 0.0 || coolantFraction > 1.0)
                throw new ArgumentException("coolant_fraction must be in (0, 1]");
            if (channel.NChannels < 1)
                throw new ArgumentException("need at least one channel");

            Propellant p = chamber.Propellant;
            double mdotC = chamber.MassFlowFuel * coolantFraction;
            double mdotPerChannel = mdotC / channel.NChannels;
            double prC = p.CoolantMu * p.CoolantCp / p.CoolantK;
            double dt = throatHydraulicDiameterMm * 1e-3;
            double rc = curvatureRadiusMm * 1e-3;
            double prG = GasPrandtl(p.Gamma);

            // film effectiveness along the wall, zero upstream of the slot
            var etaFilm = new double[n];
            double tFilm = p.CoolantInletT;
            if (film is not null && film.FuelFraction > 0.0)
            {
                double mdotFilm = chamber.MassFlowFuel * film.FuelFraction;
                double slotR = EngineAssembly.Interp(wallXMm, wallRMm, film.InjectXMm);
                double slotArea = 2.0 * Math.PI * slotR * 1e-3 * film.SlotHeightMm * 1e-3;
                double vFilm = mdotFilm / (p.DensityFuel * slotArea);
                double coreFlux = chamber.MassFlow / Math.Max(Math.PI * Math.Pow(slotR * 1e-3, 2), 1e-9);
                double blowing = Math.Max((p.DensityFuel * vFilm) / Math.Max(coreFlux, 1e-9), 0.05);
                for (int i = 0; i < n; i++)
                    etaFilm[i] = FilmEffectiveness(wallXMm[i], film.InjectXMm,
                                                   film.SlotHeightMm, blowing);
                if (film.CoolantTempK > 0.0) tFilm = film.CoolantTempK;
            }

            double rCoating = coating is not null && coating.ThicknessMm > 0.0
                ? coating.ThicknessMm * 1e-3 / coating.Conductivity : 0.0;

            var q = new double[n];
            var tCoat = new double[n];
            var tAw = new double[n];
            var twg = new double[n];
            var twc = new double[n];
            var tc = new double[n];
            var re = new double[n];

            double tWallGuess = 0.6 * chamber.TChamber;
            double pressIn = inletPressureBar is double b ? b * 1e5 : 1.5 * chamber.ChamberPressure;
            double press = pressIn;
            double tCool = p.CoolantInletT;
            double totalQ = 0.0;

            for (int k = 0; k < n; k++)
            {
                int i = n - 1 - k;                    // counterflow: aft station first
                double rM = wallRMm[i] * 1e-3;
                double pitch = 2.0 * Math.PI * rM / channel.NChannels;
                double width = Math.Min(channel.WidthMm * 1e-3,
                                        Math.Max(pitch - channel.LandMm * 1e-3, 1e-4));
                double height = channel.HeightMm * 1e-3;
                double land = Math.Max(pitch - width, 1e-5);

                double areaC = width * height;
                double dH = 2.0 * areaC / (width + height);
                double v = mdotPerChannel / (p.DensityFuel * areaC);
                double reI = p.DensityFuel * v * dH / p.CoolantMu;

                double rough = channel.RoughnessUm * 1e-6;
                double roughFactor = 1.0 + 0.3 * Math.Min(rough / dH / 0.01, 1.0);
                double hcBare = DittusBoelter(reI, prC, p.CoolantK, dH, roughFactor);

                double eta = FinEfficiency(hcBare, material.Conductivity, land, height);
                double hcI = hcBare * (width + 2.0 * eta * height) / pitch;

                double tAwI = AdiabaticWallTemperature(chamber.TChamber, mach[i], p.Gamma, prG);
                tAwI = FilmedRecoveryTemperature(tAwI, etaFilm[i], tFilm);

                bool hasChannel = channelXRange is null
                    || (wallXMm[i] >= channelXRange.Value.lo
                        && wallXMm[i] <= channelXRange.Value.hi);
                double rCool = hasChannel ? 1.0 / hcI : 1e9;   // uncooled: no path out

                double rWall = channel.HotWallMm * 1e-3 / material.Conductivity;

                double tw = tWallGuess;
                for (int it = 0; it < 12; it++)
                {
                    double hg = BartzCoefficient(chamber, dt, rc, areaRatio[i], mach[i], tw);
                    double qI2 = (tAwI - tCool) / (1.0 / hg + rCoating + rWall + rCool);
                    double twNew = tAwI - qI2 / hg;
                    if (Math.Abs(twNew - tw) < 0.01) { tw = twNew; break; }
                    tw = 0.5 * (tw + twNew);
                }
                tWallGuess = tw;

                double hgI = BartzCoefficient(chamber, dt, rc, areaRatio[i], mach[i], tw);
                double qI = (tAwI - tCool) / (1.0 / hgI + rCoating + rWall + rCool);

                double seg = k == 0
                    ? Math.Abs(wallXMm[i] - wallXMm[i - 1]) * 1e-3
                    : Math.Abs(wallXMm[n - k] - wallXMm[i]) * 1e-3;

                double heat = qI * (pitch * seg) * channel.NChannels;
                totalQ += heat;

                q[i] = qI;
                tAw[i] = tAwI;
                tCoat[i] = tw;                       // coating surface, or bare metal
                twg[i] = tw - qI * rCoating;         // metal under the coating
                twc[i] = tCool + qI / hcI;
                tc[i] = tCool;
                re[i] = reI;

                tCool += heat / (mdotC * p.CoolantCp);
                double f = FrictionFactor(reI, rough / dH);
                press -= f * (seg / dH) * 0.5 * p.DensityFuel * v * v;
            }

            return new CoolingSolution
            {
                X = wallXMm, HeatFlux = q, TWallGas = twg, TWallCoolant = twc,
                TCoolant = tc, Reynolds = re,
                TCoatingSurface = tCoat, TAdiabatic = tAw, FilmEffectiveness = etaFilm,
                WallRadiusMm = wallRMm, Coating = coating,
                Material = material, Channel = channel,
                CoolantMassFlow = mdotC, TotalHeat = totalQ,
                CoolantInletTemp = p.CoolantInletT, CoolantOutletTemp = tCool,
                CoolantMaxTemp = p.CoolantMaxT,
                PressureIn = pressIn, PressureOut = press,
            };
        }

        private static double[] Downsample(double[] a, int n)
        {
            int m = Math.Min(n, a.Length);
            var o = new double[m];
            for (int i = 0; i < m; i++)
                o[i] = a[(int)((double)i / (m - 1) * (a.Length - 1))];
            return o;
        }

        /// <summary>
        /// Search channel geometry and material for a jacket that survives.
        ///
        /// Sizing by picking a coolant velocity works on a big engine and fails
        /// on a small one: the binding constraint is how much cooled area fits
        /// on the circumference once channels are no narrower than the process
        /// can print. Chasing velocity drives the count down, the lands then
        /// carry most of the hot wall, and fin efficiency decides the wall
        /// temperature however fast the coolant goes.
        /// </summary>
        public static (CoolingCandidate? best, List<CoolingCandidate> all) Search(
            ChamberConditions chamber, double[] wallXMm, double[] wallRMm,
            double[] areaRatio, double[] mach,
            double throatHydraulicDiameterMm, double curvatureRadiusMm,
            double referenceRadiusMm, string[] materials,
            double coolantFraction = 1.0, double maxPressureDropFraction = 0.5,
            int searchStations = 80, double minChannelWidthMm = 0.4,
            double minHotWallMm = 0.3, double maxDepthMm = double.PositiveInfinity,
            FilmSpec? film = null, CoatingSpec? coating = null)
        {
            double[] xs = Downsample(wallXMm, searchStations);
            double[] rs = Downsample(wallRMm, searchStations);
            double[] ars = Downsample(areaRatio, searchStations);
            double[] ms = Downsample(mach, searchStations);

            var evaluated = new List<CoolingCandidate>();
            double dpLimit = maxPressureDropFraction * chamber.ChamberPressure;

            foreach (string matName in materials)
            {
                Material material = Materials[matName];
                foreach (double width in new[] { 0.4, 0.5, 0.6, 0.8, 1.0, 1.2 }
                             .Where(w => w >= minChannelWidthMm - 1e-9))
                    foreach (double land in new[] { 0.4, 0.6, 0.8, 1.2 })
                    {
                        int nCh = ChannelsPacked(referenceRadiusMm, width, land);
                        foreach (double aspect in new[] { 1.5, 2.0, 3.0, 4.0 })
                            foreach (double hotWall in new[] { 0.3, 0.4, 0.5, 0.7, 1.0 }
                                         .Where(h => h >= minHotWallMm - 1e-9))
                            {
                                if (hotWall + aspect * width > maxDepthMm + 1e-9) continue;
                                var ch = new ChannelSpec
                                {
                                    NChannels = nCh, WidthMm = width, HeightMm = aspect * width,
                                    LandMm = land, HotWallMm = hotWall,
                                };
                                CoolingSolution sol;
                                try
                                {
                                    sol = Solve(chamber, xs, rs, ars, ms, ch, material,
                                                throatHydraulicDiameterMm, curvatureRadiusMm,
                                                coolantFraction, null, film, coating);
                                }
                                catch (Exception e)
                                {
                                    evaluated.Add(new CoolingCandidate
                                    {
                                        Channel = ch, Material = material,
                                        Survives = false, Reason = "solver: " + e.Message,
                                        Score = -1e18,
                                    });
                                    continue;
                                }

                                var reasons = new List<string>();
                                if (sol.WallTemperatureMargin <= 0.0)
                                    reasons.Add($"hot wall {sol.PeakWallTemperature:F0} K over {material.MaxWallTemp:F0} K");
                                if (sol.CoolantTemperatureMargin <= 0.0)
                                    reasons.Add($"coolant {sol.CoolantOutletTemp:F0} K over {sol.CoolantMaxTemp:F0} K");
                                if (sol.PressureDrop > dpLimit)
                                    reasons.Add($"drop {sol.PressureDrop / 1e5:F1} bar over {dpLimit / 1e5:F1} bar");
                                if (sol.MinReynolds < 2300.0)
                                    reasons.Add($"coolant laminar at Re {sol.MinReynolds:F0}, Dittus-Boelter does not apply");
                                if (sol.CoatingTemperatureMargin <= 0.0)
                                    reasons.Add($"coating surface {sol.PeakCoatingTemperature:F0} K over " +
                                                $"{coating!.MaxSurfaceTemp:F0} K and will spall");

                                double score = Math.Min(sol.WallTemperatureMargin, sol.CoolantTemperatureMargin)
                                             - 30.0 * sol.PressureDrop / chamber.ChamberPressure;
                                evaluated.Add(new CoolingCandidate
                                {
                                    Channel = ch, Material = material, Solution = sol,
                                    Survives = reasons.Count == 0,
                                    Reason = reasons.Count == 0 ? "ok" : string.Join("; ", reasons),
                                    Score = score,
                                });
                            }
                    }
            }

            var survivors = evaluated.Where(c => c.Survives).ToList();
            CoolingCandidate? best = survivors.Count > 0
                ? survivors.Aggregate((a, b) => b.Score > a.Score ? b : a)
                : null;
            return (best, evaluated);
        }
    }
}
