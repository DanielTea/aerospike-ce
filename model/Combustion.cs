//
// Chamber conditions and delivered performance.
//
// Port of validate/combustion_ref.py. That file is authoritative.
//
// A parametric performance model for preliminary sizing: stated propellant
// properties, isentropic nozzle relations, standard efficiency factors. c* and
// T_c follow a fitted parabolic falloff either side of their peak mixture
// ratio, which reproduces the shape of a real curve near the peak and is wrong
// in the wings.
//
// This is NOT an equilibrium chemistry solver. There is no Gibbs minimisation
// and no thermochemical database; the table below is tabulated textbook data at
// one reference condition. For anything that will be fired, run CEA or RPA at
// the actual chamber pressure and mixture ratio and override the values from
// the spec. Frozen flow is assumed throughout the nozzle, so delivered Isp will
// sit a few percent above these numbers.
//
// SI throughout: Pa, K, m, kg, s.
//

namespace AerospikeCE
{
    public sealed class Propellant
    {
        public string Name = "";
        public string Fuel = "";
        public string Oxidiser = "";
        public double MrPeak;
        public double CStarPeak;          // m/s
        public double TChamberPeak;       // K
        public double Gamma;
        public double MolarMass;          // kg/kmol
        public double DensityFuel;        // kg/m^3
        public double DensityOx;
        public double CoolantCp;          // J/(kg K)
        public double CoolantK;           // W/(m K)
        public double CoolantMu;          // Pa s
        public double CoolantInletT;      // K
        public double CoolantMaxT;        // K
        public double CStarFalloff = 0.30;
        public double TChamberFalloff = 0.25;

        public Propellant Clone() => (Propellant)MemberwiseClone();
    }

    public static class Combustion
    {
        public const double G0 = 9.80665;
        public const double RUniversal = 8314.462;   // J/(kmol K)

        public static readonly Dictionary<string, Propellant> Propellants = new()
        {
            ["lox_rp1"] = new Propellant
            {
                Name = "lox_rp1", Fuel = "RP-1", Oxidiser = "LOX",
                MrPeak = 2.56, CStarPeak = 1800.0, TChamberPeak = 3670.0,
                Gamma = 1.24, MolarMass = 23.3,
                DensityFuel = 810.0, DensityOx = 1141.0,
                CoolantCp = 2000.0, CoolantK = 0.14, CoolantMu = 1.6e-3,
                CoolantInletT = 290.0, CoolantMaxT = 560.0,
            },
            ["lox_ch4"] = new Propellant
            {
                Name = "lox_ch4", Fuel = "CH4", Oxidiser = "LOX",
                MrPeak = 3.40, CStarPeak = 1857.0, TChamberPeak = 3550.0,
                Gamma = 1.20, MolarMass = 21.5,
                DensityFuel = 423.0, DensityOx = 1141.0,
                CoolantCp = 3480.0, CoolantK = 0.19, CoolantMu = 1.2e-4,
                CoolantInletT = 111.0, CoolantMaxT = 700.0,
            },
            ["n2o_ethanol"] = new Propellant
            {
                Name = "n2o_ethanol", Fuel = "ethanol", Oxidiser = "N2O",
                MrPeak = 4.00, CStarPeak = 1580.0, TChamberPeak = 3100.0,
                Gamma = 1.23, MolarMass = 25.0,
                DensityFuel = 789.0, DensityOx = 1220.0,
                CoolantCp = 2440.0, CoolantK = 0.17, CoolantMu = 1.2e-3,
                CoolantInletT = 290.0, CoolantMaxT = 460.0,
            },
        };

        /// <summary>Characteristic velocity, m/s. Parabolic falloff about the peak.</summary>
        public static double CStarIdeal(Propellant p, double mixtureRatio)
        {
            if (mixtureRatio <= 0.0)
                throw new ArgumentException("mixture_ratio must be positive");
            double d = (mixtureRatio - p.MrPeak) / p.MrPeak;
            return p.CStarPeak * Math.Max(0.0, 1.0 - p.CStarFalloff * d * d);
        }

        /// <summary>Stagnation combustion temperature, K.</summary>
        public static double ChamberTemperature(Propellant p, double mixtureRatio)
        {
            double d = (mixtureRatio - p.MrPeak) / p.MrPeak;
            return p.TChamberPeak * Math.Max(0.0, 1.0 - p.TChamberFalloff * d * d);
        }

        /// <summary>p/p0 for isentropic flow at a given Mach number.</summary>
        public static double PressureRatio(double mach, double gamma)
            => Math.Pow(1.0 + 0.5 * (gamma - 1.0) * mach * mach, -gamma / (gamma - 1.0));

        /// <summary>
        /// Nozzle thrust coefficient, momentum term plus pressure term.
        ///
        /// This is the ideal bell value. A plug's outer boundary is the ambient
        /// itself, so below the design altitude the real spike sits above this;
        /// modelling that needs a method-of-characteristics solution the repo
        /// does not have, so the number here is the conservative one.
        /// </summary>
        public static double ThrustCoefficient(double gamma, double expansionRatio,
                                               double exitMach, double chamberPressure,
                                               double ambientPressure)
        {
            double gm1 = gamma - 1.0;
            double gp1 = gamma + 1.0;
            double pePc = PressureRatio(exitMach, gamma);

            double momentum = Math.Sqrt(
                (2.0 * gamma * gamma / gm1)
                * Math.Pow(2.0 / gp1, gp1 / gm1)
                * (1.0 - Math.Pow(pePc, gm1 / gamma)));
            double pressure = (pePc - ambientPressure / chamberPressure) * expansionRatio;
            return momentum + pressure;
        }
    }

    public sealed class ChamberConditions
    {
        public Propellant Propellant = null!;
        public double MixtureRatio;
        public double ChamberPressure;    // Pa
        public double ThroatArea;         // m^2
        public double ExpansionRatio;
        public double ExitMach;
        public double CStarEfficiency;
        public double TruncationEfficiency;

        public double CStar;              // m/s, delivered
        public double TChamber;           // K
        public double MassFlow;           // kg/s
        public double MassFlowFuel;
        public double MassFlowOx;

        public double CfSeaLevel;
        public double CfVacuum;
        public double ThrustSeaLevel;     // N
        public double ThrustVacuum;
        public double IspSeaLevel;        // s
        public double IspVacuum;

        public double ExitPressure =>
            ChamberPressure * Combustion.PressureRatio(ExitMach, Propellant.Gamma);

        /// <summary>Heat the fuel can absorb before its temperature limit, W.</summary>
        public double CoolantCapacity =>
            MassFlowFuel * Propellant.CoolantCp * (Propellant.CoolantMaxT - Propellant.CoolantInletT);

        public static ChamberConditions Build(
            Propellant p, double mixtureRatio, double chamberPressureBar,
            double throatAreaMm2, double expansionRatio, double exitMach,
            double ambientPressurePa = 101325.0,
            double cStarEfficiency = 0.94, double truncationEfficiency = 0.99)
        {
            if (chamberPressureBar <= 0.0)
                throw new ArgumentException("chamber pressure must be positive");
            if (throatAreaMm2 <= 0.0)
                throw new ArgumentException("throat area must be positive");
            if (cStarEfficiency <= 0.0 || cStarEfficiency > 1.0)
                throw new ArgumentException("c_star_efficiency must be in (0, 1]");

            double pc = chamberPressureBar * 1e5;
            double at = throatAreaMm2 * 1e-6;

            double cStar = Combustion.CStarIdeal(p, mixtureRatio) * cStarEfficiency;
            if (cStar <= 0.0)
                throw new ArgumentException("mixture ratio is far enough off peak that c* collapses");

            double mdot = pc * at / cStar;          // the choked throat relation
            double cfSl = Combustion.ThrustCoefficient(p.Gamma, expansionRatio, exitMach, pc, ambientPressurePa);
            double cfVac = Combustion.ThrustCoefficient(p.Gamma, expansionRatio, exitMach, pc, 0.0);
            double fSl = cfSl * pc * at * truncationEfficiency;
            double fVac = cfVac * pc * at * truncationEfficiency;

            return new ChamberConditions
            {
                Propellant = p,
                MixtureRatio = mixtureRatio,
                ChamberPressure = pc,
                ThroatArea = at,
                ExpansionRatio = expansionRatio,
                ExitMach = exitMach,
                CStarEfficiency = cStarEfficiency,
                TruncationEfficiency = truncationEfficiency,
                CStar = cStar,
                TChamber = Combustion.ChamberTemperature(p, mixtureRatio),
                MassFlow = mdot,
                MassFlowOx = mdot * mixtureRatio / (1.0 + mixtureRatio),
                MassFlowFuel = mdot / (1.0 + mixtureRatio),
                CfSeaLevel = cfSl,
                CfVacuum = cfVac,
                ThrustSeaLevel = fSl,
                ThrustVacuum = fVac,
                IspSeaLevel = fSl / (mdot * Combustion.G0),
                IspVacuum = fVac / (mdot * Combustion.G0),
            };
        }

        public static ChamberConditions FromSpec(EngineSpec spec, double throatAreaMm2,
                                                 double expansionRatio, double exitMach)
        {
            string name = spec.Propellant.Combination;
            if (!Combustion.Propellants.TryGetValue(name, out var basep))
                throw new ArgumentException($"unknown propellant combination: {name}");

            // any tabulated value may be overridden, which is how CEA output gets in
            Propellant p = basep.Clone();
            if (spec.Propellant.MrPeak is double v1) p.MrPeak = v1;
            if (spec.Propellant.CStarPeak is double v2) p.CStarPeak = v2;
            if (spec.Propellant.TChamberPeak is double v3) p.TChamberPeak = v3;
            if (spec.Propellant.Gamma is double v4) p.Gamma = v4;
            if (spec.Propellant.MolarMass is double v5) p.MolarMass = v5;

            var o = spec.Operation;
            return Build(p, o.MixtureRatio, o.ChamberPressureBar, throatAreaMm2,
                         expansionRatio, exitMach, o.AmbientPressurePa,
                         o.CStarEfficiency, o.TruncationEfficiency);
        }
    }
}
