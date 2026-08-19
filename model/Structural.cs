//
// Thermostructural screening of the cooled wall, and a cycle life estimate.
//
// Port of validate/structural_ref.py. That file is authoritative.
//
// A regeneratively cooled chamber does not fail by bursting. It fails by
// thermal strain: the hot wall wants to expand, the cold jacket will not let
// it, so it goes into compression, yields, and returns to tension at shutdown.
// Repeat and the wall bulges into the channel, thins, and splits along the
// land. That is doghouse deformation, and it is the life-limiting mechanism for
// every regen chamber ever flown.
//
// No finite elements, no creep, no ratchetting accumulation, no build-direction
// anisotropy, no residual stress from the print. Printed material properties
// vary by more than the margins reported here. This will tell you a wall is
// obviously too thin; it will not tell you one is safe.
//

namespace AerospikeCE
{
    public sealed class StructuralResult
    {
        public double[] X = Array.Empty<double>();
        public double[] ThermalStress = Array.Empty<double>();
        public double[] PressureStress = Array.Empty<double>();
        public double[] HoopStress = Array.Empty<double>();
        public double[] CombinedStress = Array.Empty<double>();
        public double[] YieldAtTemp = Array.Empty<double>();
        public double[] Margin = Array.Empty<double>();
        public double[] StrainRange = Array.Empty<double>();
        public double CyclesToFailure;
        public Material Material = null!;
        public double RequiredCycles = 100.0;

        public double WorstMargin => Margin.Min();
        public bool StaysElastic => WorstMargin > 0.0;

        /// <summary>
        /// Life, not yield, is the criterion.
        ///
        /// A regen hot wall is expected to exceed yield: the gradient is large,
        /// the jacket restrains it, and it cycles plastically every firing.
        /// Requiring it to stay elastic would reject every chamber ever flown.
        /// </summary>
        public bool Survives =>
            CyclesToFailure >= RequiredCycles && StrainRange.Max() < 0.02;
    }

    public static class Structural
    {
        /// <summary>
        /// Yield derated linearly to zero at the melting point.
        ///
        /// Crude, and optimistic at the hot end: real alloys lose strength
        /// faster than linearly near the melt. Do not spend the last ten
        /// percent of the margin this reports.
        /// </summary>
        public static double YieldAtTemperature(Material m, double temperature)
        {
            double f = Math.Clamp(1.0 - (temperature - 293.0) / (m.MeltingPoint - 293.0),
                                  0.02, 1.0);
            return m.YieldStrength * f;
        }

        /// <summary>
        /// Screen the cooled wall for stress and cycle life.
        ///
        /// Thermal stress uses the constrained-plate result
        /// sigma = E alpha dT / (2 (1 - nu)); pressure stress treats the hot
        /// wall as a plate spanning between lands under the coolant-to-chamber
        /// differential, which is the load that pushes it into the channel.
        /// </summary>
        public static StructuralResult Analyse(
            CoolingSolution sol, double chamberPressure, double coolantPressure,
            double[] wallRadiusMm, double shellThicknessMm, double requiredCycles = 100.0)
        {
            Material m = sol.Material;
            int n = sol.X.Length;
            double tHot = sol.Channel.HotWallMm * 1e-3;
            double wCh = sol.Channel.WidthMm * 1e-3;
            double dp = Math.Abs(coolantPressure - chamberPressure);

            var th = new double[n];
            var pr = new double[n];
            var hoop = new double[n];
            var tot = new double[n];
            var sy = new double[n];
            var margin = new double[n];
            var strain = new double[n];
            double worstPlastic = 0.0;

            for (int i = 0; i < n; i++)
            {
                double dT = Math.Max(sol.TWallGas[i] - sol.TWallCoolant[i], 0.0);
                th[i] = m.YoungsModulus * m.Expansion * dT / (2.0 * (1.0 - m.Poisson));
                pr[i] = dp * Math.Pow(wCh / tHot, 2) / 2.0;
                hoop[i] = chamberPressure * (wallRadiusMm[i] * 1e-3) / (shellThicknessMm * 1e-3);

                tot[i] = Math.Sqrt(th[i] * th[i] + pr[i] * pr[i] + hoop[i] * hoop[i]
                                   - th[i] * hoop[i]);
                sy[i] = YieldAtTemperature(m, sol.TWallGas[i]);
                margin[i] = sy[i] / Math.Max(tot[i], 1.0) - 1.0;

                strain[i] = tot[i] / m.YoungsModulus;
                double plastic = Math.Max(strain[i] - sy[i] / m.YoungsModulus, 0.0);
                worstPlastic = Math.Max(worstPlastic, plastic);
            }

            double cycles = worstPlastic <= 1e-9
                ? double.PositiveInfinity
                : 0.5 * Math.Pow(worstPlastic / m.FatigueDuctility, 1.0 / m.FatigueExponent);

            return new StructuralResult
            {
                X = sol.X, ThermalStress = th, PressureStress = pr, HoopStress = hoop,
                CombinedStress = tot, YieldAtTemp = sy, Margin = margin,
                StrainRange = strain, CyclesToFailure = cycles, Material = m,
                RequiredCycles = requiredCycles,
            };
        }
    }
}
