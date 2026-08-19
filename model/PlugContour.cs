//
// Angelino approximate plug-nozzle contour.
//
// The derivation lives in validate/contour_ref.py. Short version:
//
//   alpha(M) = nu_e - nu(M) + mu(M)
//   xi       = (1 - sqrt(1 - sin(alpha) * M * eps(M) / eps_e)) / sin(alpha)
//   x        = r_e * xi * cos(alpha)
//   r        = r_e * (1 - xi * sin(alpha))
//
// This is where the engineering knowledge sits. In LEAP 71 terms this file is a
// very small piece of a Computational Engineering Model: descriptive input in,
// geometry out, no interactive CAD anywhere.
//

namespace AerospikeCE
{
    public sealed class PlugContour
    {
        public double[] X { get; }              // mm, increasing, starts at 0
        public double[] R { get; }              // mm, decreasing
        public double[] Mach { get; }
        public double ExitMach { get; }
        public double ThroatAngleRad { get; }   // nu_e, inclination of throat plane
        public double ExitRadiusMm { get; }
        public double ThroatAreaMm2 { get; }
        public double LengthMm => X[^1];
        public double TipRadiusMm => R[^1];

        private PlugContour(double[] x, double[] r, double[] mach,
                            double exitMach, double nuE, double exitRadius, double throatArea)
        {
            X = x; R = r; Mach = mach;
            ExitMach = exitMach;
            ThroatAngleRad = nuE;
            ExitRadiusMm = exitRadius;
            ThroatAreaMm2 = throatArea;
        }

        public static PlugContour Build(
            double expansionRatio,
            double exitRadiusMm,
            double gamma = 1.20,
            int nPoints = 300,
            double truncateFraction = 1.0)
        {
            if (truncateFraction <= 0.0 || truncateFraction > 1.0)
                throw new ArgumentException("truncateFraction must be in (0, 1]");

            double exitMach = GasDynamics.MachFromAreaRatio(expansionRatio, gamma);
            double nuE = GasDynamics.PrandtlMeyer(exitMach, gamma);

            var xs = new List<double>(nPoints);
            var rs = new List<double>(nPoints);
            var ms = new List<double>(nPoints);

            for (int i = 0; i < nPoints; i++)
            {
                // cluster points near the throat where curvature is highest
                double t = (double)i / (nPoints - 1);
                double mach = Math.Min(1.0 + (exitMach - 1.0) * Math.Pow(t, 1.5), exitMach);

                double nu = GasDynamics.PrandtlMeyer(mach, gamma);
                double mu = Math.Asin(Math.Min(1.0, 1.0 / mach));
                double alpha = nuE - nu + mu;
                double sinA = Math.Sin(alpha);
                double cosA = Math.Cos(alpha);

                double epsM = GasDynamics.AreaRatio(mach, gamma);
                double radicand = Math.Max(0.0, 1.0 - sinA * mach * epsM / expansionRatio);
                double xi = (1.0 - Math.Sqrt(radicand)) / sinA;

                xs.Add(exitRadiusMm * xi * cosA);
                rs.Add(exitRadiusMm * (1.0 - xi * sinA));
                ms.Add(mach);
            }

            // shift so the throat station sits at x = 0
            double x0 = xs[0];
            for (int i = 0; i < xs.Count; i++) xs[i] -= x0;

            if (truncateFraction < 1.0)
            {
                double cut = xs[^1] * truncateFraction;
                int last = 1;
                while (last < xs.Count && xs[last] <= cut) last++;
                xs = xs.GetRange(0, last);
                rs = rs.GetRange(0, last);
                ms = ms.GetRange(0, last);
            }

            double throatArea = Math.PI * exitRadiusMm * exitRadiusMm / expansionRatio;

            return new PlugContour(xs.ToArray(), rs.ToArray(), ms.ToArray(),
                                   exitMach, nuE, exitRadiusMm, throatArea);
        }

        /// <summary>
        /// Radius at a normalised station along the spike, 0 at the throat and
        /// 1 at the tip. Linear interpolation is enough: the contour is already
        /// sampled far finer than the voxel size.
        /// </summary>
        public double RadiusAtLengthRatio(double ratio)
        {
            double xTarget = Math.Clamp(ratio, 0.0, 1.0) * LengthMm;

            if (xTarget <= X[0]) return R[0];
            if (xTarget >= X[^1]) return R[^1];

            int lo = 0, hi = X.Length - 1;
            while (hi - lo > 1)
            {
                int mid = (lo + hi) / 2;
                if (X[mid] <= xTarget) lo = mid; else hi = mid;
            }

            double span = X[hi] - X[lo];
            double f = span > 1e-12 ? (xTarget - X[lo]) / span : 0.0;
            return R[lo] + f * (R[hi] - R[lo]);
        }

        public void WriteCsv(string path)
        {
            using var w = new StreamWriter(path);
            w.WriteLine("x_mm,r_mm,mach");
            for (int i = 0; i < X.Length; i++)
                w.WriteLine($"{X[i]:F6},{R[i]:F6},{Mach[i]:F6}");
        }
    }
}
