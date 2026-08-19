//
// Isentropic relations and the Prandtl-Meyer function.
//
// This is a direct mirror of validate/contour_ref.py. If you change the physics,
// change the Python first, make the pytest suite pass, then port here.
//

namespace AerospikeCE
{
    public static class GasDynamics
    {
        /// <summary>Isentropic area ratio A/A* for a given Mach number.</summary>
        public static double AreaRatio(double mach, double gamma)
        {
            if (mach <= 0.0)
                throw new ArgumentException("mach must be positive");

            double gp1 = gamma + 1.0;
            double gm1 = gamma - 1.0;
            return (1.0 / mach)
                 * Math.Pow((2.0 / gp1) * (1.0 + 0.5 * gm1 * mach * mach), gp1 / (2.0 * gm1));
        }

        /// <summary>Prandtl-Meyer function nu(M) in radians. Zero at M = 1.</summary>
        public static double PrandtlMeyer(double mach, double gamma)
        {
            if (mach < 1.0)
                throw new ArgumentException("Prandtl-Meyer requires supersonic flow");
            if (mach == 1.0)
                return 0.0;

            double gp1 = gamma + 1.0;
            double gm1 = gamma - 1.0;
            double m2m1 = mach * mach - 1.0;

            return Math.Sqrt(gp1 / gm1) * Math.Atan(Math.Sqrt(gm1 / gp1 * m2m1))
                 - Math.Atan(Math.Sqrt(m2m1));
        }

        /// <summary>
        /// Invert the area-Mach relation on the *subsonic* branch.
        ///
        /// The area ratio is double valued: one subsonic and one supersonic
        /// Mach number share every A/A* above 1. Upstream of the throat the
        /// flow is subsonic, and taking the supersonic root there would put
        /// Mach 3 in the combustion chamber and make the heat transfer estimate
        /// meaningless.
        /// </summary>
        public static double MachFromAreaRatioSubsonic(double eps, double gamma)
        {
            if (eps < 1.0)
                throw new ArgumentException("expansion ratio must be >= 1");
            double lo = 1e-6, hi = 1.0;
            for (int i = 0; i < 400; i++)
            {
                double mid = 0.5 * (lo + hi);
                if (AreaRatio(mid, gamma) > eps) lo = mid; else hi = mid;
                if (hi - lo < 1e-12) break;
            }
            return 0.5 * (lo + hi);
        }

        /// <summary>Invert the area-Mach relation on the supersonic branch.</summary>
        public static double MachFromAreaRatio(double eps, double gamma)
        {
            if (eps < 1.0)
                throw new ArgumentException("expansion ratio must be >= 1");

            double lo = 1.0, hi = 2.0;
            while (AreaRatio(hi, gamma) < eps)
            {
                hi *= 2.0;
                if (hi > 1e6)
                    throw new ArgumentException("expansion ratio out of range");
            }

            for (int i = 0; i < 400; i++)
            {
                double mid = 0.5 * (lo + hi);
                if (AreaRatio(mid, gamma) < eps) lo = mid; else hi = mid;
                if (hi - lo < 1e-12) break;
            }
            return 0.5 * (lo + hi);
        }
    }
}
