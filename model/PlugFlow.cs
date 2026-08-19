//
// Thrust from the plug surface pressure, and altitude compensation.
//
// Port of validate/plug_flow_ref.py. That file is authoritative.
//
// The thrust coefficient in Combustion.cs is the ideal bell value, which is the
// right number for a bell and the wrong one for a plug -- wrong in the
// direction that matters, since it understates the aerospike everywhere below
// the design altitude.
//
// A plug has no fixed exit area; its outer boundary is the ambient air. Every
// contour point already carries a Mach number, so it carries a static pressure,
// and the axial force is that pressure on the annular projection. Where the
// surface pressure falls below ambient the plume detaches instead, so the
// integration stops there rather than crediting the engine with the loss. That
// truncation is the altitude compensation mechanism, expressed geometrically.
//
// Not a method of characteristics solution: the contour came from the Angelino
// approximate method, and the base region behind a truncated spike is not
// modelled at all though it is worth several percent of thrust.
//

namespace AerospikeCE
{
    public sealed class PlugThrust
    {
        public double AmbientPressure;
        public double[] SurfacePressure = Array.Empty<double>();
        public double AttachedFraction;
        public double PlugForce;
        public double ThroatForce;
        public double TotalForce;
        public double ThrustCoefficient;
        public double Isp;

        public bool Separated => AttachedFraction < 0.999;
    }

    public static class PlugFlow
    {
        /// <summary>Static pressure along the plug surface, from its own Mach numbers.</summary>
        public static double[] SurfacePressure(PlugContour c, ChamberConditions ch)
        {
            var p = new double[c.Mach.Length];
            for (int i = 0; i < p.Length; i++)
                p[i] = ch.ChamberPressure * Combustion.PressureRatio(c.Mach[i], ch.Propellant.Gamma);
            return p;
        }

        public static PlugThrust Thrust(PlugContour c, ChamberConditions ch, double ambient)
        {
            double[] pSurf = SurfacePressure(c, ch);

            int attached = 0;
            while (attached < pSurf.Length && pSurf[attached] > ambient) attached++;

            double plugForce = 0.0;
            for (int i = 1; i < attached; i++)
            {
                double r0 = c.R[i - 1] * 1e-3, r1 = c.R[i] * 1e-3;
                double p0 = pSurf[i - 1] - ambient, p1 = pSurf[i] - ambient;
                // integral (p - pa) 2 pi r dr, r decreasing along the contour
                plugForce += -0.5 * (p0 * 2.0 * Math.PI * r0 + p1 * 2.0 * Math.PI * r1)
                             * (r1 - r0);
            }

            double gamma = ch.Propellant.Gamma;
            double pThroat = ch.ChamberPressure * Combustion.PressureRatio(1.0, gamma);
            double tThroat = ch.TChamber / (1.0 + 0.5 * (gamma - 1.0));
            double vThroat = Math.Sqrt(gamma * (Combustion.RUniversal / ch.Propellant.MolarMass)
                                       * tThroat);

            // The flow crosses the sonic line inclined at nu_e to the axis, not
            // at the angle the line itself makes. Resolving on the latter
            // inflates this term by 2.4 and hands the plug a 48 percent gain
            // over an ideal bell that no aerospike has ever shown.
            double axial = Math.Cos(c.ThroatAngleRad);
            double throatForce = (ch.MassFlow * vThroat
                                  + (pThroat - ambient) * ch.ThroatArea) * axial;

            double total = (plugForce + throatForce) * ch.TruncationEfficiency;
            return new PlugThrust
            {
                AmbientPressure = ambient,
                SurfacePressure = pSurf,
                AttachedFraction = (double)attached / Math.Max(pSurf.Length, 1),
                PlugForce = plugForce,
                ThroatForce = throatForce,
                TotalForce = total,
                ThrustCoefficient = total / (ch.ChamberPressure * ch.ThroatArea),
                Isp = total / (ch.MassFlow * Combustion.G0),
            };
        }
    }
}
