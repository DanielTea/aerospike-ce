//
// Injector element sizing and face layout.
//
// Port of validate/injector_ref.py. That file is authoritative.
//
// Unlike-impinging doublets on an annular face: one oxidiser and one fuel
// orifice per element, angled so the jets collide a fixed distance downstream.
// Sizing is the incompressible orifice equation solved for area at a chosen
// injection stiffness dP/pc.
//
// No spray modelling, no droplet size, no vaporisation length, no stability
// analysis. The chug check is a pressure-drop correlation, not physics: passing
// it proves nothing and failing it is a warning. Real injectors get hot fired.
//

namespace AerospikeCE
{
    public sealed class InjectorDesign
    {
        public int NElements;
        public double ElementCircleRadius;   // m
        public double DFuel;                 // m, per orifice
        public double DOx;
        public double AreaFuelTotal;         // m^2
        public double AreaOxTotal;
        public double DpFuel;                // Pa
        public double DpOx;
        public double Stiffness;
        public double VFuel;                 // m/s
        public double VOx;
        public double MomentumFuel;          // N
        public double MomentumOx;
        public double AngleFuel;             // rad from the element axis
        public double AngleOx;
        public double ImpingementDistance;   // m
        public double DischargeCoefficient;

        public double DFuelMm => DFuel * 1e3;
        public double DOxMm => DOx * 1e3;
        public double MomentumRatio => MomentumOx / MomentumFuel;
        public double IncludedAngle => AngleFuel + AngleOx;

        public double FuelRingRadius => ElementCircleRadius - ImpingementDistance * Math.Tan(AngleFuel);
        public double OxRingRadius => ElementCircleRadius + ImpingementDistance * Math.Tan(AngleOx);

        /// <summary>
        /// Angle of the combined jet from the element axis, rad.
        ///
        /// Zero by construction. Splitting the included angle evenly instead
        /// throws the resultant off axis -- about 11 degrees for a typical
        /// LOX/methane momentum split -- which streaks the chamber and drives
        /// one side of the wall hot.
        /// </summary>
        public double ResultantAngle
        {
            get
            {
                double transverse = MomentumOx * Math.Sin(AngleOx) - MomentumFuel * Math.Sin(AngleFuel);
                double axial = MomentumOx * Math.Cos(AngleOx) + MomentumFuel * Math.Cos(AngleFuel);
                return Math.Atan2(transverse, axial);
            }
        }

        /// <summary>Stiffness against the usual 0.15 chug floor. Below 1.0 is a warning.</summary>
        public double ChugMargin => Stiffness / 0.15;
    }

    public static class Injector
    {
        private static double OrificeArea(double mdot, double rho, double dp, double cd)
        {
            if (dp <= 0.0)
                throw new ArgumentException("injection pressure drop must be positive");
            return mdot / (cd * Math.Sqrt(2.0 * rho * dp));
        }

        /// <summary>Inverse of OrificeArea. Used to prove the sizing round-trips.</summary>
        public static double MassFlowThrough(double area, double rho, double dp, double cd)
            => cd * area * Math.Sqrt(2.0 * rho * dp);

        /// <summary>
        /// Split an included angle so the doublet's resultant lands on the axis.
        ///
        ///     a_f = atan2( m_ox sin(total),  m_f + m_ox cos(total) )
        ///
        /// Equal momenta collapse to the symmetric half angle. The heavier
        /// stream takes the shallower angle, which is the physical result: it
        /// needs less turning to be cancelled.
        /// </summary>
        public static (double fuel, double ox) BalancedDoubletAngles(
            double momentumFuel, double momentumOx, double includedAngle)
        {
            double aF = Math.Atan2(momentumOx * Math.Sin(includedAngle),
                                   momentumFuel + momentumOx * Math.Cos(includedAngle));
            return (aF, includedAngle - aF);
        }

        public static InjectorDesign Design(
            ChamberConditions chamber, int nElements, double elementCircleRadiusMm,
            double stiffness = 0.20, double dischargeCoefficient = 0.75,
            double includedAngleDeg = 60.0, double impingementDistanceMm = 4.0)
        {
            if (nElements < 1)
                throw new ArgumentException("need at least one element");
            if (stiffness <= 0.0 || stiffness >= 1.0)
                throw new ArgumentException("stiffness must be a fraction of pc in (0, 1)");
            if (dischargeCoefficient <= 0.0 || dischargeCoefficient > 1.0)
                throw new ArgumentException("discharge coefficient must be in (0, 1]");
            if (includedAngleDeg <= 0.0 || includedAngleDeg >= 180.0)
                throw new ArgumentException("included angle must be in (0, 180) degrees");

            Propellant p = chamber.Propellant;
            double dp = stiffness * chamber.ChamberPressure;

            double aF = OrificeArea(chamber.MassFlowFuel, p.DensityFuel, dp, dischargeCoefficient);
            double aOx = OrificeArea(chamber.MassFlowOx, p.DensityOx, dp, dischargeCoefficient);

            double vF = chamber.MassFlowFuel / (p.DensityFuel * aF);
            double vOx = chamber.MassFlowOx / (p.DensityOx * aOx);
            double momF = p.DensityFuel * vF * vF * aF;
            double momOx = p.DensityOx * vOx * vOx * aOx;

            var (angF, angOx) = BalancedDoubletAngles(momF, momOx, includedAngleDeg * Math.PI / 180.0);

            return new InjectorDesign
            {
                NElements = nElements,
                ElementCircleRadius = elementCircleRadiusMm * 1e-3,
                DFuel = Math.Sqrt(4.0 * aF / (nElements * Math.PI)),
                DOx = Math.Sqrt(4.0 * aOx / (nElements * Math.PI)),
                AreaFuelTotal = aF, AreaOxTotal = aOx,
                DpFuel = dp, DpOx = dp, Stiffness = stiffness,
                VFuel = vF, VOx = vOx,
                MomentumFuel = momF, MomentumOx = momOx,
                AngleFuel = angF, AngleOx = angOx,
                ImpingementDistance = impingementDistanceMm * 1e-3,
                DischargeCoefficient = dischargeCoefficient,
            };
        }

        /// <summary>
        /// Whether the pattern fits the annular face.
        ///
        /// On a narrow annulus the land between the two rings binds first, and
        /// on a printed part it is also the thinnest feature the process holds.
        /// </summary>
        public static (bool ok, string why) FitsOnFace(
            InjectorDesign d, double annulusInnerMm, double annulusOuterMm,
            double minLandMm = 0.8)
        {
            double rF = d.FuelRingRadius * 1e3;
            double rOx = d.OxRingRadius * 1e3;

            if (rF - d.DFuelMm / 2.0 < annulusInnerMm + minLandMm)
                return (false, "fuel orifice ring runs off the inner edge of the face");
            if (rOx + d.DOxMm / 2.0 > annulusOuterMm - minLandMm)
                return (false, "oxidiser orifice ring runs off the outer edge of the face");

            double pitchF = 2.0 * Math.PI * rF / d.NElements;
            double pitchOx = 2.0 * Math.PI * rOx / d.NElements;
            if (pitchF - d.DFuelMm < minLandMm)
                return (false, "fuel orifices collide with their neighbours");
            if (pitchOx - d.DOxMm < minLandMm)
                return (false, "oxidiser orifices collide with their neighbours");

            double gap = (rOx - d.DOxMm / 2.0) - (rF + d.DFuelMm / 2.0);
            if (gap < minLandMm)
                return (false, "no land left between the fuel and oxidiser rings");
            return (true, "ok");
        }

        /// <summary>
        /// Search for the densest doublet ring that fits the annular face.
        ///
        /// An annular chamber gives an injector far less radial room than a
        /// cylindrical one, so the pattern is usually constrained by the face
        /// rather than by the flow. Walk the element count down and return the
        /// first design that fits, or the reason nothing did.
        /// </summary>
        public static (InjectorDesign? design, string why) FitToFace(
            ChamberConditions chamber, double annulusInnerMm, double annulusOuterMm,
            double stiffness = 0.20, double dischargeCoefficient = 0.75,
            double minLandMm = 0.8, int maxElements = 48)
        {
            double rMid = 0.5 * (annulusInnerMm + annulusOuterMm);
            string last = "no candidate evaluated";
            InjectorDesign? fallback = null;

            foreach (int n in Enumerable.Range(1, maxElements).Reverse())
                foreach (double dImp in new[] { 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5 })
                    foreach (double angle in new[] { 60.0, 50.0, 40.0, 30.0 })
                    {
                        var d = Design(chamber, n, rMid, stiffness, dischargeCoefficient, angle, dImp);
                        var (ok, why) = FitsOnFace(d, annulusInnerMm, annulusOuterMm, minLandMm);
                        if (ok)
                            return (d, $"{n} elements, {angle:G} deg included, impinging {dImp:G} mm from the face");
                        last = why;
                        fallback ??= d;
                    }
            return (fallback, $"nothing fits: {last}");
        }
    }
}
