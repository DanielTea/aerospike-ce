//
// Full engine assembly geometry, derived from the plug contour.
//
// Port of validate/engine_ref.py. That file is authoritative; if the numbers
// here disagree with it, this file is wrong. Every routine below has a
// same-named counterpart there, deliberately, so the two can be diffed.
//
// The one non-obvious construction is the cowl inner wall through the
// contraction. A straight cone from the chamber radius to the lip looks
// reasonable and is wrong: it pinches the duct to roughly 0.6 * A_t just
// upstream of the design throat, so the engine chokes in the wrong place.
//
// So the wall is solved, not drawn. At each station the offset direction
// rotates from radial to the sonic-line direction, and the offset distance is
// the positive root of
//
//     pi * (2*r0 + d*cos(theta)) * d = A          (frustum between the walls)
//
// Feed it a monotonically decreasing area schedule and the contraction cannot
// choke early, by construction. The self-check is that the last station lands
// exactly on the cowl lip -- see Program.cs, which asserts it at runtime.
//
// Coordinates: x along the axis, increasing downstream, x = 0 at the spike
// shoulder. All lengths mm, all angles radians.
//

namespace AerospikeCE
{
    public sealed class EngineAssembly
    {
        public PlugContour Contour { get; }
        public EngineSpec Spec { get; }

        // flow surfaces
        public double[] InnerWallX { get; }   // centrebody side, head to spike tip
        public double[] InnerWallR { get; }
        public double[] OuterWallX { get; }   // cowl inner side, head to lip
        public double[] OuterWallR { get; }
        public double[] CowlOuterX { get; }   // cowl outer shell
        public double[] CowlOuterR { get; }

        /// <summary>
        /// Shortest distance from a point to a polyline, segment-wise.
        /// Public because the geometry side needs it to measure wall thickness
        /// perpendicular to a sloped surface rather than radially.
        /// </summary>
        public static double DistanceToPolylinePublic(double px, double pr,
                                                      double[] x, double[] r)
            => DistanceToPolyline(px, pr, x, r);
        public double[] CavityX { get; }      // centrebody internal cavity
        public double[] CavityR { get; }

        // derived stations and areas
        public double HeadXMm { get; }
        public double ConvergingStartXMm { get; }
        public double ShoulderRadiusMm { get; }
        public double ChamberOuterRadiusMm { get; }
        public double ChamberAreaMm2 { get; }
        public double FlangeRadiusMm { get; }
        public double BoreRadiusMm { get; }
        public double TipXMm { get; }
        public double CavityEndXMm { get; }

        /// <summary>
        /// Closed meridional profiles, one per part, mirroring the Profile
        /// objects in engine_ref.py. Revolved about the axis each is a solid.
        /// </summary>
        public (double[] x, double[] r) CentrebodyProfile()
        {
            var x = new List<double>(InnerWallX);
            var r = new List<double>(InnerWallR);
            x.Add(TipXMm); r.Add(0.0);              // base face to the axis
            x.Add(CavityEndXMm); r.Add(0.0);        // forward along the axis
            for (int i = CavityX.Length - 1; i >= 0; i--) { x.Add(CavityX[i]); r.Add(CavityR[i]); }
            return (x.ToArray(), r.ToArray());      // head face closes implicitly
        }

        public (double[] x, double[] r) CowlProfile()
        {
            var x = new List<double>(OuterWallX);
            var r = new List<double>(OuterWallR);
            for (int i = CowlOuterX.Length - 1; i >= 0; i--) { x.Add(CowlOuterX[i]); r.Add(CowlOuterR[i]); }
            return (x.ToArray(), r.ToArray());
        }

        public (double[] x, double[] r) HeadProfile()
        {
            double x0 = HeadXMm - Spec.Geometry.HeadThicknessMm;
            return (new[] { x0, HeadXMm, HeadXMm, x0 },
                    new[] { BoreRadiusMm, BoreRadiusMm, FlangeRadiusMm, FlangeRadiusMm });
        }

        public double OverallLengthMm => TipXMm - (HeadXMm - Spec.Geometry.HeadThicknessMm);
        public double OverallRadiusMm => FlangeRadiusMm;

        private EngineAssembly(
            PlugContour contour, EngineSpec spec,
            double[] innerWallX, double[] innerWallR,
            double[] outerWallX, double[] outerWallR,
            double[] cowlOuterX, double[] cowlOuterR,
            double[] cavityX, double[] cavityR,
            double headX, double convergingStartX, double shoulderRadius,
            double chamberOuterRadius, double chamberArea, double flangeRadius,
            double boreRadius, double tipX, double cavityEndX)
        {
            Contour = contour; Spec = spec;
            InnerWallX = innerWallX; InnerWallR = innerWallR;
            OuterWallX = outerWallX; OuterWallR = outerWallR;
            CowlOuterX = cowlOuterX; CowlOuterR = cowlOuterR;
            CavityX = cavityX; CavityR = cavityR;
            HeadXMm = headX; ConvergingStartXMm = convergingStartX;
            ShoulderRadiusMm = shoulderRadius;
            ChamberOuterRadiusMm = chamberOuterRadius; ChamberAreaMm2 = chamberArea;
            FlangeRadiusMm = flangeRadius; BoreRadiusMm = boreRadius;
            TipXMm = tipX; CavityEndXMm = cavityEndX;
        }

        // ------------------------------------------------------------------
        // helpers, mirroring engine_ref.py
        // ------------------------------------------------------------------

        /// <summary>C1 blend, zero slope at both ends.</summary>
        private static double SmoothStep(double t)
        {
            t = Math.Clamp(t, 0.0, 1.0);
            return t * t * (3.0 - 2.0 * t);
        }

        /// <summary>
        /// Unit normals at each vertex, averaged from the two adjacent segment
        /// normals so a corner gives one mitred direction rather than two
        /// overlapping offset segments.
        /// </summary>
        private static (double[] nx, double[] nr) VertexNormals(
            double[] x, double[] r, bool outward)
        {
            int n = x.Length;
            var sx = new double[n - 1];
            var sr = new double[n - 1];
            double sign = outward ? 1.0 : -1.0;

            for (int i = 0; i < n - 1; i++)
            {
                double dx = x[i + 1] - x[i];
                double dr = r[i + 1] - r[i];
                double len = Math.Max(Math.Sqrt(dx * dx + dr * dr), 1e-15);
                sx[i] = sign * (-dr / len);
                sr[i] = sign * (dx / len);
            }

            var vx = new double[n];
            var vr = new double[n];
            vx[0] = sx[0]; vr[0] = sr[0];
            vx[n - 1] = sx[n - 2]; vr[n - 1] = sr[n - 2];
            for (int i = 1; i < n - 1; i++)
            {
                vx[i] = sx[i - 1] + sx[i];
                vr[i] = sr[i - 1] + sr[i];
            }
            for (int i = 0; i < n; i++)
            {
                double len = Math.Max(Math.Sqrt(vx[i] * vx[i] + vr[i] * vr[i]), 1e-15);
                vx[i] /= len; vr[i] /= len;
            }
            return (vx, vr);
        }

        /// <summary>Shortest distance from a point to a polyline, segment-wise.</summary>
        private static double DistanceToPolyline(double px, double pr, double[] x, double[] r)
        {
            double best = double.MaxValue;
            for (int i = 0; i < x.Length - 1; i++)
            {
                double dx = x[i + 1] - x[i];
                double dr = r[i + 1] - r[i];
                double ll = Math.Max(dx * dx + dr * dr, 1e-30);
                double t = Math.Clamp(((px - x[i]) * dx + (pr - r[i]) * dr) / ll, 0.0, 1.0);
                double cx = x[i] + t * dx;
                double cr = r[i] + t * dr;
                double d = Math.Sqrt((px - cx) * (px - cx) + (pr - cr) * (pr - cr));
                if (d < best) best = d;
            }
            return best;
        }

        /// <summary>
        /// Inward offset with the folded-over part removed.
        ///
        /// A plain normal offset is valid only where the offset distance is
        /// smaller than the local radius of curvature. The spike shoulder is
        /// sharper than any sensible wall thickness, so the naive offset there
        /// turns itself inside out. Drop every point that ended up closer to
        /// the original surface than asked -- a folded one always is -- then
        /// enforce monotone x. The wall comes out thicker at the corner, never
        /// thinner, which is the safe direction for a pressure part.
        /// </summary>
        private static (double[] x, double[] r) ErodePolyline(
            double[] x, double[] r, double distance)
        {
            var (nx, nr) = VertexNormals(x, r, outward: false);
            var ox = new List<double>(x.Length);
            var orr = new List<double>(x.Length);

            double run = double.NegativeInfinity;
            for (int i = 0; i < x.Length; i++)
            {
                double px = x[i] + distance * nx[i];
                double pr = r[i] + distance * nr[i];
                if (pr <= 0.0) continue;
                if (DistanceToPolyline(px, pr, x, r) < distance - 1e-6) continue;
                if (px <= run) continue;
                run = px;
                ox.Add(px); orr.Add(pr);
            }

            if (ox.Count == 0)
                throw new Exception("wall_thickness_mm consumes the whole centrebody");
            return (ox.ToArray(), orr.ToArray());
        }

        /// <summary>Linear interpolation on a polyline with monotone x, clamped.</summary>
        public static double Interp(double[] xs, double[] rs, double x)
        {
            if (x <= xs[0]) return rs[0];
            if (x >= xs[^1]) return rs[^1];

            int lo = 0, hi = xs.Length - 1;
            while (hi - lo > 1)
            {
                int mid = (lo + hi) / 2;
                if (xs[mid] <= x) lo = mid; else hi = mid;
            }
            double span = xs[hi] - xs[lo];
            double f = span > 1e-12 ? (x - xs[lo]) / span : 0.0;
            return rs[lo] + f * (rs[hi] - rs[lo]);
        }

        /// <summary>Drop points that do not advance in x, so Interp stays well posed.</summary>
        private static (double[] x, double[] r) ForceMonotone(double[] x, double[] r)
        {
            var ox = new List<double>(x.Length);
            var orr = new List<double>(x.Length);
            double run = double.NegativeInfinity;
            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] <= run) continue;
                run = x[i];
                ox.Add(x[i]); orr.Add(r[i]);
            }
            return (ox.ToArray(), orr.ToArray());
        }

        // ------------------------------------------------------------------
        // the contraction
        // ------------------------------------------------------------------

        /// <summary>
        /// Cowl inner surface through the contraction, solved from an area
        /// schedule. Two phases, because the inner body has a convex corner at
        /// the shoulder and a plain normal offset cannot turn a corner:
        ///
        ///   A. s &lt;= cornerFanStart -- the anchor slides along the cylinder
        ///      and the offset is radial. An ordinary parallel offset.
        ///   B. s &gt;  cornerFanStart -- the anchor is pinned at the shoulder
        ///      and the direction fans round to the sonic line.
        ///
        /// Letting the anchor run past the shoulder instead looks equivalent
        /// and pinches the duct to about 0.47 * A_t just aft of it.
        ///
        /// Where the fan begins is not a free constant. Through the fan the
        /// station sits at x = d sin(theta), and starting while the offset
        /// distance d is still large makes that product overshoot the lip and
        /// come back -- the wall folds. A fixed start works at a contraction
        /// ratio of 3 and fails at 8, whatever the converging length, because
        /// the fold is set by how big d is when rotation begins and not by how
        /// much axial room there is. So the fan starts where the schedule has
        /// already brought the area down to fanAreaRatio times the throat.
        /// </summary>
        private static (double[] x, double[] r) ContractionWall(
            PlugContour c, double chamberArea, double convergingLengthMm,
            int n = 200, double fanAreaRatio = 1.5)
        {
            double r0 = c.R[0];
            double nuE = c.ThroatAngleRad;
            double aT = c.ThroatAreaMm2;

            if (fanAreaRatio <= 1.0)
                throw new ArgumentException("fanAreaRatio must exceed 1");

            // start the fan where the area schedule reaches fanAreaRatio * A_t
            double target = Math.Min(fanAreaRatio * aT, chamberArea);
            double want = (chamberArea - target) / (chamberArea - aT);
            double blo = 0.0, bhi = 1.0;
            for (int b = 0; b < 200; b++)          // invert smoothstep by bisection
            {
                double mid = 0.5 * (blo + bhi);
                if (SmoothStep(mid) < want) blo = mid; else bhi = mid;
            }
            double cornerFanStart = Math.Min(Math.Max(0.5 * (blo + bhi), 1e-6), 1.0 - 1e-6);

            var xs = new double[n];
            var rs = new double[n];

            for (int i = 0; i < n; i++)
            {
                double s = (double)i / (n - 1);
                double area = chamberArea + (aT - chamberArea) * SmoothStep(s);

                // phase A: anchor walks the cylinder to the shoulder and stops
                double xAnchor = -convergingLengthMm * (1.0 - Math.Min(s / cornerFanStart, 1.0));

                // phase B: direction fans about the shoulder
                double fan = Math.Clamp((s - cornerFanStart) / (1.0 - cornerFanStart), 0.0, 1.0);
                double theta = nuE * SmoothStep(fan);

                double cosT = Math.Cos(theta);
                // positive root of  cos(theta)*d^2 + 2*r0*d - area/pi = 0
                double d = (-r0 + Math.Sqrt(r0 * r0 + cosT * area / Math.PI)) / cosT;

                xs[i] = xAnchor + d * Math.Sin(theta);
                rs[i] = r0 + d * cosT;
            }

            // The offset locus can curl back near the throat: through the fan
            // the offset distance shrinks while the direction is still
            // rotating, so x = d sin(theta) passes a maximum before the last
            // station. Two microns at a contraction ratio of 6, and enough to
            // corrupt every interpolation that walks this wall. Drop the
            // overshoot rather than flatten it -- clamping leaves points
            // sharing one x with different r, which two interpolators resolve
            // two different ways.
            double fold = 0.0, running = double.NegativeInfinity;
            foreach (double xv in xs) { running = Math.Max(running, xv); fold = Math.Max(fold, running - xv); }
            if (fold > 0.05)
                throw new ArgumentException(
                    $"the cowl wall folds back {fold:F3} mm through the contraction. " +
                    $"The corner fan no longer fits: lower contraction_ratio or " +
                    $"lengthen converging_length_mm");

            var keep = new bool[n];
            double run = double.PositiveInfinity;
            for (int i = n - 1; i >= 0; i--)
                if (xs[i] < run) { keep[i] = true; run = xs[i]; }

            var kx = new List<double>(n);
            var kr = new List<double>(n);
            for (int i = 0; i < n; i++) if (keep[i]) { kx.Add(xs[i]); kr.Add(rs[i]); }
            return (kx.ToArray(), kr.ToArray());
        }

        // ------------------------------------------------------------------
        // assembly
        // ------------------------------------------------------------------

        public static EngineAssembly Build(PlugContour c, EngineSpec spec, int nContraction = 200)
        {
            ChamberSpec ch = spec.Chamber;
            GeometrySpec g = spec.Geometry;

            if (ch.ContractionRatio <= 1.0)
                throw new ArgumentException("contraction_ratio must exceed 1");
            if (ch.ConvergingLengthMm <= 0.0 || ch.ChamberLengthMm <= 0.0)
                throw new ArgumentException("chamber and converging lengths must be positive");

            double w = g.WallThicknessMm;
            double r0 = c.R[0];
            double aT = c.ThroatAreaMm2;
            double aCh = ch.ContractionRatio * aT;

            if (w >= r0)
                throw new ArgumentException("wall_thickness_mm must be smaller than the shoulder radius");

            double rCh = Math.Sqrt(r0 * r0 + aCh / Math.PI);
            double xConv0 = -ch.ConvergingLengthMm;
            double xHead = xConv0 - ch.ChamberLengthMm;

            // ---- inner flow surface: cylinder to the shoulder, then the spike
            int nCyl = Math.Max(2, (int)(Math.Abs(xHead) * 4));
            var innerX = new List<double>(nCyl + c.X.Length);
            var innerR = new List<double>(nCyl + c.X.Length);
            for (int i = 0; i < nCyl; i++)
            {
                innerX.Add(xHead + (0.0 - xHead) * i / nCyl);
                innerR.Add(r0);
            }
            innerX.AddRange(c.X);
            innerR.AddRange(c.R);

            // ---- outer flow surface: constant-area chamber, then the contraction
            var (convX, convR) = ContractionWall(c, aCh, ch.ConvergingLengthMm, nContraction);
            int nCh = Math.Max(2, (int)(ch.ChamberLengthMm * 4));
            var outerX = new List<double>(nCh + nContraction);
            var outerR = new List<double>(nCh + nContraction);
            for (int i = 0; i < nCh; i++)
            {
                outerX.Add(xHead + (xConv0 - xHead) * i / nCh);
                outerR.Add(rCh);
            }
            outerX.AddRange(convX);
            outerR.AddRange(convR);

            double[] innerXa = innerX.ToArray(), innerRa = innerR.ToArray();
            double[] outerXa = outerX.ToArray(), outerRa = outerR.ToArray();

            // ---- centrebody cavity
            double tipX = innerXa[^1];
            double xCavEnd = tipX - g.BaseCapThicknessMm;

            var (cavXfull, cavRfull) = ErodePolyline(innerXa, innerRa, w);
            var cavX = new List<double>();
            var cavR = new List<double>();
            for (int i = 0; i < cavXfull.Length; i++)
            {
                if (cavXfull[i] > xCavEnd) break;
                cavX.Add(cavXfull[i]); cavR.Add(cavRfull[i]);
            }
            if (cavX.Count == 0)
                throw new ArgumentException(
                    "base_cap_thickness_mm leaves no room for the internal cavity");
            double rBore = cavR[0];

            // ---- cowl outer shell, tapering to a thinner blunt lip
            int nOut = outerXa.Length;
            int taperN = Math.Max(2, (int)(0.35 * nOut));
            var thick = new double[nOut];
            for (int i = 0; i < nOut; i++) thick[i] = w;
            for (int i = 0; i < taperN; i++)
            {
                double t = SmoothStep((double)i / (taperN - 1));
                thick[nOut - taperN + i] = w + (g.LipThicknessMm - w) * t;
            }

            var (onx, onr) = VertexNormals(outerXa, outerRa, outward: true);
            var cowlOutX = new double[nOut];
            var cowlOutR = new double[nOut];
            for (int i = 0; i < nOut; i++)
            {
                cowlOutX[i] = outerXa[i] + thick[i] * onx[i];
                cowlOutR[i] = outerRa[i] + thick[i] * onr[i];
            }
            var (cowlOutXm, cowlOutRm) = ForceMonotone(cowlOutX, cowlOutR);

            double rFlange = cowlOutRm[0] + g.FlangeWidthMm;

            return new EngineAssembly(
                c, spec,
                innerXa, innerRa,
                outerXa, outerRa,
                cowlOutXm, cowlOutRm,
                cavX.ToArray(), cavR.ToArray(),
                xHead, xConv0, r0, rCh, aCh, rFlange, rBore, tipX, xCavEnd);
        }

        // ------------------------------------------------------------------
        // radius lookups, for the voxel construction
        // ------------------------------------------------------------------

        public double CentrebodyRadiusAtX(double x) => Interp(InnerWallX, InnerWallR, x);
        public double CowlInnerRadiusAtX(double x) => Interp(OuterWallX, OuterWallR, x);
        public double CowlOuterRadiusAtX(double x) => Interp(CowlOuterX, CowlOuterR, x);
        public double CavityRadiusAtX(double x) => Interp(CavityX, CavityR, x);

        /// <summary>
        /// Local flow state along one cooled wall: x, r, A/A_t and Mach.
        ///
        /// Area ratio is measured off the built geometry by the same
        /// shortest-gap frustum used to prove the duct chokes at the throat, so
        /// the thermal model and the geometry check cannot drift apart. Mach
        /// comes from the area ratio on the subsonic branch upstream of the
        /// throat; on the spike the contour's own Mach values are used, since
        /// inverting the area ratio there would discard the better answer.
        /// </summary>
        public (double[] x, double[] r, double[] areaRatio, double[] mach) WallFlowState(
            string wall, double gamma, int n = 240)
        {
            double aT = Contour.ThroatAreaMm2;

            // measured area along the cowl wall
            var ox = new double[n];
            var oar = new double[n];
            for (int i = 0; i < n; i++)
            {
                int k = (int)((double)i / (n - 1) * (OuterWallX.Length - 1));
                double px = OuterWallX[k], pr = OuterWallR[k];
                double dBest = double.MaxValue, rAt = 0.0;
                for (int j = 0; j < InnerWallX.Length; j++)
                {
                    double dx = InnerWallX[j] - px, dr = InnerWallR[j] - pr;
                    double d = Math.Sqrt(dx * dx + dr * dr);
                    if (d < dBest) { dBest = d; rAt = InnerWallR[j]; }
                }
                ox[i] = px;
                oar[i] = Math.PI * (pr + rAt) * dBest / aT;
            }

            if (wall == "cowl")
            {
                var orr = new double[n];
                var mach = new double[n];
                for (int i = 0; i < n; i++)
                {
                    orr[i] = Interp(OuterWallX, OuterWallR, ox[i]);
                    mach[i] = GasDynamics.MachFromAreaRatioSubsonic(Math.Max(oar[i], 1.0), gamma);
                }
                return (ox, orr, oar, mach);
            }

            if (wall != "centrebody")
                throw new ArgumentException($"unknown wall: {wall}");

            int nSpike = Contour.X.Length;
            int nUp = InnerWallX.Length - nSpike;
            var ar = new double[InnerWallX.Length];
            var mc = new double[InnerWallX.Length];
            for (int i = 0; i < nUp; i++)
            {
                ar[i] = Interp(ox, oar, InnerWallX[i]);
                mc[i] = GasDynamics.MachFromAreaRatioSubsonic(Math.Max(ar[i], 1.0), gamma);
            }
            for (int i = 0; i < nSpike; i++)
            {
                mc[nUp + i] = Contour.Mach[i];
                ar[nUp + i] = GasDynamics.AreaRatio(Contour.Mach[i], gamma);
            }
            return (InnerWallX, InnerWallR, ar, mc);
        }

        /// <summary>
        /// Flow area along the duct, measured from the built geometry rather
        /// than from the schedule that produced it. Shortest distance from the
        /// cowl wall to the centrebody, swept as a frustum. Independent check
        /// that the duct really contracts to A_t at the throat and not before.
        /// </summary>
        public (double xMin, double areaMin) MinimumFlowArea(int n = 400)
        {
            double bestArea = double.MaxValue, bestX = 0.0;
            for (int i = 0; i < n; i++)
            {
                int k = (int)((double)i / (n - 1) * (OuterWallX.Length - 1));
                double px = OuterWallX[k], pr = OuterWallR[k];

                double dBest = double.MaxValue, rAt = 0.0;
                for (int j = 0; j < InnerWallX.Length; j++)
                {
                    double dx = InnerWallX[j] - px;
                    double dr = InnerWallR[j] - pr;
                    double d = Math.Sqrt(dx * dx + dr * dr);
                    if (d < dBest) { dBest = d; rAt = InnerWallR[j]; }
                }
                double area = Math.PI * (pr + rAt) * dBest;
                if (area < bestArea) { bestArea = area; bestX = px; }
            }
            return (bestX, bestArea);
        }
    }
}
