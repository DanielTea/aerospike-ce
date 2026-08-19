//
// Signed distance functions for the cooled parts.
//
// Port of validate/mesh_solid.py. That file is authoritative.
//
// Cooling channels and injector orifices vary with angle, so a surface of
// revolution cannot express them. PicoGK renders any IBoundedImplicit into
// voxels, which means the same distance functions the Python mesher uses can be
// handed straight to the kernel -- no second formulation to keep in step, and
// no boolean subtraction of several hundred individual channel solids.
//
// Model x maps to PicoGK Z, matching SpikeGeometry. Radius is sqrt(X^2 + Y^2).
//

using System.Numerics;
using PicoGK;

namespace AerospikeCE
{
    /// <summary>A ring of axial cooling channels riding a wall surface.</summary>
    public sealed class ChannelCut
    {
        public double[] WallX = Array.Empty<double>();
        public double[] WallR = Array.Empty<double>();
        public int NChannels;
        public double WidthMm;
        public double HeightMm;
        public double HotWallMm;
        public double LandMm;
        public double XStart;
        public double XEnd;
        public bool Outward = true;
    }

    /// <summary>
    /// A ring of radial feed ports, one per cooling channel.
    ///
    /// Channels have to be fed. Without a port each is a blind pocket, which is
    /// unbuildable and also topologically invisible.
    /// </summary>
    public sealed class PortCut
    {
        public double XAt;
        public double DiameterMm;
        public int Count;
        public double RLo;
        public double RHi;
        public double Phase;
    }

    /// <summary>A ring of axial holes, for injector orifices.</summary>
    public sealed class HoleCut
    {
        public double RadiusMm;
        public double DiameterMm;
        public int Count;
        public double XStart;
        public double XEnd;
        public double Phase;
    }

    /// <summary>
    /// A part: a revolved meridional polygon with features subtracted.
    /// </summary>
    public sealed class CooledPart : IBoundedImplicit
    {
        private readonly double[] _px, _pr;
        private readonly List<ChannelCut> _channels;
        private readonly List<HoleCut> _holes;
        private readonly List<PortCut> _ports;
        private readonly BBox3 _bounds;

        // Tabulated profile distance over (x, r). See the constructor.
        private readonly float[,] _table;
        private readonly double _tx0, _tr0, _tStep;
        private readonly int _tnx, _tnr;

        public CooledPart((double[] x, double[] r) profile,
                          List<ChannelCut>? channels = null,
                          List<HoleCut>? holes = null,
                          List<PortCut>? ports = null,
                          float fMarginMm = 1.0f,
                          double fTableStepMm = 0.05)
        {
            _px = profile.x; _pr = profile.r;
            _channels = channels ?? new List<ChannelCut>();
            _holes = holes ?? new List<HoleCut>();
            _ports = ports ?? new List<PortCut>();

            float x0 = (float)_px.Min() - fMarginMm;
            float x1 = (float)_px.Max() + fMarginMm;
            float rr = (float)_pr.Max() + fMarginMm;
            _bounds = new BBox3(new Vector3(-rr, -rr, x0), new Vector3(rr, rr, x1));

            // The profile is a solid of revolution, so its distance field
            // depends only on (x, r) and not on angle. Walking all several
            // hundred profile edges for every voxel the kernel samples costs
            // twenty-two billion operations on the demonstrator alone, which is
            // not slow, it is a hang: the viewer opens, nothing is ever drawn,
            // and the window sits there white.
            //
            // Tabulating it once and interpolating turns that into a bilinear
            // lookup. The step is chosen well below the voxel size, so the
            // interpolation error is far smaller than the sampling the kernel
            // is about to do anyway. This mirrors the same optimisation in
            // mesh_solid.py, which evaluates the profile on a radius grid per
            // plane for exactly the same reason.
            _tStep = Math.Max(fTableStepMm, 1e-3);
            _tx0 = x0;
            _tr0 = 0.0;
            _tnx = (int)Math.Ceiling((x1 - x0) / _tStep) + 2;
            _tnr = (int)Math.Ceiling((rr * Math.Sqrt(2.0)) / _tStep) + 2;

            _table = new float[_tnx, _tnr];
            Parallel.For(0, _tnx, i =>
            {
                double xv = _tx0 + i * _tStep;
                for (int j = 0; j < _tnr; j++)
                    _table[i, j] = (float)PolygonDistanceExact(xv, _tr0 + j * _tStep);
            });
        }

        /// <summary>Bilinear lookup into the tabulated profile distance.</summary>
        private double PolygonDistance(double px, double pr)
        {
            double fi = (px - _tx0) / _tStep;
            double fj = (pr - _tr0) / _tStep;
            int i = (int)Math.Floor(fi), j = (int)Math.Floor(fj);
            if (i < 0 || j < 0 || i >= _tnx - 1 || j >= _tnr - 1)
                return PolygonDistanceExact(px, pr);   // outside the table

            double u = fi - i, v = fj - j;
            return (1 - u) * (1 - v) * _table[i, j]
                 + u * (1 - v) * _table[i + 1, j]
                 + (1 - u) * v * _table[i, j + 1]
                 + u * v * _table[i + 1, j + 1];
        }

        public BBox3 oBounds => _bounds;

        /// <summary>
        /// Signed distance to the closed meridional polygon, negative inside.
        ///
        /// Distance is measured to the nearest edge; the sign comes from a
        /// separate crossing count, because a nearest-edge normal test gets
        /// corners wrong and corners are most of this profile.
        /// </summary>
        private double PolygonDistanceExact(double px, double pr)
        {
            int n = _px.Length;
            double best = double.MaxValue;
            bool inside = false;

            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                double ex = _px[j] - _px[i], er = _pr[j] - _pr[i];
                double ll = Math.Max(ex * ex + er * er, 1e-30);
                double wx = px - _px[i], wr = pr - _pr[i];
                double t = Math.Clamp((wx * ex + wr * er) / ll, 0.0, 1.0);
                double dx = wx - t * ex, dr = wr - t * er;
                best = Math.Min(best, Math.Sqrt(dx * dx + dr * dr));

                if ((_pr[i] > pr) != (_pr[j] > pr) &&
                    px < (_px[j] - _px[i]) * (pr - _pr[i]) / (_pr[j] - _pr[i] + 1e-30) + _px[i])
                    inside = !inside;
            }
            return inside ? -best : best;
        }

        /// <summary>
        /// Distance to the nearest channel of a ring, negative inside.
        ///
        /// The width is clamped by the local pitch less the land, because the
        /// pitch shrinks with radius: on the spike the circumference runs out
        /// before the channel count does, and unclamped channels merge into one
        /// groove that cuts the plug in half. Cooling.Solve clamps the same way.
        /// </summary>
        private static double ChannelDistance(double x, double r, double th, ChannelCut c)
        {
            double wall = EngineAssembly.Interp(c.WallX, c.WallR, x);
            double sign = c.Outward ? 1.0 : -1.0;
            double floor = wall + sign * c.HotWallMm;
            double roof = floor + sign * c.HeightMm;
            double lo = Math.Min(floor, roof), hi = Math.Max(floor, roof);

            double dRadial = Math.Max(lo - r, r - hi);

            double pitch = 2.0 * Math.PI / c.NChannels;
            double arc = pitch * Math.Max(r, 1e-6);
            double width = Math.Min(c.WidthMm, Math.Max(arc - c.LandMm, 0.0));
            double local = th - pitch * Math.Floor(th / pitch + 0.5);
            double dTheta = Math.Abs(local) * Math.Max(r, 1e-6) - 0.5 * width;

            double dAxial = Math.Max(c.XStart - x, x - c.XEnd);
            return Math.Max(Math.Max(dRadial, dTheta), dAxial);
        }

        /// <summary>Distance to the nearest radial feed port. Negative inside.</summary>
        private static double PortDistance(double x, double r, double th, PortCut c)
        {
            double pitch = 2.0 * Math.PI / c.Count;
            double rel = th - c.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            double dTheta = Math.Abs(local) * Math.Max(r, 1e-6) - 0.5 * c.DiameterMm;
            double dAxial = Math.Abs(x - c.XAt) - 0.5 * c.DiameterMm;
            double dRadial = Math.Max(c.RLo - r, r - c.RHi);
            return Math.Max(Math.Max(dTheta, dAxial), dRadial);
        }

        private static double HoleDistance(double x, double y, double z, HoleCut h)
        {
            double th = Math.Atan2(z, y);
            double pitch = 2.0 * Math.PI / h.Count;
            double rel = th - h.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            double r = Math.Sqrt(y * y + z * z);

            double dr = r - h.RadiusMm;
            double dt = local * Math.Max(r, 1e-6);
            double dPlane = Math.Sqrt(dr * dr + dt * dt) - 0.5 * h.DiameterMm;

            double dAxial = Math.Max(h.XStart - x, x - h.XEnd);
            return Math.Max(dPlane, dAxial);
        }

        public float fSignedDistance(in Vector3 vec)
        {
            double x = vec.Z;                       // model x is PicoGK Z
            double y = vec.X, z = vec.Y;
            double r = Math.Sqrt(y * y + z * z);
            double th = Math.Atan2(z, y);

            double d = PolygonDistance(x, r);
            foreach (var c in _channels) d = Math.Max(d, -ChannelDistance(x, r, th, c));
            foreach (var h in _holes) d = Math.Max(d, -HoleDistance(x, y, z, h));
            foreach (var pt in _ports) d = Math.Max(d, -PortDistance(x, r, th, pt));
            return (float)d;
        }
    }

    public static class CooledFeatures
    {
        /// <summary>
        /// Longest run of x over which a wall can hold a channel.
        ///
        /// The cowl tapers to a thin lip and the centrebody runs out of section
        /// behind the truncation. A channel that outlives its wall breaks
        /// through, which shows up as a solid with wildly wrong topology rather
        /// than as an obvious hole.
        /// </summary>
        private static (double x0, double x1) ThickEnoughSpan(
            double[] wallX, double[] available, double needed)
        {
            int bestStart = -1, bestLen = 0, runStart = -1, runLen = 0;
            for (int i = 0; i < wallX.Length; i++)
            {
                if (available[i] >= needed)
                {
                    if (runStart < 0) { runStart = i; runLen = 0; }
                    runLen++;
                    if (runLen > bestLen) { bestLen = runLen; bestStart = runStart; }
                }
                else runStart = -1;
            }
            if (bestStart < 0)
                throw new Exception(
                    $"no station has {needed:F2} mm of wall for a channel " +
                    $"(most available is {available.Max():F2} mm)");
            return (wallX[bestStart], wallX[bestStart + bestLen - 1]);
        }

        public static ChannelCut Cowl(EngineAssembly a, ChannelSpec ch, double backWallMm = 0.5)
        {
            int n = a.OuterWallX.Length;
            var available = new double[n];
            for (int i = 0; i < n; i++)
                // perpendicular to the gas surface, not radial: at the lip the
                // radial gap reads three times the true wall thickness
                available[i] = EngineAssembly.DistanceToPolylinePublic(
                    a.OuterWallX[i], a.OuterWallR[i], a.CowlOuterX, a.CowlOuterR);

            var (x0, x1) = ThickEnoughSpan(a.OuterWallX, available,
                                           ch.HotWallMm + ch.HeightMm + backWallMm);
            return new ChannelCut
            {
                WallX = a.OuterWallX, WallR = a.OuterWallR,
                NChannels = ch.NChannels, WidthMm = ch.WidthMm, HeightMm = ch.HeightMm,
                HotWallMm = ch.HotWallMm, LandMm = ch.LandMm,
                XStart = x0, XEnd = x1, Outward = true,
            };
        }

        public static ChannelCut Centrebody(EngineAssembly a, ChannelSpec ch, double backWallMm = 0.5)
        {
            int n = a.InnerWallX.Length;
            double cavityEnd = a.CavityX.Max();
            var available = new double[n];
            for (int i = 0; i < n; i++)
                available[i] = a.InnerWallX[i] <= cavityEnd
                    ? EngineAssembly.DistanceToPolylinePublic(
                        a.InnerWallX[i], a.InnerWallR[i], a.CavityX, a.CavityR)
                    : a.InnerWallR[i];

            var (x0, x1) = ThickEnoughSpan(a.InnerWallX, available,
                                           ch.HotWallMm + ch.HeightMm + backWallMm);
            return new ChannelCut
            {
                WallX = a.InnerWallX, WallR = a.InnerWallR,
                NChannels = ch.NChannels, WidthMm = ch.WidthMm, HeightMm = ch.HeightMm,
                HotWallMm = ch.HotWallMm, LandMm = ch.LandMm,
                XStart = x0, XEnd = x1, Outward = false,
            };
        }

        /// <summary>
        /// Radial feed ports at the aft end of a channel ring.
        ///
        /// Coolant enters at the throat end, where the flux is worst, and
        /// leaves at the injector. The cowl draws from an external manifold
        /// through its outer skin; the centrebody draws from the central bore.
        /// Sized on flow rather than on the channel cross-section, because a
        /// port the width of a channel is a sub-millimetre bore no machine will
        /// hold.
        /// </summary>
        public static PortCut FeedPorts(EngineAssembly a, ChannelCut cut,
                                        double insetMm = 1.5)
        {
            double d = Math.Max(1.2, 1.5 * cut.HeightMm);
            double xAt = cut.XEnd - insetMm;
            double wall = EngineAssembly.Interp(cut.WallX, cut.WallR, xAt);
            double rLo, rHi;
            if (cut.Outward)
            {
                rLo = wall + cut.HotWallMm;
                rHi = a.CowlOuterR.Max() + 5.0;
            }
            else
            {
                rLo = 0.0;
                rHi = wall - cut.HotWallMm;
            }
            return new PortCut
            {
                XAt = xAt, DiameterMm = d, Count = cut.NChannels,
                RLo = rLo, RHi = rHi, Phase = 0.0,
            };
        }

        public static List<HoleCut> InjectorOrifices(EngineAssembly a, InjectorDesign inj)
        {
            double x0 = a.HeadXMm - a.Spec.Geometry.HeadThicknessMm - 1.0;
            double x1 = a.HeadXMm + 1.0;
            return new List<HoleCut>
            {
                new() { RadiusMm = inj.FuelRingRadius * 1e3, DiameterMm = inj.DFuelMm,
                        Count = inj.NElements, XStart = x0, XEnd = x1, Phase = 0.0 },
                new() { RadiusMm = inj.OxRingRadius * 1e3, DiameterMm = inj.DOxMm,
                        Count = inj.NElements, XStart = x0, XEnd = x1,
                        Phase = Math.PI / inj.NElements },
            };
        }
    }
}
