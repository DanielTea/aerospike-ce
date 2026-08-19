//
// Signed distance functions for the parts, and the features cut into them.
//
// Mirrors the distance functions in validate/mesh_solid.py, deliberately: PicoGK
// renders any bounded implicit into voxels, so the same formulation the Python
// mesher uses goes to the kernel unchanged. There is no second description of
// the geometry to keep in step, and no boolean subtraction of several hundred
// individual channel solids.
//
// Nothing here derives anything. Every dimension arrives from the build plan;
// this file only evaluates distance to shapes it is handed.
//
// Model x maps to PicoGK Z. Radius is sqrt(X^2 + Y^2).
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
        public double WidthMm, HeightMm, HotWallMm, LandMm, XStart, XEnd;
        public bool Outward = true;
    }

    /// <summary>A ring of radial feed ports, one per cooling channel.</summary>
    public sealed class PortCut
    {
        public double XAt, DiameterMm, RLo, RHi, Phase;
        public int Count;
    }

    /// <summary>A ring of axial holes: injector orifices, or lug fixings.</summary>
    public sealed class HoleCut
    {
        public double RadiusMm, DiameterMm, XStart, XEnd, Phase;
        public int Count;
    }

    /// <summary>Annular material added round the outside, to wrap a plenum.</summary>
    public sealed class RingBoss
    {
        public double XAt, RInner, ROuter, HalfX;
    }

    /// <summary>Radial pads at the head end, for bolting the engine down.</summary>
    public sealed class LugAdd
    {
        public int Count;
        public double XAt, HalfX, RInner, ROuter, HalfWidthDeg, Phase;
    }

    /// <summary>
    /// An annular manifold void with a diamond section.
    ///
    /// Diamond rather than round or square because it is an internal void in a
    /// printed part: a flat roof the width of the plenum sags, and nothing can
    /// reach inside a closed ring to support it.
    /// </summary>
    public sealed class PlenumCut
    {
        public double XAt, RInner, HalfX, HalfR;
    }

    /// <summary>A part: a revolved meridional polygon with features applied.</summary>
    public sealed class CooledPart : IBoundedImplicit
    {
        private readonly double[] _px, _pr;
        private readonly List<ChannelCut> _channels;
        private readonly List<HoleCut> _holes;
        private readonly List<PortCut> _ports;
        private readonly List<RingBoss> _bosses;
        private readonly List<LugAdd> _lugs;
        private readonly List<PlenumCut> _plenums;
        private readonly BBox3 _bounds;

        // Tabulated profile distance over (x, r). See the constructor.
        private readonly float[,] _table;
        private readonly double _tx0, _tr0, _tStep;
        private readonly int _tnx, _tnr;

        public CooledPart((double[] x, double[] r) profile,
                          List<ChannelCut>? channels = null,
                          List<HoleCut>? holes = null,
                          List<PortCut>? ports = null,
                          List<RingBoss>? bosses = null,
                          List<LugAdd>? lugs = null,
                          List<PlenumCut>? plenums = null,
                          float fMarginMm = 1.0f,
                          double fTableStepMm = 0.05)
        {
            _px = profile.x; _pr = profile.r;
            _channels = channels ?? new();
            _holes = holes ?? new();
            _ports = ports ?? new();
            _bosses = bosses ?? new();
            _lugs = lugs ?? new();
            _plenums = plenums ?? new();

            // The box has to cover material added outside the profile. Sizing it
            // from the meridional profile alone clips the mounting lugs at the
            // boundary and the mesh comes back with open edges -- silently,
            // because a clipped solid still meshes.
            double x0 = _px.Min() - fMarginMm, x1 = _px.Max() + fMarginMm;
            double rr = _pr.Max() + fMarginMm;
            foreach (var b in _bosses)
            {
                x0 = Math.Min(x0, b.XAt - b.HalfX - fMarginMm);
                x1 = Math.Max(x1, b.XAt + b.HalfX + fMarginMm);
                rr = Math.Max(rr, b.ROuter + fMarginMm);
            }
            foreach (var lg in _lugs)
            {
                x0 = Math.Min(x0, lg.XAt - lg.HalfX - fMarginMm);
                x1 = Math.Max(x1, lg.XAt + lg.HalfX + fMarginMm);
                rr = Math.Max(rr, lg.ROuter + fMarginMm);
            }
            _bounds = new BBox3(new Vector3((float)-rr, (float)-rr, (float)x0),
                                new Vector3((float)rr, (float)rr, (float)x1));

            // The profile is a solid of revolution, so its distance field depends
            // only on (x, r). Walking all several hundred profile edges for every
            // voxel the kernel samples costs twenty-two billion operations on the
            // demonstrator alone: not slow, a hang. Tabulating once and
            // interpolating turns each sample into a lookup, with the step well
            // under the voxel so the error sits below the kernel's own sampling.
            _tStep = Math.Max(fTableStepMm, 1e-3);
            _tx0 = x0;
            _tr0 = 0.0;
            _tnx = (int)Math.Ceiling((x1 - x0) / _tStep) + 2;
            _tnr = (int)Math.Ceiling(rr * Math.Sqrt(2.0) / _tStep) + 2;
            _table = new float[_tnx, _tnr];
            Parallel.For(0, _tnx, i =>
            {
                double xv = _tx0 + i * _tStep;
                for (int j = 0; j < _tnr; j++)
                    _table[i, j] = (float)PolygonDistanceExact(xv, _tr0 + j * _tStep);
            });
        }

        public BBox3 oBounds => _bounds;

        private static double Interp(double[] xs, double[] rs, double x)
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

        /// <summary>
        /// Signed distance to the closed meridional polygon, negative inside.
        ///
        /// Distance to the nearest edge; the sign from a separate crossing
        /// count, because a nearest-edge normal test gets corners wrong and
        /// corners are most of this profile.
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

        private double PolygonDistance(double px, double pr)
        {
            double fi = (px - _tx0) / _tStep, fj = (pr - _tr0) / _tStep;
            int i = (int)Math.Floor(fi), j = (int)Math.Floor(fj);
            if (i < 0 || j < 0 || i >= _tnx - 1 || j >= _tnr - 1)
                return PolygonDistanceExact(px, pr);
            double u = fi - i, v = fj - j;
            return (1 - u) * (1 - v) * _table[i, j] + u * (1 - v) * _table[i + 1, j]
                 + (1 - u) * v * _table[i, j + 1] + u * v * _table[i + 1, j + 1];
        }

        /// <summary>
        /// Distance to the nearest channel of a ring, negative inside.
        ///
        /// The width is clamped by the local pitch less the land: the pitch
        /// shrinks with radius, and on the spike the circumference runs out
        /// before the channel count does. Unclamped, the channels merge into one
        /// groove that saws the plug in half. The Python clamps identically.
        /// </summary>
        private static double ChannelDistance(double x, double r, double th, ChannelCut c)
        {
            double wall = Interp(c.WallX, c.WallR, x);
            double sign = c.Outward ? 1.0 : -1.0;
            double floor = wall + sign * c.HotWallMm;
            double roof = floor + sign * c.HeightMm;
            double lo = Math.Min(floor, roof), hi = Math.Max(floor, roof);

            double pitch = 2.0 * Math.PI / c.NChannels;
            double arc = pitch * Math.Max(r, 1e-6);
            double width = Math.Min(c.WidthMm, Math.Max(arc - c.LandMm, 0.0));
            double local = th - pitch * Math.Floor(th / pitch + 0.5);

            return Math.Max(Math.Max(Math.Max(lo - r, r - hi),
                                     Math.Abs(local) * Math.Max(r, 1e-6) - 0.5 * width),
                            Math.Max(c.XStart - x, x - c.XEnd));
        }

        private static double PortDistance(double x, double r, double th, PortCut c)
        {
            double pitch = 2.0 * Math.PI / c.Count;
            double rel = th - c.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            return Math.Max(Math.Max(Math.Abs(local) * Math.Max(r, 1e-6) - 0.5 * c.DiameterMm,
                                     Math.Abs(x - c.XAt) - 0.5 * c.DiameterMm),
                            Math.Max(c.RLo - r, r - c.RHi));
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
            return Math.Max(Math.Sqrt(dr * dr + dt * dt) - 0.5 * h.DiameterMm,
                            Math.Max(h.XStart - x, x - h.XEnd));
        }

        private static double PlenumDistance(double x, double r, PlenumCut p)
        {
            double rc = p.RInner + p.HalfR;
            double a = Math.Max(p.HalfX, 1e-6), b = Math.Max(p.HalfR, 1e-6);
            double v = Math.Abs(x - p.XAt) / a + Math.Abs(r - rc) / b - 1.0;
            return v / Math.Sqrt(1.0 / (a * a) + 1.0 / (b * b));
        }

        private static double RingBossDistance(double x, double r, RingBoss b)
        {
            double taper = Math.Max(b.HalfX - Math.Max(r - b.RInner, 0.0), 0.0);
            return Math.Max(Math.Abs(x - b.XAt) - taper,
                            Math.Max(b.RInner - r, r - b.ROuter));
        }

        private static double LugDistance(double x, double r, double th, LugAdd lg)
        {
            double pitch = 2.0 * Math.PI / lg.Count;
            double rel = th - lg.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            // The angular term is an angle until it is multiplied by a radius.
            // Comparing radians against millimetres in the same Math.Max gives a
            // lug whose width varies with how far out you measure it.
            double dTheta = (Math.Abs(local) - lg.HalfWidthDeg * Math.PI / 180.0)
                            * Math.Max(r, 1e-6);
            return Math.Max(Math.Max(dTheta, Math.Max(lg.RInner - r, r - lg.ROuter)),
                            Math.Abs(x - lg.XAt) - lg.HalfX);
        }

        public float fSignedDistance(in Vector3 vec)
        {
            double x = vec.Z;                       // model x is PicoGK Z
            double y = vec.X, z = vec.Y;
            double r = Math.Sqrt(y * y + z * z);
            double th = Math.Atan2(z, y);

            double d = PolygonDistance(x, r);

            // Material is added before anything is taken away, so a hole through
            // a lug or a plenum inside a boss cuts the metal just placed rather
            // than the air where it used to be.
            foreach (var b in _bosses) d = Math.Min(d, RingBossDistance(x, r, b));
            foreach (var lg in _lugs) d = Math.Min(d, LugDistance(x, r, th, lg));
            foreach (var p in _plenums) d = Math.Max(d, -PlenumDistance(x, r, p));
            foreach (var c in _channels) d = Math.Max(d, -ChannelDistance(x, r, th, c));
            foreach (var pt in _ports) d = Math.Max(d, -PortDistance(x, r, th, pt));
            foreach (var h in _holes) d = Math.Max(d, -HoleDistance(x, y, z, h));
            return (float)d;
        }
    }
}
