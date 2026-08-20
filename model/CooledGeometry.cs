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

    /// <summary>
    /// Stiffening ribs standing proud of a shell, one per cooling channel.
    ///
    /// They sit on the lands, where there is already material, and taper so
    /// their flanks stand above the process angle. Several hundred marginal
    /// overhangs is not a shell, it is a support problem with no way in.
    /// </summary>
    public sealed class RibAdd
    {
        public double[] BaseX = Array.Empty<double>();
        public double[] BaseR = Array.Empty<double>();
        public int Count;
        public double HeightMm, RootWidthMm, FlankDeg, XStart, XEnd, Phase;
        public bool Outward = true;
    }

    /// <summary>Splayed mounting legs, chamber down to a footed pad.</summary>
    public sealed class LegAdd
    {
        public int Count;
        public double XTop, RTop, XFoot, RFoot, ThicknessMm, HalfWidthDeg,
                      PadRadiusMm, Phase;
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
        private readonly List<RibAdd> _ribs;
        private readonly List<LegAdd> _legs;
        private readonly BBox3 _bounds;

        /// <summary>
        /// A profile vertex this close to r = 0 is on the axis of revolution.
        /// </summary>
        private const double AxisEpsMm = 1e-9;

        /// <summary>
        /// How far outside the zero set the surface is actually taken.
        ///
        /// The kernel's mesher is degenerate where a sample sits exactly on
        /// the surface, and on a solid of revolution that is not bad luck but
        /// the ordinary case: the flat faces are normal to the axis, so are
        /// the lattice planes, and a face a whole number of voxels from the
        /// grid origin lands on a plane of samples that all read exactly zero.
        /// The head does it at 0.2 mm and came back with four hundred thousand
        /// zero-area triangles and a quarter of a million edges shared by more
        /// than two faces, from a field that was entirely correct. Cura will
        /// not slice it and nothing in a viewer shows why.
        ///
        /// A tenth of a micron is four orders under the voxel and three under
        /// anything the process can hold. Mirrors SURFACE_BIAS_MM in
        /// validate/mesh_solid.py, where the same bias is applied as the
        /// marching-cubes level.
        /// </summary>
        private const double SurfaceBiasMm = 1e-4;

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
                          List<RibAdd>? ribs = null,
                          List<LegAdd>? legs = null,
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
            _ribs = ribs ?? new();
            _legs = legs ?? new();

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
            foreach (var rb in _ribs)
            {
                x0 = Math.Min(x0, rb.XStart - fMarginMm);
                x1 = Math.Max(x1, rb.XEnd + fMarginMm);
                if (rb.Outward)
                    rr = Math.Max(rr, rb.BaseR.Max() + rb.HeightMm + fMarginMm);
            }
            foreach (var lg in _legs)
            {
                x0 = Math.Min(x0, Math.Min(lg.XTop, lg.XFoot) - fMarginMm);
                x1 = Math.Max(x1, Math.Max(lg.XTop, lg.XFoot) + fMarginMm);
                rr = Math.Max(rr, Math.Max(lg.RFoot, lg.PadRadiusMm) + fMarginMm);
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
        ///
        /// Edges lying on the axis are not surfaces and are skipped. A profile
        /// that closes on r = 0 carries one -- the centrebody runs three
        /// millimetres down the axis, from the middle of the truncation face
        /// to the apex of its cavity -- and revolving it sweeps no area at
        /// all. Measured as an edge it puts the zero level set down the middle
        /// of solid metal, and the mesher then renders a zero-width sheet
        /// along the axis. They cannot affect the crossing count either, both
        /// endpoints sitting at the same radius.
        /// </summary>
        private double PolygonDistanceExact(double px, double pr)
        {
            int n = _px.Length;
            double best = double.MaxValue;
            bool inside = false;
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                if (_pr[i] <= AxisEpsMm && _pr[j] <= AxisEpsMm)
                    continue;
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

        // Exact distance to the rhombus, then offset by the fillet. Mirrors
        // _plenum_sdf in mesh_solid.py, deliberately: PicoGK renders the same
        // implicit the Python mesher samples, so the two carry one formulation
        // rather than two descriptions of the same cavity.
        //
        // The fillet is load-bearing. A diamond apex is a sharp internal edge
        // that no powder-bed machine makes, and sampled exactly on it marching
        // cubes emits a pinch: two non-manifold edges in fifteen million, no
        // boundary anywhere, and a part that fails watertightness with nothing
        // visibly wrong.
        private static double PlenumDistance(double x, double r, PlenumCut p)
        {
            double rc = p.RInner + p.HalfR;
            // Shrunk by the fillet measured perpendicular to the faces, not
            // subtracted from the half-diagonals. On a ten-to-one section those
            // differ by enough to make the rounded shape larger than the
            // diamond it is supposed to fit inside.
            double rad = Math.Min(0.4, Math.Min(0.25 * p.HalfR, 0.25 * p.HalfX));
            double face = p.HalfX * p.HalfR / Math.Sqrt(p.HalfX * p.HalfX + p.HalfR * p.HalfR);
            double k = Math.Max(1.0 - rad / face, 1e-6);
            double a = Math.Max(p.HalfX * k, 1e-6);
            double b = Math.Max(p.HalfR * k, 1e-6);

            double px = Math.Abs(x - p.XAt);
            double py = Math.Abs(r - rc);
            double h = ((a - 2.0 * px) * a - (b - 2.0 * py) * b) / (a * a + b * b);
            h = Math.Clamp(h, -1.0, 1.0);
            double qx = px - 0.5 * a * (1.0 - h);
            double qy = py - 0.5 * b * (1.0 + h);
            double d = Math.Sqrt(qx * qx + qy * qy);
            return d * Math.Sign(px * b + py * a - a * b) - rad;
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

        private static double RibDistance(double x, double r, double th, RibAdd rb)
        {
            double bas = Interp(rb.BaseX, rb.BaseR, x);
            double sign = rb.Outward ? 1.0 : -1.0;
            double h = sign * (r - bas);
            double dRadial = Math.Max(-h, h - rb.HeightMm);

            double shrink = h / Math.Max(Math.Tan(rb.FlankDeg * Math.PI / 180.0), 1e-6);
            double halfArc = Math.Max(0.5 * rb.RootWidthMm - shrink, 0.0);

            double pitch = 2.0 * Math.PI / rb.Count;
            double rel = th - rb.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            double dTheta = Math.Abs(local) * Math.Max(r, 1e-6) - halfArc;

            return Math.Max(Math.Max(dRadial, dTheta),
                            Math.Max(rb.XStart - x, x - rb.XEnd));
        }

        private static double LegDistance(double x, double r, double th, LegAdd lg)
        {
            double span = lg.XFoot - lg.XTop;
            double t = Math.Clamp((x - lg.XTop) / (Math.Abs(span) > 1e-9 ? span : 1e-9),
                                  0.0, 1.0);
            double rMid = lg.RTop + t * (lg.RFoot - lg.RTop);

            double pitch = 2.0 * Math.PI / lg.Count;
            double rel = th - lg.Phase;
            double local = rel - pitch * Math.Floor(rel / pitch + 0.5);
            double dTheta = (Math.Abs(local) - lg.HalfWidthDeg * Math.PI / 180.0)
                            * Math.Max(r, 1e-6);

            double lo = Math.Min(lg.XTop, lg.XFoot), hi = Math.Max(lg.XTop, lg.XFoot);
            double strut = Math.Max(Math.Max(dTheta, Math.Abs(r - rMid) - 0.5 * lg.ThicknessMm),
                                    Math.Max(lo - x, x - hi));

            double pad = Math.Max(Math.Max(lo - x, x - (lo + lg.ThicknessMm)),
                                  Math.Max(dTheta, Math.Max(lg.RTop - r, r - lg.PadRadiusMm)));
            return Math.Min(strut, pad);
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
            foreach (var rb in _ribs) d = Math.Min(d, RibDistance(x, r, th, rb));
            foreach (var lg in _legs) d = Math.Min(d, LegDistance(x, r, th, lg));
            foreach (var lg in _lugs) d = Math.Min(d, LugDistance(x, r, th, lg));
            foreach (var p in _plenums) d = Math.Max(d, -PlenumDistance(x, r, p));
            foreach (var c in _channels) d = Math.Max(d, -ChannelDistance(x, r, th, c));
            foreach (var pt in _ports) d = Math.Max(d, -PortDistance(x, r, th, pt));
            foreach (var h in _holes) d = Math.Max(d, -HoleDistance(x, y, z, h));
            return (float)(d - SurfaceBiasMm);
        }
    }
}
