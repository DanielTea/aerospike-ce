"""
Signed-distance meshing for the parts the revolve cannot express.

mesh_export.py revolves a meridional profile, which is exact and cheap and
cannot represent anything that varies with angle. Cooling channels and injector
orifices both do, so they need a different route: evaluate a signed distance
field on a grid, combine the features with CSG, and pull the surface out with
marching cubes. This is what PicoGK does natively; the version here exists so
the same geometry can be produced and checked without the native runtime.

Cost and resolution
-------------------
Memory goes as the cube of resolution, so the field is built one x-slab at a
time and only the finished volume is held whole. A 0.4 mm channel needs about
0.15 mm voxels to survive with its corners intact; at 0.2 mm it is recognisably
there and slightly rounded, which is the default because the alternative is a
gigabyte.

Marching cubes gives an approximate surface, so mesh volume lands within a
percent or so of the analytic value rather than on top of it. That is the
method, not a defect, and the tests check convergence with resolution instead of
demanding exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes

from engine_ref import EngineAssembly, Profile


# --------------------------------------------------------------------------
# 2D signed distance in the meridional plane
# --------------------------------------------------------------------------

def _polygon_sdf(px: np.ndarray, pr: np.ndarray,
                 vx: np.ndarray, vr: np.ndarray) -> np.ndarray:
    """
    Signed distance from points to a closed polygon in the (x, r) plane.

    Negative inside. Distance is measured to the nearest edge segment; the sign
    comes from a separate crossing count, because a nearest-edge normal test
    gets corners wrong and corners are most of this profile.
    """
    n = len(vx)
    ax, ar_ = vx, vr
    bx, br = np.roll(vx, -1), np.roll(vr, -1)

    ex, er = bx - ax, br - ar_
    ll = ex * ex + er * er
    ll[ll < 1e-30] = 1e-30

    best = np.full(px.shape, np.inf, dtype=np.float32)
    inside = np.zeros(px.shape, dtype=bool)

    for i in range(n):
        wx = px - ax[i]
        wr = pr - ar_[i]
        t = np.clip((wx * ex[i] + wr * er[i]) / ll[i], 0.0, 1.0)
        dx = wx - t * ex[i]
        dr = wr - t * er[i]
        np.minimum(best, np.sqrt(dx * dx + dr * dr, dtype=np.float32), out=best)

        # crossing count for the sign
        cond = ((ar_[i] > pr) != (br[i] > pr)) & \
               (px < (bx[i] - ax[i]) * (pr - ar_[i]) / (br[i] - ar_[i] + 1e-30) + ax[i])
        inside ^= cond

    return np.where(inside, -best, best).astype(np.float32)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelCut:
    """
    A ring of axial cooling channels riding a wall surface.

    The channel floor sits `hot_wall_mm` outboard of the gas-side wall, so the
    groove follows the contour rather than running straight, which is what keeps
    the hot wall a constant thickness through the contraction.
    """
    wall_x: np.ndarray          # gas-side wall the channels follow
    wall_r: np.ndarray
    n_channels: int
    width_mm: float
    height_mm: float
    hot_wall_mm: float
    x_start: float
    x_end: float
    outward: bool = True        # channels outboard of the wall, as on a cowl
    land_mm: float = 0.4        # needed to clamp the width at small radius


@dataclass(frozen=True)
class PortCut:
    """
    A ring of radial feed ports, one per cooling channel.

    Channels have to be fed. The cowl takes coolant from an external manifold
    through ports in its outer skin; the centrebody takes it from the central
    bore, which is already there as the drain path. Without a port each channel
    is a blind pocket, which is both unbuildable and topologically invisible --
    a blind pocket changes no genus, so the check that counts channels would
    pass on a part with none of them connected.
    """
    x_at: float
    diameter_mm: float
    count: int
    r_lo: float
    r_hi: float
    phase: float = 0.0


@dataclass(frozen=True)
class HoleCut:
    """A ring of axial holes, for injector orifices."""
    radius_mm: float            # ring radius on the face
    diameter_mm: float
    count: int
    x_start: float
    x_end: float
    phase: float = 0.0


def _channel_sdf(X, R, TH, cut: ChannelCut) -> np.ndarray:
    """Signed distance to the nearest channel of the ring. Negative inside."""
    wall = np.interp(X, cut.wall_x, cut.wall_r).astype(np.float32)
    sign = 1.0 if cut.outward else -1.0
    floor = wall + sign * cut.hot_wall_mm
    roof = floor + sign * cut.height_mm
    lo = np.minimum(floor, roof)
    hi = np.maximum(floor, roof)

    d_radial = np.maximum(lo - R, R - hi)

    # Fold angle into one channel pitch. The width is clamped by the local
    # pitch less the land, because the pitch shrinks with radius: on the spike
    # the circumference runs out before the channel count does, and unclamped
    # channels merge into one continuous groove that cuts the plug in half.
    # solve_cooling clamps the same way, so the thermal and the geometric
    # models describe the same channel.
    pitch = 2.0 * math.pi / cut.n_channels
    arc = pitch * np.maximum(R, 1e-6)
    width = np.minimum(cut.width_mm, np.maximum(arc - cut.land_mm, 0.0))
    local = np.mod(TH + 0.5 * pitch, pitch) - 0.5 * pitch
    d_theta = np.abs(local) * np.maximum(R, 1e-6) - 0.5 * width

    d_axial = np.maximum(cut.x_start - X, X - cut.x_end)
    return np.maximum(np.maximum(d_radial, d_theta), d_axial).astype(np.float32)


def _port_sdf(X, R, TH, cut: PortCut) -> np.ndarray:
    """Signed distance to the nearest radial port. Negative inside."""
    pitch = 2.0 * math.pi / cut.count
    local = np.mod(TH - cut.phase + 0.5 * pitch, pitch) - 0.5 * pitch
    d_theta = np.abs(local) * np.maximum(R, 1e-6) - 0.5 * cut.diameter_mm
    d_axial = np.abs(X - cut.x_at) - 0.5 * cut.diameter_mm
    d_radial = np.maximum(cut.r_lo - R, R - cut.r_hi)
    return np.maximum(np.maximum(d_theta, d_axial), d_radial).astype(np.float32)


def _hole_sdf(X, Y, Z, cut: HoleCut) -> np.ndarray:
    """Signed distance to the nearest hole of the ring."""
    th = np.arctan2(Z, Y)
    pitch = 2.0 * math.pi / cut.count
    local = np.mod(th - cut.phase + 0.5 * pitch, pitch) - 0.5 * pitch
    r = np.sqrt(Y * Y + Z * Z)

    # distance in the face plane to the nearest hole centre
    dr = r - cut.radius_mm
    dt = local * np.maximum(r, 1e-6)
    d_plane = np.sqrt(dr * dr + dt * dt) - 0.5 * cut.diameter_mm

    d_axial = np.maximum(cut.x_start - X, X - cut.x_end)
    return np.maximum(d_plane, d_axial).astype(np.float32)


# --------------------------------------------------------------------------
# field construction
# --------------------------------------------------------------------------

def _wedge_sdf(Y, Z, theta0: float, theta1: float) -> np.ndarray:
    """Signed distance to a wedge between two angles. Negative inside."""
    th = np.arctan2(Z, Y)
    r = np.sqrt(Y * Y + Z * Z)
    return np.maximum((theta0 - th) * r, (th - theta1) * r).astype(np.float32)


def build_field(
    profile: Profile,
    voxel_mm: float = 0.2,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    margin_mm: float = 1.0,
    cut_sector: tuple[float, float] | None = None,
    ports: list[PortCut] | None = None,
):
    """
    Signed distance field of one part, with its features subtracted.

    Built slab by slab along x: the whole volume at 0.2 mm for this engine is
    about a hundred megabytes, and holding the coordinate grids alongside it
    would be several times that.
    """
    channels = channels or []
    holes = holes or []
    ports = ports or []

    x0, x1 = float(profile.x.min()) - margin_mm, float(profile.x.max()) + margin_mm
    r_max = float(profile.r.max()) + margin_mm

    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil(2.0 * r_max / voxel_mm)) + 1)

    xs = np.linspace(x0, x1, nx, dtype=np.float32)
    ys = np.linspace(-r_max, r_max, ny, dtype=np.float32)
    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.sqrt(Y * Y + Z * Z).astype(np.float32)
    TH = np.arctan2(Z, Y).astype(np.float32)

    field = np.empty((nx, ny, ny), dtype=np.float32)
    vx = np.asarray(profile.x, dtype=np.float32)
    vr = np.asarray(profile.r, dtype=np.float32)

    for i, xv in enumerate(xs):
        X = np.full(R.shape, xv, dtype=np.float32)
        d = _polygon_sdf(X, R, vx, vr)
        for c in channels:
            d = np.maximum(d, -_channel_sdf(X, R, TH, c))
        for h in holes:
            d = np.maximum(d, -_hole_sdf(X, Y, Z, h))
        for pt in ports:
            d = np.maximum(d, -_port_sdf(X, R, TH, pt))
        if cut_sector is not None:
            d = np.maximum(d, -_wedge_sdf(Y, Z, cut_sector[0], cut_sector[1]))
        field[i] = d

    return field, (x0, -r_max, -r_max), voxel_mm


def mesh_field(field: np.ndarray, origin, voxel_mm: float):
    """Marching cubes at the zero level set, returned in model coordinates."""
    spacing = (
        (field.shape[0] - 1) and voxel_mm or voxel_mm,
        voxel_mm, voxel_mm,
    )
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=spacing)
    verts = verts + np.asarray(origin, dtype=float)
    return verts, faces.astype(np.int64)


def build_mesh(
    profile: Profile,
    voxel_mm: float = 0.2,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    cut_sector: tuple[float, float] | None = None,
    ports: list[PortCut] | None = None,
):
    field, origin, spacing = build_field(profile, voxel_mm, channels, holes,
                                         cut_sector=cut_sector, ports=ports)
    return mesh_field(field, origin, spacing)


# --------------------------------------------------------------------------
# feature construction from a solved design
# --------------------------------------------------------------------------

def _thick_enough_span(wall_x, available, needed: float):
    """
    Longest run of x over which the wall can hold a channel.

    The cowl tapers to a thin lip and the centrebody runs out of section behind
    the truncation. A channel that outlives its wall breaks through to the
    outside, which shows up as a watertight mesh with wildly wrong genus rather
    than as an obvious hole.
    """
    ok = available >= needed
    if not ok.any():
        raise ValueError(
            f"no station has {needed:.2f} mm of wall for a channel "
            f"(most available is {available.max():.2f} mm)")
    idx = np.flatnonzero(ok)
    # longest contiguous run
    splits = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    run = max(splits, key=len)
    return float(wall_x[run[0]]), float(wall_x[run[-1]])


def cowl_channels(a: EngineAssembly, channel_spec, back_wall_mm: float = 0.5,
                  x_start=None, x_end=None) -> ChannelCut:
    """
    Cooling channels riding the cowl's gas-side wall, cut outboard.

    Stopped short of the lip. The cowl tapers from the full wall thickness down
    to a thin blunt lip, and a channel run all the way to the end emerges
    through the taper somewhere before it.
    """
    from engine_ref import _distance_to_polyline

    wx = np.asarray(a.outer_wall_x, dtype=float)
    wr = np.asarray(a.outer_wall_r, dtype=float)

    # Wall available at each station, measured PERPENDICULAR to the gas surface.
    # Taking the radial difference instead overstates it wherever the wall is
    # steep: at the lip the radial gap reads 2.99 mm across a wall that is
    # actually 1.00 mm thick, so the channels are cleared to run somewhere they
    # would emerge straight through the taper.
    available = _distance_to_polyline(
        wx, wr,
        np.asarray(a.cowl_outer_x, dtype=float),
        np.asarray(a.cowl_outer_r, dtype=float),
    )

    needed = channel_spec.hot_wall_mm + channel_spec.height_mm + back_wall_mm
    xs, xe = _thick_enough_span(wx, available, needed)
    return ChannelCut(
        wall_x=wx, wall_r=wr,
        n_channels=channel_spec.n_channels,
        width_mm=channel_spec.width_mm,
        height_mm=channel_spec.height_mm,
        hot_wall_mm=channel_spec.hot_wall_mm,
        # run past the head face rather than stopping flush with it: a channel
        # whose opening coincides exactly with the face plane is a degenerate
        # marching-cubes case and sheds hundreds of loose fragments
        x_start=(xs - 2.0) if x_start is None else x_start,
        x_end=xe if x_end is None else x_end,
        outward=True,
        land_mm=channel_spec.land_mm,
    )


def centrebody_channels(a: EngineAssembly, channel_spec, back_wall_mm: float = 0.5,
                        x_start=None, x_end=None) -> ChannelCut:
    """
    Cooling channels under the centrebody's gas-side wall, cut inboard.

    Stopped where the section runs out behind the truncation, and the width is
    clamped by the local pitch so the channels do not merge as the spike closes.
    """
    from engine_ref import _distance_to_polyline

    wx = np.asarray(a.inner_wall_x, dtype=float)
    wr = np.asarray(a.inner_wall_r, dtype=float)

    # Perpendicular to the gas surface, for the same reason as the cowl: the
    # spike is steeply sloped and a radial reading flatters it.
    to_cavity = _distance_to_polyline(
        wx, wr,
        np.asarray(a.cavity_x, dtype=float),
        np.asarray(a.cavity_r, dtype=float),
    )
    # aft of the cavity the part is solid, so the whole radius is available
    available = np.where(wx <= a.cavity_x.max(), to_cavity, wr)

    needed = channel_spec.hot_wall_mm + channel_spec.height_mm + back_wall_mm
    xs, xe = _thick_enough_span(wx, available, needed)
    return ChannelCut(
        wall_x=wx, wall_r=wr,
        n_channels=channel_spec.n_channels,
        width_mm=channel_spec.width_mm,
        height_mm=channel_spec.height_mm,
        hot_wall_mm=channel_spec.hot_wall_mm,
        x_start=(xs - 2.0) if x_start is None else x_start,
        x_end=xe if x_end is None else x_end,
        outward=False,
        land_mm=channel_spec.land_mm,
    )


def injector_holes(a: EngineAssembly, injector) -> list[HoleCut]:
    """Fuel and oxidiser orifice rings through the head disc."""
    x0 = a.head_x - a.structure.head_thickness_mm - 1.0
    x1 = a.head_x + 1.0
    half_pitch = math.pi / injector.n_elements
    return [
        HoleCut(injector.fuel_ring_radius * 1e3, injector.d_fuel_mm,
                injector.n_elements, x0, x1, phase=0.0),
        HoleCut(injector.ox_ring_radius * 1e3, injector.d_ox_mm,
                injector.n_elements, x0, x1, phase=half_pitch),
    ]


# --------------------------------------------------------------------------
# streaming construction at inspection resolution
# --------------------------------------------------------------------------

def _polygon_sdf_radial(x_val: float, r_grid: np.ndarray,
                        vx: np.ndarray, vr: np.ndarray) -> np.ndarray:
    """
    Signed distance to the profile along one radial line, vectorised over r.

    The profile is a solid of revolution, so its distance field depends only on
    (x, r) and not on angle. Evaluating it once per x-plane on a radius grid and
    interpolating onto the (y, z) grid turns the cost of a plane from a few
    hundred million operations into a few million, which is the difference
    between a mesh that resolves a 0.4 mm channel and one that does not.
    """
    n = len(vx)
    ax, ar = vx, vr
    bx, br = np.roll(vx, -1), np.roll(vr, -1)
    ex, er = bx - ax, br - ar
    ll = ex * ex + er * er
    ll[ll < 1e-30] = 1e-30

    best = np.full(r_grid.shape, np.inf, dtype=np.float64)
    inside = np.zeros(r_grid.shape, dtype=bool)

    for i in range(n):
        wx = x_val - ax[i]
        wr = r_grid - ar[i]
        t = np.clip((wx * ex[i] + wr * er[i]) / ll[i], 0.0, 1.0)
        dx = wx - t * ex[i]
        dr = wr - t * er[i]
        np.minimum(best, np.hypot(dx, dr), out=best)

        cond = ((ar[i] > r_grid) != (br[i] > r_grid)) & \
               (x_val < (bx[i] - ax[i]) * (r_grid - ar[i]) / (br[i] - ar[i] + 1e-30) + ax[i])
        inside ^= cond

    return np.where(inside, -best, best)


def _weld(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-7):
    """
    Merge coincident vertices so a slab-wise mesh is index-manifold.

    Adjacent slabs share a plane of samples, so both produce the same vertices
    on it to the last bit. Without welding the surface is a triangle soup that
    reports boundary edges everywhere the slabs meet, and the topology check
    becomes meaningless.
    """
    key = np.round(verts / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    out_v = verts[first]
    out_f = inverse[faces]
    keep = (out_f[:, 0] != out_f[:, 1]) & (out_f[:, 1] != out_f[:, 2]) & \
           (out_f[:, 2] != out_f[:, 0])
    return out_v, out_f[keep]


def build_mesh_streaming(
    profile: Profile,
    voxel_mm: float = 0.15,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    margin_mm: float = 1.0,
    cut_sector: tuple[float, float] | None = None,
    ports: list[PortCut] | None = None,
    slab_planes: int = 12,
    radial_samples: int = 6000,
    progress: bool = False,
):
    """
    Marching cubes over the volume in slabs, welded into one mesh.

    Memory is bounded by the slab rather than by the whole field, so the voxel
    size can be chosen to resolve the feature rather than to fit in RAM. A
    0.4 mm cooling channel needs roughly three samples across to survive with
    its topology intact, which means about 0.13 mm.

    Consecutive slabs overlap by exactly one plane of samples: marching cubes
    consumes cells, not planes, so an overlap of one keeps every cell processed
    exactly once and leaves no seam.
    """
    channels = channels or []
    holes = holes or []
    ports = ports or []

    x0 = float(profile.x.min()) - margin_mm
    x1 = float(profile.x.max()) + margin_mm
    r_max = float(profile.r.max()) + margin_mm

    # The grid must step by exactly voxel_mm, not by a linspace interval that
    # merely averages to it. marching_cubes is told the spacing separately, so
    # any mismatch shifts each slab slightly against the next and the shared
    # plane's vertices no longer coincide -- the weld then fails and the mesh
    # reports boundary edges along every slab seam.
    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil(2.0 * r_max / voxel_mm)) + 1)

    xs = x0 + np.arange(nx) * voxel_mm
    ys = -r_max + np.arange(ny) * voxel_mm
    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.hypot(Y, Z)
    TH = np.arctan2(Z, Y)

    r_grid = np.linspace(0.0, float(np.hypot(ys, ys).max()) * 1.001, radial_samples)
    vx = np.asarray(profile.x, dtype=float)
    vr = np.asarray(profile.r, dtype=float)

    def plane(i: int) -> np.ndarray:
        prof1d = _polygon_sdf_radial(xs[i], r_grid, vx, vr)
        d = np.interp(R, r_grid, prof1d).astype(np.float32)
        X = np.full(R.shape, xs[i], dtype=np.float32)
        for c in channels:
            d = np.maximum(d, -_channel_sdf(X, R.astype(np.float32),
                                            TH.astype(np.float32), c))
        for h in holes:
            d = np.maximum(d, -_hole_sdf(X, Y.astype(np.float32),
                                         Z.astype(np.float32), h))
        for pt in ports:
            d = np.maximum(d, -_port_sdf(X, R.astype(np.float32),
                                         TH.astype(np.float32), pt))
        if cut_sector is not None:
            d = np.maximum(d, -_wedge_sdf(Y.astype(np.float32),
                                          Z.astype(np.float32),
                                          cut_sector[0], cut_sector[1]))
        return d

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    offset = 0
    step = max(1, slab_planes)

    i0 = 0
    cache = {i0: plane(i0)}
    while i0 < nx - 1:
        i1 = min(i0 + step, nx - 1)
        block = np.empty((i1 - i0 + 1,) + R.shape, dtype=np.float32)
        block[0] = cache.pop(i0)
        for k in range(i0 + 1, i1 + 1):
            block[k - i0] = plane(k)
        cache = {i1: block[-1].copy()}          # reused as the next slab's first

        if block.min() < 0.0 < block.max():
            v, f, _, _ = marching_cubes(block, level=0.0,
                                        spacing=(voxel_mm, voxel_mm, voxel_mm))
            v = v + np.array([xs[i0], ys[0], ys[0]])
            all_v.append(v)
            all_f.append(f.astype(np.int64) + offset)
            offset += len(v)
        if progress:
            print(f"  slab {i0:5d}/{nx - 1}  {len(all_f)} pieces", flush=True)
        i0 = i1

    if not all_v:
        raise ValueError("the field never crosses zero; nothing to mesh")
    return _weld(np.concatenate(all_v), np.concatenate(all_f),
                 tol=voxel_mm * 1e-5)


def decimate(verts: np.ndarray, faces: np.ndarray, keep_fraction: float = 0.10):
    """
    Quadric decimation to a target fraction of the triangles *kept*.

    Marching cubes tessellates smooth regions at voxel resolution, which is most
    of the triangles and none of the information. Collapsing edges by quadric
    error keeps the channels and throws away the flat ground between them.

    Note the argument is what to keep. The underlying library takes the fraction
    to *remove*, which reads as its opposite: asking it for 0.18 leaves 82 per
    cent of the mesh, not 18, and turns what was meant to be an eight-fold
    reduction into a two-gigabyte file.

    Decimation can merge across a thin feature and change the genus, which here
    would mean silently closing a cooling channel. The caller compares the genus
    before and after; this function only does the collapsing.
    """
    import fast_simplification

    v, f = fast_simplification.simplify(
        verts.astype(np.float32), faces.astype(np.int32),
        1.0 - float(np.clip(keep_fraction, 1e-3, 1.0)))
    return np.asarray(v, dtype=float), np.asarray(f, dtype=np.int64)


def feed_ports(a: EngineAssembly, cut: ChannelCut, diameter_mm: float | None = None,
               inset_mm: float = 1.5) -> PortCut:
    """
    Radial feed ports at the aft end of a channel ring.

    Coolant enters at the throat end, where the flux is worst, and leaves at the
    injector. The cowl draws from an external manifold through its outer skin;
    the centrebody draws from the central bore. Either way the port has to reach
    from the channel through to the free surface, so it is sized on the channel
    and run well past both.
    """
    # Sized on flow, not on the channel cross-section. A port the width of the
    # channel is a sub-millimetre bore that no drill or powder-bed machine will
    # hold, and at any sane voxel size it is too small to mesh. Real feed ports
    # are millimetres; 1.2 mm here keeps the port area comfortably above the
    # channel area so it is not the restriction.
    d = diameter_mm if diameter_mm is not None else max(1.2, 1.5 * cut.height_mm)
    x_at = cut.x_end - inset_mm
    wall = float(np.interp(x_at, cut.wall_x, cut.wall_r))
    if cut.outward:
        r_lo = wall + cut.hot_wall_mm
        r_hi = float(np.max(a.cowl_outer_r)) + 5.0
    else:
        r_lo = 0.0
        r_hi = wall - cut.hot_wall_mm
    return PortCut(x_at=x_at, diameter_mm=d, count=cut.n_channels,
                   r_lo=r_lo, r_hi=r_hi, phase=0.0)
