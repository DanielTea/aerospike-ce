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
):
    """
    Signed distance field of one part, with its features subtracted.

    Built slab by slab along x: the whole volume at 0.2 mm for this engine is
    about a hundred megabytes, and holding the coordinate grids alongside it
    would be several times that.
    """
    channels = channels or []
    holes = holes or []

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
):
    field, origin, spacing = build_field(profile, voxel_mm, channels, holes,
                                         cut_sector=cut_sector)
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
        x_start=xs if x_start is None else x_start,
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
        x_start=xs if x_start is None else x_start,
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
