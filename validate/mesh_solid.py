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
import struct
from dataclasses import dataclass

import numpy as np
from skimage.measure import marching_cubes

from engine_ref import EngineAssembly, Profile

# A profile vertex this close to r = 0 is on the axis of revolution.
AXIS_EPS = 1e-9

# How far outside the zero set the surface is actually taken.
#
# Marching cubes is degenerate wherever a sample sits exactly on the surface,
# and on a solid of revolution that is not a coincidence, it is the common
# case: every flat face is normal to the axis, so is every lattice plane, and
# a face a whole number of voxels from the grid origin lands on a plane of
# samples that all read exactly zero. The head's two faces do it at the
# default 0.2 mm, and what comes out is a million zero-area triangles with
# holes between them, from a field that is perfectly correct.
#
# A tenth of a micron is four orders under the voxel, three under anything the
# process can hold, and far enough above the float32 quantum near zero that no
# sample lands on it.
SURFACE_BIAS_MM = 1e-4


def level_guard_mm(shape, voxel_mm: float) -> float:
    """
    How far a sample has to stay from the level for its vertex to be distinct.

    Two things set it. A float32 index carries about `n * 2^-24` of absolute
    resolution, where n is how far out the indices go; eight of those puts the
    vertex unambiguously beside the sample rather than on it. And the field is
    a distance function, so it changes by at most a voxel between neighbours --
    doubled here, because a channel floor riding an inclined wall is steeper
    than one, and the band has to hold for the steepest edge in the block.
    """
    return 8.0 * max(shape) * 2.0 ** -24 * (2.0 * voxel_mm)


def hold_off_level(block: np.ndarray, voxel_mm: float,
                   level: float = SURFACE_BIAS_MM) -> np.ndarray:
    """
    Push samples out of the band around the level, each to the side it was on.

    The bias moves the level off the zero set, which is what stops a flat face
    landing on a lattice plane from meshing as a sheet of degenerate triangles.
    It does not stop the same thing happening to one sample in a hundred
    million by chance, and that is a different failure with a different cure.

    Marching cubes places a vertex along a grid edge at t = (level - v0) /
    (v1 - v0), and it works in *index* units, in single precision. Out at index
    1024 a float32 resolves about 6e-5 of an index, so a crossing with t below
    that lands exactly on the sample instead of just beside it -- and every
    other edge into that sample lands there too. The weld then merges what
    marching cubes meant to keep apart, and the surface pinches: one edge shared
    by four faces, no boundary anywhere, an odd Euler characteristic, and
    nothing whatsoever to see. The cowl did it once in 24 million triangles, at a
    sample 95 nanometres above a channel floor and so five below the level being
    meshed; sliding the lattice 0.08 mm made it vanish, which is how you tell
    arithmetic from geometry.

    Pushed to the same side, so no cell changes classification and the topology
    is exactly what it was; the surface moves by at most the band, which is
    about a micron against a 233 micron voxel. Returns the input untouched when
    nothing is in the band, and a copy otherwise -- the caller's field is not
    ours to move.
    """
    guard = level_guard_mm(block.shape, voxel_mm)
    near = np.abs(block - level) < guard
    if not near.any():
        return block
    out = block.copy()
    out[near] = np.where(block[near] >= level, level + guard, level - guard)
    return out


def _mesh_block(block: np.ndarray, voxel_mm: float, level: float = SURFACE_BIAS_MM):
    """Marching cubes over one block, with every sample held off the level."""
    block = hold_off_level(block, voxel_mm, level)
    verts, faces, _, _ = marching_cubes(block, level=level,
                                        spacing=(voxel_mm, voxel_mm, voxel_mm))
    return verts, faces


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

    Edges lying on the axis are not surfaces and are skipped. A meridional
    profile that closes on r = 0 carries one -- the centrebody runs three
    millimetres down the axis, from the middle of the truncation face to the
    apex of its cavity -- and revolving it sweeps no area at all. Measured as
    an edge it puts the zero level set down the middle of solid metal, and
    marching cubes then meshes a zero-width sheet along the axis: degenerate
    triangles, edges shared by sixteen faces, a part every slicer rejects.
    They cannot affect the crossing count either, both endpoints sitting at
    the same radius.
    """
    n = len(vx)
    ax, ar_ = vx, vr
    bx, br = np.roll(vx, -1), np.roll(vr, -1)

    ex, er = bx - ax, br - ar_
    ll = ex * ex + er * er
    ll[ll < 1e-30] = 1e-30

    on_axis = (ar_ <= AXIS_EPS) & (br <= AXIS_EPS)

    best = np.full(px.shape, np.inf, dtype=np.float32)
    inside = np.zeros(px.shape, dtype=bool)

    for i in range(n):
        if on_axis[i]:
            continue
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
class PlenumCut:
    """
    An annular manifold void with a diamond section.

    Diamond rather than round or square because it is an internal void in a
    printed part: a flat roof the width of the plenum sags, and nothing can
    reach inside a closed ring to support it. The upper faces stand at
    atan(half_x / half_r) from the plate.
    """
    x_at: float
    r_inner: float
    half_x: float
    half_r: float


@dataclass(frozen=True)
class RingBoss:
    """
    Annular material added round the outside, to wrap a plenum in metal.

    The section tapers to nothing at its outer radius, so both flanks stand at
    45 degrees and neither the leading nor the trailing shoulder is an overhang.
    A square boss would present a horizontal ledge on its underside.
    """
    x_at: float
    r_inner: float
    r_outer: float
    half_x: float


@dataclass(frozen=True)
class RibAdd:
    """
    Stiffening ribs standing proud of a shell, one per cooling channel.

    They sit on the lands, which is where there is already material, and taper
    so their flanks stand above the process angle -- several hundred marginal
    overhangs is not a shell, it is a support problem.
    """
    base_x: np.ndarray
    base_r: np.ndarray
    count: int
    height_mm: float
    root_width_mm: float
    flank_deg: float
    x_start: float
    x_end: float
    outward: bool = True
    phase: float = 0.0


@dataclass(frozen=True)
class LegAdd:
    """One ring of splayed mounting legs, chamber down to a footed pad."""
    count: int
    x_top: float
    r_top: float
    x_foot: float
    r_foot: float
    thickness_mm: float
    half_width_deg: float
    pad_radius_mm: float
    phase: float = 0.0


@dataclass(frozen=True)
class LugAdd:
    """A ring of mounting lugs: radial pads at the head end."""
    count: int
    x_at: float
    half_x: float
    r_inner: float
    r_outer: float
    half_width_deg: float
    phase: float = 0.0


@dataclass(frozen=True)
class HoleCut:
    """
    A ring of axial holes.

    Named, because the head carries two kinds -- injector orifices and mounting
    bolt holes -- in one list, and a check that wants one of them has no other
    way to tell which is which than picking an index and hoping.
    """
    radius_mm: float            # ring radius on the face
    diameter_mm: float
    count: int
    x_start: float
    x_end: float
    phase: float = 0.0
    name: str = ""


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


def plenum_fillet_mm(half_x: float, half_r: float) -> float:
    """Radius the diamond's corners are rounded to."""
    return min(0.4, 0.25 * half_r, 0.25 * half_x)


def plenum_section(half_x: float, half_r: float):
    """
    The rhombus that, offset outward by the fillet, gives the plenum section.

    Shrunk by the fillet measured perpendicular to the faces, not by subtracting
    it from the half-diagonals. On a ten-to-one section those are not remotely
    the same thing: taking a fifth of a millimetre off a 20 mm half-diagonal
    barely moves the face it belongs to, and the rounded shape ends up eighteen
    percent *larger* than the diamond it was supposed to fit inside.
    """
    rad = plenum_fillet_mm(half_x, half_r)
    face = half_x * half_r / math.hypot(half_x, half_r)   # centre to face
    k = max(1.0 - rad / face, 1e-6)
    return half_x * k, half_r * k, rad


def plenum_section_area_mm2(half_x: float, half_r: float) -> float:
    """Flow area of the filleted section: rhombus, plus its offset band."""
    a, b, rad = plenum_section(half_x, half_r)
    return 2.0 * a * b + 4.0 * math.hypot(a, b) * rad + math.pi * rad * rad


def _plenum_sdf(X, R, cut: PlenumCut) -> np.ndarray:
    """
    Distance to the diamond ring, corners filleted. Negative inside.

    The fillet is not cosmetic. A diamond's apex is a mathematically sharp
    internal edge -- the section closes to zero width there -- and no powder-bed
    machine makes one: it fills with a radius whether you draw it or not. It is
    also degenerate to mesh. Sampled exactly on the apex, marching cubes emits a
    pinch where the surface meets itself, and the part comes back with two
    non-manifold edges out of fifteen million and no boundary at all, which
    reads as a watertightness failure with nothing visibly wrong anywhere.

    Exact distance to the rhombus, after Quilez, then offset by the fillet. The
    half-diagonals are shrunk by the same radius first, so the filleted ring
    stays inside the envelope its dimensions claim rather than growing past it.
    """
    rc = cut.r_inner + cut.half_r
    a, b, rad = plenum_section(cut.half_x, cut.half_r)
    a, b = max(a, 1e-6), max(b, 1e-6)

    px = np.abs(X - cut.x_at)
    py = np.abs(R - rc)
    h = np.clip(((a - 2.0 * px) * a - (b - 2.0 * py) * b) / (a * a + b * b),
                -1.0, 1.0)
    qx = px - 0.5 * a * (1.0 - h)
    qy = py - 0.5 * b * (1.0 + h)
    d = np.sqrt(qx * qx + qy * qy)
    sign = np.sign(px * b + py * a - a * b)
    return (d * sign - rad).astype(np.float32)


def _ring_boss_sdf(X, R, boss: RingBoss) -> np.ndarray:
    """Distance to the tapered ring of added material. Negative inside."""
    taper = np.maximum(boss.half_x - np.maximum(R - boss.r_inner, 0.0), 0.0)
    d_ax = np.abs(X - boss.x_at) - taper
    d_r = np.maximum(boss.r_inner - R, R - boss.r_outer)
    return np.maximum(d_ax, d_r).astype(np.float32)


def _lug_sdf(X, R, TH, lug: LugAdd) -> np.ndarray:
    """Distance to the nearest mounting lug. Negative inside."""
    pitch = 2.0 * math.pi / lug.count
    rel = TH - lug.phase
    local = np.mod(rel + 0.5 * pitch, pitch) - 0.5 * pitch
    d_theta = np.abs(local) - math.radians(lug.half_width_deg)
    d_theta = d_theta * np.maximum(R, 1e-6)
    d_r = np.maximum(lug.r_inner - R, R - lug.r_outer)
    d_x = np.abs(X - lug.x_at) - lug.half_x
    return np.maximum(np.maximum(d_theta, d_r), d_x).astype(np.float32)


def _rib_sdf(X, R, TH, rib: RibAdd) -> np.ndarray:
    """Distance to the nearest rib. Negative inside."""
    base = np.interp(X, rib.base_x, rib.base_r).astype(np.float32)
    sign = 1.0 if rib.outward else -1.0
    h = sign * (R - base)                       # height above the shell
    d_radial = np.maximum(-h, h - rib.height_mm)

    # the rib narrows with height, so its flanks stand at flank_deg
    shrink = h / max(math.tan(math.radians(rib.flank_deg)), 1e-6)
    half_arc = np.maximum(0.5 * rib.root_width_mm - shrink, 0.0)

    pitch = 2.0 * math.pi / rib.count
    local = np.mod(TH - rib.phase + 0.5 * pitch, pitch) - 0.5 * pitch
    d_theta = np.abs(local) * np.maximum(R, 1e-6) - half_arc

    d_axial = np.maximum(rib.x_start - X, X - rib.x_end)
    return np.maximum(np.maximum(d_radial, d_theta), d_axial).astype(np.float32)


def _leg_sdf(X, R, TH, leg: LegAdd) -> np.ndarray:
    """Distance to the nearest splayed leg, foot pad included. Negative inside."""
    span = leg.x_foot - leg.x_top
    t = np.clip((X - leg.x_top) / (span if abs(span) > 1e-9 else 1e-9), 0.0, 1.0)
    r_mid = leg.r_top + t * (leg.r_foot - leg.r_top)

    pitch = 2.0 * math.pi / leg.count
    local = np.mod(TH - leg.phase + 0.5 * pitch, pitch) - 0.5 * pitch
    d_theta = (np.abs(local) - math.radians(leg.half_width_deg)) * np.maximum(R, 1e-6)

    d_radial = np.abs(R - r_mid) - 0.5 * leg.thickness_mm
    lo, hi = min(leg.x_top, leg.x_foot), max(leg.x_top, leg.x_foot)
    d_axial = np.maximum(lo - X, X - hi)
    strut = np.maximum(np.maximum(d_theta, d_radial), d_axial)

    # the foot: a pad at the bottom, wide enough to take a fixing
    pad_lo = min(leg.x_top, leg.x_foot)
    d_pad = np.maximum(np.maximum(pad_lo - X, X - (pad_lo + leg.thickness_mm)),
                       np.maximum(d_theta, np.maximum(leg.r_top - R,
                                                      R - leg.pad_radius_mm)))
    return np.minimum(strut, d_pad).astype(np.float32)


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


def sample_field(
    X, Y, Z,
    profile: Profile,
    base: np.ndarray | None = None,
    R: np.ndarray | None = None,
    TH: np.ndarray | None = None,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    ports: list[PortCut] | None = None,
    bosses: list[RingBoss] | None = None,
    lugs: list[LugAdd] | None = None,
    plenums: list[PlenumCut] | None = None,
    ribs: list[RibAdd] | None = None,
    legs: list[LegAdd] | None = None,
    cut_sector: tuple[float, float] | None = None,
):
    """
    The part's signed distance at whatever points are handed in.

    A grid plane or a scatter of mesh vertices -- the arrays only have to
    broadcast together. Pulling it out of the grid loop is what lets a mesh be
    asked how far it strays from the surface it claims to be, which is a
    question about points that are nowhere near a lattice.

    The order is load-bearing and is the reason this is one function rather
    than an idiom copied per caller: material is added before anything is taken
    away, so a hole drilled through a lug cuts the metal that was just put
    there rather than the air where it used to be.

    `base` is the part's own distance where the caller already has it by a
    cheaper route, and `R`/`TH` likewise. The polygon distance and the
    cylindrical coordinates are most of the cost on a grid, and the streaming
    mesher computes both once per plane.
    """
    if R is None:
        R = np.hypot(Y, Z)
    if TH is None:
        TH = np.arctan2(Z, Y)
    if base is None:
        # Match the caller's precision rather than imposing one: this is
        # evaluated on a float32 grid by build_field and on float64 points by
        # part_sampler, and quietly widening the grid path would change every
        # vertex the mesher places.
        vx = np.asarray(profile.x, dtype=X.dtype)
        vr = np.asarray(profile.r, dtype=X.dtype)
        d = _polygon_sdf(X, R, vx, vr)
    else:
        d = base

    for b in (bosses or []):
        d = np.minimum(d, _ring_boss_sdf(X, R, b))
    for rb in (ribs or []):
        d = np.minimum(d, _rib_sdf(X, R, TH, rb))
    for lgg in (legs or []):
        d = np.minimum(d, _leg_sdf(X, R, TH, lgg))
    for lg in (lugs or []):
        d = np.minimum(d, _lug_sdf(X, R, TH, lg))
    for pl in (plenums or []):
        d = np.maximum(d, -_plenum_sdf(X, R, pl))
    for c in (channels or []):
        d = np.maximum(d, -_channel_sdf(X, R, TH, c))
    for h in (holes or []):
        d = np.maximum(d, -_hole_sdf(X, Y, Z, h))
    for pt in (ports or []):
        d = np.maximum(d, -_port_sdf(X, R, TH, pt))
    if cut_sector is not None:
        d = np.maximum(d, -_wedge_sdf(Y, Z, cut_sector[0], cut_sector[1]))
    return d


def part_sampler(profile: Profile, **features):
    """
    A callable giving one part's signed distance at arbitrary points.

    Takes the same feature keywords the mesher takes, so a caller that meshed a
    part can ask about the field it meshed by passing the same dict through.
    """
    features = {k: v for k, v in features.items()
                if k in ("channels", "holes", "ports", "bosses", "lugs",
                         "plenums", "ribs", "legs", "cut_sector")}

    def sample(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return sample_field(p[:, 0], p[:, 1], p[:, 2], profile, **features)

    return sample


def _face_areas(verts: np.ndarray, faces: np.ndarray,
                chunk: int = 2_000_000) -> np.ndarray:
    """Triangle areas, a couple of million faces at a time."""
    out = np.empty(len(faces), dtype=np.float64)
    for i in range(0, len(faces), chunk):
        blk = faces[i:i + chunk]
        a, b, c = verts[blk[:, 0]], verts[blk[:, 1]], verts[blk[:, 2]]
        out[i:i + chunk] = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return out


def field_deviation(verts: np.ndarray, faces: np.ndarray, sample,
                    level: float = SURFACE_BIAS_MM,
                    budget: int = 1_200_000,
                    chunk: int = 200_000) -> dict:
    """
    How far a mesh strays from the surface the field says it is.

    Corners *and* centres. Decimation moves the middle of a triangle much
    further than its corners -- a collapse leaves the surviving vertices on the
    surface and the flat triangle between them cutting across whatever curve
    they spanned -- so reading only the corners reports nothing wrong with a
    mesh that has visibly lost the shape. About four vertices in ten do survive
    a collapse unmoved, which is exactly why the corners alone prove nothing.

    Sampled, not exhaustive, and aimed rather than uniform. Evaluating the
    field costs one pass over the profile per point and these profiles carry
    seventeen hundred segments, so asking every vertex and every centroid of a
    24-million-triangle part takes a quarter of an hour. What is asked instead
    is a fixed budget, half of it a systematic stride over the whole mesh --
    drift shows up wherever it is -- and half of it spent on the largest
    triangles, corners and centre each, because chordal error grows with the
    triangle and making some triangles big is the entire job of decimation. A
    defect that hides from both is smaller than the mesh around it.

    Unsigned, in millimetres, measured against the level the mesher actually
    took rather than against zero: the surface is the `SURFACE_BIAS_MM`
    contour, so a vertex exactly on it reads the bias and not nothing.

    One honest limitation. The field is composed with min and max, and the
    maximum of two distance functions is not a distance function -- outside a
    concave seam it reads short. This is therefore a lower bound on how far the
    mesh moved: tight over the smooth ground decimation actually changes,
    optimistic in the corners. It is a measurement, not a guarantee.
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces)
    half = max(1, budget // 2)

    # Half the budget, spread evenly: a stride rather than a random draw, so
    # the same mesh always gets the same answer and two runs are comparable.
    sv = max(1, (2 * len(v)) // half)
    sf = max(1, (2 * len(f)) // half)
    picked_v = [v[::sv]]
    picked_f = [f[::sf]]

    # The other half where the error concentrates. argpartition, not a sort:
    # the order among the big ones does not matter, only that they are the big
    # ones, and sorting 24 million areas to use the last few thousand is a
    # minute of nothing.
    want = max(1, half // 4)
    if len(f) > want:
        areas = _face_areas(v, f)
        big = np.argpartition(areas, len(areas) - want)[-want:]
        picked_f.append(f[big])
    else:
        picked_f.append(f)

    worst, total_sq, n = 0.0, 0.0, 0

    def take(points: np.ndarray) -> None:
        nonlocal worst, total_sq, n
        for i in range(0, len(points), chunk):
            d = np.abs(sample(points[i:i + chunk]) - level)
            if d.size:
                worst = max(worst, float(d.max()))
                total_sq += float(np.dot(d, d))
                n += d.size

    for block in picked_v:
        take(block)
    for block in picked_f:
        if not len(block):
            continue
        a, b, c = v[block[:, 0]], v[block[:, 1]], v[block[:, 2]]
        take((a + b + c) / 3.0)
        take(a)

    return {
        "max_mm": worst,
        "rms_mm": math.sqrt(total_sq / max(n, 1)),
        "points": n,
        "of_points": 2 * len(v) + 4 * len(f),
    }


def build_field(
    profile: Profile,
    voxel_mm: float = 0.2,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    margin_mm: float = 1.0,
    cut_sector: tuple[float, float] | None = None,
    ports: list[PortCut] | None = None,
    bosses: list[RingBoss] | None = None,
    lugs: list[LugAdd] | None = None,
    plenums: list[PlenumCut] | None = None,
    ribs: list[RibAdd] | None = None,
    legs: list[LegAdd] | None = None,
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
    bosses = bosses or []
    lugs = lugs or []
    plenums = plenums or []
    ribs = ribs or []
    legs = legs or []

    x0, x1, r_max = _feature_extent(profile, margin_mm, bosses, lugs, ribs, legs)

    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil(2.0 * r_max / voxel_mm)) + 1)

    # Stepped by the voxel, not fitted to the extent. linspace divides the
    # span into nx - 1 equal parts, and nx came from a ceil, so its step is
    # slightly under the voxel -- while this function returns the voxel, and
    # mesh_field then meshes with it. The result was a solid stretched by up to
    # one part in nx along each axis, which nothing caught because the
    # streaming mesher that writes print files has always stepped by arange and
    # only this path did not. The grid now runs a fraction past the extent
    # instead, which is margin either way.
    xs = (x0 + np.arange(nx) * voxel_mm).astype(np.float32)
    ys = (-r_max + np.arange(ny) * voxel_mm).astype(np.float32)
    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.sqrt(Y * Y + Z * Z).astype(np.float32)
    TH = np.arctan2(Z, Y).astype(np.float32)

    field = np.empty((nx, ny, ny), dtype=np.float32)
    vx = np.asarray(profile.x, dtype=np.float32)
    vr = np.asarray(profile.r, dtype=np.float32)

    for i, xv in enumerate(xs):
        X = np.full(R.shape, xv, dtype=np.float32)
        field[i] = sample_field(
            X, Y, Z, profile, R=R, TH=TH,
            channels=channels, holes=holes, ports=ports, bosses=bosses,
            lugs=lugs, plenums=plenums, ribs=ribs, legs=legs,
            cut_sector=cut_sector)

    return field, (x0, -r_max, -r_max), voxel_mm


def mesh_field(field: np.ndarray, origin, voxel_mm: float):
    """Marching cubes at the zero level set, returned in model coordinates."""
    verts, faces = _mesh_block(field, voxel_mm)
    verts = verts + np.asarray(origin, dtype=float)
    return verts, faces.astype(np.int64)


def _feature_extent(profile, margin_mm, bosses, lugs, ribs=None, legs=None):
    """
    Bounding extent of the part *including* material added outside its profile.

    Sizing the field from the meridional profile alone truncates anything bolted
    on beyond it: the mounting lugs reach 26 mm past the flange, get clipped at
    the field boundary, and the mesh comes back with open edges where they were
    cut. Silent, because a clipped solid still meshes.
    """
    x0 = float(profile.x.min()) - margin_mm
    x1 = float(profile.x.max()) + margin_mm
    r_max = float(profile.r.max()) + margin_mm
    for b in bosses or []:
        x0 = min(x0, b.x_at - b.half_x - margin_mm)
        x1 = max(x1, b.x_at + b.half_x + margin_mm)
        r_max = max(r_max, b.r_outer + margin_mm)
    for lg in lugs or []:
        x0 = min(x0, lg.x_at - lg.half_x - margin_mm)
        x1 = max(x1, lg.x_at + lg.half_x + margin_mm)
        r_max = max(r_max, lg.r_outer + margin_mm)
    for rb in ribs or []:
        x0 = min(x0, rb.x_start - margin_mm)
        x1 = max(x1, rb.x_end + margin_mm)
        if rb.outward:
            r_max = max(r_max, float(np.max(rb.base_r)) + rb.height_mm + margin_mm)
    for lg in legs or []:
        x0 = min(x0, min(lg.x_top, lg.x_foot) - margin_mm)
        x1 = max(x1, max(lg.x_top, lg.x_foot) + margin_mm)
        r_max = max(r_max, max(lg.r_foot, lg.pad_radius_mm) + margin_mm)
    return x0, x1, r_max


def build_mesh(
    profile: Profile,
    voxel_mm: float = 0.2,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    cut_sector: tuple[float, float] | None = None,
    ports: list[PortCut] | None = None,
    bosses: list[RingBoss] | None = None,
    lugs: list[LugAdd] | None = None,
    plenums: list[PlenumCut] | None = None,
    ribs: list[RibAdd] | None = None,
    legs: list[LegAdd] | None = None,
):
    field, origin, spacing = build_field(
        profile, voxel_mm, channels, holes, cut_sector=cut_sector, ports=ports,
        bosses=bosses, lugs=lugs, plenums=plenums, ribs=ribs, legs=legs)
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


def injector_holes(a: EngineAssembly, injector,
                   x_start_fuel: float | None = None,
                   x_start_ox: float | None = None) -> list[HoleCut]:
    """Fuel and oxidiser orifice rings through the head disc."""
    # An orifice runs from its plenum to the chamber face, not through the whole
    # block. Once the head was thickened to hold the manifolds, a hole through
    # all of it became 1.3 mm across and 36 mm long: 25:1, which no process will
    # drill or print straight, and which marching cubes cannot hold together
    # either. The plenums occupy the depth; the orifice is the short run between
    # the plenum's aft face and the chamber.
    # One start per ring, because the two rings meet their plenums at different
    # radii and the plenum section is a diamond: how far aft it reaches depends
    # on where you meet it. A single shared offset was wrong for both.
    default = a.head_x - (a.structure.wall_thickness_mm + 3.0)
    xf = default if x_start_fuel is None else x_start_fuel
    xo = default if x_start_ox is None else x_start_ox
    x1 = a.head_x + 1.0
    half_pitch = math.pi / injector.n_elements
    return [
        HoleCut(injector.fuel_ring_radius * 1e3, injector.d_fuel_mm,
                injector.n_elements, xf, x1, phase=0.0,
                name="fuel_orifice"),
        HoleCut(injector.ox_ring_radius * 1e3, injector.d_ox_mm,
                injector.n_elements, xo, x1, phase=half_pitch,
                name="ox_orifice"),
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

    Axis edges are skipped here for the reason given in `_polygon_sdf`.
    """
    n = len(vx)
    ax, ar = vx, vr
    bx, br = np.roll(vx, -1), np.roll(vr, -1)
    ex, er = bx - ax, br - ar
    ll = ex * ex + er * er
    ll[ll < 1e-30] = 1e-30

    on_axis = (ar <= AXIS_EPS) & (br <= AXIS_EPS)

    best = np.full(r_grid.shape, np.inf, dtype=np.float64)
    inside = np.zeros(r_grid.shape, dtype=bool)

    for i in range(n):
        if on_axis[i]:
            continue
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

    A lexsort over three integer columns rather than `np.unique(..., axis=0)`.
    The latter views the rows as a void dtype and sorts that, several copies
    deep; at print resolution this runs on twelve million vertices with a
    twenty-million-triangle mesh already in memory, and the difference is a
    couple of gigabytes on a machine that has been killed once for wanting
    them.
    """
    if not len(verts):
        return verts, faces
    key = np.round(verts / tol).astype(np.int64)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    sk = key[order]
    starts = np.empty(len(verts), dtype=bool)
    starts[0] = True
    np.any(sk[1:] != sk[:-1], axis=1, out=starts[1:])
    del sk, key
    group = np.cumsum(starts) - 1
    inverse = np.empty(len(verts), dtype=np.int64)
    inverse[order] = group
    out_v = verts[order[np.flatnonzero(starts)]]
    del order, group, starts
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
    bbox: tuple[float, float, float, float] | None = None,
    keep_sector: tuple[float, float] | None = None,
    x_window: tuple[float, float] | None = None,
    bosses: list[RingBoss] | None = None,
    lugs: list[LugAdd] | None = None,
    plenums: list[PlenumCut] | None = None,
    ribs: list[RibAdd] | None = None,
    legs: list[LegAdd] | None = None,
    stats: dict | None = None,
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

    Pass a dict as `stats` to get the volume of the field back as well. Every
    plane is evaluated here already, so integrating the occupancy costs one
    clip and one sum per plane -- and it gives the mesh an independent number
    to be checked against, by a different route through the same field. Working
    that volume out analytically instead means modelling every feature a second
    time and hoping the overlaps cancel.
    """
    channels = channels or []
    holes = holes or []
    ports = ports or []
    bosses = bosses or []
    lugs = lugs or []
    plenums = plenums or []
    ribs = ribs or []
    legs = legs or []

    x0, x1, r_max = _feature_extent(profile, margin_mm, bosses, lugs, ribs, legs)

    # The grid must step by exactly voxel_mm, not by a linspace interval that
    # merely averages to it. marching_cubes is told the spacing separately, so
    # any mismatch shifts each slab slightly against the next and the shared
    # plane's vertices no longer coincide -- the weld then fails and the mesh
    # reports boundary edges along every slab seam.
    # A sector needs only its own corner of the (y, z) plane. On an annular part
    # the full square is mostly the empty middle, so restricting it is worth
    # more than twenty times in both memory and time -- which is what makes
    # verifying a sector at inspection resolution affordable at all.
    y0, y1, z0, z1 = bbox if bbox is not None else (-r_max, r_max, -r_max, r_max)

    # An x window restricts the build to a slice of the part, which is what
    # makes a detail view affordable: a 0.5 mm channel wants voxels far finer
    # than the whole engine can carry, and over a short axial run it can have
    # them. The mesh is left open at the cut, which is fine for looking at and
    # not for measuring.
    if x_window is not None:
        x0 = max(x0, float(x_window[0]))
        x1 = min(x1, float(x_window[1]))
        if x1 <= x0:
            raise ValueError("x_window does not overlap the part")

    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil((y1 - y0) / voxel_mm)) + 1)
    nz = max(2, int(math.ceil((z1 - z0) / voxel_mm)) + 1)

    xs = x0 + np.arange(nx) * voxel_mm
    ys = y0 + np.arange(ny) * voxel_mm
    zs = z0 + np.arange(nz) * voxel_mm
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    R = np.hypot(Y, Z)
    TH = np.arctan2(Z, Y)

    r_grid = np.linspace(0.0, float(R.max()) * 1.001, radial_samples)
    vx = np.asarray(profile.x, dtype=float)
    vr = np.asarray(profile.r, dtype=float)

    occupied = [0.0]                     # cells' worth of metal, summed per plane

    def plane(i: int) -> np.ndarray:
        prof1d = _polygon_sdf_radial(xs[i], r_grid, vx, vr)
        d = np.interp(R, r_grid, prof1d).astype(np.float32)
        X = np.full(R.shape, xs[i], dtype=np.float32)

        # Material is added before anything is taken away, so a hole drilled
        # through a lug or a plenum hollowed inside a boss cuts the metal that
        # was just put there rather than the air where it used to be.
        for b in bosses:
            d = np.minimum(d, _ring_boss_sdf(X, R, b))
        for rb in ribs:
            d = np.minimum(d, _rib_sdf(X, R, TH, rb))
        for lgg in legs:
            d = np.minimum(d, _leg_sdf(X, R, TH, lgg))
        for lg in lugs:
            d = np.minimum(d, _lug_sdf(X, R, TH, lg))
        for pl in plenums:
            d = np.maximum(d, -_plenum_sdf(X, R, pl))
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
            # remove the wedge: this is the cutaway view
            d = np.maximum(d, -_wedge_sdf(Y.astype(np.float32),
                                          Z.astype(np.float32),
                                          cut_sector[0], cut_sector[1]))
        if keep_sector is not None:
            # intersect with the wedge: this is a sample of the part, and it is
            # the opposite operation. Subtracting where you meant to intersect
            # leaves everything except the sector, which still meshes, still
            # looks like a wedge of engine, and has entirely the wrong topology.
            d = np.maximum(d, _wedge_sdf(Y.astype(np.float32),
                                         Z.astype(np.float32),
                                         keep_sector[0], keep_sector[1]))
        if stats is not None:
            # The partial-cell ramp marching cubes itself uses to place a
            # vertex, so the two agree to well under the cell rather than to it.
            occupied[0] += float(np.clip(0.5 - d / voxel_mm, 0.0, 1.0).sum())
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

        if block.min() < SURFACE_BIAS_MM < block.max():
            v, f = _mesh_block(block, voxel_mm)
            v = v + np.array([xs[i0], ys[0], zs[0]])
            all_v.append(v)
            all_f.append(f.astype(np.int64) + offset)
            offset += len(v)
        if progress:
            print(f"  slab {i0:5d}/{nx - 1}  {len(all_f)} pieces", flush=True)
        i0 = i1

    if stats is not None:
        stats["field_volume_mm3"] = occupied[0] * voxel_mm ** 3
        stats["grid"] = (nx, ny, nz)

    if not all_v:
        raise ValueError("the field never crosses zero; nothing to mesh")
    return _weld(np.concatenate(all_v), np.concatenate(all_f),
                 tol=voxel_mm * 1e-5)


# How eagerly the collapser is allowed to take an expensive edge.
#
# The library grows its error threshold as (iteration + 3) ** agg, so a larger
# number reaches the target faster by accepting worse collapses near the end.
# At the default 7 the centrebody pinches: keep 0.40 comes back with two edges
# in four triangles, two duplicated faces and a handle missing, out of six and
# a half million -- and the volume unchanged to three decimals, so nothing was
# lost, the surface was folded. The gate refused it, correctly, and the part
# then shipped every one of its 8.2 million triangles.
#
# At 3 the same rung is clean, and so is 0.15: 2.5 million triangles, same
# genus, same volume. The collapses that broke the surface were never needed;
# they were the collapser hurrying.
COLLAPSE_AGGRESSIVENESS = 3.0


def decimate(verts: np.ndarray, faces: np.ndarray, keep_fraction: float = 0.10,
             agg: float = COLLAPSE_AGGRESSIVENESS):
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

    `agg` is how hard it is allowed to try -- see COLLAPSE_AGGRESSIVENESS. It
    is the difference between a part that decimates six-fold and one that
    refuses to decimate at all, and it costs nothing either way.
    """
    import fast_simplification

    v, f = fast_simplification.simplify(
        verts.astype(np.float32), faces.astype(np.int32),
        1.0 - float(np.clip(keep_fraction, 1e-3, 1.0)), agg=agg)
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
    if not cut.outward:
        # The spike ports draw from the central bore, so they have to sit where
        # the bore still reaches. The cavity closes in a self-supporting cone
        # short of the truncation, and a port aft of that opens into solid metal:
        # a fuel path that simply stops, with nothing in a watertightness or
        # topology check to notice.
        x_at = min(x_at, float(np.asarray(a.cavity_x).max()) - 2.0)
    wall = float(np.interp(x_at, cut.wall_x, cut.wall_r))
    if cut.outward:
        r_lo = wall + cut.hot_wall_mm
        r_hi = float(np.max(a.cowl_outer_r)) + 5.0
    else:
        r_lo = 0.0
        r_hi = wall - cut.hot_wall_mm
    return PortCut(x_at=x_at, diameter_mm=d, count=cut.n_channels,
                   r_lo=r_lo, r_hi=r_hi, phase=0.0)


def stream_mesh_to_stl(
    path: str,
    profile: Profile,
    voxel_mm: float = 0.13,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    ports: list[PortCut] | None = None,
    margin_mm: float = 1.0,
    slab_planes: int = 12,
    radial_samples: int = 6000,
):
    """
    Mesh straight to a binary STL, one slab at a time, without ever holding the
    whole mesh.

    At inspection resolution a part this size runs to tens of millions of
    triangles, and welding them needs the entire mesh plus a sort workspace --
    about twelve gigabytes for the cowl alone, which is how the first attempt
    died. STL is a triangle soup with no shared vertices, so nothing has to be
    welded to write it: each slab can be appended and dropped.

    The cost is that the file cannot be topology-checked as it is written, since
    genus needs shared vertices. `build_mesh_streaming` on a sector does that
    instead -- the features are periodic, so one sector settles the pattern.

    Returns the triangle count.
    """
    channels = channels or []
    holes = holes or []
    ports = ports or []

    x0 = float(profile.x.min()) - margin_mm
    x1 = float(profile.x.max()) + margin_mm
    r_max = float(profile.r.max()) + margin_mm

    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil(2.0 * r_max / voxel_mm)) + 1)
    xs = x0 + np.arange(nx) * voxel_mm
    ys = -r_max + np.arange(ny) * voxel_mm

    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.hypot(Y, Z).astype(np.float32)
    TH = np.arctan2(Z, Y).astype(np.float32)
    Yf, Zf = Y.astype(np.float32), Z.astype(np.float32)

    r_grid = np.linspace(0.0, float(R.max()) * 1.001, radial_samples)
    vx = np.asarray(profile.x, dtype=float)
    vr = np.asarray(profile.r, dtype=float)

    def plane(i: int) -> np.ndarray:
        d = np.interp(R, r_grid, _polygon_sdf_radial(xs[i], r_grid, vx, vr)).astype(np.float32)
        X = np.full(R.shape, xs[i], dtype=np.float32)
        for c in channels:
            d = np.maximum(d, -_channel_sdf(X, R, TH, c))
        for h in holes:
            d = np.maximum(d, -_hole_sdf(X, Yf, Zf, h))
        for pt in ports:
            d = np.maximum(d, -_port_sdf(X, R, TH, pt))
        return d

    rec_dtype = np.dtype([("n", "<f4", 3), ("a", "<f4", 3),
                          ("b", "<f4", 3), ("c", "<f4", 3), ("attr", "<u2")])
    total = 0

    with open(path, "wb") as fh:
        fh.write(b"aerospike inspection mesh".ljust(80, b"\0"))
        fh.write(struct.pack("<I", 0))              # patched at the end

        i0 = 0
        carry = plane(0)
        while i0 < nx - 1:
            i1 = min(i0 + max(1, slab_planes), nx - 1)
            block = np.empty((i1 - i0 + 1,) + R.shape, dtype=np.float32)
            block[0] = carry
            for k in range(i0 + 1, i1 + 1):
                block[k - i0] = plane(k)
            carry = block[-1].copy()

            if block.min() < SURFACE_BIAS_MM < block.max():
                v, f = _mesh_block(block, voxel_mm)
                v = v + np.array([xs[i0], ys[0], ys[0]])
                a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
                nrm = np.cross(b - a, c - a)
                ln = np.linalg.norm(nrm, axis=1)
                ln[ln < 1e-20] = 1.0
                nrm /= ln[:, None]

                rec = np.zeros(len(f), dtype=rec_dtype)
                rec["n"], rec["a"], rec["b"], rec["c"] = nrm, a, b, c
                fh.write(rec.tobytes())
                total += len(f)
                del v, f, a, b, c, nrm, rec
            del block
            i0 = i1

        fh.seek(80)
        fh.write(struct.pack("<I", total))

    return total


def sector_of(profile: Profile, n_features: int, n_take: int,
              margin_mm: float = 8.0):
    """
    An angular window holding exactly `n_take` whole features, and a bounding
    box that just contains it.

    The window edges land halfway between features, so no channel is sliced by
    the cut. That matters: a sliced channel is an open groove rather than a
    tunnel and contributes nothing to the genus, which would make the count the
    check is trying to verify come out wrong for a part that is perfectly fine.

    The margin has to clear the feed ports, which run several millimetres proud
    of the outer skin. Too tight a box truncates them, the mesh is left open
    where they leave, and the genus comes out short by however many ports were
    clipped -- a failure that looks exactly like missing channels.
    """
    pitch = 2.0 * math.pi / n_features
    th0 = 0.5 * pitch
    th1 = th0 + n_take * pitch
    r_max = float(profile.r.max()) + margin_mm

    angles = np.linspace(th0, th1, 200)
    ys = r_max * np.cos(angles)
    zs = r_max * np.sin(angles)
    corners_y = np.concatenate([ys, [0.0]])
    corners_z = np.concatenate([zs, [0.0]])
    pad = margin_mm
    return (th0, th1), (float(corners_y.min()) - pad, float(corners_y.max()) + pad,
                        float(corners_z.min()) - pad, float(corners_z.max()) + pad)


def expected_channel_volume(cut: ChannelCut) -> float:
    """
    Volume a channel ring removes, integrated along the wall it rides.

    The channel follows the contour, so its length is the arc of the wall
    between its end stations rather than the axial span, and its width is the
    clamped width at each radius. Summing over the ring gives a number the mesh
    can be checked against without any topology bookkeeping -- which matters,
    because genus is exquisitely sensitive to whether a feature happens to break
    the surface and volume simply is not.
    """
    x = np.asarray(cut.wall_x, dtype=float)
    r = np.asarray(cut.wall_r, dtype=float)
    inside = (x >= cut.x_start) & (x <= cut.x_end)
    if inside.sum() < 2:
        return 0.0
    xs, rs = x[inside], r[inside]

    ds = np.hypot(np.diff(xs), np.diff(rs))
    r_mid = 0.5 * (rs[:-1] + rs[1:])
    arc = 2.0 * math.pi * r_mid / cut.n_channels
    width = np.minimum(cut.width_mm, np.maximum(arc - cut.land_mm, 0.0))
    return float(cut.n_channels * np.sum(width * cut.height_mm * ds))


def expected_port_volume(cut: ChannelCut, port: PortCut) -> float:
    """
    Volume the feed ports remove, counting only the part inside the wall.

    The port is modelled as a square bore of the port diameter running from the
    channel floor out through the skin; beyond the skin it is cutting air and
    removes nothing.
    """
    wall = float(np.interp(port.x_at, cut.wall_x, cut.wall_r))
    if cut.outward:
        span = max(min(port.r_hi, wall + 6.0) - port.r_lo, 0.0)
    else:
        span = max(port.r_hi - max(port.r_lo, wall - 6.0), 0.0)
    return float(port.count * port.diameter_mm * port.diameter_mm * span)


def field_volume(
    profile: Profile,
    voxel_mm: float = 0.13,
    channels: list[ChannelCut] | None = None,
    holes: list[HoleCut] | None = None,
    ports: list[PortCut] | None = None,
    margin_mm: float = 1.0,
    radial_samples: int = 6000,
) -> float:
    """
    Volume enclosed by the distance field itself, by partial-cell integration.

    This is the reference the mesh should be checked against, and it is better
    than an analytic estimate because it makes no assumptions. Working out the
    channel and port volumes by hand means modelling a port as a square bore of
    some assumed depth and hoping the overlaps cancel; a couple of percent of
    error creeps in and it is not obvious whether it belongs to the estimate or
    to the mesh.

    Each cell contributes the fraction of itself that lies inside, taken as a
    linear ramp across one voxel of the signed distance. That is the same
    approximation marching cubes makes when it places a vertex, so the two agree
    to well under the cell size rather than to the cell size.
    """
    channels = channels or []
    holes = holes or []
    ports = ports or []

    x0 = float(profile.x.min()) - margin_mm
    x1 = float(profile.x.max()) + margin_mm
    r_max = float(profile.r.max()) + margin_mm

    nx = max(2, int(math.ceil((x1 - x0) / voxel_mm)) + 1)
    ny = max(2, int(math.ceil(2.0 * r_max / voxel_mm)) + 1)
    xs = x0 + np.arange(nx) * voxel_mm
    ys = -r_max + np.arange(ny) * voxel_mm

    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    R = np.hypot(Y, Z).astype(np.float32)
    TH = np.arctan2(Z, Y).astype(np.float32)
    Yf, Zf = Y.astype(np.float32), Z.astype(np.float32)

    r_grid = np.linspace(0.0, float(R.max()) * 1.001, radial_samples)
    vx = np.asarray(profile.x, dtype=float)
    vr = np.asarray(profile.r, dtype=float)

    cell = voxel_mm ** 3
    total = 0.0
    for i in range(nx):
        d = np.interp(R, r_grid, _polygon_sdf_radial(xs[i], r_grid, vx, vr)).astype(np.float32)
        X = np.full(R.shape, xs[i], dtype=np.float32)
        for c in channels:
            d = np.maximum(d, -_channel_sdf(X, R, TH, c))
        for h in holes:
            d = np.maximum(d, -_hole_sdf(X, Yf, Zf, h))
        for pt in ports:
            d = np.maximum(d, -_port_sdf(X, R, TH, pt))
        total += float(np.clip(0.5 - d / voxel_mm, 0.0, 1.0).sum())
    return total * cell
