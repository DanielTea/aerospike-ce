"""
The printability gate itself.

`print_ready.py` refuses to write a file that a slicer would accept and a
machine could not build. That refusal is only worth having if the checks behind
it are right, so they are exercised here on shapes whose answer is known by
construction rather than on the engine, where nobody can independently say how
many cavities there ought to be.

The distinction being tested is the one that took two wrong attempts to get
right: a part with internal cavities is not broken, and counting connected
components rejects every hollow part ever printed. What is broken is a cavity
with no way out, because it stays full of powder for ever.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import marching_cubes

from print_ready import sealed_voids, surfaces


# Fields here are signed distances in the convention the rest of the model
# uses: negative inside the material. Written the other way up every surface
# comes out inside-out, and a solid ball reports itself as a cavity.
def _step(n: int = 48, half: float = 12.0) -> float:
    return 2.0 * half / (n - 1)


def _mesh(field: np.ndarray, n: int = 48, half: float = 12.0):
    # Spacing matters: left at 1.0 the mesh comes out in grid indices rather
    # than millimetres, every surface count still passes, and only the volume
    # gives it away -- by the cube of the step.
    h = _step(n, half)
    v, f, _, _ = marching_cubes(field, level=0.0, spacing=(h, h, h))
    return v, f


def _grid(n: int = 48, half: float = 12.0):
    g = np.linspace(-half, half, n)
    return np.meshgrid(g, g, g, indexing="ij")


def _radius(n: int = 48, half: float = 12.0):
    X, Y, Z = _grid(n, half)
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2)


def test_a_solid_ball_is_one_surface_with_no_cavity():
    v, f = _mesh(_radius() - 8.0)
    assert len(surfaces(v, f)) == 1
    assert sealed_voids(v, f) == []


def test_a_hollow_shell_reads_as_one_sealed_cavity():
    """
    Two closed surfaces, and the inner one carries negative volume. This is the
    shape the head disc turned out to be, and the reason `components == 1` was
    the wrong test: nothing is wrong with a shell except that it is sealed.
    """
    r = _radius()
    v, f = _mesh(np.maximum(r - 8.0, 4.0 - r))
    surf = surfaces(v, f)
    assert len(surf) == 2
    volumes = sorted(vol for _, vol in surf)
    assert volumes[0] < 0.0 < volumes[1]
    assert len(sealed_voids(v, f)) == 1


def test_drilling_the_shell_open_clears_it():
    """
    The same shell with a bore to the outside. Still hollow, still two-walled,
    but the powder has somewhere to go -- so it passes.
    """
    X, Y, Z = _grid()
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    shell = np.maximum(r - 8.0, 4.0 - r)
    bore = np.sqrt(Y ** 2 + Z ** 2) - 1.5          # a rod along x, cut away
    v, f = _mesh(np.maximum(shell, -bore))
    assert sealed_voids(v, f) == []


def test_two_separate_balls_are_two_surfaces_and_neither_is_a_cavity():
    """
    Disconnected solids also give more than one surface, and must not be
    confused with cavities: both volumes are positive.
    """
    X, Y, Z = _grid()
    a = np.sqrt((X - 6.0) ** 2 + Y ** 2 + Z ** 2) - 4.0
    b = np.sqrt((X + 6.0) ** 2 + Y ** 2 + Z ** 2) - 4.0
    v, f = _mesh(np.minimum(a, b))
    surf = surfaces(v, f)
    assert len(surf) == 2
    assert all(vol > 0.0 for _, vol in surf)
    assert sealed_voids(v, f) == []


def test_the_cavity_volume_is_reported_at_its_true_size():
    """
    The refusal message quotes how much powder is trapped, so the number has to
    mean something. A 4 mm inner radius is 268 mm3.
    """
    r = _radius(n=96, half=12.0)
    v, f = _mesh(np.maximum(r - 8.0, 4.0 - r), n=96)
    trapped = abs(sealed_voids(v, f)[0])
    assert trapped == pytest.approx(4.0 / 3.0 * np.pi * 4.0 ** 3, rel=0.03)


def test_decimation_walks_from_mild_to_aggressive(monkeypatch):
    """
    The ladder is tried gentlest first and stops at the target.

    Written the other way round -- skip everything gentler than the target,
    open with the target itself -- a part that cannot survive an eight-fold
    collapse keeps every one of its triangles, because the one ratio that was
    tried failed and there was nothing milder left to fall back to. The
    centrebody did exactly that: 16.4 million triangles in, 16.4 million out.
    """
    import print_ready

    r = _radius()
    v, f = _mesh(r - 8.0)

    tried = []

    def spy(verts, faces, ratio):
        tried.append(ratio)
        return verts, faces          # refuse to actually reduce anything

    monkeypatch.setattr(print_ready, "decimate", spy)
    print_ready.reduce_safely(v, f, target_ratio=0.12)

    assert tried == sorted(tried, reverse=True), "ladder is not mildest-first"
    assert tried[0] == 0.85
    assert min(tried) == 0.12, "ladder went past the requested target"


def test_decimation_keeps_the_last_mesh_that_passed(monkeypatch):
    """
    When a step breaks the topology the previous one is returned, not the
    broken one and not the undecimated original.
    """
    import print_ready

    from mesh_solid import decimate as real_decimate

    r = _radius()
    v, f = _mesh(r - 8.0)

    def spy(verts, faces, ratio):
        if ratio >= 0.3:
            return real_decimate(verts, faces, ratio)
        # Below 0.3, hand back something that is no longer closed. A real
        # quadric collapse fails this way too: it shuts a cooling channel and
        # the result is still a perfectly plausible-looking mesh.
        return verts, faces[: len(faces) // 2]

    monkeypatch.setattr(print_ready, "decimate", spy)
    _, out, rep = print_ready.reduce_safely(v, f, target_ratio=0.12)
    assert len(out) < len(f), "kept the original when 0.5 and 0.3 both worked"
    assert rep["watertight"], "returned the mesh that failed the check"
    assert len(out) > len(f) // 4, "accepted a step past the one that broke"


def test_a_loose_fragment_is_not_mistaken_for_a_part():
    """
    Two solids, both closed, neither hollow. Watertightness passes and the
    sealed-void test passes, because the fragment is metal rather than a
    cavity -- so this needed its own check, and did not have one. A 27 cm3
    ring of copper attached to nothing shipped inside the cowl.
    """
    from print_ready import loose_pieces

    X, Y, Z = _grid()
    a = np.sqrt((X - 6.0) ** 2 + Y ** 2 + Z ** 2) - 5.0
    b = np.sqrt((X + 7.0) ** 2 + Y ** 2 + Z ** 2) - 2.5
    v, f = _mesh(np.minimum(a, b))

    loose = loose_pieces(v, f)
    assert len(loose) == 1
    assert loose[0] == pytest.approx(4.0 / 3.0 * np.pi * 2.5 ** 3, rel=0.06)
    assert sealed_voids(v, f) == [], "a second solid is not a cavity"


def test_one_solid_leaves_nothing_loose():
    v, f = _mesh(_radius() - 8.0)
    from print_ready import loose_pieces
    assert loose_pieces(v, f) == []


def test_decimation_refuses_a_step_that_flattens_triangles(monkeypatch):
    """
    Quadric collapse can slide three vertices into a line without breaking any
    edge pairing. The mesh stays watertight, keeps its genus, keeps its volume,
    and acquires faces with no normal -- which is the one thing a slicer reads
    and none of the other checks do.
    """
    import print_ready

    r = _radius()
    v, f = _mesh(r - 8.0)

    def spy(verts, faces, ratio):
        out = verts.copy()
        # collapse one triangle onto a line, leaving the topology untouched
        a, b, c = faces[0]
        out[c] = 0.5 * (out[a] + out[b])
        return out, faces

    monkeypatch.setattr(print_ready, "decimate", spy)
    _, out, rep = print_ready.reduce_safely(v, f, target_ratio=0.5)
    assert rep["degenerate_faces"] == 0, "kept a mesh with a flattened face"
    assert len(out) == len(f), "should have fallen back to the base mesh"
