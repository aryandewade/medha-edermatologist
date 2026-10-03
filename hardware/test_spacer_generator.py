"""
Tests for hardware/spacer_generator.py

Verifies the generated mesh is a closed (watertight) manifold, matches the
specified physical dimensions, and that both STL and OpenSCAD artifacts are
written correctly.
"""

import os
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from spacer_generator import (  # noqa: E402
    build_spacer,
    bounding_box,
    write_ascii_stl,
    generate,
)


def _edge_counts(faces):
    counts = defaultdict(int)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((u, v)))] += 1
    return counts


def test_mesh_is_watertight_manifold():
    vertices, faces = build_spacer(segments=64)
    counts = _edge_counts(faces)

    assert len(counts) > 0, "Mesh has no edges"
    bad = {e: c for e, c in counts.items() if c != 2}
    assert not bad, f"Non-manifold edges (should each appear twice): {list(bad.items())[:5]}"
    print(f"[PASS] watertight manifold: {len(faces)} faces, {len(counts)} edges (all shared by 2)")


def test_dimensions_match_spec():
    vertices, faces = build_spacer(
        height=35.0, top_aperture=14.0, base_clear=38.0, base_outer=48.0, wall_top=2.0
    )
    (x0, y0, z0), (x1, y1, z1) = bounding_box(vertices)

    assert abs(x1 - 24.0) < 1e-6 and abs(x0 + 24.0) < 1e-6, "Base outer diameter must be 48 mm"
    assert abs(z0) < 1e-6 and abs(z1 - 35.0) < 1e-6, "Focal height must be 35 mm"

    # Top aperture: min radius among top-ring vertices == 7 mm (14 mm dia)
    top_radii = [ (v[0] ** 2 + v[1] ** 2) ** 0.5 for v in vertices if abs(v[2] - 35.0) < 1e-6 ]
    assert abs(min(top_radii) - 7.0) < 1e-6, "Top aperture must be 14 mm diameter"
    print(f"[PASS] dimensions: base {x1-x0:.1f}mm, height {z1-z0:.1f}mm, aperture {min(top_radii)*2:.1f}mm")


def test_stl_roundtrip():
    vertices, faces = build_spacer(segments=48)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "spacer_35mm.stl")
        written = write_ascii_stl(path, vertices, faces, solid_name="optical_spacer")
        assert written == len(faces)

        with open(path, "r", encoding="ascii") as f:
            content = f.read()
        assert content.startswith("solid optical_spacer")
        assert content.rstrip().endswith("endsolid optical_spacer")
        assert content.count("facet normal") == len(faces)
        assert content.count("vertex") == len(faces) * 3
    print(f"[PASS] STL roundtrip: {written} facets")


def test_generate_writes_scad():
    with tempfile.TemporaryDirectory() as tmp:
        summary = generate(out_dir=os.path.join(tmp, "stl"), segments=32, write_scad=True)
        assert os.path.exists(summary["stl"])
        assert os.path.exists(summary["scad"])
        with open(summary["scad"], "r", encoding="utf-8") as f:
            scad = f.read()
        assert "module spacer()" in scad and "difference()" in scad and "$fn" in scad
    print("[PASS] generate() writes STL + OpenSCAD")


if __name__ == "__main__":
    test_mesh_is_watertight_manifold()
    test_dimensions_match_spec()
    test_stl_roundtrip()
    test_generate_writes_scad()
    print("\nAll spacer_generator tests passed.")