"""
Parametric 3D-Printed Optical Spacer Generator (Mobile Spacer Edition)

Emits watertight ASCII STL meshes for the smartphone optical spacer described in
hardware-spacer.md, plus a parametric OpenSCAD source file. No external CAD
dependency is required - meshes are constructed directly in pure Python.

Geometry (all millimetres):
  * Hollow conical frustum  : focal height 35 mm, top camera aperture 14 mm,
                              base clear aperture 38 mm, 2 mm wall.
  * Contact flange ring     : flat annulus at the skin-contact face,
                              outer diameter 48 mm, 3 mm tall.

Usage:
    python hardware/spacer_generator.py --out-dir hardware/stl
"""

import os
import math
import argparse
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Triangle = Tuple[Vec3, Vec3, Vec3]


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------
def _ring_points(radius: float, z: float, segments: int) -> List[Vec3]:
    pts = []
    for i in range(segments):
        ang = 2.0 * math.pi * i / segments
        pts.append((radius * math.cos(ang), radius * math.sin(ang), z))
    return pts


def annular_frustum(
    outer_bottom_r: float,
    outer_top_r: float,
    inner_bottom_r: float,
    inner_top_r: float,
    z_bottom: float,
    z_top: float,
    segments: int = 128,
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """
    Build a closed hollow (annular) frustum shell.

    Returns (vertices, faces) where each face is a triangle of vertex indices.
    The mesh is manifold: outer wall + inner wall + top annulus + bottom annulus.
    """
    s = segments
    ot = _ring_points(outer_top_r, z_top, s)      # outer top
    ob = _ring_points(outer_bottom_r, z_bottom, s)  # outer bottom
    it = _ring_points(inner_top_r, z_top, s)      # inner top
    ib = _ring_points(inner_bottom_r, z_bottom, s)  # inner bottom

    vertices: List[Vec3] = ot + ob + it + ib
    idx_ot = 0
    idx_ob = s
    idx_it = 2 * s
    idx_ib = 3 * s

    faces: List[Tuple[int, int, int]] = []
    for i in range(s):
        j = (i + 1) % s

        ot_i, ot_j = idx_ot + i, idx_ot + j
        ob_i, ob_j = idx_ob + i, idx_ob + j
        it_i, it_j = idx_it + i, idx_it + j
        ib_i, ib_j = idx_ib + i, idx_ib + j

        # Outer wall (outward-facing)
        faces.append((ot_i, ob_i, ob_j))
        faces.append((ot_i, ob_j, ot_j))
        # Inner wall (inward-facing, reversed winding)
        faces.append((it_i, it_j, ib_j))
        faces.append((it_i, ib_j, ib_i))
        # Top annulus (camera aperture face)
        faces.append((ot_i, it_i, it_j))
        faces.append((ot_i, it_j, ot_j))
        # Bottom annulus (skin-contact face)
        faces.append((ob_i, ob_j, ib_j))
        faces.append((ob_i, ib_j, ib_i))

    return vertices, faces


def _triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def merge_meshes(meshes: List[Tuple[List[Vec3], List[Tuple[int, int, int]]]]):
    """Combine several (vertices, faces) meshes into one with re-indexed faces."""
    all_vertices: List[Vec3] = []
    all_faces: List[Tuple[int, int, int]] = []
    offset = 0
    for vertices, faces in meshes:
        all_vertices.extend(vertices)
        all_faces.extend((a + offset, b + offset, c + offset) for a, b, c in faces)
        offset += len(vertices)
    return all_vertices, all_faces


def write_ascii_stl(path: str, vertices: List[Vec3], faces: List[Tuple[int, int, int]],
                    solid_name: str = "spacer") -> int:
    """Write an ASCII STL file. Returns the number of facets written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"solid {solid_name}\n")
        for a, b, c in faces:
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            nx, ny, nz = _triangle_normal(va, vb, vc)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for v in (va, vb, vc):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")
    return len(faces)


# ---------------------------------------------------------------------------
# Spacer construction
# ---------------------------------------------------------------------------
def build_spacer(
    height: float = 35.0,
    top_aperture: float = 14.0,
    base_clear: float = 38.0,
    base_outer: float = 48.0,
    wall_top: float = 2.0,
    segments: int = 128,
) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """
    Build the spacer as a single watertight hollow frustum.

    The wall is thin (2 mm) at the camera aperture and widens toward the skin,
    forming the integrated contact flange. All dimensions in millimetres.
    """
    top_inner_r = top_aperture / 2.0
    top_outer_r = top_inner_r + wall_top
    base_inner_r = base_clear / 2.0
    base_outer_r = base_outer / 2.0
    return annular_frustum(
        outer_bottom_r=base_outer_r,
        outer_top_r=top_outer_r,
        inner_bottom_r=base_inner_r,
        inner_top_r=top_inner_r,
        z_bottom=0.0,
        z_top=height,
        segments=segments,
    )


def bounding_box(vertices: List[Vec3]) -> Tuple[Vec3, Vec3]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def write_openscad(path: str, params: dict) -> None:
    """Emit a parametric OpenSCAD source equivalent to the generated STL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    src = f"""// Parametric Optical Spacer - E-Dermatologist (Mobile Spacer Edition)
// Auto-generated by hardware/spacer_generator.py
// Units: millimetres. Render with F6 then export STL.

HEIGHT        = {params['height']};
TOP_APERTURE  = {params['top_aperture']};
BASE_CLEAR    = {params['base_clear']};
BASE_OUTER    = {params['base_outer']};
WALL_TOP      = {params['wall_top']};
SEG           = {params['segments']};

module spacer() {{
    difference() {{
        // Outer body: widening base flange tapering to the camera aperture.
        cylinder(h = HEIGHT,
                 d1 = BASE_OUTER,
                 d2 = TOP_APERTURE + 2 * WALL_TOP,
                 $fn = SEG);
        // Inner cavity: clear optical path.
        translate([0, 0, -0.01])
            cylinder(h = HEIGHT + 0.02,
                     d1 = BASE_CLEAR,
                     d2 = TOP_APERTURE,
                     $fn = SEG);
    }}
}}

spacer();
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[SpacerGen] Wrote OpenSCAD source to {path}")


def generate(
    out_dir: str = os.path.join("hardware", "stl"),
    height: float = 35.0,
    top_aperture: float = 14.0,
    base_clear: float = 38.0,
    base_outer: float = 48.0,
    wall_top: float = 2.0,
    segments: int = 128,
    write_scad: bool = True
) -> dict:
    """Generate the STL(s) and optional OpenSCAD source. Returns a summary dict."""
    params = dict(height=height, top_aperture=top_aperture, base_clear=base_clear,
                  base_outer=base_outer, wall_top=wall_top, segments=segments)

    vertices, faces = build_spacer(height, top_aperture, base_clear, base_outer, wall_top, segments)

    stl_path = os.path.join(out_dir, f"spacer_{int(round(height))}mm.stl")
    n_facets = write_ascii_stl(stl_path, vertices, faces, solid_name="optical_spacer")
    print(f"[SpacerGen] Wrote {n_facets} facets ({len(vertices)} vertices) to {stl_path}")

    (x0, y0, z0), (x1, y1, z1) = bounding_box(vertices)
    print(f"[SpacerGen] Bounding box: "
          f"X[{x0:.1f},{x1:.1f}] Y[{y0:.1f},{y1:.1f}] Z[{z0:.1f},{z1:.1f}] mm")

    summary = {
        "stl": stl_path,
        "facets": n_facets,
        "vertices": len(vertices),
        "bbox_min": [x0, y0, z0],
        "bbox_max": [x1, y1, z1],
        "params": params,
    }

    if write_scad:
        scad_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "spacer.scad")
        write_openscad(scad_path, params)
        summary["scad"] = scad_path

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate parametric optical spacer STL")
    parser.add_argument("--out-dir", type=str, default=os.path.join("hardware", "stl"))
    parser.add_argument("--height", type=float, default=35.0, help="Focal height (mm)")
    parser.add_argument("--top-aperture", type=float, default=14.0, help="Camera aperture dia (mm)")
    parser.add_argument("--base-clear", type=float, default=38.0, help="Inner clear aperture dia (mm)")
    parser.add_argument("--base-outer", type=float, default=48.0, help="Contact face outer dia (mm)")
    parser.add_argument("--wall-top", type=float, default=2.0, help="Wall thickness at aperture (mm)")
    parser.add_argument("--segments", type=int, default=128, help="Radial tessellation")
    parser.add_argument("--no-scad", action="store_true", help="Skip OpenSCAD source generation")
    args = parser.parse_args()

    generate(
        out_dir=args.out_dir,
        height=args.height,
        top_aperture=args.top_aperture,
        base_clear=args.base_clear,
        base_outer=args.base_outer,
        wall_top=args.wall_top,
        segments=args.segments,
        write_scad=not args.no_scad,
    )


if __name__ == "__main__":
    main()