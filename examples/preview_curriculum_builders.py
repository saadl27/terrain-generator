import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.assembled_parts import (
    make_linear_slopes_mesh,
    make_linear_stairs_mesh,
    make_rotating_slopes_mesh,
    make_rotating_stairs_mesh,
)
from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import build_mesh
from terrain_generator.trimesh_tiles.mesh_parts.part_presets import (
    make_corner_cfg,
    make_platform_cfg,
    make_platform_for_stage,
    make_slope_cfg,
    make_stairs_cfg,
)
from terrain_generator.utils import merge_meshes


TERRAIN_ORDER = (
    "stairs",
    "slopes",
    "linear_stairs",
    "linear_slopes",
    "turning_stairs",
    "turning_slopes",
    "corner",
)


def _round_float(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


def _mesh_extents(mesh) -> dict:
    extents = mesh.bounds[1] - mesh.bounds[0]
    return {"x": _round_float(extents[0]), "y": _round_float(extents[1]), "z": _round_float(extents[2])}


def _normalize_ground_center(mesh):
    normalized = mesh.copy()
    if len(normalized.vertices) == 0:
        return normalized
    bounds = normalized.bounds
    normalized.apply_translation([0.0, 0.0, -bounds[0, 2]])
    bounds = normalized.bounds
    normalized.apply_translation(
        [
            -0.5 * (bounds[0, 0] + bounds[1, 0]),
            -0.5 * (bounds[0, 1] + bounds[1, 1]),
            0.0,
        ]
    )
    return normalized


def _stairs_cfg(args):
    return make_stairs_cfg(
        name="preview_stairs",
        direction="front",
        corridor_width=args.corridor_width,
        wall_thickness=args.wall_thickness,
        floor_thickness=args.floor_thickness,
        num_steps=args.num_steps,
        step_height=args.step_height,
        step_depth=args.step_depth,
        wall_height=max(args.wall_height, args.num_steps * args.step_height + args.floor_thickness),
    )


def _slope_cfg(args):
    slope_height = np.tan(np.deg2rad(args.slope_angle_deg)) * args.slope_length
    return make_slope_cfg(
        name="preview_slope",
        corridor_width=args.corridor_width,
        wall_thickness=args.wall_thickness,
        floor_thickness=args.floor_thickness,
        structure_height=max(args.structure_height, slope_height + args.floor_thickness),
        slope_length=args.slope_length,
        slope_angle_deg=args.slope_angle_deg,
        slope_resolution=args.slope_resolution,
        wall_height=max(args.wall_height, args.floor_thickness + slope_height),
    )


def _corner_cfg(args):
    return make_corner_cfg(
        name="preview_corner",
        corridor_width=args.corner_corridor_width,
        pre_corridor_width=args.corner_pre_corridor_width,
        post_corridor_width=args.corner_post_corridor_width,
        wall_thickness=args.wall_thickness,
        wall_height=args.wall_height,
        floor_thickness=args.floor_thickness,
        structure_height=max(args.structure_height, args.wall_height),
        pre_length=args.corner_pre_length,
        post_length=args.corner_post_length,
        turn_angle_deg=args.corner_angle_deg,
    )


def _linear_stairs_platform_cfg(stairs_cfg, args):
    return make_platform_cfg(
        name="preview_linear_stairs_platform",
        width=stairs_cfg.dim[0],
        length=args.flat_length,
        height=stairs_cfg.stairs[0].total_height,
        floor_thickness=args.floor_thickness,
        wall_thickness=args.wall_thickness,
        wall_height=max(args.wall_height, stairs_cfg.stairs[0].total_height),
        wall_edges=("left", "right"),
    )


def _linear_slopes_platform_cfg(slope_cfg, args):
    slope_height = np.tan(np.deg2rad(args.slope_angle_deg)) * args.slope_length
    return make_platform_cfg(
        name="preview_linear_slopes_platform",
        width=slope_cfg.dim[0],
        length=args.flat_length,
        height=slope_height,
        floor_thickness=args.floor_thickness,
        wall_thickness=args.wall_thickness,
        wall_height=max(args.wall_height, slope_height),
        wall_edges=("left", "right"),
    )


def _build_named_meshes(args):
    stairs_cfg = _stairs_cfg(args)
    slope_cfg = _slope_cfg(args)
    corner_cfg = _corner_cfg(args)

    linear_stairs_platform_cfg = _linear_stairs_platform_cfg(stairs_cfg, args)
    linear_slopes_platform_cfg = _linear_slopes_platform_cfg(slope_cfg, args)
    turning_stairs_platform_cfg = make_platform_for_stage(
        stairs_cfg,
        name="preview_turning_stairs_platform",
        turn_direction=args.turn_direction,
        floor_thickness=args.floor_thickness,
        wall_thickness=args.wall_thickness,
        wall_height=max(args.wall_height, stairs_cfg.stairs[0].total_height),
    )
    turning_slopes_platform_cfg = make_platform_for_stage(
        slope_cfg,
        name="preview_turning_slopes_platform",
        turn_direction=args.turn_direction,
        floor_thickness=args.floor_thickness,
        wall_thickness=args.wall_thickness,
        wall_height=max(args.wall_height, np.tan(np.deg2rad(args.slope_angle_deg)) * args.slope_length),
    )

    all_meshes = {
        "stairs": _normalize_ground_center(build_mesh(stairs_cfg)),
        "slopes": _normalize_ground_center(build_mesh(slope_cfg)),
        "linear_stairs": _normalize_ground_center(
            make_linear_stairs_mesh(
                stairs_cfg,
                linear_stairs_platform_cfg,
                num_stages=args.num_segments,
            )
        ),
        "linear_slopes": _normalize_ground_center(
            make_linear_slopes_mesh(
                slope_cfg,
                linear_slopes_platform_cfg,
                num_stages=args.num_segments,
            )
        ),
        "turning_stairs": _normalize_ground_center(
            make_rotating_stairs_mesh(
                stairs_cfg,
                turning_stairs_platform_cfg,
                num_stages=args.num_segments,
                turn_direction=args.turn_direction,
            )
        ),
        "turning_slopes": _normalize_ground_center(
            make_rotating_slopes_mesh(
                slope_cfg,
                turning_slopes_platform_cfg,
                num_stages=args.num_segments,
                turn_direction=args.turn_direction,
            )
        ),
        "corner": _normalize_ground_center(build_mesh(corner_cfg)),
    }
    return [(name, all_meshes[name]) for name in TERRAIN_ORDER if name in args.terrains]


def _build_gallery(named_meshes, gap: float):
    placed_meshes = []
    layout = []
    cursor_x = 0.0

    for name, mesh in named_meshes:
        placed = mesh.copy()
        bounds = placed.bounds
        extents = bounds[1] - bounds[0]
        translation_x = cursor_x - bounds[0, 0]
        placed.apply_translation([translation_x, 0.0, 0.0])
        placed_meshes.append(placed)
        layout.append(
            {
                "terrain_id": name,
                "origin_x": _round_float(translation_x + bounds[0, 0]),
                "width": _round_float(extents[0]),
                "length": _round_float(extents[1]),
                "height": _round_float(extents[2]),
            }
        )
        cursor_x += float(extents[0]) + gap

    return merge_meshes(placed_meshes, False), layout


def _export_named_meshes(named_meshes, output_dir: Path):
    exported = []
    for name, mesh in named_meshes:
        path = output_dir / f"{name}.obj"
        mesh.export(path)
        exported.append(
            {
                "terrain_id": name,
                "file": path.name,
                "mesh_extents": _mesh_extents(mesh),
            }
        )
        print(f"Exported {name}: {path}")
    return exported


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview the same stair/slope/corner builders used by the curriculum generator."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/curriculum_builder_preview"))
    parser.add_argument("--terrains", nargs="+", choices=TERRAIN_ORDER, default=list(TERRAIN_ORDER))
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--step-height", type=float, default=0.18)
    parser.add_argument("--step-depth", type=float, default=0.40)
    parser.add_argument("--corridor-width", type=float, default=4.0)
    parser.add_argument("--flat-length", type=float, default=2.5)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--slope-length", type=float, default=3.5)
    parser.add_argument("--slope-angle-deg", type=float, default=18.0)
    parser.add_argument("--slope-resolution", type=int, default=36)
    parser.add_argument("--corner-angle-deg", type=float, default=35.0)
    parser.add_argument("--corner-corridor-width", type=float, default=4.0)
    parser.add_argument("--corner-pre-corridor-width", type=float, default=4.0)
    parser.add_argument("--corner-post-corridor-width", type=float, default=4.0)
    parser.add_argument("--corner-pre-length", type=float, default=4.0)
    parser.add_argument("--corner-post-length", type=float, default=4.0)
    parser.add_argument("--turn-direction", choices=("left", "right"), default="left")
    parser.add_argument("--wall-thickness", type=float, default=0.18)
    parser.add_argument("--floor-thickness", type=float, default=0.12)
    parser.add_argument("--wall-height", type=float, default=1.2)
    parser.add_argument("--structure-height", type=float, default=4.0)
    parser.add_argument("--gallery-gap", type=float, default=3.0)
    parser.add_argument("--show", action="store_true", help="Open the exported gallery mesh in the viewer.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    named_meshes = _build_named_meshes(args)
    exported_meshes = _export_named_meshes(named_meshes, args.output_dir)

    gallery_mesh, gallery_layout = _build_gallery(named_meshes, args.gallery_gap)
    gallery_path = args.output_dir / "gallery.obj"
    gallery_mesh.export(gallery_path)
    print(f"Exported gallery: {gallery_path}")

    manifest = {
        "script": Path(__file__).name,
        "parameters": vars(args),
        "terrains": exported_meshes,
        "gallery": {
            "file": gallery_path.name,
            "mesh_extents": _mesh_extents(gallery_mesh),
            "layout": gallery_layout,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Exported manifest: {manifest_path}")

    if args.show:
        gallery_mesh.show()


if __name__ == "__main__":
    main()
