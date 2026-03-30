import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import get_mesh_gen
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    BoxMeshPartsCfg,
    CornerMeshPartsCfg,
    HeightMapMeshPartsCfg,
    StairMeshPartsCfg,
    WallPartsCfg,
)
from terrain_generator.utils import yaw_rotate_mesh


OUTPUT_DIR = Path("results/basic_parts_preview")
SHOW_MESHES = True


def build_mesh(cfg):
    if isinstance(cfg, trimesh.Trimesh):
        return cfg
    return get_mesh_gen(cfg)(cfg)


def make_stairs_cfg(
    name="stairs_demo",
    direction="front",
    corridor_width=1.2,
    wall_thickness=0.12,
    floor_thickness=0.1,
    num_steps=5,
    step_height=0.15,
    step_depth=0.32,
    wall_height=2,
    extra_length=0.5,
    minimal_triangles=False,
    load_from_cache=False,
    stair_type="standard",
    attach_side="",
    add_residual_side_up=True,
    height_offset=0.0,
    wall_edges=("left", "right"),
):
    total_height = num_steps * step_height
    structure_height = total_height + height_offset
    if wall_height is None:
        wall_height = total_height + height_offset

    tile_width = corridor_width + 2.0 * wall_thickness
    tile_length = num_steps * step_depth + extra_length
    wall_length = num_steps * step_depth
    dim = (tile_width, tile_length, structure_height)
    dim_wall = (tile_width, wall_length, structure_height)
    stairs_cfg = StairMeshPartsCfg(
        name=name,
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        load_from_cache=load_from_cache,
        stairs=(
            StairMeshPartsCfg.Stair(
                dim=dim,
                floor_thickness=floor_thickness,
                minimal_triangles=minimal_triangles,
                step_width=corridor_width,
                step_depth=step_depth,
                n_steps=num_steps,
                total_height=total_height,
                height_offset=height_offset,
                stair_type=stair_type,
                direction=direction,
                attach_side=attach_side,
                add_residual_side_up=add_residual_side_up,
            ),
        ),
        wall=WallPartsCfg(
            dim=dim_wall,
            floor_thickness=floor_thickness,
            minimal_triangles=minimal_triangles,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_edges=wall_edges,
            height_offset=height_offset,
            load_from_cache=load_from_cache,
        ),
    )
    mesh = build_mesh(stairs_cfg)

    residual_depth = extra_length
    if residual_depth > 0.0:
        base_z = -structure_height / 2.0
        landing_top_z = base_z + total_height + height_offset
        shoulder_bottom_z = landing_top_z - floor_thickness
        wall_center_x = corridor_width / 2.0 + wall_thickness / 2.0
        run_length = num_steps * step_depth
        landing_start_y = -tile_length / 2.0 + (run_length if add_residual_side_up else 0.0)
        landing_center_y = landing_start_y + residual_depth / 2.0

        shoulder_dims = []
        shoulder_transforms = []

        if "left" in wall_edges:
            left_t = np.eye(4)
            left_t[:3, 3] = [-wall_center_x, landing_center_y, shoulder_bottom_z + floor_thickness / 2.0]
            shoulder_dims.append((wall_thickness, residual_depth, floor_thickness))
            shoulder_transforms.append(left_t)
        if "right" in wall_edges:
            right_t = np.eye(4)
            right_t[:3, 3] = [wall_center_x, landing_center_y, shoulder_bottom_z + floor_thickness / 2.0]
            shoulder_dims.append((wall_thickness, residual_depth, floor_thickness))
            shoulder_transforms.append(right_t)

        if shoulder_dims:
            shoulder_cfg = BoxMeshPartsCfg(
                name=f"{name}_landing_shoulders",
                dim=dim,
                minimal_triangles=minimal_triangles,
                add_floor=False,
                box_dims=tuple(shoulder_dims),
                transformations=tuple(shoulder_transforms),
                load_from_cache=load_from_cache,
            )
            shoulder_mesh = build_mesh(shoulder_cfg)

            yaw_by_direction = {"front": 0, "left": 90, "back": 180, "right": 270}
            yaw_deg = yaw_by_direction[direction]
            if yaw_deg != 0:
                shoulder_mesh = yaw_rotate_mesh(shoulder_mesh, yaw_deg)

            stair_dim = np.array([corridor_width, tile_length, total_height])
            if direction in ("left", "right"):
                stair_dim = stair_dim[np.array([1, 0, 2])]

            translation = np.zeros(3)
            if "left" in attach_side:
                translation[0] -= dim[0] / 2.0 - stair_dim[0] / 2.0
            if "right" in attach_side:
                translation[0] += dim[0] / 2.0 - stair_dim[0] / 2.0
            if "front" in attach_side:
                translation[1] += dim[1] / 2.0 - stair_dim[1] / 2.0
            if "back" in attach_side:
                translation[1] -= dim[1] / 2.0 - stair_dim[1] / 2.0
            shoulder_mesh.apply_translation(translation)
            mesh = trimesh.util.concatenate([mesh, shoulder_mesh])

    return mesh


def make_slope_cfg(
    name="slope_demo",
    corridor_width=5.0,
    wall_thickness=0.12,
    floor_thickness=0.1,
    structure_height=2.0,
    slope_length=3.0,
    slope_angle_deg=45.0,
    wall_height=None,
    extra_length=0.5,
    slope_resolution=24,
    minimal_triangles=False,
    load_from_cache=False,
    fill_borders=True,
    slope_threshold=4.0,
    simplify=False,
):
    slope_height = np.tan(np.deg2rad(slope_angle_deg)) * slope_length
    if wall_height is None:
        wall_height = floor_thickness + slope_height

    total_length = slope_length + extra_length
    outer_width = corridor_width + 2.0 * wall_thickness
    structure_height = max(structure_height, wall_height)
    square_side = total_length
    dim = (square_side, square_side, structure_height)

    sample_positions = np.linspace(0.0, total_length, slope_resolution)
    ramp_profile = floor_thickness + np.minimum(sample_positions, slope_length) * np.tan(np.deg2rad(slope_angle_deg))
    height_map = np.tile(ramp_profile, (slope_resolution, 1))

    slope_cfg = HeightMapMeshPartsCfg(
        name=f"{name}_surface",
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        height_map=height_map,
        fill_borders=fill_borders,
        slope_threshold=slope_threshold,
        simplify=simplify,
        load_from_cache=load_from_cache,
    )

    slope_mesh = get_mesh_gen(slope_cfg)(slope_cfg)
    width_scale = outer_width / total_length
    scale_t = np.eye(4)
    scale_t[0, 0] = width_scale
    slope_mesh.apply_transform(scale_t)

    wall_x = outer_width / 2.0 - wall_thickness / 2.0
    wall_z = wall_height / 2.0
    left_wall_t = np.eye(4)
    left_wall_t[:3, 3] = [-wall_x, 0.0, wall_z]
    right_wall_t = np.eye(4)
    right_wall_t[:3, 3] = [wall_x, 0.0, wall_z]

    wall_cfg = BoxMeshPartsCfg(
        name=f"{name}_walls",
        dim=(outer_width, total_length, structure_height),
        minimal_triangles=minimal_triangles,
        add_floor=False,
        box_dims=(
            (wall_thickness, total_length, wall_height),
            (wall_thickness, total_length, wall_height),
        ),
        transformations=(left_wall_t, right_wall_t),
        load_from_cache=load_from_cache,
    )
    wall_mesh = get_mesh_gen(wall_cfg)(wall_cfg)
    return trimesh.util.concatenate([slope_mesh, wall_mesh])


def make_corner_cfg(
    name="corner_demo",
    corridor_width=2.0,
    pre_corridor_width=2.0,
    post_corridor_width=1.0,
    wall_thickness=0.12,
    wall_height=1.0,
    floor_thickness=0.1,
    structure_height=2.0,
    pre_length=2.0,
    post_length=2.0,
    turn_angle_deg=90.0,
    minimal_triangles=False,
    load_from_cache=False,
):
    base_width = max(corridor_width, pre_corridor_width, post_corridor_width)

    return CornerMeshPartsCfg(
        name=name,
        dim=(base_width + 2.0 * wall_thickness, pre_length + post_length, structure_height),
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        corridor_width=corridor_width,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=turn_angle_deg,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        load_from_cache=load_from_cache,
    )


def export_and_optionally_show(name: str, cfg):
    mesh = build_mesh(cfg)
    output_path = OUTPUT_DIR / f"{name}.obj"
    mesh.export(output_path)
    print(f"Exported {name}: {output_path}")
    if SHOW_MESHES:
        mesh.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_and_optionally_show("stairs_demo", make_stairs_cfg())
    export_and_optionally_show("slope_demo", make_slope_cfg())
    export_and_optionally_show("corner_demo", make_corner_cfg())


if __name__ == "__main__":
    main()
