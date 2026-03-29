import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import get_mesh_gen
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    BoxMeshPartsCfg,
    CombinedMeshPartsCfg,
    CornerMeshPartsCfg,
    HeightMapMeshPartsCfg,
    StairMeshPartsCfg,
    WallPartsCfg,
)


OUTPUT_DIR = Path("results/basic_parts_preview")
SHOW_MESHES = True


def build_mesh(cfg):
    return get_mesh_gen(cfg)(cfg)


def make_stairs_cfg():
    corridor_width = 1.2
    wall_thickness = 0.12
    floor_thickness = 0.1
    structure_height = 2.0

    num_steps = 5
    step_depth = 0.32
    total_height = 0.75
    wall_height = total_height
    extra_length = 0.5

    tile_width = corridor_width + 2.0 * wall_thickness
    tile_length = (num_steps + 1) * step_depth + extra_length
    dim = (tile_width, tile_length, structure_height)

    return StairMeshPartsCfg(
        name="stairs_demo",
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=False,
        load_from_cache=False,
        stairs=(
            StairMeshPartsCfg.Stair(
                dim=dim,
                floor_thickness=floor_thickness,
                minimal_triangles=False,
                step_width=corridor_width,
                step_depth=step_depth,
                n_steps=num_steps,
                total_height=total_height,
                stair_type="standard",
                direction="front",
                attach_side="",
                add_residual_side_up=True,
            ),
        ),
        wall=WallPartsCfg(
            dim=dim,
            floor_thickness=floor_thickness,
            minimal_triangles=False,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_edges=("left", "right"),
            load_from_cache=False,
        ),
    )


def make_slope_cfg():
    corridor_width = 5
    wall_thickness = 0.12
    floor_thickness = 0.1
    structure_height = 2.0

    slope_length = 3.0
    slope_angle_deg = 45.0
    slope_height = np.tan(np.deg2rad(slope_angle_deg)) * slope_length
    wall_height = floor_thickness + slope_height
    # Number of samples per side in the square height map used to build the ramp surface.
    slope_resolution = 24

    tile_side = max(corridor_width + 2.0 * wall_thickness, slope_length)
    dim = (tile_side, tile_side, structure_height)

    ramp_profile = np.linspace(floor_thickness, floor_thickness + slope_height, slope_resolution)
    height_map = np.tile(ramp_profile, (slope_resolution, 1))

    slope_cfg = HeightMapMeshPartsCfg(
        name="slope_surface_demo",
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=False,
        height_map=height_map,
        fill_borders=True,
        slope_threshold=4.0,
        simplify=False,
        load_from_cache=False,
    )

    wall_x = tile_side / 2.0 - wall_thickness / 2.0
    wall_z = wall_height / 2.0
    left_wall_t = np.eye(4)
    left_wall_t[:3, 3] = [-wall_x, 0.0, wall_z]
    right_wall_t = np.eye(4)
    right_wall_t[:3, 3] = [wall_x, 0.0, wall_z]

    wall_cfg = BoxMeshPartsCfg(
        name="slope_walls_demo",
        dim=dim,
        minimal_triangles=False,
        add_floor=False,
        box_dims=(
            (wall_thickness, tile_side, wall_height),
            (wall_thickness, tile_side, wall_height),
        ),
        transformations=(left_wall_t, right_wall_t),
        load_from_cache=False,
    )
    return CombinedMeshPartsCfg(
        name="slope_demo",
        dim=dim,
        minimal_triangles=False,
        load_from_cache=False,
        cfgs=(slope_cfg, wall_cfg),
    )


def make_corner_cfg():
    corridor_width = 2
    pre_corridor_width = 2
    post_corridor_width = 1
    wall_thickness = 0.12
    wall_height = 1.0
    floor_thickness = 0.1
    structure_height = 2.0

    pre_length = 2.0
    post_length = 2.0
    turn_angle_deg = 90.0

    return CornerMeshPartsCfg(
        name="corner_demo",
        dim=(corridor_width + 2.0 * wall_thickness, pre_length + post_length, structure_height),
        floor_thickness=floor_thickness,
        minimal_triangles=False,
        corridor_width=corridor_width,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=turn_angle_deg,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        load_from_cache=False,
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
