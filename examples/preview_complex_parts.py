import sys
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.part_presets import (
    make_platform_for_stage,
    make_slope_cfg,
    make_stairs_cfg,
)
from terrain_generator.trimesh_tiles.mesh_parts.assembled_parts import (
    make_linear_slopes_mesh,
    make_linear_stairs_mesh,
    make_rotating_slopes_mesh,
    make_rotating_stairs_mesh,
    make_slopes_platform_slopes_mesh,
    make_slopes_turn_90_mesh,
    make_stairs_platform_stairs_mesh,
    make_stairs_turn_90_mesh,
)


OUTPUT_DIR = Path("results/complex_parts_preview")
SHOW_MESHES = True
USE_GROUNDED_SIDE_WALLS = True
USE_COMMON_GROUND = True
SIDE_WALL_EXTRA_HEIGHT = 2.0
ADD_FINAL_STAIR_END_WALL = True
ADD_FINAL_LINEAR_SLOPE_END_WALL = True


def _stairs_wall_height(stairs_cfg, num_stages):
    return num_stages * stairs_cfg.stairs[0].total_height + SIDE_WALL_EXTRA_HEIGHT


def _slope_wall_height(slope_cfg, num_stages):
    stage_rise = slope_cfg.slope_length * math.tan(math.radians(slope_cfg.slope_angle_deg))
    return num_stages * stage_rise + SIDE_WALL_EXTRA_HEIGHT


def get_preview_meshes():
    stairs_cfg = make_stairs_cfg(
        name="stairs_stage",
        direction="front",
        corridor_width=10.0,
        wall_thickness=0.12,
        floor_thickness=0.1,
        num_steps=20,
        step_height=0.15,
        step_depth=0.32,
    )
    stairs_linear_platform_cfg = make_platform_for_stage(stairs_cfg, name="stairs_linear_platform")
    stairs_turn_platform_cfg = make_platform_for_stage(
        stairs_cfg,
        name="stairs_turn_platform",
        turn_direction="left",
    )

    slope_cfg = make_slope_cfg(
        name="slope_stage",
        corridor_width=5.0,
        wall_thickness=0.12,
        floor_thickness=0.1,
        structure_height=2.0,
        slope_length=3.0,
        slope_angle_deg=45.0,
    )
    slopes_linear_platform_cfg = make_platform_for_stage(slope_cfg, name="slopes_linear_platform")
    slopes_turn_platform_cfg = make_platform_for_stage(
        slope_cfg,
        name="slopes_turn_platform",
        turn_direction="left",
    )

    return (
        (
            "stairs_platform_stairs_demo",
            make_stairs_platform_stairs_mesh(
                stairs_cfg,
                stairs_linear_platform_cfg,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 2),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            ),
        ),
        (
            "stairs_platform_4_demo",
            make_linear_stairs_mesh(
                stairs_cfg,
                stairs_linear_platform_cfg,
                num_stages=4,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 4),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            ),
        ),
        (
            "stairs_turn_90_demo",
            make_stairs_turn_90_mesh(
                stairs_cfg,
                stairs_turn_platform_cfg,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 2),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            ),
        ),
        (
            "stairs_turn_4_demo",
            make_rotating_stairs_mesh(
                stairs_cfg,
                stairs_turn_platform_cfg,
                num_stages=4,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 4),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            ),
        ),
        (
            "slopes_platform_slopes_demo",
            make_slopes_platform_slopes_mesh(
                slope_cfg,
                slopes_linear_platform_cfg,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_slope_wall_height(slope_cfg, 2),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
            ),
        ),
        (
            "slopes_platform_4_demo",
            make_linear_slopes_mesh(
                slope_cfg,
                slopes_linear_platform_cfg,
                num_stages=4,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_slope_wall_height(slope_cfg, 4),
                common_ground=USE_COMMON_GROUND,
                add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
            ),
        ),
        (
            "slopes_turn_90_demo",
            make_slopes_turn_90_mesh(
                slope_cfg,
                slopes_turn_platform_cfg,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_slope_wall_height(slope_cfg, 2),
                common_ground=USE_COMMON_GROUND,
            ),
        ),
        (
            "slopes_turn_4_demo",
            make_rotating_slopes_mesh(
                slope_cfg,
                slopes_turn_platform_cfg,
                num_stages=4,
                grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
                grounded_wall_height=_slope_wall_height(slope_cfg, 4),
                common_ground=USE_COMMON_GROUND,
            ),
        ),
    )


def export_and_optionally_show(name, mesh):
    output_path = OUTPUT_DIR / f"{name}.obj"
    mesh.export(output_path)
    print(f"Exported {name}: {output_path}")
    if SHOW_MESHES:
        mesh.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, mesh in get_preview_meshes():
        export_and_optionally_show(name, mesh)


if __name__ == "__main__":
    main()
