import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.assembled_parts import (
    make_linear_slopes_mesh,
    make_linear_stairs_mesh,
    make_repeating_u_turn_stairs_mesh,
    make_rotating_slopes_mesh,
    make_rotating_stairs_mesh,
    make_slopes_platform_slopes_mesh,
    make_slopes_turn_90_mesh,
    make_slopes_u_turn_mesh,
    make_stairs_platform_stairs_mesh,
    make_stairs_turn_90_mesh,
    make_stairs_u_turn_mesh,
)
from terrain_generator.trimesh_tiles.mesh_parts.part_presets import (
    make_platform_cfg,
    make_platform_for_stage,
    make_slope_cfg,
    make_stairs_cfg,
)


def _stairs_wall_height(stairs_cfg, num_stages, side_wall_extra_height):
    return num_stages * stairs_cfg.stairs[0].total_height + side_wall_extra_height


def _slope_wall_height(slope_cfg, num_stages, side_wall_extra_height):
    stage_rise = slope_cfg.slope_length * math.tan(math.radians(slope_cfg.slope_angle_deg))
    return num_stages * stage_rise + side_wall_extra_height


def _stage_rise(stage_cfg):
    if hasattr(stage_cfg, "stairs"):
        return stage_cfg.stairs[0].total_height
    return stage_cfg.slope_length * math.tan(math.radians(stage_cfg.slope_angle_deg))


def _make_u_turn_platform(stage_cfg, name, return_stage_cfg=None, stage_gap=0.0):
    return_stage_cfg = stage_cfg if return_stage_cfg is None else return_stage_cfg
    return make_platform_cfg(
        name=name,
        width=stage_cfg.dim[0] + return_stage_cfg.dim[0] + stage_gap,
        length=stage_cfg.dim[0],
        height=_stage_rise(stage_cfg),
        floor_thickness=stage_cfg.floor_thickness,
        wall_thickness=stage_cfg.wall.wall_thickness,
        wall_height=max(1.0, _stage_rise(stage_cfg)),
        wall_edges=("left", "right"),
        surface_thickness=stage_cfg.floor_thickness,
    )


def _make_u_turn_final_platform(stage_cfg, name):
    return make_platform_for_stage(
        stage_cfg,
        name=name,
        surface_thickness=stage_cfg.floor_thickness,
    )


def _build_stage_cfgs(u_turn_stage_gap):
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
    slope_cfg = make_slope_cfg(
        name="slope_stage",
        corridor_width=5.0,
        wall_thickness=0.12,
        floor_thickness=0.1,
        structure_height=2.0,
        slope_length=3.0,
        slope_angle_deg=45.0,
    )

    return {
        "stairs": {
            "stage": stairs_cfg,
            "linear_platform": make_platform_for_stage(stairs_cfg, name="stairs_linear_platform"),
            "turn_platform": make_platform_for_stage(
                stairs_cfg,
                name="stairs_turn_platform",
                turn_direction="left",
            ),
            "u_turn_platform": _make_u_turn_platform(
                stairs_cfg,
                name="stairs_u_turn_platform",
                stage_gap=u_turn_stage_gap,
            ),
            "u_turn_final_platform": _make_u_turn_final_platform(
                stairs_cfg,
                name="stairs_u_turn_final_platform",
            ),
        },
        "slopes": {
            "stage": slope_cfg,
            "linear_platform": make_platform_for_stage(slope_cfg, name="slopes_linear_platform"),
            "turn_platform": make_platform_for_stage(
                slope_cfg,
                name="slopes_turn_platform",
                turn_direction="left",
            ),
            "u_turn_platform": _make_u_turn_platform(
                slope_cfg,
                name="slopes_u_turn_platform",
                stage_gap=u_turn_stage_gap,
            ),
            "u_turn_final_platform": _make_u_turn_final_platform(
                slope_cfg,
                name="slopes_u_turn_final_platform",
            ),
        },
    }


def _build_stairs_previews(
    cfgs,
    *,
    use_grounded_side_walls,
    use_common_ground,
    use_u_turn_common_ground,
    side_wall_extra_height,
    add_final_stair_end_wall,
    add_u_turn_platform_end_wall,
    add_u_turn_gap_wall,
    u_turn_repeating_stair_stages,
):
    stairs_cfg = cfgs["stage"]
    linear_kwargs = {
        "grounded_side_walls": use_grounded_side_walls,
        "common_ground": use_common_ground,
        "add_final_end_wall": add_final_stair_end_wall,
    }
    u_turn_kwargs = {
        "return_side": "left",
        "grounded_side_walls": use_grounded_side_walls,
        "common_ground": use_u_turn_common_ground,
        "add_final_end_wall": add_final_stair_end_wall,
        "add_turn_platform_end_wall": add_u_turn_platform_end_wall,
        "add_turn_gap_wall": add_u_turn_gap_wall,
    }

    return [
        (
            "stairs_platform_stairs_demo",
            make_stairs_platform_stairs_mesh(
                stairs_cfg,
                cfgs["linear_platform"],
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 2, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "stairs_platform_4_demo",
            make_linear_stairs_mesh(
                stairs_cfg,
                cfgs["linear_platform"],
                num_stages=4,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 4, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "stairs_turn_90_demo",
            make_stairs_turn_90_mesh(
                stairs_cfg,
                cfgs["turn_platform"],
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 2, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "stairs_turn_4_demo",
            make_rotating_stairs_mesh(
                stairs_cfg,
                cfgs["turn_platform"],
                num_stages=4,
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 4, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "stairs_u_turn_demo",
            make_stairs_u_turn_mesh(
                stairs_cfg,
                cfgs["u_turn_platform"],
                cfgs["u_turn_final_platform"],
                grounded_wall_height=_stairs_wall_height(stairs_cfg, 2, side_wall_extra_height),
                **u_turn_kwargs,
            ),
        ),
        (
            "stairs_u_turn_4_demo",
            make_repeating_u_turn_stairs_mesh(
                stairs_cfg,
                cfgs["u_turn_platform"],
                cfgs["u_turn_final_platform"],
                num_stages=u_turn_repeating_stair_stages,
                grounded_wall_height=_stairs_wall_height(
                    stairs_cfg,
                    u_turn_repeating_stair_stages,
                    side_wall_extra_height,
                ),
                **u_turn_kwargs,
            ),
        ),
    ]


def _build_slope_previews(
    cfgs,
    *,
    use_grounded_side_walls,
    use_common_ground,
    use_u_turn_common_ground,
    side_wall_extra_height,
    add_final_linear_slope_end_wall,
    add_u_turn_platform_end_wall,
    add_u_turn_gap_wall,
):
    slope_cfg = cfgs["stage"]
    linear_kwargs = {
        "grounded_side_walls": use_grounded_side_walls,
        "common_ground": use_common_ground,
        "add_final_end_wall": add_final_linear_slope_end_wall,
    }
    u_turn_kwargs = {
        "return_side": "left",
        "grounded_side_walls": use_grounded_side_walls,
        "common_ground": use_u_turn_common_ground,
        "add_final_end_wall": add_final_linear_slope_end_wall,
        "add_turn_platform_end_wall": add_u_turn_platform_end_wall,
        "add_turn_gap_wall": add_u_turn_gap_wall,
    }

    return [
        (
            "slopes_platform_slopes_demo",
            make_slopes_platform_slopes_mesh(
                slope_cfg,
                cfgs["linear_platform"],
                grounded_wall_height=_slope_wall_height(slope_cfg, 2, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "slopes_platform_4_demo",
            make_linear_slopes_mesh(
                slope_cfg,
                cfgs["linear_platform"],
                num_stages=4,
                grounded_wall_height=_slope_wall_height(slope_cfg, 4, side_wall_extra_height),
                **linear_kwargs,
            ),
        ),
        (
            "slopes_turn_90_demo",
            make_slopes_turn_90_mesh(
                slope_cfg,
                cfgs["turn_platform"],
                grounded_wall_height=_slope_wall_height(slope_cfg, 2, side_wall_extra_height),
                grounded_side_walls=use_grounded_side_walls,
                common_ground=use_common_ground,
            ),
        ),
        (
            "slopes_turn_4_demo",
            make_rotating_slopes_mesh(
                slope_cfg,
                cfgs["turn_platform"],
                num_stages=4,
                grounded_wall_height=_slope_wall_height(slope_cfg, 4, side_wall_extra_height),
                grounded_side_walls=use_grounded_side_walls,
                common_ground=use_common_ground,
            ),
        ),
        (
            "slopes_u_turn_demo",
            make_slopes_u_turn_mesh(
                slope_cfg,
                cfgs["u_turn_platform"],
                cfgs["u_turn_final_platform"],
                grounded_wall_height=_slope_wall_height(slope_cfg, 2, side_wall_extra_height),
                **u_turn_kwargs,
            ),
        ),
    ]


def get_preview_meshes(
    *,
    use_grounded_side_walls=True,
    use_common_ground=True,
    use_u_turn_common_ground=False,
    side_wall_extra_height=2.0,
    add_final_stair_end_wall=True,
    add_final_linear_slope_end_wall=True,
    u_turn_stage_gap=2.0,
    add_u_turn_platform_end_wall=True,
    add_u_turn_gap_wall=True,
    u_turn_repeating_stair_stages=4,
):
    cfgs = _build_stage_cfgs(u_turn_stage_gap)
    meshes = []
    meshes.extend(
        _build_stairs_previews(
            cfgs["stairs"],
            use_grounded_side_walls=use_grounded_side_walls,
            use_common_ground=use_common_ground,
            use_u_turn_common_ground=use_u_turn_common_ground,
            side_wall_extra_height=side_wall_extra_height,
            add_final_stair_end_wall=add_final_stair_end_wall,
            add_u_turn_platform_end_wall=add_u_turn_platform_end_wall,
            add_u_turn_gap_wall=add_u_turn_gap_wall,
            u_turn_repeating_stair_stages=u_turn_repeating_stair_stages,
        )
    )
    meshes.extend(
        _build_slope_previews(
            cfgs["slopes"],
            use_grounded_side_walls=use_grounded_side_walls,
            use_common_ground=use_common_ground,
            use_u_turn_common_ground=use_u_turn_common_ground,
            side_wall_extra_height=side_wall_extra_height,
            add_final_linear_slope_end_wall=add_final_linear_slope_end_wall,
            add_u_turn_platform_end_wall=add_u_turn_platform_end_wall,
            add_u_turn_gap_wall=add_u_turn_gap_wall,
        )
    )
    return tuple(meshes)


def export_and_optionally_show(name, mesh, *, output_dir, show_meshes):
    output_path = output_dir / f"{name}.obj"
    mesh.export(output_path)
    print(f"Exported {name}: {output_path}")
    if show_meshes:
        mesh.show()


def main(
    *,
    output_dir=Path("results/complex_parts_preview"),
    show_meshes=True,
    use_grounded_side_walls=True,
    use_common_ground=True,
    use_u_turn_common_ground=False,
    side_wall_extra_height=2.0,
    add_final_stair_end_wall=True,
    add_final_linear_slope_end_wall=True,
    u_turn_stage_gap=2.0,
    add_u_turn_platform_end_wall=True,
    add_u_turn_gap_wall=True,
    u_turn_repeating_stair_stages=4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meshes = get_preview_meshes(
        use_grounded_side_walls=use_grounded_side_walls,
        use_common_ground=use_common_ground,
        use_u_turn_common_ground=use_u_turn_common_ground,
        side_wall_extra_height=side_wall_extra_height,
        add_final_stair_end_wall=add_final_stair_end_wall,
        add_final_linear_slope_end_wall=add_final_linear_slope_end_wall,
        u_turn_stage_gap=u_turn_stage_gap,
        add_u_turn_platform_end_wall=add_u_turn_platform_end_wall,
        add_u_turn_gap_wall=add_u_turn_gap_wall,
        u_turn_repeating_stair_stages=u_turn_repeating_stair_stages,
    )
    for name, mesh in meshes:
        export_and_optionally_show(name, mesh, output_dir=output_dir, show_meshes=show_meshes)


if __name__ == "__main__":
    main()
