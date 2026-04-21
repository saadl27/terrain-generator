import numpy as np

from .common import (
    ADD_FINAL_LINEAR_SLOPE_END_WALL,
    FLOOR_THICKNESS,
    SIDE_WALL_EXTRA_HEIGHT,
    U_TURN_COMMON_GROUND,
    USE_COMMON_GROUND,
    USE_GROUNDED_SIDE_WALLS,
    WALL_THICKNESS,
    TerrainScene,
    difficulty_ratio,
    effective_corridor_width,
    lerp,
    make_final_platform_for_stage,
    make_u_turn_platform_for_stage,
    merge_feature_with_flat_base,
    mesh_extents,
    round_float,
    turn_angle_for_level,
    turn_angle_with_direction,
)
from ..mesh_parts.assembled_parts import (
    make_angled_slopes_mesh,
    make_linear_slopes_mesh,
    make_rotating_slopes_mesh,
    make_slopes_u_turn_mesh,
)
from ..mesh_parts.part_presets import make_corner_for_stage, make_platform_for_stage, make_slope_cfg


def _level_params(level: int):
    t = difficulty_ratio(level)
    turn_angle_deg = turn_angle_for_level(level)
    return {
        "slope_length": lerp(3.6, 6.84, t),
        "slope_angle_deg": lerp(10.0, 31.6, t),
        "corridor_width": lerp(3.4, 2.5, t),
        "entry_length": lerp(2.8, 1.6, t),
        "turn_angle_deg": turn_angle_deg,
        "num_segments": 3 if abs(turn_angle_deg - 90.0) <= 1.0e-6 else 2,
    }


def build_category_terrain(level: int) -> TerrainScene:
    params = _level_params(level)
    turn_direction = "left" if level % 2 == 0 else "right"
    effective_width = effective_corridor_width(params["corridor_width"])
    slope_cfg = make_slope_cfg(
        name="turning_slopes_slope",
        corridor_width=effective_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=4.0,
        slope_length=float(params["slope_length"]),
        slope_angle_deg=float(params["slope_angle_deg"]),
        slope_resolution=36,
        wall_height=max(1.0, FLOOR_THICKNESS + np.tan(np.deg2rad(params["slope_angle_deg"])) * params["slope_length"]),
    )
    stage_rise = np.tan(np.deg2rad(slope_cfg.slope_angle_deg)) * slope_cfg.slope_length
    grounded_wall_height = int(params["num_segments"]) * stage_rise + SIDE_WALL_EXTRA_HEIGHT
    common_ground = USE_COMMON_GROUND
    abs_turn_angle_deg = abs(float(params["turn_angle_deg"]))

    if abs_turn_angle_deg <= 1.0e-6:
        platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_straight_platform")
        feature_mesh = make_linear_slopes_mesh(
            slope_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(slope_cfg.wall.wall_height),
        )
        terrain_type = "turning_slopes"
        pattern = "slope -> flat platform -> slope -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating slopes"
    elif abs_turn_angle_deg >= 180.0 - 1.0e-6:
        turn_platform_cfg = make_u_turn_platform_for_stage(slope_cfg, name="turning_slopes_u_turn_platform")
        final_platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_u_turn_final_platform")
        feature_mesh = make_slopes_u_turn_mesh(
            slope_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            return_side=turn_direction,
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=U_TURN_COMMON_GROUND,
            add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(slope_cfg.wall.wall_height),
        )
        terrain_type = "u_turn_slopes"
        pattern = "slope -> U-turn landing -> slope -> final platform"
        turn_pattern = "u_turn"
        common_ground = U_TURN_COMMON_GROUND
        label = f"Level {level} U-turn slopes"
    elif abs(abs_turn_angle_deg - 90.0) <= 1.0e-6:
        platform_cfg = make_platform_for_stage(
            slope_cfg,
            name="turning_slopes_turn_platform",
            turn_direction=turn_direction,
            floor_thickness=FLOOR_THICKNESS,
            wall_thickness=WALL_THICKNESS,
            wall_height=max(1.0, stage_rise),
            load_from_cache=False,
        )
        feature_mesh = make_rotating_slopes_mesh(
            slope_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            turn_direction=turn_direction,
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(slope_cfg.wall.wall_height),
        )
        terrain_type = "turning_slopes"
        pattern = "slope -> 90-degree turn platform -> slope -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating slopes"
    else:
        turn_platform_cfg = make_corner_for_stage(
            slope_cfg,
            name="turning_slopes_angled_turn_platform",
            turn_angle_deg=turn_angle_with_direction(turn_direction, abs_turn_angle_deg),
            floor_thickness=FLOOR_THICKNESS,
            wall_thickness=WALL_THICKNESS,
            wall_height=max(1.0, stage_rise),
            load_from_cache=False,
        )
        final_platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_angled_final_platform")
        feature_mesh = make_angled_slopes_mesh(
            slope_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(slope_cfg.wall.wall_height),
        )
        terrain_type = "turning_slopes"
        pattern = "slope -> angled turn platform -> slope -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating slopes"
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id="turning_slopes",
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": "turning_slopes",
            "label": label,
            "type": terrain_type,
            "pattern": f"entry corridor -> {pattern}",
            "num_segments": int(params["num_segments"]),
            "slope_length": round_float(params["slope_length"]),
            "slope_angle_deg": round_float(params["slope_angle_deg"]),
            "corridor_width": round_float(effective_width),
            "nominal_corridor_width": round_float(params["corridor_width"]),
            "entry_corridor_length": round_float(params["entry_length"]),
            "turn_direction": turn_direction,
            "turn_angle_deg": round_float(abs_turn_angle_deg),
            "turn_pattern": turn_pattern,
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": common_ground,
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_LINEAR_SLOPE_END_WALL,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )
