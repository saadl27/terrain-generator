from .common import (
    ADD_FINAL_STAIR_END_WALL,
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
    lerp_int,
    make_final_platform_for_stage,
    make_u_turn_platform_for_stage,
    merge_feature_with_flat_base,
    mesh_extents,
    round_float,
    turn_angle_for_level,
    turn_angle_with_direction,
)
from ..mesh_parts.assembled_parts import (
    make_angled_stairs_mesh,
    make_linear_stairs_mesh,
    make_rotating_stairs_mesh,
    make_stairs_u_turn_mesh,
)
from ..mesh_parts.part_presets import make_corner_for_stage, make_platform_for_stage, make_stairs_cfg


def _level_params(level: int):
    t = difficulty_ratio(level)
    turn_angle_deg = turn_angle_for_level(level)
    return {
        "num_steps": lerp_int(2, 11, t),
        "step_height": lerp(0.11, 0.20, t),
        "step_depth": lerp(0.45, 0.63, t),
        "corridor_width": lerp(3.4, 2.5, t),
        "entry_length": lerp(2.8, 1.6, t),
        "turn_angle_deg": turn_angle_deg,
        "num_segments": 3 if abs(turn_angle_deg - 90.0) <= 1.0e-6 else 2,
    }


def build_category_terrain(level: int) -> TerrainScene:
    params = _level_params(level)
    turn_direction = "left" if level % 2 == 0 else "right"
    effective_width = effective_corridor_width(params["corridor_width"])
    stairs_cfg = make_stairs_cfg(
        name="turning_stairs_stairs",
        direction="front",
        corridor_width=effective_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        num_steps=int(params["num_steps"]),
        step_height=float(params["step_height"]),
        step_depth=float(params["step_depth"]),
        wall_height=max(1.0, float(params["num_steps"]) * float(params["step_height"]) + FLOOR_THICKNESS),
    )
    stage_rise = stairs_cfg.stairs[0].total_height
    grounded_wall_height = int(params["num_segments"]) * stage_rise + SIDE_WALL_EXTRA_HEIGHT
    common_ground = USE_COMMON_GROUND
    abs_turn_angle_deg = abs(float(params["turn_angle_deg"]))

    if abs_turn_angle_deg <= 1.0e-6:
        platform_cfg = make_final_platform_for_stage(stairs_cfg, name="turning_stairs_straight_platform")
        feature_mesh = make_linear_stairs_mesh(
            stairs_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(stairs_cfg.wall.wall_height),
        )
        terrain_type = "turning_stairs"
        pattern = "stairs -> flat platform -> stairs -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating stairs"
    elif abs_turn_angle_deg >= 180.0 - 1.0e-6:
        turn_platform_cfg = make_u_turn_platform_for_stage(stairs_cfg, name="turning_stairs_u_turn_platform")
        final_platform_cfg = make_final_platform_for_stage(stairs_cfg, name="turning_stairs_u_turn_final_platform")
        feature_mesh = make_stairs_u_turn_mesh(
            stairs_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            return_side=turn_direction,
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=U_TURN_COMMON_GROUND,
            add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(stairs_cfg.wall.wall_height),
        )
        terrain_type = "u_turn_stairs"
        pattern = "stairs -> U-turn landing -> stairs -> final platform"
        turn_pattern = "u_turn"
        common_ground = U_TURN_COMMON_GROUND
        label = f"Level {level} U-turn stairs"
    elif abs(abs_turn_angle_deg - 90.0) <= 1.0e-6:
        platform_cfg = make_platform_for_stage(
            stairs_cfg,
            name="turning_stairs_turn_platform",
            turn_direction=turn_direction,
            floor_thickness=FLOOR_THICKNESS,
            wall_thickness=WALL_THICKNESS,
            wall_height=max(1.0, stage_rise),
            load_from_cache=False,
        )
        feature_mesh = make_rotating_stairs_mesh(
            stairs_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            turn_direction=turn_direction,
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(stairs_cfg.wall.wall_height),
        )
        terrain_type = "turning_stairs"
        pattern = "stairs -> 90-degree turn platform -> stairs -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating stairs"
    else:
        turn_platform_cfg = make_corner_for_stage(
            stairs_cfg,
            name="turning_stairs_angled_turn_platform",
            turn_angle_deg=turn_angle_with_direction(turn_direction, abs_turn_angle_deg),
            floor_thickness=FLOOR_THICKNESS,
            wall_thickness=WALL_THICKNESS,
            wall_height=max(1.0, stage_rise),
            load_from_cache=False,
        )
        final_platform_cfg = make_final_platform_for_stage(stairs_cfg, name="turning_stairs_angled_final_platform")
        feature_mesh = make_angled_stairs_mesh(
            stairs_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
            entry_length=float(params["entry_length"]),
            entry_wall_height=float(stairs_cfg.wall.wall_height),
        )
        terrain_type = "turning_stairs"
        pattern = "stairs -> angled turn platform -> stairs -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating stairs"
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id="turning_stairs",
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": "turning_stairs",
            "label": label,
            "type": terrain_type,
            "pattern": f"entry corridor -> {pattern}",
            "num_segments": int(params["num_segments"]),
            "num_steps_per_segment": int(params["num_steps"]),
            "step_height": round_float(params["step_height"]),
            "step_depth": round_float(params["step_depth"]),
            "corridor_width": round_float(effective_width),
            "nominal_corridor_width": round_float(params["corridor_width"]),
            "entry_corridor_length": round_float(params["entry_length"]),
            "turn_direction": turn_direction,
            "turn_angle_deg": round_float(abs_turn_angle_deg),
            "turn_pattern": turn_pattern,
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": common_ground,
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_STAIR_END_WALL,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )
