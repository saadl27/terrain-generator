import numpy as np

from .common import (
    ADD_FINAL_LINEAR_SLOPE_END_WALL,
    FILTER_UNUSED_COMPLEX_AREA,
    FLOOR_THICKNESS,
    MAX_TURNING_STAGES,
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
    prepare_footprint_filter_source,
    round_float,
    strip_part_walls,
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
    nominal_num_segments = 3 if abs(turn_angle_deg - 90.0) <= 1.0e-6 else 2
    return {
        "slope_length": lerp(3.6, 6.84, t),
        "slope_angle_deg": lerp(10.0, 31.6, t),
        "corridor_width": lerp(3.4, 2.5, t),
        "entry_length": lerp(2.8, 1.6, t),
        "turn_angle_deg": turn_angle_deg,
        "num_segments": min(MAX_TURNING_STAGES, nominal_num_segments),
    }


def build_category_terrain(level: int, include_walls: bool = True) -> TerrainScene:
    params = _level_params(level)
    turn_direction = "left" if level % 2 == 0 else "right"
    effective_width = effective_corridor_width(params["corridor_width"])
    wall_thickness = WALL_THICKNESS if include_walls else 0.0
    add_final_end_wall = ADD_FINAL_LINEAR_SLOPE_END_WALL if include_walls else False
    grounded_side_walls = USE_GROUNDED_SIDE_WALLS if include_walls else False
    slope_cfg = make_slope_cfg(
        name="turning_slopes_slope",
        corridor_width=effective_width,
        wall_thickness=wall_thickness,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=4.0,
        slope_length=float(params["slope_length"]),
        slope_angle_deg=float(params["slope_angle_deg"]),
        slope_resolution=36,
        wall_height=max(1.0, FLOOR_THICKNESS + np.tan(np.deg2rad(params["slope_angle_deg"])) * params["slope_length"])
        if include_walls
        else 0.0,
    )
    if not include_walls:
        strip_part_walls(slope_cfg)
    filter_slope_cfg = (
        make_slope_cfg(
            name="turning_slopes_filter_slope",
            corridor_width=effective_width,
            wall_thickness=0.0,
            floor_thickness=FLOOR_THICKNESS,
            structure_height=4.0,
            slope_length=float(params["slope_length"]),
            slope_angle_deg=float(params["slope_angle_deg"]),
            slope_resolution=36,
            wall_height=0.0,
        )
        if include_walls
        else None
    )
    if filter_slope_cfg is not None:
        strip_part_walls(filter_slope_cfg)
    stage_rise = np.tan(np.deg2rad(slope_cfg.slope_angle_deg)) * slope_cfg.slope_length
    grounded_wall_height = int(params["num_segments"]) * stage_rise + SIDE_WALL_EXTRA_HEIGHT if include_walls else 0.0
    common_ground = USE_COMMON_GROUND
    abs_turn_angle_deg = abs(float(params["turn_angle_deg"]))

    if abs_turn_angle_deg <= 1.0e-6:
        platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_straight_platform")
        if not include_walls:
            strip_part_walls(platform_cfg)
        filter_platform_cfg = None
        if filter_slope_cfg is not None:
            filter_platform_cfg = make_final_platform_for_stage(
                filter_slope_cfg,
                name="turning_slopes_straight_filter_platform",
            )
            strip_part_walls(filter_platform_cfg)
        feature_mesh = make_linear_slopes_mesh(
            slope_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=grounded_side_walls,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=add_final_end_wall,
            entry_length=float(params["entry_length"]),
            entry_wall_height=grounded_wall_height if include_walls else 0.0,
        )
        filter_feature_mesh = (
            make_linear_slopes_mesh(
                filter_slope_cfg,
                filter_platform_cfg,
                num_stages=int(params["num_segments"]),
                grounded_side_walls=False,
                grounded_wall_height=0.0,
                common_ground=False,
                add_final_end_wall=False,
                entry_length=float(params["entry_length"]),
                entry_wall_height=0.0,
            )
            if filter_slope_cfg is not None and filter_platform_cfg is not None
            else None
        )
        terrain_type = "turning_slopes"
        pattern = "slope -> flat platform -> slope -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating slopes"
    elif abs_turn_angle_deg >= 180.0 - 1.0e-6:
        turn_platform_cfg = make_u_turn_platform_for_stage(slope_cfg, name="turning_slopes_u_turn_platform")
        final_platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_u_turn_final_platform")
        if not include_walls:
            strip_part_walls(turn_platform_cfg)
            strip_part_walls(final_platform_cfg)
        filter_turn_platform_cfg = None
        filter_final_platform_cfg = None
        if filter_slope_cfg is not None:
            filter_turn_platform_cfg = make_u_turn_platform_for_stage(
                filter_slope_cfg,
                name="turning_slopes_u_turn_filter_platform",
            )
            filter_final_platform_cfg = make_final_platform_for_stage(
                filter_slope_cfg,
                name="turning_slopes_u_turn_filter_final_platform",
            )
            strip_part_walls(filter_turn_platform_cfg)
            strip_part_walls(filter_final_platform_cfg)
        feature_mesh = make_slopes_u_turn_mesh(
            slope_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            return_side=turn_direction,
            grounded_side_walls=grounded_side_walls,
            grounded_wall_height=grounded_wall_height,
            common_ground=U_TURN_COMMON_GROUND,
            add_final_end_wall=add_final_end_wall,
            entry_length=float(params["entry_length"]),
            entry_wall_height=grounded_wall_height if include_walls else 0.0,
        )
        filter_feature_mesh = (
            make_slopes_u_turn_mesh(
                filter_slope_cfg,
                filter_turn_platform_cfg,
                filter_final_platform_cfg,
                return_side=turn_direction,
                grounded_side_walls=False,
                grounded_wall_height=0.0,
                common_ground=False,
                add_final_end_wall=False,
                entry_length=float(params["entry_length"]),
                entry_wall_height=0.0,
            )
            if (
                filter_slope_cfg is not None
                and filter_turn_platform_cfg is not None
                and filter_final_platform_cfg is not None
            )
            else None
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
            wall_thickness=wall_thickness,
            wall_height=max(1.0, stage_rise) if include_walls else 0.0,
            load_from_cache=False,
        )
        if not include_walls:
            strip_part_walls(platform_cfg)
        filter_platform_cfg = None
        if filter_slope_cfg is not None:
            filter_platform_cfg = make_platform_for_stage(
                filter_slope_cfg,
                name="turning_slopes_filter_turn_platform",
                turn_direction=turn_direction,
                floor_thickness=FLOOR_THICKNESS,
                wall_thickness=0.0,
                wall_height=0.0,
                load_from_cache=False,
            )
            strip_part_walls(filter_platform_cfg)
        feature_mesh = make_rotating_slopes_mesh(
            slope_cfg,
            platform_cfg,
            num_stages=int(params["num_segments"]),
            turn_direction=turn_direction,
            grounded_side_walls=grounded_side_walls,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            entry_length=float(params["entry_length"]),
            entry_wall_height=grounded_wall_height if include_walls else 0.0,
        )
        filter_feature_mesh = (
            make_rotating_slopes_mesh(
                filter_slope_cfg,
                filter_platform_cfg,
                num_stages=int(params["num_segments"]),
                turn_direction=turn_direction,
                grounded_side_walls=False,
                grounded_wall_height=0.0,
                common_ground=False,
                entry_length=float(params["entry_length"]),
                entry_wall_height=0.0,
            )
            if filter_slope_cfg is not None and filter_platform_cfg is not None
            else None
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
            wall_thickness=wall_thickness,
            wall_height=max(1.0, stage_rise) if include_walls else 0.0,
            load_from_cache=False,
        )
        final_platform_cfg = make_final_platform_for_stage(slope_cfg, name="turning_slopes_angled_final_platform")
        if not include_walls:
            strip_part_walls(turn_platform_cfg)
            strip_part_walls(final_platform_cfg)
        filter_turn_platform_cfg = None
        filter_final_platform_cfg = None
        if filter_slope_cfg is not None:
            filter_turn_platform_cfg = make_corner_for_stage(
                filter_slope_cfg,
                name="turning_slopes_filter_angled_turn_platform",
                turn_angle_deg=turn_angle_with_direction(turn_direction, abs_turn_angle_deg),
                floor_thickness=FLOOR_THICKNESS,
                wall_thickness=0.0,
                wall_height=0.0,
                load_from_cache=False,
            )
            filter_final_platform_cfg = make_final_platform_for_stage(
                filter_slope_cfg,
                name="turning_slopes_filter_angled_final_platform",
            )
            strip_part_walls(filter_turn_platform_cfg)
            strip_part_walls(filter_final_platform_cfg)
        feature_mesh = make_angled_slopes_mesh(
            slope_cfg,
            turn_platform_cfg,
            final_platform_cfg,
            num_stages=int(params["num_segments"]),
            grounded_side_walls=grounded_side_walls,
            grounded_wall_height=grounded_wall_height,
            common_ground=common_ground,
            add_final_end_wall=add_final_end_wall,
            entry_length=float(params["entry_length"]),
            entry_wall_height=grounded_wall_height if include_walls else 0.0,
        )
        filter_feature_mesh = (
            make_angled_slopes_mesh(
                filter_slope_cfg,
                filter_turn_platform_cfg,
                filter_final_platform_cfg,
                num_stages=int(params["num_segments"]),
                grounded_side_walls=False,
                grounded_wall_height=0.0,
                common_ground=False,
                add_final_end_wall=False,
                entry_length=float(params["entry_length"]),
                entry_wall_height=0.0,
            )
            if (
                filter_slope_cfg is not None
                and filter_turn_platform_cfg is not None
                and filter_final_platform_cfg is not None
            )
            else None
        )
        terrain_type = "turning_slopes"
        pattern = "slope -> angled turn platform -> slope -> final platform"
        turn_pattern = "turning"
        label = f"Level {level} rotating slopes"
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
    filter_mesh = None
    if filter_feature_mesh is not None:
        filter_mesh = prepare_footprint_filter_source(filter_feature_mesh)
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
            "max_turning_stages": MAX_TURNING_STAGES,
            "slope_length": round_float(params["slope_length"]),
            "slope_angle_deg": round_float(params["slope_angle_deg"]),
            "corridor_width": round_float(effective_width),
            "nominal_corridor_width": round_float(params["corridor_width"]),
            "entry_corridor_length": round_float(params["entry_length"]),
            "turn_direction": turn_direction,
            "turn_angle_deg": round_float(abs_turn_angle_deg),
            "turn_pattern": turn_pattern,
            "has_walls": include_walls,
            "grounded_side_walls": grounded_side_walls,
            "common_ground": common_ground,
            "grounded_wall_height": round_float(grounded_wall_height),
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT if include_walls else 0.0),
            "add_final_end_wall": add_final_end_wall,
            "filter_unused_area": FILTER_UNUSED_COMPLEX_AREA if include_walls else False,
            "filter_unused_strategy": "footprint",
            "filter_unused_keepout_width": base_dim["feature_width"],
            "filter_unused_keepout_length": base_dim["feature_length"],
            "arena_walls": include_walls,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
        filter_mesh=filter_mesh,
    )
