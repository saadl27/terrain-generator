import numpy as np

from .common import (
    EXTEND_LINEAR_FINAL_PLATEAU,
    FILTER_UNUSED_LINEAR_AREA,
    FLOOR_THICKNESS,
    LINEAR_SHARED_GROUNDED_WALL_HEIGHT,
    SIDE_WALL_EXTRA_HEIGHT,
    USE_COMMON_GROUND,
    USE_GROUNDED_SIDE_WALLS,
    WALL_THICKNESS,
    TerrainScene,
    difficulty_ratio,
    effective_corridor_width,
    lerp,
    merge_feature_with_flat_base,
    mesh_extents,
    round_float,
    strip_part_walls,
)
from ..mesh_parts.assembled_parts import make_linear_slopes_mesh
from ..mesh_parts.part_presets import make_platform_cfg, make_slope_cfg


def _level_params(level: int):
    t = difficulty_ratio(level)
    return {
        "slope_length": lerp(2.5, 6.2, t),
        "slope_angle_deg": lerp(5.0, 28.0, t),
        "corridor_width": lerp(4.0, 2.6, t),
        "flat_length": lerp(2.0, 1.3, t),
        "entry_length": lerp(2.8, 1.6, t),
        "num_segments": 3,
    }


def _stage_rise(params) -> float:
    return float(np.tan(np.deg2rad(params["slope_angle_deg"])) * params["slope_length"])


def build_category_terrain(level: int, include_walls: bool = True) -> TerrainScene:
    params = _level_params(level)
    effective_width = effective_corridor_width(params["corridor_width"])
    wall_thickness = WALL_THICKNESS if include_walls else 0.0
    slope_cfg = make_slope_cfg(
        name="linear_slopes_slope",
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
    platform_height = _stage_rise(params)
    grounded_wall_height = LINEAR_SHARED_GROUNDED_WALL_HEIGHT if include_walls else 0.0
    final_plateau_height = int(params["num_segments"]) * platform_height
    platform_cfg = make_platform_cfg(
        name="linear_slopes_platform",
        width=slope_cfg.dim[0],
        length=float(params["flat_length"]),
        height=platform_height,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=wall_thickness,
        wall_height=max(0.8, platform_height) if include_walls else 0.0,
        wall_edges=("left", "right") if include_walls else (),
        load_from_cache=False,
        surface_thickness=FLOOR_THICKNESS,
    )
    if not include_walls:
        strip_part_walls(platform_cfg)
    feature_mesh = make_linear_slopes_mesh(
        slope_cfg,
        platform_cfg,
        num_stages=int(params["num_segments"]),
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS if include_walls else False,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=False,
        entry_length=float(params["entry_length"]),
        entry_wall_height=grounded_wall_height if include_walls else 0.0,
    )
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh, min_width=0.0, side_padding=0.0)
    return TerrainScene(
        terrain_id="linear_slopes",
        label=f"Level {level} linear slopes",
        mesh=mesh,
        metadata={
            "terrain_id": "linear_slopes",
            "label": f"Level {level} linear slopes",
            "type": "linear_slopes",
            "pattern": "entry corridor -> slope -> platform -> slope -> platform -> slope -> platform",
            "num_segments": int(params["num_segments"]),
            "slope_length": round_float(params["slope_length"]),
            "slope_angle_deg": round_float(params["slope_angle_deg"]),
            "corridor_width": round_float(effective_width),
            "nominal_corridor_width": round_float(params["corridor_width"]),
            "entry_corridor_length": round_float(params["entry_length"]),
            "flat_length_between_segments": round_float(params["flat_length"]),
            "has_walls": include_walls,
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS if include_walls else False,
            "common_ground": USE_COMMON_GROUND,
            "grounded_wall_height": round_float(grounded_wall_height),
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT if include_walls else 0.0),
            "fixed_max_height_across_levels": True,
            "filter_unused_area": FILTER_UNUSED_LINEAR_AREA if include_walls else False,
            "filter_unused_outer_width": round_float(effective_width),
            "extend_final_plateau_to_arena": EXTEND_LINEAR_FINAL_PLATEAU if include_walls else False,
            "final_plateau_width": round_float(effective_width),
            "final_plateau_height": round_float(final_plateau_height),
            "add_final_end_wall": False,
            "arena_walls": include_walls,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )
