import numpy as np

from .common import (
    ADD_FINAL_LINEAR_SLOPE_END_WALL,
    FLOOR_THICKNESS,
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


def build_category_terrain(level: int) -> TerrainScene:
    params = _level_params(level)
    effective_width = effective_corridor_width(params["corridor_width"])
    slope_cfg = make_slope_cfg(
        name="linear_slopes_slope",
        corridor_width=effective_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=4.0,
        slope_length=float(params["slope_length"]),
        slope_angle_deg=float(params["slope_angle_deg"]),
        slope_resolution=36,
        wall_height=max(1.0, FLOOR_THICKNESS + np.tan(np.deg2rad(params["slope_angle_deg"])) * params["slope_length"]),
    )
    platform_height = np.tan(np.deg2rad(params["slope_angle_deg"])) * params["slope_length"]
    grounded_wall_height = int(params["num_segments"]) * platform_height + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_cfg(
        name="linear_slopes_platform",
        width=slope_cfg.dim[0],
        length=float(params["flat_length"]),
        height=platform_height,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(0.8, platform_height),
        wall_edges=("left", "right"),
        load_from_cache=False,
        surface_thickness=FLOOR_THICKNESS,
    )
    feature_mesh = make_linear_slopes_mesh(
        slope_cfg,
        platform_cfg,
        num_stages=int(params["num_segments"]),
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
        entry_length=float(params["entry_length"]),
        entry_wall_height=float(slope_cfg.wall.wall_height),
    )
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
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
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_LINEAR_SLOPE_END_WALL,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )
