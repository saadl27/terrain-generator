from .common import (
    ADD_FINAL_STAIR_END_WALL,
    FLOOR_THICKNESS,
    SIDE_WALL_EXTRA_HEIGHT,
    USE_COMMON_GROUND,
    USE_GROUNDED_SIDE_WALLS,
    WALL_THICKNESS,
    TerrainScene,
    difficulty_ratio,
    effective_corridor_width,
    lerp,
    lerp_int,
    merge_feature_with_flat_base,
    mesh_extents,
    round_float,
)
from ..mesh_parts.assembled_parts import make_linear_stairs_mesh
from ..mesh_parts.part_presets import make_platform_cfg, make_stairs_cfg


def _level_params(level: int):
    t = difficulty_ratio(level)
    return {
        "num_steps": lerp_int(1, 12, t),
        "step_height": lerp(0.08, 0.18, t),
        "step_depth": lerp(0.45, 0.60, t),
        "corridor_width": lerp(4.0, 2.6, t),
        "flat_length": lerp(2.0, 1.3, t),
        "num_segments": 3,
    }


def build_category_terrain(level: int) -> TerrainScene:
    params = _level_params(level)
    effective_width = effective_corridor_width(params["corridor_width"])
    stairs_cfg = make_stairs_cfg(
        name="linear_stairs_stairs",
        direction="front",
        corridor_width=effective_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        num_steps=int(params["num_steps"]),
        step_height=float(params["step_height"]),
        step_depth=float(params["step_depth"]),
        wall_height=max(1.0, float(params["num_steps"]) * float(params["step_height"]) + FLOOR_THICKNESS),
    )
    grounded_wall_height = int(params["num_segments"]) * stairs_cfg.stairs[0].total_height + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_cfg(
        name="linear_stairs_platform",
        width=stairs_cfg.dim[0],
        length=float(params["flat_length"]),
        height=stairs_cfg.stairs[0].total_height,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(0.8, stairs_cfg.stairs[0].total_height),
        wall_edges=("left", "right"),
        load_from_cache=False,
        surface_thickness=FLOOR_THICKNESS,
    )
    feature_mesh = make_linear_stairs_mesh(
        stairs_cfg,
        platform_cfg,
        num_stages=int(params["num_segments"]),
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
    )
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id="linear_stairs",
        label=f"Level {level} linear stairs",
        mesh=mesh,
        metadata={
            "terrain_id": "linear_stairs",
            "label": f"Level {level} linear stairs",
            "type": "linear_stairs",
            "pattern": "stairs -> platform -> stairs -> platform -> stairs -> platform",
            "num_segments": int(params["num_segments"]),
            "num_steps_per_segment": int(params["num_steps"]),
            "step_height": round_float(params["step_height"]),
            "step_depth": round_float(params["step_depth"]),
            "corridor_width": round_float(effective_width),
            "nominal_corridor_width": round_float(params["corridor_width"]),
            "flat_length_between_segments": round_float(params["flat_length"]),
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_STAIR_END_WALL,
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )
