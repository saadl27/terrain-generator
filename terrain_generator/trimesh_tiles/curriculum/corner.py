from .common import (
    FLOOR_THICKNESS,
    WALL_THICKNESS,
    TerrainScene,
    build_corner_loop_feature_mesh,
    closed_corner_loop_spec,
    difficulty_ratio,
    lerp,
    merge_feature_with_flat_base,
    mesh_extents,
    round_float,
)
from ..mesh_parts.part_presets import make_corner_cfg


def _level_params(level: int):
    t = difficulty_ratio(level)
    return {
        "corridor_width": lerp(5.2, 2.8, t),
        "pre_corridor_width": lerp(5.2, 2.8, t),
        "post_corridor_width": lerp(5.2, 2.8, t),
        "turn_angle_deg": lerp(20.0, 90.0, t),
        "pre_length": lerp(1.6, 2.4, t),
        "post_length": lerp(1.6, 2.4, t),
        "wall_height": lerp(1.2, 1.75, t),
    }


def build_corner_terrain(
    *,
    terrain_id: str,
    label: str,
    corridor_width: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    pre_length: float,
    post_length: float,
    wall_height: float = 1.4,
) -> TerrainScene:
    num_corners, adjusted_turn_angle_deg = closed_corner_loop_spec(turn_angle_deg)
    loop_corridor_width = corridor_width
    corner_cfg = make_corner_cfg(
        name=f"{terrain_id}_corner_loop",
        corridor_width=loop_corridor_width,
        pre_corridor_width=loop_corridor_width,
        post_corridor_width=loop_corridor_width,
        wall_thickness=WALL_THICKNESS,
        wall_height=wall_height,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=max(2.0, wall_height),
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=adjusted_turn_angle_deg,
        cap_ends=False,
        load_from_cache=False,
    )
    feature_mesh = build_corner_loop_feature_mesh(corner_cfg, num_corners=num_corners)
    mesh, base_dim = merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "corner_loop",
            "pattern": "closed polygon loop built from repeated corner segments",
            "corridor_width": round_float(loop_corridor_width),
            "pre_corridor_width": round_float(loop_corridor_width),
            "post_corridor_width": round_float(loop_corridor_width),
            "nominal_corridor_width": round_float(corridor_width),
            "nominal_pre_corridor_width": round_float(pre_corridor_width),
            "nominal_post_corridor_width": round_float(post_corridor_width),
            "loop_corridor_width": round_float(loop_corridor_width),
            "turn_angle_deg": round_float(adjusted_turn_angle_deg),
            "nominal_turn_angle_deg": round_float(turn_angle_deg),
            "num_corners": num_corners,
            "pre_length": round_float(pre_length),
            "post_length": round_float(post_length),
            "wall_height": round_float(wall_height),
            **base_dim,
            "mesh_extents": mesh_extents(mesh),
        },
    )


def build_category_terrain(level: int) -> TerrainScene:
    params = _level_params(level)
    return build_corner_terrain(
        terrain_id="corner",
        label=f"Level {level} corner loop",
        corridor_width=float(params["corridor_width"]),
        pre_corridor_width=float(params["pre_corridor_width"]),
        post_corridor_width=float(params["post_corridor_width"]),
        turn_angle_deg=float(params["turn_angle_deg"]),
        pre_length=float(params["pre_length"]),
        post_length=float(params["post_length"]),
        wall_height=float(params["wall_height"]),
    )
