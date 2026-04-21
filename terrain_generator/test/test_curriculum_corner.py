from ..trimesh_tiles.curriculum.common import (
    CORNER_LOOP_CELL_MARGIN,
    WALL_THICKNESS,
    corner_loop_outline_extents,
    expand_terrain_to_cell,
)
from ..trimesh_tiles.curriculum.corner import build_category_terrain


def test_corner_loop_expansion_scales_to_allocated_cell():
    terrain = build_category_terrain(38)
    target_size = max(float(terrain.metadata["base_width"]), float(terrain.metadata["base_length"])) + 6.0

    raw_feature_extents = corner_loop_outline_extents(
        num_corners=int(terrain.metadata["num_corners"]),
        pre_length=float(terrain.metadata["pre_length"]),
        post_length=float(terrain.metadata["post_length"]),
        pre_corridor_width=float(terrain.metadata["pre_corridor_width"]),
        post_corridor_width=float(terrain.metadata["post_corridor_width"]),
        turn_angle_deg=float(terrain.metadata["turn_angle_deg"]),
        wall_thickness=WALL_THICKNESS,
    )

    expanded = expand_terrain_to_cell(terrain, target_size, target_size)
    expanded_feature_extents = expanded.metadata["loop_feature_extents"]

    assert expanded.metadata["outer_polygon_uses_allocated_cell"] is True
    assert float(expanded.metadata["loop_scale"]) > 1.0
    assert float(expanded.metadata["pre_length"]) > float(expanded.metadata["nominal_pre_length"])
    assert float(expanded.metadata["post_length"]) > float(expanded.metadata["nominal_post_length"])
    assert float(expanded_feature_extents["x"]) > float(raw_feature_extents[0])
    assert float(expanded_feature_extents["y"]) > float(raw_feature_extents[1])
    assert float(expanded_feature_extents["x"]) <= target_size - 2.0 * CORNER_LOOP_CELL_MARGIN + 1.0e-3
    assert float(expanded_feature_extents["y"]) <= target_size - 2.0 * CORNER_LOOP_CELL_MARGIN + 1.0e-3
    assert float(expanded.metadata["mesh_extents"]["x"]) == target_size
    assert float(expanded.metadata["mesh_extents"]["y"]) == target_size
