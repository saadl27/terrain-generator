import pytest

from ..trimesh_tiles.curriculum.common import (
    LINEAR_SHARED_GROUNDED_WALL_HEIGHT,
    TOTAL_CURRICULUM_LEVELS,
    expand_terrain_to_cell,
)
from ..trimesh_tiles.curriculum.linear_slopes import build_category_terrain as build_linear_slopes_terrain
from ..trimesh_tiles.curriculum.linear_stairs import build_category_terrain as build_linear_stairs_terrain
from ..trimesh_tiles.mesh_parts import assembled_parts
from ..trimesh_tiles.mesh_parts.part_presets import make_platform_cfg


def test_linear_stairs_curriculum_uses_grounded_stage_platform(monkeypatch):
    captured_kwargs = {}
    original_make_platform_cfg = make_platform_cfg

    def capture_platform_cfg(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_make_platform_cfg(*args, **kwargs)

    monkeypatch.setattr(
        "terrain_generator.trimesh_tiles.curriculum.linear_stairs.make_platform_cfg",
        capture_platform_cfg,
    )

    build_linear_stairs_terrain(1)

    assert captured_kwargs["name"] == "linear_stairs_platform"
    assert "surface_thickness" not in captured_kwargs


@pytest.mark.parametrize(
    "build_terrain",
    (build_linear_stairs_terrain, build_linear_slopes_terrain),
)
def test_linear_curriculum_max_height_is_fixed_across_levels(build_terrain):
    terrains = [build_terrain(level) for level in range(1, TOTAL_CURRICULUM_LEVELS + 1)]
    heights = [terrain.metadata["mesh_extents"]["z"] for terrain in terrains]
    target_width = max(float(terrain.metadata["mesh_extents"]["x"]) for terrain in terrains)
    target_length = max(float(terrain.metadata["mesh_extents"]["y"]) for terrain in terrains)
    expanded_heights = [
        expand_terrain_to_cell(terrain, target_width, target_length).metadata["mesh_extents"]["z"]
        for terrain in terrains
    ]

    assert max(heights) == pytest.approx(min(heights), abs=1.0e-6)
    assert max(expanded_heights) == pytest.approx(min(expanded_heights), abs=1.0e-6)


def test_linear_stairs_and_slopes_share_max_height():
    terrains = [
        builder(level)
        for builder in (build_linear_stairs_terrain, build_linear_slopes_terrain)
        for level in range(1, TOTAL_CURRICULUM_LEVELS + 1)
    ]
    target_width = max(float(terrain.metadata["mesh_extents"]["x"]) for terrain in terrains)
    target_length = max(float(terrain.metadata["mesh_extents"]["y"]) for terrain in terrains)
    heights = [
        expand_terrain_to_cell(terrain, target_width, target_length).metadata["mesh_extents"]["z"]
        for terrain in terrains
    ]

    assert max(heights) == pytest.approx(min(heights), abs=1.0e-6)


@pytest.mark.parametrize(
    "build_terrain",
    (build_linear_stairs_terrain, build_linear_slopes_terrain),
)
def test_linear_curriculum_entry_walls_extend_to_shared_height(monkeypatch, build_terrain):
    entry_wall_heights = []
    original_entry_corridor = assembled_parts._build_entry_corridor_meshes

    def capture_entry_corridor(part, yaw_deg, start_edge_center_world, floor_base_z, entry_length, wall_height=None):
        entry_wall_heights.append(float(wall_height))
        return original_entry_corridor(
            part,
            yaw_deg,
            start_edge_center_world,
            floor_base_z,
            entry_length,
            wall_height=wall_height,
        )

    monkeypatch.setattr(assembled_parts, "_build_entry_corridor_meshes", capture_entry_corridor)

    terrain = build_terrain(1)

    assert entry_wall_heights == [pytest.approx(LINEAR_SHARED_GROUNDED_WALL_HEIGHT)]
    assert terrain.metadata["grounded_wall_height"] == pytest.approx(
        LINEAR_SHARED_GROUNDED_WALL_HEIGHT,
        abs=1.0e-3,
    )


@pytest.mark.parametrize(
    "build_terrain",
    (build_linear_stairs_terrain, build_linear_slopes_terrain),
)
def test_linear_curriculum_filter_unused_area_raises_outside_side_walls(build_terrain):
    terrain = build_terrain(1)
    expanded = expand_terrain_to_cell(
        terrain,
        float(terrain.metadata["mesh_extents"]["x"]) + 4.0,
        float(terrain.metadata["mesh_extents"]["y"]) + 4.0,
    )
    vertices = expanded.mesh.vertices
    outer_half_width = 0.5 * float(expanded.metadata["filter_unused_outer_width"])
    outside_walls = abs(vertices[:, 0]) > outer_half_width + 0.05

    assert expanded.metadata["filtered_unused_area_to_max_height"] is True
    assert outside_walls.any()
    assert vertices[outside_walls, 2].max() == pytest.approx(expanded.mesh.bounds[1, 2], abs=1.0e-6)


@pytest.mark.parametrize(
    "build_terrain",
    (build_linear_stairs_terrain, build_linear_slopes_terrain),
)
def test_linear_curriculum_common_ground_uses_one_support_plane(monkeypatch, build_terrain):
    fill_support_bases = []
    wall_support_bases = []
    original_ground_fill = assembled_parts._build_ground_fill_mesh
    original_grounded_wall = assembled_parts._build_extended_grounded_wall_mesh

    def capture_ground_fill(part, yaw_deg, translation_xy, bottom_world_z, support_base_z):
        fill_support_bases.append(float(support_base_z))
        return original_ground_fill(part, yaw_deg, translation_xy, bottom_world_z, support_base_z)

    def capture_grounded_wall(
        part,
        yaw_deg,
        translation_xy,
        translation_z,
        support_base_z,
        wall_height,
        extra_length,
        wall_edges_override=None,
    ):
        wall_support_bases.append(float(support_base_z))
        return original_grounded_wall(
            part,
            yaw_deg,
            translation_xy,
            translation_z,
            support_base_z,
            wall_height,
            extra_length,
            wall_edges_override=wall_edges_override,
        )

    monkeypatch.setattr(assembled_parts, "_build_ground_fill_mesh", capture_ground_fill)
    monkeypatch.setattr(assembled_parts, "_build_extended_grounded_wall_mesh", capture_grounded_wall)

    build_terrain(20)

    assert len(fill_support_bases) > 0
    assert len(wall_support_bases) > 0
    assert max(fill_support_bases) == pytest.approx(min(fill_support_bases), abs=1.0e-6)
    assert max(wall_support_bases) == pytest.approx(min(wall_support_bases), abs=1.0e-6)
