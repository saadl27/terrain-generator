import pytest
import numpy as np

from ..utils import merge_meshes
from ..trimesh_tiles.curriculum.common import (
    ARENA_WALL_HEIGHT,
    FLOOR_THICKNESS,
    MAX_TURNING_STAGES,
    OVERLAY_EPS,
    box_mesh,
    build_flat_mesh,
    fit_mesh_to_dimensions,
    build_rectangular_unused_area_filter_mesh,
    expand_terrain_to_cell,
)
from ..trimesh_tiles.curriculum.corner import build_category_terrain as build_corner_terrain
from ..trimesh_tiles.curriculum.turning_slopes import build_category_terrain as build_turning_slopes_terrain
from ..trimesh_tiles.curriculum.turning_stairs import build_category_terrain as build_turning_stairs_terrain
from ..trimesh_tiles.mesh_parts import assembled_parts


def _top_height_at(mesh, x, y):
    origins = np.array([[x, y, float(mesh.bounds[1, 2]) + 1.0]])
    vectors = np.array([[0.0, 0.0, -1.0]])
    points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)
    if len(points) == 0 or 0 not in index_ray:
        return -np.inf
    return float(points[np.where(index_ray == 0)[0][0], 2])


@pytest.mark.parametrize(
    "build_terrain",
    (build_turning_stairs_terrain, build_turning_slopes_terrain),
)
@pytest.mark.parametrize("level", (1, 10, 20, 38))
def test_turning_curriculum_uses_common_ground_and_caps_stages(build_terrain, level):
    terrain = build_terrain(level)

    assert terrain.metadata["common_ground"] is True
    assert int(terrain.metadata["num_segments"]) <= MAX_TURNING_STAGES
    assert int(terrain.metadata["max_turning_stages"]) == MAX_TURNING_STAGES


@pytest.mark.parametrize(
    "build_terrain",
    (build_corner_terrain, build_turning_stairs_terrain, build_turning_slopes_terrain),
)
def test_complex_curriculum_filter_unused_area_raises_margins(build_terrain):
    terrain = build_terrain(20)
    target_width = float(terrain.metadata["mesh_extents"]["x"]) + 6.0
    target_length = float(terrain.metadata["mesh_extents"]["y"]) + 6.0
    expanded = expand_terrain_to_cell(terrain, target_width, target_length)

    assert expanded.metadata["filtered_unused_area_to_max_height"] is True
    assert expanded.metadata["mesh_extents"]["x"] == pytest.approx(target_width)
    assert expanded.metadata["mesh_extents"]["y"] == pytest.approx(target_length)


def test_corner_loop_filter_raises_inner_unused_region():
    terrain = build_corner_terrain(20)
    expanded = expand_terrain_to_cell(
        terrain,
        float(terrain.metadata["mesh_extents"]["x"]) + 6.0,
        float(terrain.metadata["mesh_extents"]["y"]) + 6.0,
    )

    assert expanded.metadata["filter_unused_strategy"] == "footprint"
    assert _top_height_at(expanded.mesh, 0.0, 0.0) == pytest.approx(expanded.mesh.bounds[1, 2], abs=1.0e-6)


def test_footprint_filter_raises_residual_flat_terrain_outside_course():
    base = build_flat_mesh(12.0, 12.0)
    feature = box_mesh(2.0, 2.0, FLOOR_THICKNESS, (0.0, 0.0, 0.5 * FLOOR_THICKNESS))
    feature.apply_translation([0.0, 0.0, OVERLAY_EPS])
    mesh = merge_meshes([base, feature], False)

    filtered = fit_mesh_to_dimensions(
        mesh,
        12.0,
        12.0,
        filter_unused_area=True,
        filter_unused_strategy="footprint",
    )

    assert _top_height_at(filtered, 4.0, 0.0) == pytest.approx(ARENA_WALL_HEIGHT + OVERLAY_EPS, abs=1.0e-6)
    assert _top_height_at(filtered, 0.0, 0.0) < 1.0


def test_rectangular_unused_area_filter_raises_space_outside_keepout():
    filter_mesh = build_rectangular_unused_area_filter_mesh(
        width=16.0,
        length=18.0,
        keepout_width=8.0,
        keepout_length=10.0,
        height=4.0,
    )
    vertices = filter_mesh.vertices
    outside_keepout_x = abs(vertices[:, 0]) > 4.0 + 1.0e-6

    assert outside_keepout_x.any()
    assert vertices[outside_keepout_x, 2].max() == pytest.approx(4.0)


@pytest.mark.parametrize(
    "build_terrain",
    (build_turning_stairs_terrain, build_turning_slopes_terrain),
)
@pytest.mark.parametrize("level", (10, 20, 38))
def test_turning_curriculum_common_ground_uses_one_support_plane(monkeypatch, build_terrain, level):
    fill_support_bases = []
    wall_support_bases = []
    original_ground_fill = assembled_parts._build_ground_fill_mesh
    original_grounded_wall = assembled_parts._build_extended_grounded_wall_mesh
    original_grounded_wall_segment = assembled_parts._build_grounded_wall_segment

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

    def capture_grounded_wall_segment(
        part,
        yaw_deg,
        translation_xy,
        translation_z,
        support_base_z,
        wall_height,
        edge,
        segment_span,
        center_offset=0.0,
    ):
        wall_support_bases.append(float(support_base_z))
        return original_grounded_wall_segment(
            part,
            yaw_deg,
            translation_xy,
            translation_z,
            support_base_z,
            wall_height,
            edge,
            segment_span,
            center_offset=center_offset,
        )

    monkeypatch.setattr(assembled_parts, "_build_ground_fill_mesh", capture_ground_fill)
    monkeypatch.setattr(assembled_parts, "_build_extended_grounded_wall_mesh", capture_grounded_wall)
    monkeypatch.setattr(assembled_parts, "_build_grounded_wall_segment", capture_grounded_wall_segment)

    terrain = build_terrain(level)

    assert terrain.metadata["common_ground"] is True
    assert len(fill_support_bases) > 0
    assert len(wall_support_bases) > 0
    assert max(fill_support_bases) == pytest.approx(min(fill_support_bases), abs=1.0e-6)
    assert max(wall_support_bases) == pytest.approx(min(wall_support_bases), abs=1.0e-6)
