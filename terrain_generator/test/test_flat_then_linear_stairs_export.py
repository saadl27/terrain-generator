import json
import sys

import numpy as np
import trimesh

from examples import export_curriculum as export_curriculum_example
from examples.export_flat_then_linear_stairs import (
    CATEGORY_ID,
    CATEGORY_LABEL,
    FLAT_INTRO_LEVELS,
    build_flat_then_linear_stairs_terrain,
)
from terrain_generator.trimesh_tiles.curriculum.common import FLOOR_THICKNESS
from terrain_generator.trimesh_tiles.curriculum_generator import (
    TOTAL_CURRICULUM_LEVELS,
    CurriculumLayoutCfg,
    export_curriculum,
)


def _upward_horizontal_faces(mesh):
    triangles = mesh.triangles
    return (mesh.face_normals[:, 2] > 0.999) & (np.ptp(triangles[:, :, 2], axis=1) < 1.0e-6)


def _horizontal_face_areas_and_z(mesh):
    horizontal = _upward_horizontal_faces(mesh)
    triangles = mesh.triangles[horizontal]
    return trimesh.triangles.area(triangles), triangles[:, :, 2].mean(axis=1)


def _horizontal_area_near_z(mesh, target_z):
    areas, z_values = _horizontal_face_areas_and_z(mesh)
    return float(areas[np.isclose(z_values, target_z, atol=1.0e-5)].sum())


def test_flat_then_linear_stairs_builder_uses_flat_intro_levels():
    first_level = build_flat_then_linear_stairs_terrain(1)
    last_flat_level = build_flat_then_linear_stairs_terrain(FLAT_INTRO_LEVELS)
    first_stair_level = build_flat_then_linear_stairs_terrain(FLAT_INTRO_LEVELS + 1)
    final_level = build_flat_then_linear_stairs_terrain(TOTAL_CURRICULUM_LEVELS)

    assert first_level.metadata["type"] == "flat"
    assert last_flat_level.metadata["type"] == "flat"
    assert first_stair_level.metadata["type"] == "linear_stairs"
    assert first_stair_level.metadata["linear_stairs_source_level"] == 1
    assert final_level.metadata["type"] == "linear_stairs"
    assert final_level.metadata["linear_stairs_source_level"] == TOTAL_CURRICULUM_LEVELS


def test_flat_then_linear_stairs_exports_curriculum_layout(tmp_path):
    output_dir = tmp_path / "flat_then_linear_stairs_export"

    manifest = export_curriculum(
        output_dir,
        levels=(1, FLAT_INTRO_LEVELS, FLAT_INTRO_LEVELS + 1),
        layout_cfg=CurriculumLayoutCfg(add_category_base_floor=False),
        category_order=(CATEGORY_ID,),
        category_builders={CATEGORY_ID: build_flat_then_linear_stairs_terrain},
        category_labels={CATEGORY_ID: CATEGORY_LABEL},
    )

    assert manifest["total_levels"] == 3
    assert manifest["category_count"] == 1
    assert (output_dir / "full_curriculum" / "mesh.obj").is_file()
    assert (output_dir / "categories" / CATEGORY_ID / "mesh.obj").is_file()
    assert (output_dir / "categories" / CATEGORY_ID / "level_01" / "mesh.obj").is_file()
    assert (output_dir / "categories" / CATEGORY_ID / f"level_{FLAT_INTRO_LEVELS + 1:02d}" / "mesh.obj").is_file()

    stair_metadata = json.loads(
        (output_dir / "categories" / CATEGORY_ID / f"level_{FLAT_INTRO_LEVELS + 1:02d}" / "metadata.json").read_text()
    )
    assert stair_metadata["type"] == "linear_stairs"
    assert stair_metadata["linear_stairs_source_level"] == 1
    assert stair_metadata["has_walls"] is False
    assert stair_metadata["arena_walls"] is False

    category_layout = json.loads((output_dir / "categories" / CATEGORY_ID / "layout.json").read_text())
    assert category_layout["add_category_base_floor"] is False

    category_mesh = trimesh.load(output_dir / "categories" / CATEGORY_ID / "mesh.obj", force="mesh", process=False)
    areas, _ = _horizontal_face_areas_and_z(category_mesh)
    row_lengths = [float(cell["y_max"]) - float(cell["y_min"]) for cell in category_layout["level_cells"]]
    full_column_floor_triangle_area = 0.5 * float(category_layout["column_width"]) * float(
        category_layout["total_height"]
    )
    largest_expected_cell_floor_triangle_area = 0.5 * float(category_layout["column_width"]) * max(row_lengths)

    assert areas.max() <= largest_expected_cell_floor_triangle_area + 1.0e-5
    assert not np.any(np.isclose(areas, full_column_floor_triangle_area, rtol=1.0e-5, atol=1.0e-5))


def test_flat_then_linear_stairs_export_has_no_stacked_flat_intro_floors(tmp_path):
    output_dir = tmp_path / "flat_intro_floor_export"

    export_curriculum(
        output_dir,
        levels=(1, FLAT_INTRO_LEVELS),
        layout_cfg=CurriculumLayoutCfg(add_category_base_floor=False),
        category_order=(CATEGORY_ID,),
        category_builders={CATEGORY_ID: build_flat_then_linear_stairs_terrain},
        category_labels={CATEGORY_ID: CATEGORY_LABEL},
    )

    category_mesh = trimesh.load(output_dir / "categories" / CATEGORY_ID / "mesh.obj", force="mesh", process=False)

    assert _horizontal_area_near_z(category_mesh, FLOOR_THICKNESS) > 0.0
    assert _horizontal_area_near_z(category_mesh, 2.0 * FLOOR_THICKNESS) == 0.0


def test_export_curriculum_cli_can_disable_category_base_floor(monkeypatch, tmp_path):
    captured = {}

    def fake_export_curriculum(output_dir, levels=None, mesh_extension="obj", layout_cfg=None):
        captured["output_dir"] = output_dir
        captured["levels"] = tuple(levels)
        captured["mesh_extension"] = mesh_extension
        captured["layout_cfg"] = layout_cfg
        return {"total_levels": len(captured["levels"]), "category_count": 0}

    monkeypatch.setattr(export_curriculum_example, "export_curriculum", fake_export_curriculum)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_curriculum.py",
            "--start-level",
            "2",
            "--end-level",
            "2",
            "--output-dir",
            str(tmp_path),
            "--no-category-base-floor",
        ],
    )

    export_curriculum_example.main()

    assert captured["levels"] == (2,)
    assert captured["layout_cfg"].add_category_base_floor is False
