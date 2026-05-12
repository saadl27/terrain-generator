import json

from examples.export_terrain_category_examples import (
    CATEGORY_ORDER,
    DEFAULT_LEVEL,
    export_category_examples,
)


def test_export_category_examples_writes_raw_category_meshes(tmp_path):
    output_dir = tmp_path / "terrain_category_examples"

    manifest = export_category_examples(output_dir, level=DEFAULT_LEVEL)

    assert manifest["source_curriculum_level"] == DEFAULT_LEVEL
    assert manifest["category_count"] == len(CATEGORY_ORDER)
    assert not (output_dir / "full_curriculum").exists()

    exported_ids = [category["category_id"] for category in manifest["categories"]]
    assert exported_ids == list(CATEGORY_ORDER)

    for category_id in CATEGORY_ORDER:
        category_dir = output_dir / "categories" / category_id
        metadata = json.loads((category_dir / "metadata.json").read_text())

        assert (category_dir / "mesh.obj").is_file()
        assert metadata["level"] == DEFAULT_LEVEL
        assert metadata["category_id"] == category_id
        assert metadata["source_curriculum_level"] == DEFAULT_LEVEL
        assert metadata["has_walls"] is (category_id == "corner")
        assert metadata["exported_without_curriculum_layout"] is True
        assert metadata["curriculum_arena_walls"] is False
        assert metadata["curriculum_divider_walls"] is False
        assert metadata["curriculum_category_base_floor"] is False
        assert "allocated_cell_width" not in metadata
        assert "allocated_cell_length" not in metadata
