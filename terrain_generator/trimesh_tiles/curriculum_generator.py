import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import trimesh

from ..utils import merge_meshes
from .curriculum.common import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    DEFAULT_MESH_EXTENSION,
    FLOOR_THICKNESS,
    TOTAL_CURRICULUM_LEVELS,
    CurriculumCategory,
    CurriculumLayoutCfg,
    CurriculumLevel,
    TerrainScene,
    annotate_terrain,
    box_mesh,
    expand_terrain_to_cell,
    expand_terrains_to_layout,
    mesh_extents,
    normalize_ground_center,
    normalize_levels,
    round_float,
    validate_level,
)
from .curriculum.corner import build_category_terrain as build_corner_category_terrain
from .curriculum.corner import build_corner_terrain as _build_corner_terrain
from .curriculum.flat import build_category_terrain as build_flat_category_terrain
from .curriculum.linear_slopes import build_category_terrain as build_linear_slopes_category_terrain
from .curriculum.linear_stairs import build_category_terrain as build_linear_stairs_category_terrain
from .curriculum.turning_slopes import build_category_terrain as build_turning_slopes_category_terrain
from .curriculum.turning_stairs import build_category_terrain as build_turning_stairs_category_terrain


_CATEGORY_BUILDERS = {
    "flat": build_flat_category_terrain,
    "linear_stairs": build_linear_stairs_category_terrain,
    "linear_slopes": build_linear_slopes_category_terrain,
    "corner": build_corner_category_terrain,
    "turning_stairs": build_turning_stairs_category_terrain,
    "turning_slopes": build_turning_slopes_category_terrain,
}


def _build_level_terrain_map(level: int) -> Dict[str, TerrainScene]:
    validate_level(level)
    terrain_map = {}
    for category_id in CATEGORY_ORDER:
        terrain = _CATEGORY_BUILDERS[category_id](level)
        terrain_map[category_id] = annotate_terrain(terrain, level=level, category_id=category_id)
    return terrain_map


def _build_level_terrains(level: int) -> Tuple[TerrainScene, ...]:
    terrain_map = _build_level_terrain_map(level)
    return tuple(terrain_map[category_id] for category_id in CATEGORY_ORDER)


def _assemble_level_mesh(
    level: int,
    terrains: Tuple[TerrainScene, ...],
    layout_cfg: CurriculumLayoutCfg,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    column_widths: List[float] = []
    max_length = 0.0
    for terrain in terrains:
        cell_width = float(terrain.metadata.get("allocated_cell_width", terrain.metadata["mesh_extents"]["x"]))
        cell_length = float(terrain.metadata.get("allocated_cell_length", terrain.metadata["mesh_extents"]["y"]))
        column_widths.append(cell_width)
        max_length = max(max_length, cell_length)

    row_length = max(layout_cfg.min_cell_length, max_length)
    total_width = sum(column_widths)
    meshes = [
        box_mesh(
            total_width,
            row_length,
            FLOOR_THICKNESS,
            (0.5 * total_width, 0.0, -0.5 * FLOOR_THICKNESS),
        )
    ]

    terrain_cells = []
    cursor_x = 0.0
    for idx, terrain in enumerate(terrains):
        width = column_widths[idx]
        cell_center_x = cursor_x + 0.5 * width
        terrain_mesh = terrain.mesh.copy()
        terrain_mesh.apply_translation([cell_center_x, 0.0, 0.0])
        meshes.append(terrain_mesh)
        terrain_cells.append(
            {
                "terrain_id": terrain.terrain_id,
                "label": terrain.label,
                "column_index": idx,
                "x_min": round_float(cursor_x),
                "x_max": round_float(cursor_x + width),
                "y_min": round_float(-0.5 * row_length),
                "y_max": round_float(0.5 * row_length),
                "mesh_extents": dict(terrain.metadata["mesh_extents"]),
            }
        )
        cursor_x += width

    boundaries = [0.0]
    running = 0.0
    for width in column_widths:
        running += width
        boundaries.append(running)

    for boundary_x in boundaries:
        meshes.append(
            box_mesh(
                layout_cfg.divider_wall_thickness,
                row_length,
                layout_cfg.divider_wall_height,
                (boundary_x, 0.0, 0.5 * layout_cfg.divider_wall_height),
            )
        )
    for boundary_y in (-0.5 * row_length, 0.5 * row_length):
        meshes.append(
            box_mesh(
                total_width,
                layout_cfg.divider_wall_thickness,
                layout_cfg.divider_wall_height,
                (0.5 * total_width, boundary_y, 0.5 * layout_cfg.divider_wall_height),
            )
        )

    level_mesh = merge_meshes(meshes, False)
    level_mesh = normalize_ground_center(level_mesh)

    if layout_cfg.center_rows_on_origin:
        offset_x = 0.5 * total_width
        for cell in terrain_cells:
            cell["x_min"] = round_float(cell["x_min"] - offset_x)
            cell["x_max"] = round_float(cell["x_max"] - offset_x)

    metadata = {
        "level": level,
        "terrain_count": len(terrains),
        "row_width": round_float(total_width),
        "row_length": round_float(row_length),
        "terrain_cells": terrain_cells,
        "mesh_extents": mesh_extents(level_mesh),
    }
    return level_mesh, metadata


def build_curriculum_level(level: int, layout_cfg: Optional[CurriculumLayoutCfg] = None) -> CurriculumLevel:
    validate_level(level)
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    terrains = expand_terrains_to_layout(_build_level_terrains(level), layout_cfg)
    level_mesh, level_layout = _assemble_level_mesh(level, terrains, layout_cfg)
    metadata = {
        "level": level,
        "terrain_labels": [terrain.label for terrain in terrains],
        "terrain_ids": [terrain.terrain_id for terrain in terrains],
        "level_layout": level_layout,
    }
    return CurriculumLevel(level=level, terrains=terrains, level_mesh=level_mesh, metadata=metadata)


def build_curriculum(
    levels: Optional[Iterable[int]] = None,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Tuple[CurriculumLevel, ...]:
    level_numbers = normalize_levels(levels)
    return tuple(build_curriculum_level(level, layout_cfg=layout_cfg) for level in level_numbers)


def _compute_category_widths(
    raw_terrain_map: Dict[int, Dict[str, TerrainScene]],
    level_numbers: List[int],
    layout_cfg: CurriculumLayoutCfg,
) -> Dict[str, float]:
    widths: Dict[str, float] = {category_id: layout_cfg.min_cell_width for category_id in CATEGORY_ORDER}
    for category_id in CATEGORY_ORDER:
        for level in level_numbers:
            extents = raw_terrain_map[level][category_id].metadata["mesh_extents"]
            widths[category_id] = max(
                widths[category_id],
                float(extents["x"]) + 2.0 * layout_cfg.terrain_padding_x,
            )
    return widths


def _compute_level_lengths(
    raw_terrain_map: Dict[int, Dict[str, TerrainScene]],
    level_numbers: List[int],
    layout_cfg: CurriculumLayoutCfg,
    category_widths: Dict[str, float],
) -> Dict[int, float]:
    shared_row_length = layout_cfg.min_cell_length
    for level in level_numbers:
        for category_id in CATEGORY_ORDER:
            extents = raw_terrain_map[level][category_id].metadata["mesh_extents"]
            shared_row_length = max(
                shared_row_length,
                float(extents["y"]) + 2.0 * layout_cfg.terrain_padding_y,
            )
            if category_id == "corner":
                shared_row_length = max(shared_row_length, category_widths[category_id])
    return {level: shared_row_length for level in level_numbers}


def _assemble_category_mesh(
    category_id: str,
    terrains: Tuple[TerrainScene, ...],
    level_numbers: List[int],
    column_width: float,
    level_lengths: Dict[int, float],
    layout_cfg: CurriculumLayoutCfg,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    total_height = sum(level_lengths[level] for level in level_numbers)
    total_height += layout_cfg.row_gap * max(len(level_numbers) - 1, 0)

    meshes = [
        box_mesh(
            column_width,
            total_height,
            FLOOR_THICKNESS,
            (0.0, 0.0, -0.5 * FLOOR_THICKNESS),
        )
    ]

    level_cells = []
    row_boundaries = [0.5 * total_height]
    current_top = 0.5 * total_height
    for idx, terrain in enumerate(terrains):
        level = int(terrain.metadata["level"])
        row_length = level_lengths[level]
        row_bottom = current_top - row_length
        row_center_y = 0.5 * (current_top + row_bottom)

        terrain_mesh = terrain.mesh.copy()
        terrain_mesh.apply_translation([0.0, row_center_y, 0.0])
        meshes.append(terrain_mesh)

        level_cells.append(
            {
                "level": level,
                "row_index": idx,
                "terrain_id": terrain.terrain_id,
                "label": terrain.label,
                "x_min": round_float(-0.5 * column_width),
                "x_max": round_float(0.5 * column_width),
                "y_min": round_float(row_bottom),
                "y_max": round_float(current_top),
                "mesh_extents": dict(terrain.metadata["mesh_extents"]),
            }
        )

        row_boundaries.append(row_bottom)
        current_top = row_bottom - layout_cfg.row_gap

    for boundary_x in (-0.5 * column_width, 0.5 * column_width):
        meshes.append(
            box_mesh(
                layout_cfg.divider_wall_thickness,
                total_height,
                layout_cfg.divider_wall_height,
                (boundary_x, 0.0, 0.5 * layout_cfg.divider_wall_height),
            )
        )

    for boundary_y in row_boundaries:
        meshes.append(
            box_mesh(
                column_width,
                layout_cfg.divider_wall_thickness,
                layout_cfg.divider_wall_height,
                (0.0, boundary_y, 0.5 * layout_cfg.divider_wall_height),
            )
        )

    category_mesh = merge_meshes(meshes, False)
    category_mesh = normalize_ground_center(category_mesh)

    metadata = {
        "category_id": category_id,
        "label": CATEGORY_LABELS[category_id],
        "level_count": len(level_numbers),
        "column_width": round_float(column_width),
        "total_height": round_float(total_height),
        "level_cells": level_cells,
        "mesh_extents": mesh_extents(category_mesh),
    }
    return category_mesh, metadata


def build_curriculum_categories(
    levels: Optional[Iterable[int]] = None,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Tuple[CurriculumCategory, ...]:
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    level_numbers = normalize_levels(levels)

    raw_terrain_map = {level: _build_level_terrain_map(level) for level in level_numbers}
    category_widths = _compute_category_widths(raw_terrain_map, level_numbers, layout_cfg)
    level_lengths = _compute_level_lengths(raw_terrain_map, level_numbers, layout_cfg, category_widths)

    curriculum_categories = []
    for category_id in CATEGORY_ORDER:
        expanded_terrains = tuple(
            expand_terrain_to_cell(
                raw_terrain_map[level][category_id],
                category_widths[category_id],
                level_lengths[level],
            )
            for level in level_numbers
        )
        category_mesh, category_layout = _assemble_category_mesh(
            category_id,
            expanded_terrains,
            level_numbers,
            category_widths[category_id],
            level_lengths,
            layout_cfg,
        )
        curriculum_categories.append(
            CurriculumCategory(
                category_id=category_id,
                label=CATEGORY_LABELS[category_id],
                terrains=expanded_terrains,
                category_mesh=category_mesh,
                metadata={
                    "category_id": category_id,
                    "label": CATEGORY_LABELS[category_id],
                    "levels": level_numbers,
                    "category_layout": category_layout,
                },
            )
        )
    return tuple(curriculum_categories)


def build_full_curriculum_mesh(
    levels: Optional[Iterable[int]] = None,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Tuple[trimesh.Trimesh, Dict[str, object], Tuple[CurriculumCategory, ...]]:
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    curriculum_categories = build_curriculum_categories(levels=levels, layout_cfg=layout_cfg)

    total_width = sum(float(category.metadata["category_layout"]["column_width"]) for category in curriculum_categories)
    total_width += layout_cfg.category_gap * max(len(curriculum_categories) - 1, 0)

    meshes = []
    category_columns = []
    current_left = -0.5 * total_width
    for idx, category in enumerate(curriculum_categories):
        column_width = float(category.metadata["category_layout"]["column_width"])
        column_center_x = current_left + 0.5 * column_width
        category_mesh = category.category_mesh.copy()
        category_mesh.apply_translation([column_center_x, 0.0, 0.0])
        meshes.append(category_mesh)
        category_columns.append(
            {
                "category_id": category.category_id,
                "label": category.label,
                "column_index": idx,
                "x_min": round_float(current_left),
                "x_max": round_float(current_left + column_width),
                "mesh_extents": dict(category.metadata["category_layout"]["mesh_extents"]),
            }
        )
        current_left += column_width + layout_cfg.category_gap

    full_mesh = merge_meshes(meshes, False)
    full_mesh = normalize_ground_center(full_mesh)

    metadata = {
        "total_levels": len(curriculum_categories[0].terrains) if len(curriculum_categories) > 0 else 0,
        "category_count": len(curriculum_categories),
        "category_gap": round_float(layout_cfg.category_gap),
        "category_columns": category_columns,
        "mesh_extents": mesh_extents(full_mesh),
        "layout_cfg": asdict(layout_cfg),
    }
    return full_mesh, metadata, curriculum_categories


def export_curriculum(
    output_dir: Path,
    levels: Optional[Iterable[int]] = None,
    mesh_extension: str = DEFAULT_MESH_EXTENSION,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Dict[str, object]:
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_mesh, full_layout, curriculum_categories = build_full_curriculum_mesh(levels=levels, layout_cfg=layout_cfg)

    full_dir = output_path / "full_curriculum"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_mesh_file = f"mesh.{mesh_extension}"
    full_mesh.export(full_dir / full_mesh_file)
    with open(full_dir / "layout.json", "w", encoding="utf-8") as f:
        json.dump(full_layout, f, indent=2, sort_keys=True)

    categories_dir = output_path / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "total_levels": full_layout["total_levels"],
        "category_count": len(curriculum_categories),
        "mesh_extension": mesh_extension,
        "full_curriculum_directory": full_dir.name,
        "full_curriculum_mesh_file": full_mesh_file,
        "categories": [],
    }

    for category in curriculum_categories:
        category_dir = categories_dir / category.category_id
        category_dir.mkdir(parents=True, exist_ok=True)

        category_mesh_file = f"mesh.{mesh_extension}"
        category.category_mesh.export(category_dir / category_mesh_file)
        with open(category_dir / "layout.json", "w", encoding="utf-8") as f:
            json.dump(category.metadata["category_layout"], f, indent=2, sort_keys=True)

        level_records = []
        for terrain in category.terrains:
            level = int(terrain.metadata["level"])
            level_dir = category_dir / f"level_{level:02d}"
            level_dir.mkdir(parents=True, exist_ok=True)
            terrain_mesh_file = f"mesh.{mesh_extension}"
            terrain.mesh.export(level_dir / terrain_mesh_file)
            terrain_record = dict(terrain.metadata)
            terrain_record["mesh_file"] = terrain_mesh_file
            with open(level_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(terrain_record, f, indent=2, sort_keys=True)
            level_records.append(
                {
                    "level": level,
                    "directory": level_dir.name,
                    "mesh_file": terrain_mesh_file,
                    "label": terrain.label,
                }
            )

        category_record = {
            "category_id": category.category_id,
            "label": category.label,
            "directory": str(Path("categories") / category.category_id),
            "category_mesh_file": category_mesh_file,
            "level_count": len(level_records),
            "levels": level_records,
        }
        with open(category_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(category_record, f, indent=2, sort_keys=True)
        manifest["categories"].append(category_record)

    with open(output_path / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_ORDER",
    "DEFAULT_MESH_EXTENSION",
    "TOTAL_CURRICULUM_LEVELS",
    "CurriculumCategory",
    "CurriculumLayoutCfg",
    "CurriculumLevel",
    "TerrainScene",
    "_build_corner_terrain",
    "build_curriculum",
    "build_curriculum_categories",
    "build_curriculum_level",
    "build_full_curriculum_mesh",
    "export_curriculum",
]
