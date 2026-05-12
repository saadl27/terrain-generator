import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.curriculum.common import build_flat_terrain
from terrain_generator.trimesh_tiles.curriculum.linear_stairs import (
    build_category_terrain as build_linear_stairs_terrain,
)
from terrain_generator.trimesh_tiles.curriculum_generator import (
    TOTAL_CURRICULUM_LEVELS,
    CurriculumLayoutCfg,
    TerrainScene,
    export_curriculum,
)


DEFAULT_OUTPUT_DIR = Path("results/flat_then_linear_stairs_export")
FLAT_INTRO_LEVELS = 5
CATEGORY_ID = "linear_stairs"
CATEGORY_LABEL = "Flat Then Linear Stairs"


def _linear_stairs_source_level(level: int) -> int:
    first_stair_level = FLAT_INTRO_LEVELS + 1
    stair_level_count = TOTAL_CURRICULUM_LEVELS - FLAT_INTRO_LEVELS
    if stair_level_count <= 1:
        return TOTAL_CURRICULUM_LEVELS
    ratio = float(level - first_stair_level) / float(stair_level_count - 1)
    source_level = 1 + round(ratio * float(TOTAL_CURRICULUM_LEVELS - 1))
    return max(1, min(TOTAL_CURRICULUM_LEVELS, int(source_level)))


def build_flat_then_linear_stairs_terrain(level: int) -> TerrainScene:
    if level <= FLAT_INTRO_LEVELS:
        label = f"Level {level} flat intro terrain"
        terrain = build_flat_terrain(CATEGORY_ID, label, width=14.0, length=14.0)
        terrain.metadata.update(
            {
                "label": label,
                "flat_intro_levels": FLAT_INTRO_LEVELS,
                "curriculum_type": "flat_then_linear_stairs",
                "has_walls": False,
                "arena_walls": False,
            }
        )
        return terrain

    source_level = _linear_stairs_source_level(level)
    terrain = build_linear_stairs_terrain(source_level, include_walls=False)
    label = f"Level {level} linear stairs"
    terrain.label = label
    terrain.metadata.update(
        {
            "label": label,
            "flat_intro_levels": FLAT_INTRO_LEVELS,
            "curriculum_type": "flat_then_linear_stairs",
            "linear_stairs_source_level": source_level,
            "has_walls": False,
            "arena_walls": False,
        }
    )
    return terrain


def main():
    parser = argparse.ArgumentParser(
        description="Export a 38-level curriculum with five flat intro levels followed by linear stairs."
    )
    parser.add_argument("--start-level", type=int, default=1, help="First curriculum level to export.")
    parser.add_argument(
        "--end-level",
        type=int,
        default=TOTAL_CURRICULUM_LEVELS,
        help="Last curriculum level to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where meshes and metadata will be written.",
    )
    parser.add_argument(
        "--min-cell-width",
        type=float,
        default=12.0,
        help="Minimum width in meters for each exported subterrain cell.",
    )
    parser.add_argument(
        "--min-cell-length",
        type=float,
        default=12.0,
        help="Minimum length in meters for each exported subterrain cell.",
    )
    parser.add_argument(
        "--terrain-padding",
        type=float,
        default=0.0,
        help="Padding in meters around each terrain before it is laid out in the curriculum.",
    )
    parser.add_argument(
        "--divider-wall-height",
        type=float,
        default=0.0,
        help="Height in meters for the walls delimiting curriculum cells. Defaults to 0 for wall-free exports.",
    )
    parser.add_argument(
        "--divider-wall-thickness",
        type=float,
        default=0.0,
        help="Thickness in meters for the walls delimiting curriculum cells. Defaults to 0 for wall-free exports.",
    )
    parser.add_argument(
        "--row-gap",
        type=float,
        default=0.0,
        help="Gap in meters between consecutive levels inside each subterrain category column.",
    )
    parser.add_argument(
        "--category-gap",
        type=float,
        default=0.0,
        help="Gap in meters between merged subterrain categories in the full curriculum mesh.",
    )
    args = parser.parse_args()

    if args.start_level > args.end_level:
        raise ValueError("start-level must be less than or equal to end-level.")

    layout_cfg = CurriculumLayoutCfg(
        min_cell_width=args.min_cell_width,
        min_cell_length=args.min_cell_length,
        terrain_padding_x=args.terrain_padding,
        terrain_padding_y=args.terrain_padding,
        divider_wall_height=args.divider_wall_height,
        divider_wall_thickness=args.divider_wall_thickness,
        row_gap=args.row_gap,
        category_gap=args.category_gap,
        add_category_base_floor=False,
    )
    manifest = export_curriculum(
        args.output_dir,
        levels=range(args.start_level, args.end_level + 1),
        layout_cfg=layout_cfg,
        category_order=(CATEGORY_ID,),
        category_builders={CATEGORY_ID: build_flat_then_linear_stairs_terrain},
        category_labels={CATEGORY_ID: CATEGORY_LABEL},
    )
    print(
        f"Exported {manifest['total_levels']} levels for {CATEGORY_LABEL}, with individual meshes, the "
        f"per-category curriculum mesh, and the full merged curriculum to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
