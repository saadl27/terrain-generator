import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.curriculum_generator import (
    TOTAL_CURRICULUM_LEVELS,
    CurriculumLayoutCfg,
    export_curriculum,
)


DEFAULT_OUTPUT_DIR = Path("results/curriculum_export")


def main():
    parser = argparse.ArgumentParser(description="Export the hand-authored terrain curriculum.")
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
        default=1.2,
        help="Height in meters for the walls delimiting curriculum cells.",
    )
    parser.add_argument(
        "--divider-wall-thickness",
        type=float,
        default=0.2,
        help="Thickness in meters for the walls delimiting curriculum cells.",
    )
    parser.add_argument(
        "--row-gap",
        type=float,
        default=1.5,
        help="Gap in meters between consecutive levels inside each subterrain category column.",
    )
    parser.add_argument(
        "--category-gap",
        type=float,
        default=2.0,
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
    )
    manifest = export_curriculum(
        args.output_dir,
        levels=range(args.start_level, args.end_level + 1),
        layout_cfg=layout_cfg,
    )
    print(
        f"Exported {manifest['total_levels']} levels across {manifest['category_count']} subterrain categories, "
        f"with individual subterrain meshes, per-category curriculum meshes, and the full merged curriculum to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
