import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.curriculum.common import (
    CATEGORY_LABELS,
    DEFAULT_MESH_EXTENSION,
    TOTAL_CURRICULUM_LEVELS,
    TerrainScene,
    annotate_terrain,
    mesh_extents,
    validate_level,
)
from terrain_generator.trimesh_tiles.curriculum.corner import (
    build_category_terrain as build_corner_terrain,
)
from terrain_generator.trimesh_tiles.curriculum.linear_slopes import (
    build_category_terrain as build_linear_slopes_terrain,
)
from terrain_generator.trimesh_tiles.curriculum.linear_stairs import (
    build_category_terrain as build_linear_stairs_terrain,
)
from terrain_generator.trimesh_tiles.curriculum.turning_slopes import (
    build_category_terrain as build_turning_slopes_terrain,
)
from terrain_generator.trimesh_tiles.curriculum.turning_stairs import (
    build_category_terrain as build_turning_stairs_terrain,
)


DEFAULT_LEVEL = 30
DEFAULT_OUTPUT_DIR = Path("results/terrain_category_examples")
CATEGORY_ORDER = (
    "linear_stairs",
    "linear_slopes",
    "turning_stairs",
    "turning_slopes",
    "corner",
)
CATEGORY_BUILDERS = {
    "linear_stairs": build_linear_stairs_terrain,
    "linear_slopes": build_linear_slopes_terrain,
    "turning_stairs": build_turning_stairs_terrain,
    "turning_slopes": build_turning_slopes_terrain,
    "corner": build_corner_terrain,
}


def build_category_example(category_id: str, level: int) -> TerrainScene:
    validate_level(level)
    if category_id not in CATEGORY_BUILDERS:
        raise ValueError(f"Unknown category '{category_id}'.")

    include_walls = category_id == "corner"
    terrain = annotate_terrain(
        CATEGORY_BUILDERS[category_id](level, include_walls=include_walls),
        level=level,
        category_id=category_id,
    )
    terrain.metadata.update(
        {
            "source_curriculum_level": level,
            "export_style": "single_category_example",
            "exported_without_curriculum_layout": True,
            "has_walls": include_walls,
            "curriculum_arena_walls": False,
            "curriculum_divider_walls": False,
            "curriculum_category_base_floor": False,
            "mesh_extents": mesh_extents(terrain.mesh),
        }
    )
    return terrain


def export_category_examples(
    output_dir: Path,
    level: int = DEFAULT_LEVEL,
    mesh_extension: str = DEFAULT_MESH_EXTENSION,
    category_order: Sequence[str] = CATEGORY_ORDER,
) -> Dict[str, object]:
    validate_level(level)
    output_path = Path(output_dir)
    categories_path = output_path / "categories"
    categories_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_curriculum_level": level,
        "category_count": len(category_order),
        "mesh_extension": mesh_extension,
        "export_style": "single_category_examples_without_curriculum_layout",
        "categories": [],
    }

    for category_id in category_order:
        terrain = build_category_example(category_id, level)
        category_dir = categories_path / category_id
        category_dir.mkdir(parents=True, exist_ok=True)

        mesh_file = f"mesh.{mesh_extension}"
        terrain.mesh.export(category_dir / mesh_file)

        terrain_record = dict(terrain.metadata)
        terrain_record["mesh_file"] = mesh_file
        with open(category_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(terrain_record, f, indent=2, sort_keys=True)

        manifest["categories"].append(
            {
                "category_id": category_id,
                "label": CATEGORY_LABELS[category_id],
                "directory": str(Path("categories") / category_id),
                "mesh_file": mesh_file,
                "metadata_file": "metadata.json",
            }
        )

    with open(output_path / "terrain_category_examples_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export one raw terrain example per curriculum category, without curriculum cell expansion, "
            "arena walls, divider walls, or a merged curriculum layout."
        )
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"Curriculum difficulty level to sample for each category, from 1 to {TOTAL_CURRICULUM_LEVELS}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the individual category meshes and metadata will be written.",
    )
    parser.add_argument(
        "--mesh-extension",
        default=DEFAULT_MESH_EXTENSION,
        help="Mesh extension supported by trimesh export, for example obj, ply, or stl.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORY_ORDER,
        default=list(CATEGORY_ORDER),
        help="Terrain categories to export. Defaults to all slide-example categories.",
    )
    args = parser.parse_args()

    manifest = export_category_examples(
        args.output_dir,
        level=args.level,
        mesh_extension=args.mesh_extension,
        category_order=tuple(args.categories),
    )
    print(
        f"Exported {manifest['category_count']} terrain category examples from curriculum level "
        f"{manifest['source_curriculum_level']} to {args.output_dir}"
    )
    for category in manifest["categories"]:
        print(f"  {category['category_id']}: {Path(category['directory']) / category['mesh_file']}")


if __name__ == "__main__":
    main()
