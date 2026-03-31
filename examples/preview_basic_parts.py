import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import build_mesh
from terrain_generator.trimesh_tiles.mesh_parts.part_presets import (
    make_corner_cfg,
    make_platform_cfg,
    make_slope_cfg,
    make_stairs_cfg,
)


OUTPUT_DIR = Path("results/basic_parts_preview")
SHOW_MESHES = True


def export_and_optionally_show(name: str, cfg):
    mesh = build_mesh(cfg)
    output_path = OUTPUT_DIR / f"{name}.obj"
    mesh.export(output_path)
    print(f"Exported {name}: {output_path}")
    if SHOW_MESHES:
        mesh.show()


def get_preview_cfgs():
    return (
        ("stairs_demo", make_stairs_cfg(name="stairs_demo")),
        ("platform_demo", make_platform_cfg(name="platform_demo")),
        ("slope_demo", make_slope_cfg(name="slope_demo")),
        ("corner_demo", make_corner_cfg(name="corner_demo")),
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in get_preview_cfgs():
        export_and_optionally_show(name, cfg)


if __name__ == "__main__":
    main()
