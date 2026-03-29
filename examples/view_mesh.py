#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import argparse
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("results/generated_terrain")


def resolve_mesh_path(mesh_arg: str, mesh_dir: Path) -> Path:
    mesh_path = Path(mesh_arg)
    candidates = []

    if mesh_path.suffix:
        candidates.append(mesh_path)
        candidates.append(mesh_dir / mesh_path)
    else:
        candidates.append(mesh_dir / mesh_path / "mesh.obj")
        candidates.append(mesh_dir / f"{mesh_path}.obj")
        candidates.append(mesh_path / "mesh.obj")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find a mesh for '{mesh_arg}'. Tried: {', '.join(str(path) for path in candidates)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="View a generated terrain mesh.")
    parser.add_argument(
        "mesh",
        help="Mesh name or path. Examples: mesh_0, mesh_0/mesh.obj, results/generated_terrain/mesh_0/mesh.obj",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base directory containing generated meshes.",
    )
    args = parser.parse_args()

    mesh_path = resolve_mesh_path(args.mesh, args.mesh_dir)

    try:
        import trimesh
        from terrain_generator.utils import visualize_mesh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Required mesh dependencies are not installed in this Python environment. Activate the environment "
            "you use for terrain generation before running this viewer."
        ) from exc

    mesh = trimesh.load_mesh(str(mesh_path))
    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {mesh_path}")

    print(f"Viewing mesh: {mesh_path}")
    visualize_mesh(mesh)


if __name__ == "__main__":
    main()
