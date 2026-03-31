#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import trimesh
import numpy as np
from ...utils import (
    merge_meshes,
    yaw_rotate_mesh,
    get_height_array_of_mesh,
)
from .mesh_parts_cfg import (
    StairMeshPartsCfg,
)
from .basic_parts import create_floor, create_standard_wall


def create_standard_stairs(cfg: StairMeshPartsCfg.Stair):
    """Create a standard stair with a given number of steps and a given height. This will fill bottom."""
    n_steps = cfg.n_steps
    step_height = cfg.total_height / n_steps
    step_depth = cfg.step_depth
    residual_depth = cfg.dim[1] - n_steps * step_depth
    mesh = trimesh.Trimesh()
    stair_start_pos = np.array([0.0, -cfg.dim[1] / 2.0, -cfg.dim[2] / 2.0])
    current_pos = stair_start_pos
    if cfg.add_residual_side_up is False:
        dims = np.array([cfg.step_width, residual_depth, cfg.height_offset])
        current_pos += np.array([0.0, dims[1], 0.0])
        if cfg.height_offset != 0.0:
            pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
            step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
            mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    for n in range(n_steps):
        dims = [cfg.step_width, cfg.step_depth, (n + 1) * step_height + cfg.height_offset]
        pos = current_pos + np.array([0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        current_pos += np.array([0.0, dims[1], 0.0])
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    if cfg.add_residual_side_up is True:
        dims = np.array([cfg.step_width, residual_depth, n_steps * step_height + cfg.height_offset])
        pos = current_pos + np.array([0.0, dims[1] / 2.0, dims[2] / 2.0])
        step = trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
        mesh = merge_meshes([mesh, step], cfg.minimal_triangles)
    return mesh


def create_stairs(cfg: StairMeshPartsCfg.Stair):
    if cfg.stair_type == "standard":
        mesh = create_standard_stairs(cfg)
    else:
        raise NotImplementedError(f"stair_type '{cfg.stair_type}' is not implemented.")

    dim = np.array([cfg.step_width, cfg.dim[1], cfg.total_height])
    if cfg.direction == "front":
        mesh = mesh
    elif cfg.direction == "left":
        mesh = yaw_rotate_mesh(mesh, 90)
        dim = dim[np.array([1, 0, 2])]
    elif cfg.direction == "back":
        mesh = yaw_rotate_mesh(mesh, 180)
    elif cfg.direction == "right":
        mesh = yaw_rotate_mesh(mesh, 270)
        dim = dim[np.array([1, 0, 2])]

    if "left" in cfg.attach_side:
        mesh.apply_translation([-cfg.dim[0] / 2.0 + dim[0] / 2.0, 0, 0])
    if "right" in cfg.attach_side:
        mesh.apply_translation([cfg.dim[0] / 2.0 - dim[0] / 2.0, 0, 0])
    if "front" in cfg.attach_side:
        mesh.apply_translation([0, cfg.dim[1] / 2.0 - dim[1] / 2.0, 0])
    if "back" in cfg.attach_side:
        mesh.apply_translation([0, -cfg.dim[1] / 2.0 + dim[1] / 2.0, 0])
    return mesh


def create_stairs_mesh(cfg: StairMeshPartsCfg):
    mesh = create_floor(cfg)
    for stair in cfg.stairs:
        stairs = create_stairs(stair)
        mesh = merge_meshes([mesh, stairs], cfg.minimal_triangles)
    if cfg.wall is not None:
        for wall_edges in cfg.wall.wall_edges:
            wall = create_standard_wall(cfg.wall, wall_edges)
            mesh = merge_meshes([mesh, wall], cfg.minimal_triangles)

    return mesh


if __name__ == "__main__":

    # cfg = WallMeshPartsCfg(wall_edges=("left", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    # #
    # cfg = WallMeshPartsCfg(wall_edges=("up", ))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    #
    # cfg = WallMeshPartsCfg(wall_edges=("right", "bottom"))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()

    # cfg = WallMeshPartsCfg(wall_edges=("bottom_right", "right_bottom"))
    # mesh = create_wall_mesh(cfg)
    # mesh.show()
    #
    # for i in range(10):
    #     cfg = WallMeshPartsCfg(wall_edges=("middle_right", "middle_left"), door_direction="up")
    #     mesh = create_wall_mesh(cfg)
    #     mesh.show()

    # cfg = StairMeshPartsCfg()
    # mesh = create_stairs(cfg)
    # mesh.show()
    #
    # cfg = StairMeshPartsCfg()
    # mesh = create_stairs(cfg.stairs[0])
    # mesh.show()

    # stair_straight = StairMeshPartsCfg(
    #     name="stair_s",
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    #     stairs=(
    #         StairMeshPartsCfg.Stair(
    #             step_width=2.0,
    #             # step_height=0.15,
    #             step_depth=0.3,
    #             total_height=1.0,
    #             stair_type="standard",
    #             direction="up",
    #             add_residual_side_up=True,
    #             attach_side="front",
    #             add_rail=False,
    #         ),
    #     ),
    #     # wall=WallMeshPartsCfg(
    #     #     name="wall",
    #     #     wall_edges=("middle_left", "middle_right"),
    #     #     )
    # )
    stair_wide = StairMeshPartsCfg(
        name="stair_w",
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=2.0,
                step_depth=0.3,
                total_height=1.0,
                stair_type="standard",
                direction="up",
                add_residual_side_up=False,
                attach_side="front",
                add_rail=False,
            ),
        ),
    )
    # from mesh_parts.mesh_parts_cfg import StairPattern
    # pattern = StairPattern(name="stairs")
    mesh = create_stairs_mesh(stair_wide)
    mesh.show()
    print(get_height_array_of_mesh(mesh, stair_wide.dim, 5))

    stair_straight = StairMeshPartsCfg(
        name="stair_s",
        rotations=(90, 180, 270),
        flips=(),
        weight=0.1,
        stairs=(
            StairMeshPartsCfg.Stair(
                step_width=1.0,
                # step_height=0.15,
                step_depth=0.3,
                total_height=1.0,
                height_offset=1.0,
                stair_type="standard",
                direction="up",
                add_residual_side_up=True,
                attach_side="front_right",
                add_rail=False,
            ),
        ),
    )
    mesh = create_stairs_mesh(stair_straight)
    mesh.show()
    print(get_height_array_of_mesh(mesh, stair_straight.dim, 5))
    #
    # stair_straight = StairMeshPartsCfg(
    #     name="stair_s",
    #     rotations=(90, 180, 270),
    #     flips=(),
    #     weight=0.1,
    #     stairs=(
    #         StairMeshPartsCfg.Stair(
    #             step_width=1.0,
    #             # step_height=0.15,
    #             step_depth=0.3,
    #             total_height=1.0,
    #             stair_type="standard",
    #             direction="up",
    #             gap_direction="up",
    #             add_residual_side_up=True,
    #             height_offset=1.0,
    #             attach_side="front_right",
    #             add_rail=False,
    #             fill_bottom=True,
    #         ),
    #     ),
    # )
    # # from mesh_parts.mesh_parts_cfg import StairPattern
    # # pattern = StairPattern(name="stairs")
    # mesh = create_stairs_mesh(stair_straight)
    # mesh.show()
