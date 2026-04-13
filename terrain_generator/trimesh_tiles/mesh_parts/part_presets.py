#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np

from .mesh_parts_cfg import (
    CornerMeshPartsCfg,
    PlatformMeshPartsCfg,
    SlopeMeshPartsCfg,
    StairMeshPartsCfg,
    WallPartsCfg,
)


def make_wall_cfg(
    *,
    dim,
    floor_thickness,
    wall_thickness,
    wall_height,
    wall_edges,
    minimal_triangles=False,
    load_from_cache=False,
    height_offset=0.0,
    wall_x_offset=0.0,
    wall_y_offset=0.0,
    wall_z_offset=0.0,
):
    return WallPartsCfg(
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        wall_edges=wall_edges,
        wall_x_offset=wall_x_offset,
        wall_y_offset=wall_y_offset,
        wall_z_offset=wall_z_offset,
        height_offset=height_offset,
        load_from_cache=load_from_cache,
    )


def make_stairs_cfg(
    name="stairs_demo",
    direction="front",
    corridor_width=1.2,
    wall_thickness=0.12,
    floor_thickness=0.1,
    num_steps=5,
    step_height=0.15,
    step_depth=0.32,
    minimal_triangles=False,
    load_from_cache=False,
    stair_type="standard",
    attach_side="",
    add_residual_side_up=False,
    height_offset=0.0,
    wall_edges=("left", "right"),
    wall_x_offset=0.0,
    wall_y_offset=0.0,
    wall_z_offset=0.0,
    wall_height=5,
):
    structure_height = num_steps * step_height + height_offset
    wall_height = structure_height if wall_height is None else wall_height

    tile_width = corridor_width + 2.0 * wall_thickness
    tile_length = num_steps * step_depth
    dim = (tile_width, tile_length, structure_height)

    return StairMeshPartsCfg(
        name=name,
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        load_from_cache=load_from_cache,
        stairs=(
            StairMeshPartsCfg.Stair(
                dim=dim,
                floor_thickness=floor_thickness,
                minimal_triangles=minimal_triangles,
                step_width=corridor_width,
                step_depth=step_depth,
                n_steps=num_steps,
                total_height=num_steps * step_height,
                height_offset=height_offset,
                stair_type=stair_type,
                direction=direction,
                attach_side=attach_side,
                add_residual_side_up=add_residual_side_up,
            ),
        ),
        wall=make_wall_cfg(
            dim=dim,
            floor_thickness=floor_thickness,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_edges=wall_edges,
            minimal_triangles=minimal_triangles,
            load_from_cache=load_from_cache,
            height_offset=height_offset,
            wall_x_offset=wall_x_offset,
            wall_y_offset=wall_y_offset,
            wall_z_offset=wall_z_offset,
        ),
    )


def make_platform_cfg(
    name="platform_demo",
    width=1.44,
    length=1.44,
    height=0.75,
    floor_thickness=0.1,
    wall_thickness=0.12,
    wall_height=0.0,
    wall_edges=("left", "right", "bottom", "up"),
    wall_x_offset=0.0,
    wall_y_offset=0.0,
    wall_z_offset=0.0,
    minimal_triangles=False,
    load_from_cache=False,
):
    dim = (width, length, height)
    return PlatformMeshPartsCfg(
        name=name,
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        add_floor=False,
        array=np.array([[height]]),
        wall=make_wall_cfg(
            dim=dim,
            floor_thickness=floor_thickness,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_edges=wall_edges,
            minimal_triangles=minimal_triangles,
            load_from_cache=load_from_cache,
            wall_x_offset=wall_x_offset,
            wall_y_offset=wall_y_offset,
            wall_z_offset=height + wall_z_offset,
        ),
        load_from_cache=load_from_cache,
    )


def _turn_wall_edges(turn_direction):
    if turn_direction == "left":
        return ("up", "right")
    if turn_direction == "right":
        return ("up", "left")
    raise ValueError("turn_direction must be 'left' or 'right'.")


def make_platform_for_stage(
    stage_cfg,
    *,
    name,
    turn_direction=None,
    floor_thickness=0.1,
    wall_thickness=0.12,
    wall_height=None,
    minimal_triangles=False,
    load_from_cache=False,
):
    if isinstance(stage_cfg, StairMeshPartsCfg):
        if len(stage_cfg.stairs) == 0:
            raise ValueError("StairMeshPartsCfg must contain at least one stair.")
        stage_rise = stage_cfg.stairs[0].total_height
    elif isinstance(stage_cfg, SlopeMeshPartsCfg):
        stage_rise = np.tan(np.deg2rad(stage_cfg.slope_angle_deg)) * stage_cfg.slope_length
    else:
        raise ValueError("stage_cfg must be a StairMeshPartsCfg or SlopeMeshPartsCfg.")

    if wall_height is None:
        wall_height = stage_rise

    wall_edges = ("left", "right") if turn_direction is None else _turn_wall_edges(turn_direction)
    return make_platform_cfg(
        name=name,
        width=stage_cfg.dim[0],
        length=stage_cfg.dim[0],
        height=stage_rise,
        floor_thickness=floor_thickness,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        wall_edges=wall_edges,
        minimal_triangles=minimal_triangles,
        load_from_cache=load_from_cache,
    )


def make_slope_cfg(
    name="slope_demo",
    corridor_width=5.0,
    wall_thickness=0.12,
    floor_thickness=0.1,
    structure_height=2.0,
    slope_length=3.0,
    slope_angle_deg=45.0,
    slope_resolution=24,
    minimal_triangles=False,
    load_from_cache=False,
    fill_borders=True,
    slope_threshold=4.0,
    simplify=False,
    wall_height=None,
):
    slope_height = np.tan(np.deg2rad(slope_angle_deg)) * slope_length
    wall_height = floor_thickness + slope_height if wall_height is None else wall_height
    outer_width = corridor_width + 2.0 * wall_thickness
    dim = (outer_width, slope_length, max(structure_height, wall_height))

    return SlopeMeshPartsCfg(
        name=name,
        dim=dim,
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        corridor_width=corridor_width,
        slope_length=slope_length,
        slope_angle_deg=slope_angle_deg,
        slope_resolution=slope_resolution,
        fill_borders=fill_borders,
        slope_threshold=slope_threshold,
        simplify=simplify,
        wall=make_wall_cfg(
            dim=dim,
            floor_thickness=floor_thickness,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_edges=("left", "right"),
            minimal_triangles=minimal_triangles,
            load_from_cache=load_from_cache,
        ),
        load_from_cache=load_from_cache,
    )


def make_corner_cfg(
    name="corner_demo",
    corridor_width=2.0,
    pre_corridor_width=2.0,
    post_corridor_width=1.0,
    wall_thickness=0.12,
    wall_height=1.0,
    floor_thickness=0.1,
    structure_height=2.0,
    pre_length=2.0,
    post_length=2.0,
    turn_angle_deg=90.0,
    minimal_triangles=False,
    load_from_cache=False,
):
    base_width = max(corridor_width, pre_corridor_width, post_corridor_width)

    return CornerMeshPartsCfg(
        name=name,
        dim=(base_width + 2.0 * wall_thickness, pre_length + post_length, structure_height),
        floor_thickness=floor_thickness,
        minimal_triangles=minimal_triangles,
        corridor_width=corridor_width,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=turn_angle_deg,
        wall_thickness=wall_thickness,
        wall_height=wall_height,
        load_from_cache=load_from_cache,
    )
