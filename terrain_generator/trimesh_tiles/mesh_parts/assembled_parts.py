#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import replace

import numpy as np
import trimesh

from ...utils import merge_meshes, yaw_rotate_mesh
from .basic_parts import create_standard_wall
from .create_tiles import build_mesh
from .mesh_parts_cfg import MeshPartsCfg, PlatformMeshPartsCfg, SlopeMeshPartsCfg, StairMeshPartsCfg


SEAM_OVERLAP = 1.0e-3


def _build_mesh(part):
    return build_mesh(part).copy()


def _copy_part_without_walls(part):
    if isinstance(part, trimesh.Trimesh):
        raise ValueError("grounded_side_walls requires cfg inputs, not pre-built meshes.")
    if isinstance(part, (PlatformMeshPartsCfg, SlopeMeshPartsCfg, StairMeshPartsCfg)):
        if getattr(part, "wall", None) is None:
            return part
        return replace(part, wall=None)
    return part


def _infer_stage_rise(part, stage_rise=None):
    if stage_rise is not None:
        return stage_rise
    if isinstance(part, StairMeshPartsCfg):
        if len(part.stairs) == 0:
            raise ValueError("StairMeshPartsCfg must contain at least one stair.")
        return part.stairs[0].total_height
    if isinstance(part, SlopeMeshPartsCfg):
        return np.tan(np.deg2rad(part.slope_angle_deg)) * part.slope_length
    raise ValueError("stage_rise must be provided when the stage is not a known stair/slope cfg.")


def _infer_platform_top_z(part, platform_top_z=None):
    if platform_top_z is not None:
        return platform_top_z
    if isinstance(part, PlatformMeshPartsCfg):
        return float(np.max(part.array)) - part.dim[2] / 2.0
    raise ValueError("platform_top_z must be provided when the platform is not a PlatformMeshPartsCfg.")


def _rotate_mesh(mesh, yaw_deg):
    yaw_deg = yaw_deg % 360
    if yaw_deg == 0:
        return mesh.copy()
    if yaw_deg == 90:
        return yaw_rotate_mesh(mesh, 90)
    if yaw_deg == 180:
        return yaw_rotate_mesh(mesh, 180)
    if yaw_deg == 270:
        return yaw_rotate_mesh(mesh, 270)
    raise ValueError(f"Unsupported yaw_deg={yaw_deg}. Only 90-degree turns are supported.")


def _rotate_point_xy(point_xy, yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg % 360)
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ]
    )
    return rotation @ point_xy


def _rotate_left(vec):
    return np.array([-vec[1], vec[0]])


def _rotate_right(vec):
    return np.array([vec[1], -vec[0]])


def _final_turn_platform_side_edge(turn_direction):
    if turn_direction == "left":
        return "left"
    if turn_direction == "right":
        return "right"
    raise ValueError("turn_direction must be 'left' or 'right'.")


def _build_grounded_wall_mesh(part, yaw_deg, translation_xy, translation_z, ground_z, wall_height):
    wall_cfg = getattr(part, "wall", None)
    if wall_cfg is None or len(wall_cfg.wall_edges) == 0:
        return None

    grounded_wall_cfg = replace(
        wall_cfg,
        dim=part.dim,
        floor_thickness=part.floor_thickness,
        minimal_triangles=part.minimal_triangles,
        load_from_cache=part.load_from_cache,
        height_offset=part.height_offset,
        wall_height=wall_height,
        wall_z_offset=ground_z + part.dim[2] / 2.0 - translation_z - part.height_offset,
    )
    wall_meshes = [create_standard_wall(grounded_wall_cfg, edge) for edge in grounded_wall_cfg.wall_edges]
    wall_mesh = merge_meshes(wall_meshes, False)
    wall_mesh = _rotate_mesh(wall_mesh, yaw_deg)
    wall_mesh.apply_translation([translation_xy[0], translation_xy[1], translation_z])
    return wall_mesh


def _wall_dim_with_extra_length(part_dim, edge, extra_length):
    if extra_length <= 0.0:
        return part_dim

    dim_x, dim_y, dim_z = part_dim
    if edge in ("left", "right", "middle_bottom", "middle_up", "right_bottom", "right_up"):
        return (dim_x, dim_y + extra_length, dim_z)
    if edge in ("bottom", "up", "middle_left", "middle_right", "bottom_left", "bottom_right"):
        return (dim_x + extra_length, dim_y, dim_z)
    return part_dim


def _build_extended_grounded_wall_mesh(
    part,
    yaw_deg,
    translation_xy,
    translation_z,
    ground_z,
    wall_height,
    extra_length,
    wall_edges_override=None,
):
    wall_cfg = getattr(part, "wall", None)
    if wall_cfg is None:
        return None
    wall_edges = tuple(wall_edges_override) if wall_edges_override is not None else tuple(wall_cfg.wall_edges)
    if len(wall_edges) == 0:
        return None

    wall_meshes = []
    for edge in wall_edges:
        edge_wall_cfg = replace(
            wall_cfg,
            dim=_wall_dim_with_extra_length(part.dim, edge, extra_length),
            floor_thickness=part.floor_thickness,
            minimal_triangles=part.minimal_triangles,
            load_from_cache=part.load_from_cache,
            height_offset=part.height_offset,
            wall_height=wall_height,
            wall_z_offset=ground_z + part.dim[2] / 2.0 - translation_z - part.height_offset,
        )
        wall_meshes.append(create_standard_wall(edge_wall_cfg, edge))

    wall_mesh = merge_meshes(wall_meshes, False)
    wall_mesh = _rotate_mesh(wall_mesh, yaw_deg)
    wall_mesh.apply_translation([translation_xy[0], translation_xy[1], translation_z])
    return wall_mesh


def _build_ground_fill_mesh(part, yaw_deg, translation_xy, bottom_world_z, ground_z):
    support_height = bottom_world_z - ground_z
    if support_height <= 1.0e-6:
        return None

    support_mesh = trimesh.creation.box((part.dim[0], part.dim[1], support_height))
    support_mesh = _rotate_mesh(support_mesh, yaw_deg)
    support_mesh.apply_translation(
        [
            translation_xy[0],
            translation_xy[1],
            ground_z + support_height / 2.0,
        ]
    )
    return support_mesh


def assemble_linear_sequence(
    stage,
    platform,
    num_stages=4,
    stage_rise=None,
    platform_top_z=None,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1.")

    stage_rise = _infer_stage_rise(stage, stage_rise)
    wall_top_height = grounded_wall_height if grounded_wall_height is not None else num_stages * stage_rise

    stage_for_geometry = _copy_part_without_walls(stage) if grounded_side_walls else stage
    platform_for_geometry = _copy_part_without_walls(platform) if grounded_side_walls else platform

    base_stage_mesh = _build_mesh(stage_for_geometry)
    base_platform_mesh = _build_mesh(platform_for_geometry)
    platform_top_z = _infer_platform_top_z(platform_for_geometry, platform_top_z)

    stage_bounds = base_stage_mesh.bounds
    platform_bounds = base_platform_mesh.bounds
    local_stage_start_edge_center = np.array([0.0, stage_bounds[0, 1]])
    local_stage_end_edge_center = np.array([0.0, stage_bounds[1, 1]])
    ground_z = stage_bounds[0, 2]
    platform_length = platform_bounds[1, 1] - platform_bounds[0, 1]

    meshes = []
    wall_meshes = []
    fill_meshes = []
    current_start_edge_center = local_stage_start_edge_center.copy()
    current_height = 0.0

    for stage_idx in range(num_stages):
        stage_mesh = base_stage_mesh.copy()
        stage_translation_xy = current_start_edge_center - local_stage_start_edge_center
        stage_mesh.apply_translation([stage_translation_xy[0], stage_translation_xy[1], current_height])
        meshes.append(stage_mesh)
        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                stage,
                0,
                stage_translation_xy,
                current_height + stage_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                stage,
                0,
                stage_translation_xy,
                current_height,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)

        stage_end_world = local_stage_end_edge_center + stage_translation_xy
        platform_center_world = stage_end_world + np.array([0.0, platform_length / 2.0 - SEAM_OVERLAP])
        platform_mesh = base_platform_mesh.copy()
        platform_top_world_z = stage_mesh.bounds[1, 2]
        platform_translation_z = platform_top_world_z - platform_top_z
        platform_mesh.apply_translation([platform_center_world[0], platform_center_world[1], platform_translation_z])
        meshes.append(platform_mesh)
        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                platform,
                0,
                platform_center_world,
                platform_translation_z + platform_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                platform,
                0,
                platform_center_world,
                platform_translation_z,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)
            if add_final_end_wall and stage_idx == num_stages - 1:
                end_wall_mesh = _build_extended_grounded_wall_mesh(
                    platform,
                    0,
                    platform_center_world,
                    platform_translation_z,
                    ground_z,
                    wall_top_height,
                    0.0,
                    wall_edges_override=("up",),
                )
                if end_wall_mesh is not None:
                    wall_meshes.append(end_wall_mesh)

        if stage_idx == num_stages - 1:
            continue

        current_start_edge_center = platform_center_world + np.array([0.0, platform_length / 2.0 - SEAM_OVERLAP])
        current_height += stage_rise

    return merge_meshes(fill_meshes + meshes + wall_meshes, False)


def assemble_rotating_sequence(
    stage,
    platform,
    num_stages=4,
    turn_direction="left",
    stage_rise=None,
    platform_top_z=None,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1.")
    if turn_direction not in ("left", "right"):
        raise ValueError("turn_direction must be 'left' or 'right'.")

    stage_rise = _infer_stage_rise(stage, stage_rise)
    wall_top_height = grounded_wall_height if grounded_wall_height is not None else num_stages * stage_rise

    stage_for_geometry = _copy_part_without_walls(stage) if grounded_side_walls else stage
    platform_for_geometry = _copy_part_without_walls(platform) if grounded_side_walls else platform

    base_stage_mesh = _build_mesh(stage_for_geometry)
    base_platform_mesh = _build_mesh(platform_for_geometry)
    platform_top_z = _infer_platform_top_z(platform_for_geometry, platform_top_z)

    stage_bounds = base_stage_mesh.bounds
    platform_bounds = base_platform_mesh.bounds
    local_stage_start_edge_center = np.array([0.0, stage_bounds[0, 1]])
    local_stage_end_edge_center = np.array([0.0, stage_bounds[1, 1]])
    ground_z = stage_bounds[0, 2]
    platform_width = platform_bounds[1, 0] - platform_bounds[0, 0]
    platform_length = platform_bounds[1, 1] - platform_bounds[0, 1]

    meshes = []
    wall_meshes = []
    fill_meshes = []
    current_start_edge_center = local_stage_start_edge_center.copy()
    current_heading = np.array([0.0, 1.0])
    current_yaw_deg = 0
    current_height = 0.0

    for stage_idx in range(num_stages):
        stage_mesh = _rotate_mesh(base_stage_mesh, current_yaw_deg)
        rotated_stage_start = _rotate_point_xy(local_stage_start_edge_center, current_yaw_deg)
        stage_translation_xy = current_start_edge_center - rotated_stage_start
        stage_mesh.apply_translation([stage_translation_xy[0], stage_translation_xy[1], current_height])
        meshes.append(stage_mesh)
        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                stage,
                current_yaw_deg,
                stage_translation_xy,
                current_height + stage_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                stage,
                current_yaw_deg,
                stage_translation_xy,
                current_height,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)

        stage_end_world = _rotate_point_xy(local_stage_end_edge_center, current_yaw_deg) + stage_translation_xy
        platform_center_world = stage_end_world + current_heading * (platform_length / 2.0 - SEAM_OVERLAP)
        platform_mesh = _rotate_mesh(base_platform_mesh, current_yaw_deg)
        platform_top_world_z = stage_mesh.bounds[1, 2]
        platform_translation_z = platform_top_world_z - platform_top_z
        platform_mesh.apply_translation([platform_center_world[0], platform_center_world[1], platform_translation_z])
        meshes.append(platform_mesh)
        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                platform,
                current_yaw_deg,
                platform_center_world,
                platform_translation_z + platform_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                platform,
                current_yaw_deg,
                platform_center_world,
                platform_translation_z,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)
            if stage_idx == num_stages - 1:
                closing_side_wall = _build_extended_grounded_wall_mesh(
                    platform,
                    current_yaw_deg,
                    platform_center_world,
                    platform_translation_z,
                    ground_z,
                    wall_top_height,
                    grounded_wall_extra_length,
                    wall_edges_override=(_final_turn_platform_side_edge(turn_direction),),
                )
                if closing_side_wall is not None:
                    wall_meshes.append(closing_side_wall)
            if add_final_end_wall and stage_idx == num_stages - 1:
                end_wall_mesh = _build_extended_grounded_wall_mesh(
                    platform,
                    current_yaw_deg,
                    platform_center_world,
                    platform_translation_z,
                    ground_z,
                    wall_top_height,
                    0.0,
                    wall_edges_override=("up",),
                )
                if end_wall_mesh is not None:
                    wall_meshes.append(end_wall_mesh)

        if stage_idx == num_stages - 1:
            continue

        turn_normal = _rotate_left(current_heading) if turn_direction == "left" else _rotate_right(current_heading)
        current_start_edge_center = platform_center_world + turn_normal * (platform_width / 2.0 - SEAM_OVERLAP)
        current_heading = turn_normal
        current_yaw_deg = (current_yaw_deg + (90 if turn_direction == "left" else 270)) % 360
        current_height += stage_rise

    return merge_meshes(fill_meshes + meshes + wall_meshes, False)


def make_linear_stairs_mesh(
    stairs,
    platform,
    num_stages=4,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return assemble_linear_sequence(
        stairs,
        platform,
        num_stages=num_stages,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_stairs_platform_stairs_mesh(
    stairs,
    platform,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return make_linear_stairs_mesh(
        stairs,
        platform,
        num_stages=2,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_rotating_stairs_mesh(
    stairs,
    platform,
    num_stages=4,
    turn_direction="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return assemble_rotating_sequence(
        stairs,
        platform,
        num_stages=num_stages,
        turn_direction=turn_direction,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_stairs_turn_90_mesh(
    stairs,
    platform,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return make_rotating_stairs_mesh(
        stairs,
        platform,
        num_stages=2,
        turn_direction="left",
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_linear_slopes_mesh(
    slope,
    platform,
    num_stages=4,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return assemble_linear_sequence(
        slope,
        platform,
        num_stages=num_stages,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_slopes_platform_slopes_mesh(
    slope,
    platform,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
):
    return make_linear_slopes_mesh(
        slope,
        platform,
        num_stages=2,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
    )


def make_rotating_slopes_mesh(
    slope,
    platform,
    num_stages=4,
    turn_direction="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
):
    return assemble_rotating_sequence(
        slope,
        platform,
        num_stages=num_stages,
        turn_direction=turn_direction,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
    )


def make_slopes_turn_90_mesh(
    slope,
    platform,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
):
    return make_rotating_slopes_mesh(
        slope,
        platform,
        num_stages=2,
        turn_direction="left",
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
    )


def compose_transformed_parts(part_specs, center_xy=False):
    """Compose cfgs or meshes with explicit yaw and translation transforms."""
    meshes = []
    for spec in part_specs:
        part = spec["part"]
        yaw_deg = spec.get("yaw_deg", 0)
        translation = np.asarray(spec.get("translation", (0.0, 0.0, 0.0)), dtype=float)
        align_bottom = spec.get("align_bottom", False)

        mesh = _rotate_mesh(_build_mesh(part), yaw_deg)
        if align_bottom:
            mesh.apply_translation([0.0, 0.0, -mesh.bounds[0, 2]])
        mesh.apply_translation(translation)
        meshes.append(mesh)

    if len(meshes) == 0:
        raise ValueError("part_specs must contain at least one part.")

    mesh = merge_meshes(meshes, False)
    if center_xy and len(mesh.vertices) > 0:
        bounds = mesh.bounds
        center = np.array(
            [
                0.5 * (bounds[0, 0] + bounds[1, 0]),
                0.5 * (bounds[0, 1] + bounds[1, 1]),
                0.0,
            ]
        )
        mesh.apply_translation(-center)
    return mesh
