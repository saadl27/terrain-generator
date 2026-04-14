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


def _build_grounded_wall_segment(
    part,
    yaw_deg,
    translation_xy,
    translation_z,
    ground_z,
    wall_height,
    edge,
    segment_span,
    center_offset=0.0,
):
    wall_cfg = getattr(part, "wall", None)
    if wall_cfg is None or segment_span <= 1.0e-6:
        return None

    if edge in ("bottom", "up"):
        dim = (segment_span, part.dim[1], part.dim[2])
        local_offset = np.array([center_offset, 0.0, 0.0])
    elif edge in ("left", "right"):
        dim = (part.dim[0], segment_span, part.dim[2])
        local_offset = np.array([0.0, center_offset, 0.0])
    else:
        raise ValueError(f"Unsupported edge {edge} for wall segment.")

    segment_wall_cfg = replace(
        wall_cfg,
        dim=dim,
        floor_thickness=part.floor_thickness,
        minimal_triangles=part.minimal_triangles,
        load_from_cache=part.load_from_cache,
        height_offset=part.height_offset,
        wall_height=wall_height,
        wall_z_offset=ground_z + part.dim[2] / 2.0 - translation_z - part.height_offset,
    )
    wall_mesh = create_standard_wall(segment_wall_cfg, edge)
    wall_mesh.apply_translation(local_offset)
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


def assemble_u_turn_sequence(
    outbound_stage,
    turn_platform,
    final_platform,
    return_stage=None,
    return_side="left",
    stage_rise=None,
    turn_platform_top_z=None,
    final_platform_top_z=None,
    return_stage_yaw_deg=180,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    if return_side not in ("left", "right"):
        raise ValueError("return_side must be 'left' or 'right'.")

    return_stage = outbound_stage if return_stage is None else return_stage
    stage_rise = _infer_stage_rise(outbound_stage, stage_rise)
    wall_top_height = grounded_wall_height if grounded_wall_height is not None else 2.0 * stage_rise

    outbound_for_geometry = _copy_part_without_walls(outbound_stage) if grounded_side_walls else outbound_stage
    turn_platform_for_geometry = _copy_part_without_walls(turn_platform) if grounded_side_walls else turn_platform
    return_for_geometry = _copy_part_without_walls(return_stage) if grounded_side_walls else return_stage
    final_platform_for_geometry = _copy_part_without_walls(final_platform) if grounded_side_walls else final_platform

    outbound_base_mesh = _build_mesh(outbound_for_geometry)
    turn_platform_base_mesh = _build_mesh(turn_platform_for_geometry)
    return_base_mesh = _build_mesh(return_for_geometry)
    final_platform_base_mesh = _build_mesh(final_platform_for_geometry)

    turn_platform_top_z = _infer_platform_top_z(turn_platform_for_geometry, turn_platform_top_z)
    final_platform_top_z = _infer_platform_top_z(final_platform_for_geometry, final_platform_top_z)

    outbound_bounds = outbound_base_mesh.bounds
    turn_platform_bounds = turn_platform_base_mesh.bounds
    return_bounds = return_base_mesh.bounds
    final_platform_bounds = final_platform_base_mesh.bounds

    outbound_start_edge_center = np.array([0.0, outbound_bounds[0, 1]])
    outbound_end_edge_center = np.array([0.0, outbound_bounds[1, 1]])
    return_start_edge_center = _rotate_point_xy(np.array([0.0, return_bounds[0, 1]]), return_stage_yaw_deg)
    return_end_edge_center = _rotate_point_xy(np.array([0.0, return_bounds[1, 1]]), return_stage_yaw_deg)

    ground_z = outbound_bounds[0, 2]
    outbound_stage_width = outbound_bounds[1, 0] - outbound_bounds[0, 0]
    turn_platform_width = turn_platform_bounds[1, 0] - turn_platform_bounds[0, 0]
    turn_platform_length = turn_platform_bounds[1, 1] - turn_platform_bounds[0, 1]
    return_stage_length = return_bounds[1, 1] - return_bounds[0, 1]
    return_stage_width = return_bounds[1, 0] - return_bounds[0, 0]
    final_platform_length = final_platform_bounds[1, 1] - final_platform_bounds[0, 1]
    return_heading = _rotate_point_xy(np.array([0.0, 1.0]), return_stage_yaw_deg)

    meshes = []
    wall_meshes = []
    fill_meshes = []

    outbound_translation_xy = np.zeros(2)
    outbound_mesh = outbound_base_mesh.copy()
    outbound_mesh.apply_translation([outbound_translation_xy[0], outbound_translation_xy[1], 0.0])
    meshes.append(outbound_mesh)
    if common_ground:
        support_mesh = _build_ground_fill_mesh(
            outbound_stage,
            0,
            outbound_translation_xy,
            outbound_bounds[0, 2],
            ground_z,
        )
        if support_mesh is not None:
            fill_meshes.append(support_mesh)
    if grounded_side_walls:
        wall_mesh = _build_extended_grounded_wall_mesh(
            outbound_stage,
            0,
            outbound_translation_xy,
            0.0,
            ground_z,
            wall_top_height,
            grounded_wall_extra_length,
        )
        if wall_mesh is not None:
            wall_meshes.append(wall_mesh)

    side_sign = -1.0 if return_side == "left" else 1.0
    turn_platform_lateral_offset = side_sign * (turn_platform_width / 2.0 - outbound_stage_width / 2.0)

    outbound_end_world = outbound_end_edge_center + outbound_translation_xy
    turn_platform_center_world = outbound_end_world + np.array(
        [turn_platform_lateral_offset, turn_platform_length / 2.0 - SEAM_OVERLAP]
    )
    turn_platform_mesh = turn_platform_base_mesh.copy()
    turn_platform_translation_z = outbound_mesh.bounds[1, 2] - turn_platform_top_z
    turn_platform_mesh.apply_translation(
        [turn_platform_center_world[0], turn_platform_center_world[1], turn_platform_translation_z]
    )
    meshes.append(turn_platform_mesh)
    if common_ground:
        support_mesh = _build_ground_fill_mesh(
            turn_platform,
            0,
            turn_platform_center_world,
            turn_platform_translation_z + turn_platform_bounds[0, 2],
            ground_z,
        )
        if support_mesh is not None:
            fill_meshes.append(support_mesh)
    if grounded_side_walls:
        wall_mesh = _build_extended_grounded_wall_mesh(
            turn_platform,
            0,
            turn_platform_center_world,
            turn_platform_translation_z,
            ground_z,
            wall_top_height,
            grounded_wall_extra_length,
        )
        if wall_mesh is not None:
            wall_meshes.append(wall_mesh)
        if add_turn_platform_end_wall:
            end_wall_mesh = _build_extended_grounded_wall_mesh(
                turn_platform,
                0,
                turn_platform_center_world,
                turn_platform_translation_z,
                ground_z,
                wall_top_height,
                0.0,
                wall_edges_override=("up",),
            )
            if end_wall_mesh is not None:
                wall_meshes.append(end_wall_mesh)

    return_start_world = turn_platform_center_world + np.array(
        [
            side_sign * (turn_platform_width / 2.0 - return_stage_width / 2.0),
            -turn_platform_length / 2.0 + SEAM_OVERLAP,
        ]
    )
    return_translation_xy = return_start_world - return_start_edge_center
    return_mesh = _rotate_mesh(return_base_mesh, return_stage_yaw_deg)
    return_mesh.apply_translation([return_translation_xy[0], return_translation_xy[1], stage_rise])
    meshes.append(return_mesh)
    if grounded_side_walls and add_turn_gap_wall:
        outbound_span = np.array(
            [
                outbound_end_world[0] - outbound_stage_width / 2.0,
                outbound_end_world[0] + outbound_stage_width / 2.0,
            ]
        )
        return_span = np.array(
            [
                return_start_world[0] - return_stage_width / 2.0,
                return_start_world[0] + return_stage_width / 2.0,
            ]
        )
        first_span, second_span = sorted((outbound_span, return_span), key=lambda span: span[0])
        gap_start = first_span[1]
        gap_end = second_span[0]
        gap_width = gap_end - gap_start
        if gap_width > 1.0e-6:
            gap_center_offset = 0.5 * (gap_start + gap_end) - turn_platform_center_world[0]
            gap_wall_mesh = _build_grounded_wall_segment(
                turn_platform,
                0,
                turn_platform_center_world,
                turn_platform_translation_z,
                ground_z,
                wall_top_height,
                edge="bottom",
                segment_span=gap_width,
                center_offset=gap_center_offset,
            )
            if gap_wall_mesh is not None:
                wall_meshes.append(gap_wall_mesh)
    if common_ground:
        support_mesh = _build_ground_fill_mesh(
            return_stage,
            return_stage_yaw_deg,
            return_translation_xy,
            stage_rise + return_bounds[0, 2],
            ground_z,
        )
        if support_mesh is not None:
            fill_meshes.append(support_mesh)
    if grounded_side_walls:
        wall_mesh = _build_extended_grounded_wall_mesh(
            return_stage,
            return_stage_yaw_deg,
            return_translation_xy,
            stage_rise,
            ground_z,
            wall_top_height,
            grounded_wall_extra_length,
        )
        if wall_mesh is not None:
            wall_meshes.append(wall_mesh)

    return_end_world = return_end_edge_center + return_translation_xy
    final_platform_center_world = return_end_world + return_heading * (final_platform_length / 2.0 - SEAM_OVERLAP)
    final_platform_mesh = _rotate_mesh(final_platform_base_mesh, return_stage_yaw_deg)
    final_platform_translation_z = return_mesh.bounds[1, 2] - final_platform_top_z
    final_platform_mesh.apply_translation(
        [final_platform_center_world[0], final_platform_center_world[1], final_platform_translation_z]
    )
    meshes.append(final_platform_mesh)
    if common_ground:
        support_mesh = _build_ground_fill_mesh(
            final_platform,
            return_stage_yaw_deg,
            final_platform_center_world,
            final_platform_translation_z + final_platform_bounds[0, 2],
            ground_z,
        )
        if support_mesh is not None:
            fill_meshes.append(support_mesh)
    if grounded_side_walls:
        wall_mesh = _build_extended_grounded_wall_mesh(
            final_platform,
            return_stage_yaw_deg,
            final_platform_center_world,
            final_platform_translation_z,
            ground_z,
            wall_top_height,
            grounded_wall_extra_length,
        )
        if wall_mesh is not None:
            wall_meshes.append(wall_mesh)
        if add_final_end_wall:
            end_wall_mesh = _build_extended_grounded_wall_mesh(
                final_platform,
                return_stage_yaw_deg,
                final_platform_center_world,
                final_platform_translation_z,
                ground_z,
                wall_top_height,
                0.0,
                wall_edges_override=("up",),
            )
            if end_wall_mesh is not None:
                wall_meshes.append(end_wall_mesh)

    return merge_meshes(fill_meshes + meshes + wall_meshes, False)


def assemble_repeating_u_turn_sequence(
    outbound_stage,
    turn_platform,
    final_platform,
    num_stages=4,
    return_stage=None,
    return_side="left",
    stage_rise=None,
    turn_platform_top_z=None,
    final_platform_top_z=None,
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1.")
    if return_side not in ("left", "right"):
        raise ValueError("return_side must be 'left' or 'right'.")

    return_stage = outbound_stage if return_stage is None else return_stage
    stage_rise = _infer_stage_rise(outbound_stage, stage_rise)
    wall_top_height = grounded_wall_height if grounded_wall_height is not None else num_stages * stage_rise

    outbound_for_geometry = _copy_part_without_walls(outbound_stage) if grounded_side_walls else outbound_stage
    return_for_geometry = _copy_part_without_walls(return_stage) if grounded_side_walls else return_stage
    turn_platform_for_geometry = _copy_part_without_walls(turn_platform) if grounded_side_walls else turn_platform
    final_platform_for_geometry = _copy_part_without_walls(final_platform) if grounded_side_walls else final_platform

    outbound_base_mesh = _build_mesh(outbound_for_geometry)
    return_base_mesh = _build_mesh(return_for_geometry)
    turn_platform_base_mesh = _build_mesh(turn_platform_for_geometry)
    final_platform_base_mesh = _build_mesh(final_platform_for_geometry)

    turn_platform_top_z = _infer_platform_top_z(turn_platform_for_geometry, turn_platform_top_z)
    final_platform_top_z = _infer_platform_top_z(final_platform_for_geometry, final_platform_top_z)

    outbound_bounds = outbound_base_mesh.bounds
    return_bounds = return_base_mesh.bounds
    turn_platform_bounds = turn_platform_base_mesh.bounds
    final_platform_bounds = final_platform_base_mesh.bounds

    outbound_start_edge_center = np.array([0.0, outbound_bounds[0, 1]])
    outbound_end_edge_center = np.array([0.0, outbound_bounds[1, 1]])
    return_start_edge_center = np.array([0.0, return_bounds[0, 1]])
    return_end_edge_center = np.array([0.0, return_bounds[1, 1]])

    outbound_stage_width = outbound_bounds[1, 0] - outbound_bounds[0, 0]
    return_stage_width = return_bounds[1, 0] - return_bounds[0, 0]
    turn_platform_width = turn_platform_bounds[1, 0] - turn_platform_bounds[0, 0]
    turn_platform_length = turn_platform_bounds[1, 1] - turn_platform_bounds[0, 1]
    final_platform_length = final_platform_bounds[1, 1] - final_platform_bounds[0, 1]
    ground_z = outbound_bounds[0, 2]

    meshes = []
    wall_meshes = []
    fill_meshes = []

    current_stage_start_world = outbound_start_edge_center.copy()
    current_yaw_deg = 0
    current_height = 0.0
    current_is_outbound = True
    side_sign = -1.0 if return_side == "left" else 1.0

    last_stage_mesh = None
    last_stage_cfg = None
    last_stage_bounds = None
    last_stage_translation_xy = None
    last_stage_end_world = None
    last_heading = None

    for stage_idx in range(num_stages):
        if current_is_outbound:
            stage_cfg = outbound_stage
            stage_base_mesh = outbound_base_mesh
            stage_bounds = outbound_bounds
            local_start_edge = outbound_start_edge_center
            local_end_edge = outbound_end_edge_center
            stage_width = outbound_stage_width
        else:
            stage_cfg = return_stage
            stage_base_mesh = return_base_mesh
            stage_bounds = return_bounds
            local_start_edge = return_start_edge_center
            local_end_edge = return_end_edge_center
            stage_width = return_stage_width

        stage_mesh = _rotate_mesh(stage_base_mesh, current_yaw_deg)
        rotated_stage_start = _rotate_point_xy(local_start_edge, current_yaw_deg)
        stage_translation_xy = current_stage_start_world - rotated_stage_start
        stage_mesh.apply_translation([stage_translation_xy[0], stage_translation_xy[1], current_height])
        meshes.append(stage_mesh)

        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                stage_cfg,
                current_yaw_deg,
                stage_translation_xy,
                current_height + stage_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                stage_cfg,
                current_yaw_deg,
                stage_translation_xy,
                current_height,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)

        current_stage_end_world = _rotate_point_xy(local_end_edge, current_yaw_deg) + stage_translation_xy
        current_heading = _rotate_point_xy(np.array([0.0, 1.0]), current_yaw_deg)

        last_stage_mesh = stage_mesh
        last_stage_cfg = stage_cfg
        last_stage_bounds = stage_bounds
        last_stage_translation_xy = stage_translation_xy
        last_stage_end_world = current_stage_end_world
        last_heading = current_heading

        if stage_idx == num_stages - 1:
            continue

        next_is_outbound = not current_is_outbound
        next_stage_width = outbound_stage_width if next_is_outbound else return_stage_width
        next_local_start_edge = outbound_start_edge_center if next_is_outbound else return_start_edge_center
        next_yaw_deg = (current_yaw_deg + 180) % 360
        next_side_direction = _rotate_left(current_heading) if return_side == "left" else _rotate_right(current_heading)
        lateral_axis = _rotate_point_xy(np.array([1.0, 0.0]), current_yaw_deg)

        turn_platform_lateral_offset = side_sign * (turn_platform_width / 2.0 - stage_width / 2.0)
        turn_platform_center_world = current_stage_end_world + current_heading * (turn_platform_length / 2.0 - SEAM_OVERLAP)
        turn_platform_center_world = turn_platform_center_world + lateral_axis * turn_platform_lateral_offset

        turn_platform_mesh = _rotate_mesh(turn_platform_base_mesh, current_yaw_deg)
        turn_platform_translation_z = stage_mesh.bounds[1, 2] - turn_platform_top_z
        turn_platform_mesh.apply_translation(
            [turn_platform_center_world[0], turn_platform_center_world[1], turn_platform_translation_z]
        )
        meshes.append(turn_platform_mesh)

        if common_ground:
            support_mesh = _build_ground_fill_mesh(
                turn_platform,
                current_yaw_deg,
                turn_platform_center_world,
                turn_platform_translation_z + turn_platform_bounds[0, 2],
                ground_z,
            )
            if support_mesh is not None:
                fill_meshes.append(support_mesh)
        if grounded_side_walls:
            wall_mesh = _build_extended_grounded_wall_mesh(
                turn_platform,
                current_yaw_deg,
                turn_platform_center_world,
                turn_platform_translation_z,
                ground_z,
                wall_top_height,
                grounded_wall_extra_length,
            )
            if wall_mesh is not None:
                wall_meshes.append(wall_mesh)
            if add_turn_platform_end_wall:
                end_wall_mesh = _build_extended_grounded_wall_mesh(
                    turn_platform,
                    current_yaw_deg,
                    turn_platform_center_world,
                    turn_platform_translation_z,
                    ground_z,
                    wall_top_height,
                    0.0,
                    wall_edges_override=("up",),
                )
                if end_wall_mesh is not None:
                    wall_meshes.append(end_wall_mesh)

        next_stage_start_world = turn_platform_center_world + lateral_axis * side_sign * (
            turn_platform_width / 2.0 - next_stage_width / 2.0
        )
        next_stage_start_world = next_stage_start_world - current_heading * (turn_platform_length / 2.0 - SEAM_OVERLAP)

        if grounded_side_walls and add_turn_gap_wall:
            current_offset = np.dot(current_stage_end_world - turn_platform_center_world, lateral_axis)
            next_offset = np.dot(next_stage_start_world - turn_platform_center_world, lateral_axis)
            current_span = np.array([current_offset - stage_width / 2.0, current_offset + stage_width / 2.0])
            next_span = np.array([next_offset - next_stage_width / 2.0, next_offset + next_stage_width / 2.0])
            first_span, second_span = sorted((current_span, next_span), key=lambda span: span[0])
            gap_start = first_span[1]
            gap_end = second_span[0]
            gap_width = gap_end - gap_start
            if gap_width > 1.0e-6:
                gap_center_offset = 0.5 * (gap_start + gap_end)
                gap_wall_mesh = _build_grounded_wall_segment(
                    turn_platform,
                    current_yaw_deg,
                    turn_platform_center_world,
                    turn_platform_translation_z,
                    ground_z,
                    wall_top_height,
                    edge="bottom",
                    segment_span=gap_width,
                    center_offset=gap_center_offset,
                )
                if gap_wall_mesh is not None:
                    wall_meshes.append(gap_wall_mesh)

        current_stage_start_world = next_stage_start_world
        current_yaw_deg = next_yaw_deg
        current_height += stage_rise
        current_is_outbound = next_is_outbound

    final_platform_center_world = last_stage_end_world + last_heading * (final_platform_length / 2.0 - SEAM_OVERLAP)
    final_platform_mesh = _rotate_mesh(final_platform_base_mesh, current_yaw_deg)
    final_platform_translation_z = last_stage_mesh.bounds[1, 2] - final_platform_top_z
    final_platform_mesh.apply_translation(
        [final_platform_center_world[0], final_platform_center_world[1], final_platform_translation_z]
    )
    meshes.append(final_platform_mesh)

    if common_ground:
        support_mesh = _build_ground_fill_mesh(
            final_platform,
            current_yaw_deg,
            final_platform_center_world,
            final_platform_translation_z + final_platform_bounds[0, 2],
            ground_z,
        )
        if support_mesh is not None:
            fill_meshes.append(support_mesh)
    if grounded_side_walls:
        wall_mesh = _build_extended_grounded_wall_mesh(
            final_platform,
            current_yaw_deg,
            final_platform_center_world,
            final_platform_translation_z,
            ground_z,
            wall_top_height,
            grounded_wall_extra_length,
        )
        if wall_mesh is not None:
            wall_meshes.append(wall_mesh)
        if add_final_end_wall:
            end_wall_mesh = _build_extended_grounded_wall_mesh(
                final_platform,
                current_yaw_deg,
                final_platform_center_world,
                final_platform_translation_z,
                ground_z,
                wall_top_height,
                0.0,
                wall_edges_override=("up",),
            )
            if end_wall_mesh is not None:
                wall_meshes.append(end_wall_mesh)

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


def make_stairs_u_turn_mesh(
    stairs,
    turn_platform,
    final_platform,
    return_side="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    return assemble_u_turn_sequence(
        stairs,
        turn_platform,
        final_platform,
        return_stage=stairs,
        return_side=return_side,
        return_stage_yaw_deg=180,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
        add_turn_platform_end_wall=add_turn_platform_end_wall,
        add_turn_gap_wall=add_turn_gap_wall,
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


def make_slopes_u_turn_mesh(
    slope,
    turn_platform,
    final_platform,
    return_side="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    return assemble_u_turn_sequence(
        slope,
        turn_platform,
        final_platform,
        return_stage=slope,
        return_side=return_side,
        return_stage_yaw_deg=180,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
        add_turn_platform_end_wall=add_turn_platform_end_wall,
        add_turn_gap_wall=add_turn_gap_wall,
    )


def make_repeating_u_turn_stairs_mesh(
    stairs,
    turn_platform,
    final_platform,
    num_stages=4,
    return_side="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    return assemble_repeating_u_turn_sequence(
        stairs,
        turn_platform,
        final_platform,
        num_stages=num_stages,
        return_stage=stairs,
        return_side=return_side,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
        add_turn_platform_end_wall=add_turn_platform_end_wall,
        add_turn_gap_wall=add_turn_gap_wall,
    )


def make_repeating_u_turn_slopes_mesh(
    slope,
    turn_platform,
    final_platform,
    num_stages=4,
    return_side="left",
    grounded_side_walls=False,
    grounded_wall_height=None,
    grounded_wall_extra_length=0.0,
    common_ground=False,
    add_final_end_wall=False,
    add_turn_platform_end_wall=True,
    add_turn_gap_wall=True,
):
    return assemble_repeating_u_turn_sequence(
        slope,
        turn_platform,
        final_platform,
        num_stages=num_stages,
        return_stage=slope,
        return_side=return_side,
        grounded_side_walls=grounded_side_walls,
        grounded_wall_height=grounded_wall_height,
        grounded_wall_extra_length=grounded_wall_extra_length,
        common_ground=common_ground,
        add_final_end_wall=add_final_end_wall,
        add_turn_platform_end_wall=add_turn_platform_end_wall,
        add_turn_gap_wall=add_turn_gap_wall,
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
