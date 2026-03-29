#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import numpy as np
import trimesh
from typing import Tuple, Optional, Union, Literal
from dataclasses import dataclass


@dataclass
class MeshPartsCfg:
    name: str = "mesh"
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    floor_thickness: float = 0.1
    minimal_triangles: bool = True
    weight: float = 1.0
    rotations: Tuple[Literal[90, 180, 270], ...] = ()  # (90, 180, 270)
    flips: Tuple[Literal["x", "y"], ...] = ()  # ("x", "y")
    height_offset: float = 0.0
    edge_array: Optional[np.ndarray] = None  # Array for edge definition. If None, height of the mesh is used.
    use_generator: bool = True
    load_from_cache: bool = True


@dataclass
class WallPartsCfg(MeshPartsCfg):
    wall_thickness: float = 0.4
    wall_height: float = 3.0
    wall_edges: Tuple[str, ...] = ()  # bottom, up, left, right, middle_left, middle_right, middle_up, middle_bottom
    wall_type: str = "wall"  # wall, window, door
    # wall_type_probs: Tuple[float, ...] = (0.6, 0.2, 0.2)  # wall, window, door
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5
    door_direction: str = ""  # left, right, up, down, none
    # wall_array: np.ndarray = np.zeros((3, 3))


@dataclass
class StairMeshPartsCfg(MeshPartsCfg):
    @dataclass
    class Stair(MeshPartsCfg):
        step_width: float = 1.0
        step_depth: float = 0.3
        n_steps: int = 5
        total_height: float = 1.0
        height_offset: float = 0.0
        stair_type: str = "standard"  # stair, open, ramp
        add_residual_side_up: bool = True  # If false, add to bottom.
        add_rail: bool = False
        direction: str = "up"
        attach_side: str = "left"

    stairs: Tuple[Stair, ...] = (Stair(),)
    wall: Optional[WallPartsCfg] = None


@dataclass
class CornerMeshPartsCfg(MeshPartsCfg):
    pre_length: float = 2.0
    post_length: float = 2.0
    corridor_width: float = 1.0
    pre_corridor_width: Optional[float] = None
    post_corridor_width: Optional[float] = None
    turn_angle_deg: float = 90.0  # Positive is a left turn, negative is a right turn.
    wall_thickness: float = 0.4
    wall_height: float = 3.0

    def __post_init__(self):
        self.pre_corridor_width = self.corridor_width if self.pre_corridor_width is None else self.pre_corridor_width
        self.post_corridor_width = self.corridor_width if self.post_corridor_width is None else self.post_corridor_width

        if self.pre_length <= 0.0:
            raise ValueError("pre_length must be positive.")
        if self.post_length <= 0.0:
            raise ValueError("post_length must be positive.")
        if self.pre_corridor_width <= 0.0:
            raise ValueError("pre_corridor_width must be positive.")
        if self.post_corridor_width <= 0.0:
            raise ValueError("post_corridor_width must be positive.")
        if abs(self.turn_angle_deg) < 1e-6:
            raise ValueError("turn_angle_deg must be non-zero.")

        pre_outer_width = self.pre_corridor_width + 2.0 * self.wall_thickness
        post_outer_width = self.post_corridor_width + 2.0 * self.wall_thickness
        angle_rad = np.deg2rad(self.turn_angle_deg)
        incoming_dir = np.array([0.0, 1.0])
        outgoing_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
        outgoing_dir = outgoing_dir / np.linalg.norm(outgoing_dir)

        def left_normal(vec):
            return np.array([-vec[1], vec[0]])

        def cross_2d(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def line_intersection(point_a, dir_a, point_b, dir_b):
            denom = cross_2d(dir_a, dir_b)
            if abs(denom) < 1e-8:
                raise ValueError("Corner turn angle creates parallel walls.")
            diff = point_b - point_a
            scale_a = cross_2d(diff, dir_b) / denom
            return point_a + scale_a * dir_a

        outer_sign = -1.0 if self.turn_angle_deg > 0.0 else 1.0
        incoming_floor_offset = pre_outer_width / 2.0
        outgoing_floor_offset = post_outer_width / 2.0

        incoming_outer_offset = outer_sign * incoming_floor_offset * left_normal(incoming_dir)
        outgoing_outer_offset = outer_sign * outgoing_floor_offset * left_normal(outgoing_dir)
        incoming_inner_offset = -outer_sign * incoming_floor_offset * left_normal(incoming_dir)
        outgoing_inner_offset = -outer_sign * outgoing_floor_offset * left_normal(outgoing_dir)

        outer_join = line_intersection(incoming_outer_offset, incoming_dir, outgoing_outer_offset, outgoing_dir)
        inner_join = line_intersection(incoming_inner_offset, incoming_dir, outgoing_inner_offset, outgoing_dir)

        incoming_outer_start = incoming_outer_offset - incoming_dir * self.pre_length
        incoming_inner_start = incoming_inner_offset - incoming_dir * self.pre_length
        outgoing_outer_end = outgoing_outer_offset + outgoing_dir * self.post_length
        outgoing_inner_end = outgoing_inner_offset + outgoing_dir * self.post_length

        all_corners = np.array(
            [
                incoming_outer_start,
                outer_join,
                outgoing_outer_end,
                outgoing_inner_end,
                inner_join,
                incoming_inner_start,
            ]
        )

        half_extent_x = float(np.max(np.abs(all_corners[:, 0])))
        half_extent_y = float(np.max(np.abs(all_corners[:, 1])))
        z_dim = max(self.dim[2], self.wall_height)
        self.dim = (
            max(self.dim[0], 2.0 * half_extent_x),
            max(self.dim[1], 2.0 * half_extent_y),
            z_dim,
        )


@dataclass
class PlatformMeshPartsCfg(MeshPartsCfg):
    array: np.ndarray = np.zeros((2, 2))
    z_dim_array: np.ndarray = np.zeros((2, 2))
    arrays: Optional[Tuple[np.ndarray, ...]] = None  # Additional arrays
    z_dim_arrays: Optional[Tuple[np.ndarray, ...]] = None  # Additional arrays
    add_floor: bool = True
    use_z_dim_array: bool = False  # If true, the box height is determined by the z_dim_array.
    wall: Optional[WallPartsCfg] = None  # It will be used to create the walls.


@dataclass
class HeightMapMeshPartsCfg(MeshPartsCfg):
    height_map: np.ndarray = np.ones((10, 10))
    add_floor: bool = True
    vertical_scale: float = 1.0
    slope_threshold: float = 4.0
    fill_borders: bool = True
    simplify: bool = True
    target_num_faces: int = 500

    def __post_init__(self):
        self.horizontal_scale = self.dim[0] / (self.height_map.shape[0])


@dataclass
class OverhangingMeshPartsCfg(MeshPartsCfg):
    connection_array: np.ndarray = np.zeros((3, 3))
    height_array: Optional[np.ndarray] = np.zeros((3, 3))  # Height array of the terrain.
    mesh: Optional[trimesh.Trimesh] = None  # Mesh of the terrain.
    obstacle_type: str = "wall"  # wall, window, door


@dataclass
class WallMeshPartsCfg(OverhangingMeshPartsCfg):
    wall_thickness: float = 0.4
    wall_height: float = 3.0
    create_door: bool = False
    door_width: float = 0.8
    door_height: float = 1.5


@dataclass
class OverhangingBoxesPartsCfg(OverhangingMeshPartsCfg):
    gap_mean: float = 0.8
    gap_std: float = 0.2
    box_height: float = 0.5
    box_grid_n: int = 6
    # box_prob: float = 1.0


@dataclass
class FloatingBoxesPartsCfg(OverhangingMeshPartsCfg):
    n_boxes: int = 5
    box_dim_min: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    box_dim_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roll_pitch_range: Tuple[float, float] = (0.0, np.pi / 3)  # in rad
    yaw_range: Tuple[float, float] = (0.0, 2 * np.pi)  # in rad
    min_height: float = 0.5
    max_height: float = 1.0


@dataclass
class MeshPattern:
    # name: str
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    mesh_parts: Tuple[MeshPartsCfg, ...] = (MeshPartsCfg(),)


@dataclass
class CapsuleMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    radii: Tuple[float, ...] = ()
    heights: Tuple[float, ...] = ()
    transformations: Tuple[np.ndarray, ...] = ()


@dataclass
class CylinderMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    radii: Tuple[float, ...] = ()
    heights: Tuple[float, ...] = ()
    transformations: Tuple[np.ndarray, ...] = ()


@dataclass
class BoxMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    box_dims: Tuple[Tuple[float, float, float], ...] = ()
    transformations: Tuple[np.ndarray, ...] = ()


@dataclass
class RandomMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    meshes: Tuple[Union[CapsuleMeshPartsCfg, BoxMeshPartsCfg], ...] = ()


@dataclass
class CombinedMeshPartsCfg(MeshPartsCfg):
    add_floor: bool = True
    cfgs: Tuple[MeshPartsCfg, ...] = ()
