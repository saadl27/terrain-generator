import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh

from ...utils import merge_meshes
from ..mesh_parts.create_tiles import build_mesh
from ..mesh_parts.mesh_parts_cfg import PlatformMeshPartsCfg
from ..mesh_parts.part_presets import make_corner_cfg, make_platform_cfg, make_platform_for_stage


TOTAL_CURRICULUM_LEVELS = 38
DEFAULT_MESH_EXTENSION = "obj"

FLOOR_THICKNESS = 0.12
WALL_THICKNESS = 0.18
MIN_TERRAIN_WIDTH = 12.0
MIN_TERRAIN_LENGTH = 12.0
SIDE_PADDING = 1.0
END_PADDING = 1.0
OVERLAY_EPS = 1.0e-3
CORNER_LOOP_CELL_MARGIN = WALL_THICKNESS

USE_GROUNDED_SIDE_WALLS = True
USE_COMMON_GROUND = True
FILTER_UNUSED_LINEAR_AREA = True
USE_ARENA_WALLS = True
EXTEND_LINEAR_FINAL_PLATEAU = True
SIDE_WALL_EXTRA_HEIGHT = 2.0
ADD_FINAL_STAIR_END_WALL = True
ADD_FINAL_LINEAR_SLOPE_END_WALL = True
COURSE_WIDTH_BOOST = 4.0
MIN_EFFECTIVE_CORRIDOR_WIDTH = 8.0
U_TURN_STAGE_GAP = 2.0
U_TURN_COMMON_GROUND = False
DEFAULT_ROW_GAP = 0.0
DEFAULT_CATEGORY_GAP = 0.0

LINEAR_NUM_SEGMENTS = 3
LINEAR_STAIRS_MAX_STAGE_RISE = 12 * 0.18
LINEAR_SLOPES_MAX_STAGE_RISE = float(np.tan(np.deg2rad(28.0)) * 6.2)
LINEAR_SHARED_GROUNDED_WALL_HEIGHT = (
    LINEAR_NUM_SEGMENTS * max(LINEAR_STAIRS_MAX_STAGE_RISE, LINEAR_SLOPES_MAX_STAGE_RISE)
    + SIDE_WALL_EXTRA_HEIGHT
)
ARENA_WALL_HEIGHT = LINEAR_SHARED_GROUNDED_WALL_HEIGHT

CATEGORY_ORDER = (
    "flat",
    "linear_stairs",
    "linear_slopes",
    "corner",
    "turning_stairs",
    "turning_slopes",
)

CATEGORY_LABELS = {
    "flat": "Flat",
    "linear_stairs": "Linear Stairs",
    "linear_slopes": "Linear Slopes",
    "corner": "Corner Loops",
    "turning_stairs": "Rotating Stairs",
    "turning_slopes": "Rotating Slopes",
}

TURN_ANGLE_SEQUENCE = tuple(float(angle) for angle in range(0, 91, 10))
TURNING_RAMP_LEVELS = max(2, TOTAL_CURRICULUM_LEVELS // 2 + 1)


@dataclass
class TerrainScene:
    terrain_id: str
    label: str
    mesh: trimesh.Trimesh
    metadata: Dict[str, object]


@dataclass
class CurriculumLevel:
    level: int
    terrains: Tuple[TerrainScene, ...]
    level_mesh: trimesh.Trimesh
    metadata: Dict[str, object]


@dataclass
class CurriculumCategory:
    category_id: str
    label: str
    terrains: Tuple[TerrainScene, ...]
    category_mesh: trimesh.Trimesh
    metadata: Dict[str, object]


@dataclass(frozen=True)
class CurriculumLayoutCfg:
    min_cell_width: float = MIN_TERRAIN_WIDTH
    min_cell_length: float = MIN_TERRAIN_LENGTH
    terrain_padding_x: float = 0.0
    terrain_padding_y: float = 0.0
    divider_wall_thickness: float = 0.20
    divider_wall_height: float = 1.20
    row_gap: float = DEFAULT_ROW_GAP
    category_gap: float = DEFAULT_CATEGORY_GAP
    center_rows_on_origin: bool = True


def validate_level(level: int) -> None:
    if level < 1 or level > TOTAL_CURRICULUM_LEVELS:
        raise ValueError(f"level must be in [1, {TOTAL_CURRICULUM_LEVELS}], got {level}.")


def normalize_levels(levels: Optional[Iterable[int]]) -> List[int]:
    if levels is None:
        return list(range(1, TOTAL_CURRICULUM_LEVELS + 1))
    normalized = sorted({int(level) for level in levels})
    for level in normalized:
        validate_level(level)
    return normalized


def round_float(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


def mesh_extents(mesh: trimesh.Trimesh) -> Dict[str, float]:
    extents = mesh.bounds[1] - mesh.bounds[0]
    return {"x": round_float(extents[0]), "y": round_float(extents[1]), "z": round_float(extents[2])}


def effective_corridor_width(width: float) -> float:
    return max(MIN_EFFECTIVE_CORRIDOR_WIDTH, width + COURSE_WIDTH_BOOST)


def lerp(start: float, end: float, t: float) -> float:
    return float(start + (end - start) * t)


def lerp_int(start: int, end: int, t: float) -> int:
    return int(round(lerp(float(start), float(end), t)))


def difficulty_ratio(level: int) -> float:
    validate_level(level)
    if TOTAL_CURRICULUM_LEVELS <= 1:
        return 0.0
    return float(level - 1) / float(TOTAL_CURRICULUM_LEVELS - 1)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1.0e-8:
        raise ValueError("Zero-length vector is not allowed.")
    return vec / norm


def left_normal(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[1], vec[0]])


def cross_2d(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])


def line_intersection(point_a: np.ndarray, dir_a: np.ndarray, point_b: np.ndarray, dir_b: np.ndarray) -> np.ndarray:
    denom = cross_2d(dir_a, dir_b)
    if abs(denom) < 1.0e-8:
        raise ValueError("Line intersection is undefined for parallel directions.")
    diff = point_b - point_a
    scale_a = cross_2d(diff, dir_b) / denom
    return point_a + scale_a * dir_a


def ray_to_rectangle_boundary(
    point: np.ndarray,
    direction: np.ndarray,
    half_width: float,
    half_length: float,
) -> np.ndarray:
    candidates = []
    if abs(direction[0]) > 1.0e-8:
        for bound_x in (-half_width, half_width):
            scale = (bound_x - point[0]) / direction[0]
            if scale > 0.0:
                hit_y = point[1] + scale * direction[1]
                if -half_length - 1.0e-8 <= hit_y <= half_length + 1.0e-8:
                    candidates.append(scale)
    if abs(direction[1]) > 1.0e-8:
        for bound_y in (-half_length, half_length):
            scale = (bound_y - point[1]) / direction[1]
            if scale > 0.0:
                hit_x = point[0] + scale * direction[0]
                if -half_width - 1.0e-8 <= hit_x <= half_width + 1.0e-8:
                    candidates.append(scale)
    if len(candidates) == 0:
        raise ValueError("Failed to extend corridor ray to the rectangle frontier.")
    return point + min(candidates) * direction


def polygon_mask(xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(xs.shape, dtype=bool)
    px = polygon[:, 0]
    py = polygon[:, 1]
    prev_idx = len(polygon) - 1
    for idx in range(len(polygon)):
        xi, yi = px[idx], py[idx]
        xj, yj = px[prev_idx], py[prev_idx]
        intersects = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / (yj - yi + 1.0e-12) + xi
        )
        mask ^= intersects
        prev_idx = idx
    return mask


def build_corner_plateau_mesh(
    *,
    width: float,
    length: float,
    corridor_width: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    wall_height: float,
) -> trimesh.Trimesh:
    half_width = 0.5 * width
    half_length = 0.5 * length

    incoming_dir = np.array([0.0, 1.0])
    outgoing_dir = unit(
        np.array(
            [
                -np.sin(np.deg2rad(turn_angle_deg)),
                np.cos(np.deg2rad(turn_angle_deg)),
            ]
        )
    )

    outer_sign = -1.0 if turn_angle_deg > 0.0 else 1.0
    incoming_outer_offset = outer_sign * (pre_corridor_width / 2.0) * left_normal(incoming_dir)
    outgoing_outer_offset = outer_sign * (post_corridor_width / 2.0) * left_normal(outgoing_dir)
    incoming_inner_offset = -outer_sign * (pre_corridor_width / 2.0) * left_normal(incoming_dir)
    outgoing_inner_offset = -outer_sign * (post_corridor_width / 2.0) * left_normal(outgoing_dir)

    outer_join = line_intersection(incoming_outer_offset, incoming_dir, outgoing_outer_offset, outgoing_dir)
    inner_join = line_intersection(incoming_inner_offset, incoming_dir, outgoing_inner_offset, outgoing_dir)

    incoming_outer_start = ray_to_rectangle_boundary(incoming_outer_offset, -incoming_dir, half_width, half_length)
    incoming_inner_start = ray_to_rectangle_boundary(incoming_inner_offset, -incoming_dir, half_width, half_length)
    outgoing_outer_end = ray_to_rectangle_boundary(outgoing_outer_offset, outgoing_dir, half_width, half_length)
    outgoing_inner_end = ray_to_rectangle_boundary(outgoing_inner_offset, outgoing_dir, half_width, half_length)

    corridor_polygon = np.array(
        [
            incoming_outer_start,
            outer_join,
            outgoing_outer_end,
            outgoing_inner_end,
            inner_join,
            incoming_inner_start,
        ]
    )

    resolution = max(72, int(round(max(width, length) * 6)))
    dx = width / resolution
    dy = length / resolution
    xs = np.linspace(-half_width + 0.5 * dx, half_width - 0.5 * dx, resolution)
    ys = np.linspace(-half_length + 0.5 * dy, half_length - 0.5 * dy, resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    corridor_mask = polygon_mask(xx, yy, corridor_polygon)

    height_array = np.full((resolution, resolution), wall_height, dtype=float)
    height_array[corridor_mask] = FLOOR_THICKNESS

    cfg = PlatformMeshPartsCfg(
        name="corner_plateau",
        dim=(width, length, wall_height),
        floor_thickness=FLOOR_THICKNESS,
        minimal_triangles=False,
        array=height_array,
        add_floor=True,
        load_from_cache=False,
    )
    return normalize_ground_center(build_mesh(cfg))


def corner_floor_outline(
    *,
    pre_length: float,
    post_length: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    wall_thickness: float,
) -> np.ndarray:
    incoming_outer_width = pre_corridor_width + 2.0 * wall_thickness
    outgoing_outer_width = post_corridor_width + 2.0 * wall_thickness

    incoming_dir = np.array([0.0, 1.0])
    outgoing_dir = unit(
        np.array(
            [
                -np.sin(np.deg2rad(turn_angle_deg)),
                np.cos(np.deg2rad(turn_angle_deg)),
            ]
        )
    )

    outer_sign = -1.0 if turn_angle_deg > 0.0 else 1.0
    incoming_outer_offset = outer_sign * (incoming_outer_width / 2.0) * left_normal(incoming_dir)
    outgoing_outer_offset = outer_sign * (outgoing_outer_width / 2.0) * left_normal(outgoing_dir)
    incoming_inner_offset = -outer_sign * (incoming_outer_width / 2.0) * left_normal(incoming_dir)
    outgoing_inner_offset = -outer_sign * (outgoing_outer_width / 2.0) * left_normal(outgoing_dir)

    outer_floor_join = line_intersection(incoming_outer_offset, incoming_dir, outgoing_outer_offset, outgoing_dir)
    inner_floor_join = line_intersection(incoming_inner_offset, incoming_dir, outgoing_inner_offset, outgoing_dir)

    incoming_outer_start = incoming_outer_offset - incoming_dir * pre_length
    incoming_inner_start = incoming_inner_offset - incoming_dir * pre_length
    outgoing_outer_end = outgoing_outer_offset + outgoing_dir * post_length
    outgoing_inner_end = outgoing_inner_offset + outgoing_dir * post_length

    return np.array(
        [
            incoming_outer_start,
            outer_floor_join,
            outgoing_outer_end,
            outgoing_inner_end,
            inner_floor_join,
            incoming_inner_start,
        ]
    )


def normalize_ground_center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    normalized = mesh.copy()
    if len(normalized.vertices) == 0:
        return normalized
    bounds = normalized.bounds
    normalized.apply_translation([0.0, 0.0, -bounds[0, 2]])
    bounds = normalized.bounds
    normalized.apply_translation(
        [
            -0.5 * (bounds[0, 0] + bounds[1, 0]),
            -0.5 * (bounds[0, 1] + bounds[1, 1]),
            0.0,
        ]
    )
    return normalized


def box_mesh(size_x: float, size_y: float, size_z: float, center: Tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box((size_x, size_y, size_z), trimesh.transformations.translation_matrix(center))


def make_flat_cfg(name: str, width: float, length: float) -> PlatformMeshPartsCfg:
    return PlatformMeshPartsCfg(
        name=name,
        dim=(width, length, FLOOR_THICKNESS),
        floor_thickness=FLOOR_THICKNESS,
        minimal_triangles=False,
        array=np.zeros((1, 1)),
        add_floor=True,
        load_from_cache=False,
    )


def build_flat_mesh(width: float, length: float) -> trimesh.Trimesh:
    return normalize_ground_center(build_mesh(make_flat_cfg("flat_base", width, length)))


def build_flat_terrain(terrain_id: str, label: str, width: float = 14.0, length: float = 14.0) -> TerrainScene:
    mesh = build_flat_mesh(width, length)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "flat",
            "base_width": round_float(width),
            "base_length": round_float(length),
            "mesh_extents": mesh_extents(mesh),
        },
    )


def copy_terrain_scene(terrain: TerrainScene) -> TerrainScene:
    return TerrainScene(
        terrain_id=terrain.terrain_id,
        label=terrain.label,
        mesh=terrain.mesh.copy(),
        metadata=copy.deepcopy(terrain.metadata),
    )


def merge_feature_with_flat_base(feature_mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, float]]:
    feature_mesh = normalize_ground_center(feature_mesh)
    feature_extents = feature_mesh.bounds[1] - feature_mesh.bounds[0]
    base_width = max(MIN_TERRAIN_WIDTH, float(feature_extents[0]) + 2.0 * SIDE_PADDING)
    base_length = max(MIN_TERRAIN_LENGTH, float(feature_extents[1]) + 2.0 * END_PADDING)
    base_mesh = build_flat_mesh(base_width, base_length)
    lifted_feature = feature_mesh.copy()
    lifted_feature.apply_translation([0.0, 0.0, OVERLAY_EPS])
    mesh = merge_meshes([base_mesh, lifted_feature], False)
    return mesh, {
        "base_width": round_float(base_width),
        "base_length": round_float(base_length),
        "feature_width": round_float(feature_extents[0]),
        "feature_length": round_float(feature_extents[1]),
    }


def build_linear_unused_area_filter_mesh(
    width: float,
    length: float,
    course_outer_width: float,
    height: float,
) -> trimesh.Trimesh:
    side_width = 0.5 * (float(width) - float(course_outer_width))
    if side_width <= 1.0e-6 or height <= 1.0e-6:
        return trimesh.Trimesh()

    half_width = 0.5 * float(width)
    centers_x = (
        -half_width + 0.5 * side_width,
        half_width - 0.5 * side_width,
    )
    return merge_meshes(
        [
            box_mesh(
                side_width,
                float(length),
                float(height),
                (center_x, 0.0, 0.5 * float(height)),
            )
            for center_x in centers_x
        ],
        False,
    )


def build_arena_wall_mesh(
    width: float,
    length: float,
    height: float,
    wall_thickness: float = WALL_THICKNESS,
) -> trimesh.Trimesh:
    width = float(width)
    length = float(length)
    height = float(height)
    wall_thickness = float(wall_thickness)
    if width <= wall_thickness or length <= wall_thickness or height <= 1.0e-6:
        return trimesh.Trimesh()

    half_width = 0.5 * width
    half_length = 0.5 * length
    half_wall = 0.5 * wall_thickness
    return merge_meshes(
        [
            box_mesh(
                wall_thickness,
                length,
                height,
                (-half_width + half_wall, 0.0, 0.5 * height),
            ),
            box_mesh(
                wall_thickness,
                length,
                height,
                (half_width - half_wall, 0.0, 0.5 * height),
            ),
            box_mesh(
                width,
                wall_thickness,
                height,
                (0.0, -half_length + half_wall, 0.5 * height),
            ),
            box_mesh(
                width,
                wall_thickness,
                height,
                (0.0, half_length - half_wall, 0.5 * height),
            ),
        ],
        False,
    )


def build_linear_final_plateau_extension_mesh(
    cell_length: float,
    feature_length: float,
    plateau_width: float,
    plateau_height: float,
    arena_wall_thickness: float = WALL_THICKNESS,
) -> trimesh.Trimesh:
    plateau_height = float(plateau_height)
    plateau_width = float(plateau_width)
    y_start = 0.5 * float(feature_length) - OVERLAY_EPS
    y_end = 0.5 * float(cell_length) - float(arena_wall_thickness)
    plateau_length = y_end - y_start
    if plateau_width <= 1.0e-6 or plateau_height <= 1.0e-6 or plateau_length <= 1.0e-6:
        return trimesh.Trimesh()

    return box_mesh(
        plateau_width,
        plateau_length,
        plateau_height,
        (0.0, 0.5 * (y_start + y_end), 0.5 * plateau_height),
    )


def fit_mesh_to_dimensions(
    mesh: trimesh.Trimesh,
    target_width: float,
    target_length: float,
    force_overlay: bool = False,
    filter_unused_area: bool = False,
    filter_unused_outer_width: Optional[float] = None,
) -> trimesh.Trimesh:
    normalized = normalize_ground_center(mesh)
    extents = normalized.bounds[1] - normalized.bounds[0]
    if (
        not force_overlay
        and not filter_unused_area
        and extents[0] >= target_width - 1.0e-6
        and extents[1] >= target_length - 1.0e-6
    ):
        return normalized

    fitted_width = max(target_width, float(extents[0]))
    fitted_length = max(target_length, float(extents[1]))
    base_mesh = build_flat_mesh(fitted_width, fitted_length)
    lifted = normalized.copy()
    lifted.apply_translation([0.0, 0.0, OVERLAY_EPS])
    meshes = [base_mesh]
    if filter_unused_area:
        filter_mesh = build_linear_unused_area_filter_mesh(
            fitted_width,
            fitted_length,
            float(filter_unused_outer_width) if filter_unused_outer_width is not None else float(extents[0]),
            float(extents[2]) + OVERLAY_EPS,
        )
        if len(filter_mesh.vertices) > 0:
            meshes.append(filter_mesh)
    meshes.append(lifted)
    return merge_meshes(meshes, False)


def build_corner_loop_mesh_for_cell(
    *,
    width: float,
    length: float,
    corridor_width: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    wall_height: float,
    num_corners: int,
    nominal_pre_length: float,
    nominal_post_length: float,
) -> Tuple[trimesh.Trimesh, Dict[str, float]]:
    scale = solve_corner_loop_scale_to_cell(
        target_width=width,
        target_length=length,
        num_corners=num_corners,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        turn_angle_deg=turn_angle_deg,
        nominal_pre_length=nominal_pre_length,
        nominal_post_length=nominal_post_length,
        wall_thickness=WALL_THICKNESS,
        margin=CORNER_LOOP_CELL_MARGIN,
    )
    pre_length = max(1.0e-3, float(nominal_pre_length) * scale)
    post_length = max(1.0e-3, float(nominal_post_length) * scale)
    corner_cfg = make_corner_cfg(
        name="corner_loop_cell_fill",
        corridor_width=corridor_width,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        wall_thickness=WALL_THICKNESS,
        wall_height=wall_height,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=max(2.0, wall_height),
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=turn_angle_deg,
        cap_ends=False,
        load_from_cache=False,
    )
    feature_mesh = normalize_ground_center(build_corner_loop_feature_mesh(corner_cfg, num_corners=num_corners))
    base_mesh = build_flat_mesh(width, length)
    lifted_feature = feature_mesh.copy()
    lifted_feature.apply_translation([0.0, 0.0, OVERLAY_EPS])
    mesh = normalize_ground_center(merge_meshes([base_mesh, lifted_feature], False))
    return mesh, {
        "loop_scale": round_float(scale),
        "pre_length": round_float(pre_length),
        "post_length": round_float(post_length),
        "loop_feature_extents": mesh_extents(feature_mesh),
    }


def expand_terrain_to_cell(terrain: TerrainScene, target_width: float, target_length: float) -> TerrainScene:
    expanded = copy_terrain_scene(terrain)
    terrain_type = expanded.metadata.get("type")
    if terrain_type == "corner":
        expanded.mesh = build_corner_plateau_mesh(
            width=target_width,
            length=target_length,
            corridor_width=float(expanded.metadata["corridor_width"]),
            pre_corridor_width=float(expanded.metadata["pre_corridor_width"]),
            post_corridor_width=float(expanded.metadata["post_corridor_width"]),
            turn_angle_deg=float(expanded.metadata["turn_angle_deg"]),
            wall_height=float(expanded.metadata["wall_height"]),
        )
        expanded.metadata["pattern"] = "cell-filled corner corridor"
        expanded.metadata["fills_non_corridor_area_to_wall_height"] = True
    elif terrain_type == "corner_loop":
        nominal_pre_length = float(expanded.metadata["pre_length"])
        nominal_post_length = float(expanded.metadata["post_length"])
        expanded.mesh, loop_metadata = build_corner_loop_mesh_for_cell(
            width=target_width,
            length=target_length,
            corridor_width=float(expanded.metadata["corridor_width"]),
            pre_corridor_width=float(expanded.metadata["pre_corridor_width"]),
            post_corridor_width=float(expanded.metadata["post_corridor_width"]),
            turn_angle_deg=float(expanded.metadata["turn_angle_deg"]),
            wall_height=float(expanded.metadata["wall_height"]),
            num_corners=int(expanded.metadata["num_corners"]),
            nominal_pre_length=nominal_pre_length,
            nominal_post_length=nominal_post_length,
        )
        expanded.metadata["pattern"] = "closed polygon loop scaled to the allocated cell"
        expanded.metadata["preserves_loop_geometry"] = True
        expanded.metadata["nominal_pre_length"] = round_float(nominal_pre_length)
        expanded.metadata["nominal_post_length"] = round_float(nominal_post_length)
        expanded.metadata["outer_polygon_uses_allocated_cell"] = True
        expanded.metadata.update(loop_metadata)
    else:
        filter_unused_area = bool(expanded.metadata.get("filter_unused_area", False))
        expanded.mesh = fit_mesh_to_dimensions(
            expanded.mesh,
            target_width,
            target_length,
            force_overlay=bool(expanded.metadata.get("fixed_max_height_across_levels", False)),
            filter_unused_area=filter_unused_area,
            filter_unused_outer_width=expanded.metadata.get("filter_unused_outer_width"),
        )
        if filter_unused_area:
            expanded.metadata["filtered_unused_area_to_max_height"] = True
        extend_final_plateau = bool(
            expanded.metadata.get("extend_final_plateau_to_arena", False)
        )
        if extend_final_plateau:
            expanded.mesh = normalize_ground_center(expanded.mesh)
            fitted_length = float(expanded.mesh.bounds[1, 1] - expanded.mesh.bounds[0, 1])
            plateau_mesh = build_linear_final_plateau_extension_mesh(
                fitted_length,
                float(expanded.metadata["feature_length"]),
                float(expanded.metadata["final_plateau_width"]),
                float(expanded.metadata["final_plateau_height"]) + OVERLAY_EPS,
                WALL_THICKNESS,
            )
            if len(plateau_mesh.vertices) > 0:
                expanded.mesh = normalize_ground_center(merge_meshes([expanded.mesh, plateau_mesh], False))
                expanded.metadata["extended_final_plateau_to_arena"] = True
    arena_walls = bool(expanded.metadata.get("arena_walls", USE_ARENA_WALLS))
    if arena_walls:
        expanded.mesh = normalize_ground_center(expanded.mesh)
        arena_height = max(
            float(expanded.mesh.bounds[1, 2] - expanded.mesh.bounds[0, 2]),
            float(expanded.metadata.get("arena_wall_height", ARENA_WALL_HEIGHT)),
        )
        arena_mesh = build_arena_wall_mesh(
            target_width,
            target_length,
            arena_height,
            WALL_THICKNESS,
        )
        if len(arena_mesh.vertices) > 0:
            expanded.mesh = normalize_ground_center(merge_meshes([expanded.mesh, arena_mesh], False))
        expanded.metadata["arena_walls"] = True
        expanded.metadata["arena_wall_height"] = round_float(arena_height)
    expanded.metadata["mesh_extents"] = mesh_extents(expanded.mesh)
    expanded.metadata["allocated_cell_width"] = round_float(target_width)
    expanded.metadata["allocated_cell_length"] = round_float(target_length)
    expanded.metadata["base_width"] = max(
        round_float(target_width),
        float(expanded.metadata.get("base_width", 0.0)),
    )
    expanded.metadata["base_length"] = max(
        round_float(target_length),
        float(expanded.metadata.get("base_length", 0.0)),
    )
    return expanded


def expand_terrains_to_layout(
    terrains: Tuple[TerrainScene, ...],
    layout_cfg: CurriculumLayoutCfg,
) -> Tuple[TerrainScene, ...]:
    column_widths: List[float] = []
    row_length = layout_cfg.min_cell_length

    for terrain in terrains:
        extents = terrain.metadata["mesh_extents"]
        column_widths.append(max(layout_cfg.min_cell_width, extents["x"] + 2.0 * layout_cfg.terrain_padding_x))
        row_length = max(row_length, extents["y"] + 2.0 * layout_cfg.terrain_padding_y)

    return tuple(
        expand_terrain_to_cell(terrain, column_widths[idx], row_length) for idx, terrain in enumerate(terrains)
    )


def stage_rise_from_stairs_cfg(stairs_cfg) -> float:
    return float(stairs_cfg.stairs[0].total_height)


def stage_rise_from_slope_cfg(slope_cfg) -> float:
    return float(np.tan(np.deg2rad(slope_cfg.slope_angle_deg)) * slope_cfg.slope_length)


def turn_angle_with_direction(turn_direction: str, turn_angle_deg: float) -> float:
    if turn_direction == "left":
        return abs(turn_angle_deg)
    if turn_direction == "right":
        return -abs(turn_angle_deg)
    raise ValueError("turn_direction must be 'left' or 'right'.")


def rotate_mesh_z(mesh: trimesh.Trimesh, yaw_deg: float) -> trimesh.Trimesh:
    rotated = mesh.copy()
    rotation = trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0.0, 0.0, 1.0])
    rotated.apply_transform(rotation)
    return rotated


def rotate_point_xy(point_xy: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(yaw_deg)
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ]
    )
    return rotation @ point_xy


def rotate_points_xy(points_xy: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(yaw_deg)
    rotation = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad), np.cos(yaw_rad)],
        ]
    )
    return points_xy @ rotation.T


def corner_connection_geometry(cfg) -> Tuple[np.ndarray, np.ndarray]:
    turn_angle_rad = np.deg2rad(cfg.turn_angle_deg)
    outgoing_heading = np.array([-np.sin(turn_angle_rad), np.cos(turn_angle_rad)])
    outgoing_heading = outgoing_heading / np.linalg.norm(outgoing_heading)
    incoming_edge_center = np.array([0.0, -cfg.pre_length])
    outgoing_edge_center = outgoing_heading * cfg.post_length
    return incoming_edge_center, outgoing_edge_center


def closed_corner_loop_spec(requested_turn_angle_deg: float) -> Tuple[int, float]:
    abs_turn_angle_deg = abs(float(requested_turn_angle_deg))
    if abs_turn_angle_deg <= 1.0e-6:
        raise ValueError("Corner loop turn angle must be non-zero.")
    num_corners = max(3, int(round(360.0 / abs_turn_angle_deg)))
    adjusted_turn_angle_deg = np.sign(requested_turn_angle_deg) * (360.0 / float(num_corners))
    return num_corners, float(adjusted_turn_angle_deg)


def corner_loop_outline_points(
    *,
    num_corners: int,
    pre_length: float,
    post_length: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    wall_thickness: float,
) -> np.ndarray:
    segment_outline = corner_floor_outline(
        pre_length=pre_length,
        post_length=post_length,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        turn_angle_deg=turn_angle_deg,
        wall_thickness=wall_thickness,
    )
    turn_angle_rad = np.deg2rad(turn_angle_deg)
    outgoing_heading = unit(np.array([-np.sin(turn_angle_rad), np.cos(turn_angle_rad)]))
    incoming_edge_center = np.array([0.0, -pre_length])
    outgoing_edge_center = outgoing_heading * post_length

    outlines = []
    current_entry_center = np.array([0.0, 0.0])
    current_yaw_deg = 0.0
    for _ in range(num_corners):
        rotated_outline = rotate_points_xy(segment_outline, current_yaw_deg)
        rotated_entry_center = rotate_point_xy(incoming_edge_center, current_yaw_deg)
        translation_xy = current_entry_center - rotated_entry_center
        outlines.append(rotated_outline + translation_xy)

        rotated_exit_center = rotate_point_xy(outgoing_edge_center, current_yaw_deg) + translation_xy
        current_entry_center = rotated_exit_center
        current_yaw_deg += turn_angle_deg
    return np.vstack(outlines)


def corner_loop_outline_extents(
    *,
    num_corners: int,
    pre_length: float,
    post_length: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    wall_thickness: float,
) -> np.ndarray:
    outline_points = corner_loop_outline_points(
        num_corners=num_corners,
        pre_length=pre_length,
        post_length=post_length,
        pre_corridor_width=pre_corridor_width,
        post_corridor_width=post_corridor_width,
        turn_angle_deg=turn_angle_deg,
        wall_thickness=wall_thickness,
    )
    return np.max(outline_points, axis=0) - np.min(outline_points, axis=0)


def solve_corner_loop_scale_to_cell(
    *,
    target_width: float,
    target_length: float,
    num_corners: int,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    nominal_pre_length: float,
    nominal_post_length: float,
    wall_thickness: float,
    margin: float = 0.0,
) -> float:
    available_width = max(target_width - 2.0 * margin, 1.0e-3)
    available_length = max(target_length - 2.0 * margin, 1.0e-3)

    def fits(scale: float) -> bool:
        extents = corner_loop_outline_extents(
            num_corners=num_corners,
            pre_length=float(nominal_pre_length) * scale,
            post_length=float(nominal_post_length) * scale,
            pre_corridor_width=pre_corridor_width,
            post_corridor_width=post_corridor_width,
            turn_angle_deg=turn_angle_deg,
            wall_thickness=wall_thickness,
        )
        return bool(extents[0] <= available_width + 1.0e-6 and extents[1] <= available_length + 1.0e-6)

    low = 0.0
    high = 1.0
    if fits(high):
        low = high
        while fits(high):
            low = high
            high *= 2.0
            if high > 1024.0:
                break

    for _ in range(40):
        mid = 0.5 * (low + high)
        if fits(mid):
            low = mid
        else:
            high = mid
    return low


def build_corner_loop_feature_mesh(cfg, num_corners: int) -> trimesh.Trimesh:
    corner_mesh = build_mesh(cfg)
    incoming_edge_center, outgoing_edge_center = corner_connection_geometry(cfg)

    meshes = []
    current_entry_center = np.array([0.0, 0.0])
    current_yaw_deg = 0.0

    for _ in range(num_corners):
        rotated_mesh = rotate_mesh_z(corner_mesh, current_yaw_deg)
        rotated_entry_center = rotate_point_xy(incoming_edge_center, current_yaw_deg)
        translation_xy = current_entry_center - rotated_entry_center
        rotated_mesh.apply_translation([translation_xy[0], translation_xy[1], 0.0])
        meshes.append(rotated_mesh)

        rotated_exit_center = rotate_point_xy(outgoing_edge_center, current_yaw_deg) + translation_xy
        current_entry_center = rotated_exit_center
        current_yaw_deg += cfg.turn_angle_deg

    return merge_meshes(meshes, False)


def make_u_turn_platform_for_stage(stage_cfg, *, name: str, stage_gap: float = U_TURN_STAGE_GAP):
    stage_rise = (
        stage_rise_from_stairs_cfg(stage_cfg)
        if hasattr(stage_cfg, "stairs")
        else stage_rise_from_slope_cfg(stage_cfg)
    )
    wall_height = max(1.0, stage_rise)
    return make_platform_cfg(
        name=name,
        width=stage_cfg.dim[0] + stage_cfg.dim[0] + stage_gap,
        length=stage_cfg.dim[0],
        height=stage_rise,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=wall_height,
        wall_edges=("left", "right"),
        load_from_cache=False,
        surface_thickness=FLOOR_THICKNESS,
    )


def make_final_platform_for_stage(stage_cfg, *, name: str):
    stage_rise = (
        stage_rise_from_stairs_cfg(stage_cfg)
        if hasattr(stage_cfg, "stairs")
        else stage_rise_from_slope_cfg(stage_cfg)
    )
    return make_platform_for_stage(
        stage_cfg,
        name=name,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(1.0, stage_rise),
        load_from_cache=False,
        surface_thickness=FLOOR_THICKNESS,
    )


def turn_angle_for_level(level: int) -> float:
    if level > TURNING_RAMP_LEVELS:
        return 180.0
    if TURNING_RAMP_LEVELS <= 1:
        return TURN_ANGLE_SEQUENCE[-1]
    idx = int(round((level - 1) * (len(TURN_ANGLE_SEQUENCE) - 1) / float(TURNING_RAMP_LEVELS - 1)))
    idx = max(0, min(idx, len(TURN_ANGLE_SEQUENCE) - 1))
    return TURN_ANGLE_SEQUENCE[idx]


def annotate_terrain(terrain: TerrainScene, *, level: int, category_id: str) -> TerrainScene:
    annotated = copy_terrain_scene(terrain)
    annotated.metadata["level"] = level
    annotated.metadata["category_id"] = category_id
    return annotated
