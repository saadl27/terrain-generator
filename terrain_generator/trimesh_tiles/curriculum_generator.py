import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
from perlin_numpy import generate_fractal_noise_2d, generate_perlin_noise_2d

from ..utils import merge_meshes
from .mesh_parts.assembled_parts import (
    make_linear_slopes_mesh,
    make_linear_stairs_mesh,
    make_rotating_slopes_mesh,
    make_rotating_stairs_mesh,
)
from .mesh_parts.create_tiles import build_mesh
from .mesh_parts.mesh_parts_cfg import HeightMapMeshPartsCfg, PlatformMeshPartsCfg
from .mesh_parts.part_presets import (
    make_corner_cfg,
    make_platform_cfg,
    make_platform_for_stage,
    make_slope_cfg,
    make_stairs_cfg,
)


TOTAL_CURRICULUM_LEVELS = 34
DEFAULT_MESH_EXTENSION = "obj"

FLOOR_THICKNESS = 0.12
WALL_THICKNESS = 0.18
MIN_TERRAIN_WIDTH = 12.0
MIN_TERRAIN_LENGTH = 12.0
SIDE_PADDING = 1.0
END_PADDING = 1.0
OVERLAY_EPS = 1.0e-3

USE_GROUNDED_SIDE_WALLS = True
USE_COMMON_GROUND = True
SIDE_WALL_EXTRA_HEIGHT = 2.0
ADD_FINAL_STAIR_END_WALL = True
ADD_FINAL_LINEAR_SLOPE_END_WALL = True
COURSE_WIDTH_BOOST = 4.0
MIN_EFFECTIVE_CORRIDOR_WIDTH = 8.0


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


@dataclass(frozen=True)
class CurriculumLayoutCfg:
    min_cell_width: float = MIN_TERRAIN_WIDTH
    min_cell_length: float = MIN_TERRAIN_LENGTH
    terrain_padding_x: float = 0.0
    terrain_padding_y: float = 0.0
    divider_wall_thickness: float = 0.20
    divider_wall_height: float = 1.20
    row_gap: float = 1.50
    center_rows_on_origin: bool = True


def _validate_level(level: int) -> None:
    if level < 1 or level > TOTAL_CURRICULUM_LEVELS:
        raise ValueError(f"level must be in [1, {TOTAL_CURRICULUM_LEVELS}], got {level}.")


def _normalize_levels(levels: Optional[Iterable[int]]) -> List[int]:
    if levels is None:
        return list(range(1, TOTAL_CURRICULUM_LEVELS + 1))
    normalized = sorted({int(level) for level in levels})
    for level in normalized:
        _validate_level(level)
    return normalized


def _round_float(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)


def _mesh_extents(mesh: trimesh.Trimesh) -> Dict[str, float]:
    extents = mesh.bounds[1] - mesh.bounds[0]
    return {"x": _round_float(extents[0]), "y": _round_float(extents[1]), "z": _round_float(extents[2])}


def _effective_corridor_width(width: float) -> float:
    return max(MIN_EFFECTIVE_CORRIDOR_WIDTH, width + COURSE_WIDTH_BOOST)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1.0e-8:
        raise ValueError("Zero-length vector is not allowed.")
    return vec / norm


def _left_normal(vec: np.ndarray) -> np.ndarray:
    return np.array([-vec[1], vec[0]])


def _cross_2d(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])


def _line_intersection(point_a: np.ndarray, dir_a: np.ndarray, point_b: np.ndarray, dir_b: np.ndarray) -> np.ndarray:
    denom = _cross_2d(dir_a, dir_b)
    if abs(denom) < 1.0e-8:
        raise ValueError("Line intersection is undefined for parallel directions.")
    diff = point_b - point_a
    scale_a = _cross_2d(diff, dir_b) / denom
    return point_a + scale_a * dir_a


def _ray_to_rectangle_boundary(point: np.ndarray, direction: np.ndarray, half_width: float, half_length: float) -> np.ndarray:
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


def _polygon_mask(xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray) -> np.ndarray:
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


def _build_corner_plateau_mesh(
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
    outgoing_dir = _unit(
        np.array(
            [
                -np.sin(np.deg2rad(turn_angle_deg)),
                np.cos(np.deg2rad(turn_angle_deg)),
            ]
        )
    )

    outer_sign = -1.0 if turn_angle_deg > 0.0 else 1.0
    incoming_outer_offset = outer_sign * (pre_corridor_width / 2.0) * _left_normal(incoming_dir)
    outgoing_outer_offset = outer_sign * (post_corridor_width / 2.0) * _left_normal(outgoing_dir)
    incoming_inner_offset = -outer_sign * (pre_corridor_width / 2.0) * _left_normal(incoming_dir)
    outgoing_inner_offset = -outer_sign * (post_corridor_width / 2.0) * _left_normal(outgoing_dir)

    outer_join = _line_intersection(incoming_outer_offset, incoming_dir, outgoing_outer_offset, outgoing_dir)
    inner_join = _line_intersection(incoming_inner_offset, incoming_dir, outgoing_inner_offset, outgoing_dir)

    incoming_outer_start = _ray_to_rectangle_boundary(incoming_outer_offset, -incoming_dir, half_width, half_length)
    incoming_inner_start = _ray_to_rectangle_boundary(incoming_inner_offset, -incoming_dir, half_width, half_length)
    outgoing_outer_end = _ray_to_rectangle_boundary(outgoing_outer_offset, outgoing_dir, half_width, half_length)
    outgoing_inner_end = _ray_to_rectangle_boundary(outgoing_inner_offset, outgoing_dir, half_width, half_length)

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
    corridor_mask = _polygon_mask(xx, yy, corridor_polygon)

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
    return _normalize_ground_center(build_mesh(cfg))


def _normalize_ground_center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
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


def _box_mesh(size_x: float, size_y: float, size_z: float, center: Tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box((size_x, size_y, size_z), trimesh.transformations.translation_matrix(center))


def _make_flat_cfg(name: str, width: float, length: float) -> PlatformMeshPartsCfg:
    return PlatformMeshPartsCfg(
        name=name,
        dim=(width, length, FLOOR_THICKNESS),
        floor_thickness=FLOOR_THICKNESS,
        minimal_triangles=False,
        array=np.zeros((1, 1)),
        add_floor=True,
        load_from_cache=False,
    )


def _build_flat_mesh(width: float, length: float) -> trimesh.Trimesh:
    return _normalize_ground_center(build_mesh(_make_flat_cfg("flat_base", width, length)))


def _build_flat_terrain(terrain_id: str, label: str, width: float = 12.0, length: float = 12.0) -> TerrainScene:
    mesh = _build_flat_mesh(width, length)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "flat",
            "base_width": _round_float(width),
            "base_length": _round_float(length),
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _generate_border_rough_height_map(
    shape: Tuple[int, int],
    rough_band_ratio: float,
    amplitude: float,
    seed: int,
) -> np.ndarray:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    noise = generate_perlin_noise_2d(shape, (4, 4), tileable=(True, True)) * 0.6
    noise += generate_fractal_noise_2d(shape, (4, 4), 3, tileable=(True, True)) * 0.4
    np.random.set_state(rng_state)

    noise = noise / max(np.max(np.abs(noise)), 1.0e-6)
    positive_noise = 0.5 * (noise + 1.0)

    gx = np.linspace(0.0, 1.0, shape[0])
    gy = np.linspace(0.0, 1.0, shape[1])
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    distance_to_edge = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy])
    rough_mask = np.clip((rough_band_ratio - distance_to_edge) / max(rough_band_ratio, 1.0e-6), 0.0, 1.0)
    rough_mask = rough_mask**1.25

    return FLOOR_THICKNESS + amplitude * (0.25 + 0.75 * positive_noise) * rough_mask


def _build_border_rough_terrain(
    terrain_id: str,
    label: str,
    width: float,
    length: float,
    rough_band: float,
    amplitude: float,
    seed: int,
) -> TerrainScene:
    shape = (160, 160)
    rough_band_ratio = rough_band / min(width, length)
    height_map = _generate_border_rough_height_map(shape, rough_band_ratio, amplitude, seed)
    cfg = HeightMapMeshPartsCfg(
        name=terrain_id,
        dim=(width, length, FLOOR_THICKNESS + amplitude + 0.4),
        floor_thickness=FLOOR_THICKNESS,
        minimal_triangles=False,
        height_map=height_map,
        fill_borders=True,
        slope_threshold=1.5,
        simplify=True,
        target_num_faces=6000,
        load_from_cache=False,
    )
    mesh = _normalize_ground_center(build_mesh(cfg))
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "rough_border",
            "base_width": _round_float(width),
            "base_length": _round_float(length),
            "rough_band": _round_float(rough_band),
            "rough_amplitude": _round_float(amplitude),
            "seed": seed,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _merge_feature_with_flat_base(feature_mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, float]]:
    feature_mesh = _normalize_ground_center(feature_mesh)
    feature_extents = feature_mesh.bounds[1] - feature_mesh.bounds[0]
    base_width = max(MIN_TERRAIN_WIDTH, float(feature_extents[0]) + 2.0 * SIDE_PADDING)
    base_length = max(MIN_TERRAIN_LENGTH, float(feature_extents[1]) + 2.0 * END_PADDING)
    base_mesh = _build_flat_mesh(base_width, base_length)
    lifted_feature = feature_mesh.copy()
    lifted_feature.apply_translation([0.0, 0.0, OVERLAY_EPS])
    mesh = merge_meshes([base_mesh, lifted_feature], False)
    return mesh, {"base_width": _round_float(base_width), "base_length": _round_float(base_length)}


def _fit_mesh_to_dimensions(mesh: trimesh.Trimesh, target_width: float, target_length: float) -> trimesh.Trimesh:
    normalized = _normalize_ground_center(mesh)
    extents = normalized.bounds[1] - normalized.bounds[0]
    if extents[0] >= target_width - 1.0e-6 and extents[1] >= target_length - 1.0e-6:
        return normalized

    base_mesh = _build_flat_mesh(max(target_width, float(extents[0])), max(target_length, float(extents[1])))
    lifted = normalized.copy()
    lifted.apply_translation([0.0, 0.0, OVERLAY_EPS])
    return merge_meshes([base_mesh, lifted], False)


def _expand_terrain_to_cell(terrain: TerrainScene, target_width: float, target_length: float) -> TerrainScene:
    if terrain.metadata.get("type") == "corner":
        terrain.mesh = _build_corner_plateau_mesh(
            width=target_width,
            length=target_length,
            corridor_width=float(terrain.metadata["corridor_width"]),
            pre_corridor_width=float(terrain.metadata["pre_corridor_width"]),
            post_corridor_width=float(terrain.metadata["post_corridor_width"]),
            turn_angle_deg=float(terrain.metadata["turn_angle_deg"]),
            wall_height=float(terrain.metadata["wall_height"]),
        )
        terrain.metadata["pattern"] = "cell-filled corner corridor"
        terrain.metadata["fills_non_corridor_area_to_wall_height"] = True
    else:
        terrain.mesh = _fit_mesh_to_dimensions(terrain.mesh, target_width, target_length)
    terrain.metadata["mesh_extents"] = _mesh_extents(terrain.mesh)
    terrain.metadata["allocated_cell_width"] = _round_float(target_width)
    terrain.metadata["allocated_cell_length"] = _round_float(target_length)
    terrain.metadata["base_width"] = max(
        _round_float(target_width),
        float(terrain.metadata.get("base_width", 0.0)),
    )
    terrain.metadata["base_length"] = max(
        _round_float(target_length),
        float(terrain.metadata.get("base_length", 0.0)),
    )
    return terrain


def _expand_terrains_to_layout(
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
        _expand_terrain_to_cell(terrain, column_widths[idx], row_length) for idx, terrain in enumerate(terrains)
    )


def _build_linear_stairs_terrain(
    *,
    terrain_id: str,
    label: str,
    num_steps: int,
    step_height: float,
    step_depth: float,
    corridor_width: float,
    flat_length: float,
    num_segments: int = 3,
) -> TerrainScene:
    effective_corridor_width = _effective_corridor_width(corridor_width)
    stairs_cfg = make_stairs_cfg(
        name=f"{terrain_id}_stairs",
        direction="front",
        corridor_width=effective_corridor_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        num_steps=num_steps,
        step_height=step_height,
        step_depth=step_depth,
        wall_height=max(1.0, num_steps * step_height + FLOOR_THICKNESS),
    )
    grounded_wall_height = num_segments * stairs_cfg.stairs[0].total_height + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_cfg(
        name=f"{terrain_id}_platform",
        width=stairs_cfg.dim[0],
        length=flat_length,
        height=stairs_cfg.stairs[0].total_height,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(0.8, stairs_cfg.stairs[0].total_height),
        wall_edges=("left", "right"),
    )
    feature_mesh = make_linear_stairs_mesh(
        stairs_cfg,
        platform_cfg,
        num_stages=num_segments,
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
    )
    mesh, base_dim = _merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "linear_stairs",
            "pattern": "stairs -> platform -> stairs -> platform -> stairs -> platform",
            "num_segments": num_segments,
            "num_steps_per_segment": num_steps,
            "step_height": _round_float(step_height),
            "step_depth": _round_float(step_depth),
            "corridor_width": _round_float(effective_corridor_width),
            "nominal_corridor_width": _round_float(corridor_width),
            "flat_length_between_segments": _round_float(flat_length),
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": _round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_STAIR_END_WALL,
            **base_dim,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _build_linear_slopes_terrain(
    *,
    terrain_id: str,
    label: str,
    slope_length: float,
    slope_angle_deg: float,
    corridor_width: float,
    flat_length: float,
    num_segments: int = 3,
) -> TerrainScene:
    effective_corridor_width = _effective_corridor_width(corridor_width)
    slope_cfg = make_slope_cfg(
        name=f"{terrain_id}_slope",
        corridor_width=effective_corridor_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=4.0,
        slope_length=slope_length,
        slope_angle_deg=slope_angle_deg,
        slope_resolution=36,
        wall_height=max(1.0, FLOOR_THICKNESS + np.tan(np.deg2rad(slope_angle_deg)) * slope_length),
    )
    platform_height = np.tan(np.deg2rad(slope_angle_deg)) * slope_length
    grounded_wall_height = num_segments * platform_height + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_cfg(
        name=f"{terrain_id}_platform",
        width=slope_cfg.dim[0],
        length=flat_length,
        height=platform_height,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(0.8, platform_height),
        wall_edges=("left", "right"),
    )
    feature_mesh = make_linear_slopes_mesh(
        slope_cfg,
        platform_cfg,
        num_stages=num_segments,
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=ADD_FINAL_LINEAR_SLOPE_END_WALL,
    )
    mesh, base_dim = _merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "linear_slopes",
            "pattern": "slope -> platform -> slope -> platform -> slope -> platform",
            "num_segments": num_segments,
            "slope_length": _round_float(slope_length),
            "slope_angle_deg": _round_float(slope_angle_deg),
            "corridor_width": _round_float(effective_corridor_width),
            "nominal_corridor_width": _round_float(corridor_width),
            "flat_length_between_segments": _round_float(flat_length),
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": _round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_LINEAR_SLOPE_END_WALL,
            **base_dim,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _build_corner_terrain(
    *,
    terrain_id: str,
    label: str,
    corridor_width: float,
    pre_corridor_width: float,
    post_corridor_width: float,
    turn_angle_deg: float,
    pre_length: float,
    post_length: float,
    wall_height: float = 1.4,
) -> TerrainScene:
    effective_corridor_width = _effective_corridor_width(corridor_width)
    effective_pre_corridor_width = _effective_corridor_width(pre_corridor_width)
    effective_post_corridor_width = _effective_corridor_width(post_corridor_width)
    corner_cfg = make_corner_cfg(
        name=f"{terrain_id}_corner",
        corridor_width=effective_corridor_width,
        pre_corridor_width=effective_pre_corridor_width,
        post_corridor_width=effective_post_corridor_width,
        wall_thickness=WALL_THICKNESS,
        wall_height=wall_height,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=max(2.0, wall_height),
        pre_length=pre_length,
        post_length=post_length,
        turn_angle_deg=turn_angle_deg,
    )
    feature_mesh = build_mesh(corner_cfg)
    mesh, base_dim = _merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "corner",
            "corridor_width": _round_float(effective_corridor_width),
            "pre_corridor_width": _round_float(effective_pre_corridor_width),
            "post_corridor_width": _round_float(effective_post_corridor_width),
            "nominal_corridor_width": _round_float(corridor_width),
            "nominal_pre_corridor_width": _round_float(pre_corridor_width),
            "nominal_post_corridor_width": _round_float(post_corridor_width),
            "turn_angle_deg": _round_float(turn_angle_deg),
            "pre_length": _round_float(pre_length),
            "post_length": _round_float(post_length),
            "wall_height": _round_float(wall_height),
            **base_dim,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _build_turning_stairs_terrain(
    *,
    terrain_id: str,
    label: str,
    num_steps: int,
    step_height: float,
    step_depth: float,
    corridor_width: float,
    num_segments: int,
    turn_direction: str,
) -> TerrainScene:
    effective_corridor_width = _effective_corridor_width(corridor_width)
    stairs_cfg = make_stairs_cfg(
        name=f"{terrain_id}_stairs",
        direction="front",
        corridor_width=effective_corridor_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        num_steps=num_steps,
        step_height=step_height,
        step_depth=step_depth,
        wall_height=max(1.0, num_steps * step_height + FLOOR_THICKNESS),
    )
    grounded_wall_height = num_segments * stairs_cfg.stairs[0].total_height + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_for_stage(
        stairs_cfg,
        name=f"{terrain_id}_turn_platform",
        turn_direction=turn_direction,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(1.0, stairs_cfg.stairs[0].total_height),
    )
    feature_mesh = make_rotating_stairs_mesh(
        stairs_cfg,
        platform_cfg,
        num_stages=num_segments,
        turn_direction=turn_direction,
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
        add_final_end_wall=ADD_FINAL_STAIR_END_WALL,
    )
    mesh, base_dim = _merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "turning_stairs",
            "num_segments": num_segments,
            "num_steps_per_segment": num_steps,
            "step_height": _round_float(step_height),
            "step_depth": _round_float(step_depth),
            "corridor_width": _round_float(effective_corridor_width),
            "nominal_corridor_width": _round_float(corridor_width),
            "turn_direction": turn_direction,
            "turn_angle_deg": 90.0,
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": _round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": ADD_FINAL_STAIR_END_WALL,
            **base_dim,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _build_turning_slopes_terrain(
    *,
    terrain_id: str,
    label: str,
    slope_length: float,
    slope_angle_deg: float,
    corridor_width: float,
    num_segments: int,
    turn_direction: str,
) -> TerrainScene:
    effective_corridor_width = _effective_corridor_width(corridor_width)
    slope_cfg = make_slope_cfg(
        name=f"{terrain_id}_slope",
        corridor_width=effective_corridor_width,
        wall_thickness=WALL_THICKNESS,
        floor_thickness=FLOOR_THICKNESS,
        structure_height=4.0,
        slope_length=slope_length,
        slope_angle_deg=slope_angle_deg,
        slope_resolution=36,
        wall_height=max(1.0, FLOOR_THICKNESS + np.tan(np.deg2rad(slope_angle_deg)) * slope_length),
    )
    stage_rise = np.tan(np.deg2rad(slope_angle_deg)) * slope_length
    grounded_wall_height = num_segments * stage_rise + SIDE_WALL_EXTRA_HEIGHT
    platform_cfg = make_platform_for_stage(
        slope_cfg,
        name=f"{terrain_id}_turn_platform",
        turn_direction=turn_direction,
        floor_thickness=FLOOR_THICKNESS,
        wall_thickness=WALL_THICKNESS,
        wall_height=max(1.0, np.tan(np.deg2rad(slope_angle_deg)) * slope_length),
    )
    feature_mesh = make_rotating_slopes_mesh(
        slope_cfg,
        platform_cfg,
        num_stages=num_segments,
        turn_direction=turn_direction,
        grounded_side_walls=USE_GROUNDED_SIDE_WALLS,
        grounded_wall_height=grounded_wall_height,
        common_ground=USE_COMMON_GROUND,
    )
    mesh, base_dim = _merge_feature_with_flat_base(feature_mesh)
    return TerrainScene(
        terrain_id=terrain_id,
        label=label,
        mesh=mesh,
        metadata={
            "terrain_id": terrain_id,
            "label": label,
            "type": "turning_slopes",
            "num_segments": num_segments,
            "slope_length": _round_float(slope_length),
            "slope_angle_deg": _round_float(slope_angle_deg),
            "corridor_width": _round_float(effective_corridor_width),
            "nominal_corridor_width": _round_float(corridor_width),
            "turn_direction": turn_direction,
            "turn_angle_deg": 90.0,
            "grounded_side_walls": USE_GROUNDED_SIDE_WALLS,
            "common_ground": USE_COMMON_GROUND,
            "side_wall_extra_height": _round_float(SIDE_WALL_EXTRA_HEIGHT),
            "add_final_end_wall": False,
            **base_dim,
            "mesh_extents": _mesh_extents(mesh),
        },
    )


def _build_linear_level(level: int, params: Dict[str, float]) -> Tuple[TerrainScene, ...]:
    return (
        _build_linear_stairs_terrain(
            terrain_id="stairs_course",
            label=f"Level {level} stairs path",
            num_steps=int(params["stair_num_steps"]),
            step_height=params["stair_step_height"],
            step_depth=params["stair_step_depth"],
            corridor_width=params["corridor_width"],
            flat_length=params["flat_length"],
        ),
        _build_linear_slopes_terrain(
            terrain_id="slopes_course",
            label=f"Level {level} slope path",
            slope_length=params["slope_length"],
            slope_angle_deg=params["slope_angle_deg"],
            corridor_width=params["corridor_width"],
            flat_length=params["flat_length"],
        ),
        _build_corner_terrain(
            terrain_id="corner_course",
            label=f"Level {level} corner path",
            corridor_width=params["corridor_width"],
            pre_corridor_width=params["corner_pre_corridor_width"],
            post_corridor_width=params["corner_post_corridor_width"],
            turn_angle_deg=params["corner_turn_angle_deg"],
            pre_length=params["corner_pre_length"],
            post_length=params["corner_post_length"],
            wall_height=params["corner_wall_height"],
        ),
    )


def _build_turning_level(level: int, params: Dict[str, float]) -> Tuple[TerrainScene, ...]:
    turn_direction = "left" if level % 2 == 0 else "right"
    return (
        _build_turning_stairs_terrain(
            terrain_id="turning_stairs_course",
            label=f"Level {level} turning stairs",
            num_steps=int(params["stair_num_steps"]),
            step_height=params["stair_step_height"],
            step_depth=params["stair_step_depth"],
            corridor_width=params["corridor_width"],
            num_segments=int(params["num_segments"]),
            turn_direction=turn_direction,
        ),
        _build_turning_slopes_terrain(
            terrain_id="turning_slopes_course",
            label=f"Level {level} turning slopes",
            slope_length=params["slope_length"],
            slope_angle_deg=params["slope_angle_deg"],
            corridor_width=params["corridor_width"],
            num_segments=int(params["num_segments"]),
            turn_direction=turn_direction,
        ),
    )


LINEAR_LEVEL_PARAMS: Dict[int, Dict[str, float]] = {
    4: {
        "stair_num_steps": 1,
        "stair_step_height": 0.08,
        "stair_step_depth": 0.45,
        "slope_length": 2.5,
        "slope_angle_deg": 5.0,
        "corridor_width": 4.0,
        "flat_length": 2.0,
        "corner_turn_angle_deg": 20.0,
        "corner_pre_corridor_width": 4.4,
        "corner_post_corridor_width": 4.2,
        "corner_pre_length": 4.0,
        "corner_post_length": 4.0,
        "corner_wall_height": 1.2,
    },
    5: {
        "stair_num_steps": 3,
        "stair_step_height": 0.08,
        "stair_step_depth": 0.45,
        "slope_length": 3.0,
        "slope_angle_deg": 8.0,
        "corridor_width": 4.0,
        "flat_length": 2.0,
        "corner_turn_angle_deg": 45.0,
        "corner_pre_corridor_width": 4.2,
        "corner_post_corridor_width": 4.0,
        "corner_pre_length": 4.0,
        "corner_post_length": 4.0,
        "corner_wall_height": 1.2,
    },
    6: {
        "stair_num_steps": 5,
        "stair_step_height": 0.08,
        "stair_step_depth": 0.45,
        "slope_length": 3.2,
        "slope_angle_deg": 12.0,
        "corridor_width": 3.9,
        "flat_length": 1.8,
        "corner_turn_angle_deg": 70.0,
        "corner_pre_corridor_width": 4.0,
        "corner_post_corridor_width": 3.8,
        "corner_pre_length": 4.0,
        "corner_post_length": 4.0,
        "corner_wall_height": 1.3,
    },
    7: {
        "stair_num_steps": 5,
        "stair_step_height": 0.12,
        "stair_step_depth": 0.45,
        "slope_length": 3.8,
        "slope_angle_deg": 15.0,
        "corridor_width": 3.8,
        "flat_length": 1.8,
        "corner_turn_angle_deg": 90.0,
        "corner_pre_corridor_width": 3.8,
        "corner_post_corridor_width": 3.6,
        "corner_pre_length": 4.2,
        "corner_post_length": 4.2,
        "corner_wall_height": 1.4,
    },
    8: {
        "stair_num_steps": 5,
        "stair_step_height": 0.12,
        "stair_step_depth": 0.55,
        "slope_length": 4.2,
        "slope_angle_deg": 17.0,
        "corridor_width": 3.6,
        "flat_length": 1.6,
        "corner_turn_angle_deg": 90.0,
        "corner_pre_corridor_width": 3.5,
        "corner_post_corridor_width": 3.1,
        "corner_pre_length": 4.2,
        "corner_post_length": 4.2,
        "corner_wall_height": 1.4,
    },
}

for idx, level in enumerate(range(9, 20)):
    LINEAR_LEVEL_PARAMS[level] = {
        "stair_num_steps": 7 + idx // 2,
        "stair_step_height": 0.13 + 0.005 * idx,
        "stair_step_depth": 0.50 + 0.01 * idx,
        "slope_length": 4.4 + 0.18 * idx,
        "slope_angle_deg": 18.0 + idx,
        "corridor_width": 3.5 - 0.08 * idx,
        "flat_length": max(1.1, 1.6 - 0.03 * idx),
        "corner_turn_angle_deg": 90.0,
        "corner_pre_corridor_width": 3.4 - 0.08 * idx,
        "corner_post_corridor_width": 3.0 - 0.08 * idx,
        "corner_pre_length": 4.2 + 0.1 * idx,
        "corner_post_length": 4.2 + 0.1 * idx,
        "corner_wall_height": 1.45 + 0.03 * idx,
    }


TURNING_LEVEL_PARAMS: Dict[int, Dict[str, float]] = {}
for idx, level in enumerate(range(20, TOTAL_CURRICULUM_LEVELS + 1)):
    TURNING_LEVEL_PARAMS[level] = {
        "stair_num_steps": 2 + idx // 2,
        "stair_step_height": 0.11 + 0.005 * idx,
        "stair_step_depth": 0.45 + 0.01 * idx,
        "slope_length": 3.6 + 0.18 * idx,
        "slope_angle_deg": 10.0 + 1.2 * idx,
        "corridor_width": 3.4 - 0.05 * idx,
        "num_segments": 2 if level < 28 else 3,
    }


def build_level_01() -> Tuple[TerrainScene, ...]:
    return (_build_flat_terrain("flat", "Level 1 flat terrain", width=14.0, length=14.0),)


def build_level_02() -> Tuple[TerrainScene, ...]:
    return (
        _build_border_rough_terrain(
            "flat_with_rough_border_light",
            "Level 2 flat terrain with light rough border",
            width=14.0,
            length=14.0,
            rough_band=2.2,
            amplitude=0.32,
            seed=2,
        ),
    )


def build_level_03() -> Tuple[TerrainScene, ...]:
    return (
        _build_border_rough_terrain(
            "flat_with_rough_border_medium",
            "Level 3 flat terrain with more rough border",
            width=14.0,
            length=14.0,
            rough_band=3.2,
            amplitude=0.52,
            seed=3,
        ),
    )


def build_level_04() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(4, LINEAR_LEVEL_PARAMS[4])


def build_level_05() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(5, LINEAR_LEVEL_PARAMS[5])


def build_level_06() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(6, LINEAR_LEVEL_PARAMS[6])


def build_level_07() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(7, LINEAR_LEVEL_PARAMS[7])


def build_level_08() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(8, LINEAR_LEVEL_PARAMS[8])


def build_level_09() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(9, LINEAR_LEVEL_PARAMS[9])


def build_level_10() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(10, LINEAR_LEVEL_PARAMS[10])


def build_level_11() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(11, LINEAR_LEVEL_PARAMS[11])


def build_level_12() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(12, LINEAR_LEVEL_PARAMS[12])


def build_level_13() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(13, LINEAR_LEVEL_PARAMS[13])


def build_level_14() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(14, LINEAR_LEVEL_PARAMS[14])


def build_level_15() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(15, LINEAR_LEVEL_PARAMS[15])


def build_level_16() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(16, LINEAR_LEVEL_PARAMS[16])


def build_level_17() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(17, LINEAR_LEVEL_PARAMS[17])


def build_level_18() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(18, LINEAR_LEVEL_PARAMS[18])


def build_level_19() -> Tuple[TerrainScene, ...]:
    return _build_linear_level(19, LINEAR_LEVEL_PARAMS[19])


def build_level_20() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(20, TURNING_LEVEL_PARAMS[20])


def build_level_21() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(21, TURNING_LEVEL_PARAMS[21])


def build_level_22() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(22, TURNING_LEVEL_PARAMS[22])


def build_level_23() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(23, TURNING_LEVEL_PARAMS[23])


def build_level_24() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(24, TURNING_LEVEL_PARAMS[24])


def build_level_25() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(25, TURNING_LEVEL_PARAMS[25])


def build_level_26() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(26, TURNING_LEVEL_PARAMS[26])


def build_level_27() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(27, TURNING_LEVEL_PARAMS[27])


def build_level_28() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(28, TURNING_LEVEL_PARAMS[28])


def build_level_29() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(29, TURNING_LEVEL_PARAMS[29])


def build_level_30() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(30, TURNING_LEVEL_PARAMS[30])


def build_level_31() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(31, TURNING_LEVEL_PARAMS[31])


def build_level_32() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(32, TURNING_LEVEL_PARAMS[32])


def build_level_33() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(33, TURNING_LEVEL_PARAMS[33])


def build_level_34() -> Tuple[TerrainScene, ...]:
    return _build_turning_level(34, TURNING_LEVEL_PARAMS[34])


LEVEL_BUILDERS = {
    1: build_level_01,
    2: build_level_02,
    3: build_level_03,
    4: build_level_04,
    5: build_level_05,
    6: build_level_06,
    7: build_level_07,
    8: build_level_08,
    9: build_level_09,
    10: build_level_10,
    11: build_level_11,
    12: build_level_12,
    13: build_level_13,
    14: build_level_14,
    15: build_level_15,
    16: build_level_16,
    17: build_level_17,
    18: build_level_18,
    19: build_level_19,
    20: build_level_20,
    21: build_level_21,
    22: build_level_22,
    23: build_level_23,
    24: build_level_24,
    25: build_level_25,
    26: build_level_26,
    27: build_level_27,
    28: build_level_28,
    29: build_level_29,
    30: build_level_30,
    31: build_level_31,
    32: build_level_32,
    33: build_level_33,
    34: build_level_34,
}


def _assemble_level_mesh(
    level: int,
    terrains: Tuple[TerrainScene, ...],
    layout_cfg: CurriculumLayoutCfg,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    column_widths: List[float] = []
    max_length = 0.0
    for terrain in terrains:
        cell_width = float(terrain.metadata.get("allocated_cell_width", terrain.metadata["mesh_extents"]["x"]))
        cell_length = float(terrain.metadata.get("allocated_cell_length", terrain.metadata["mesh_extents"]["y"]))
        column_widths.append(cell_width)
        max_length = max(max_length, cell_length)

    row_length = max(layout_cfg.min_cell_length, max_length)
    total_width = sum(column_widths)
    meshes = [
        _box_mesh(
            total_width,
            row_length,
            FLOOR_THICKNESS,
            (0.5 * total_width, 0.0, -0.5 * FLOOR_THICKNESS),
        )
    ]

    terrain_cells = []
    cursor_x = 0.0
    for idx, terrain in enumerate(terrains):
        width = column_widths[idx]
        cell_center_x = cursor_x + 0.5 * width
        terrain_mesh = terrain.mesh.copy()
        terrain_mesh.apply_translation([cell_center_x, 0.0, 0.0])
        meshes.append(terrain_mesh)
        terrain_cells.append(
            {
                "terrain_id": terrain.terrain_id,
                "label": terrain.label,
                "column_index": idx,
                "x_min": _round_float(cursor_x),
                "x_max": _round_float(cursor_x + width),
                "y_min": _round_float(-0.5 * row_length),
                "y_max": _round_float(0.5 * row_length),
                "mesh_extents": dict(terrain.metadata["mesh_extents"]),
            }
        )
        cursor_x += width

    boundaries = [0.0]
    running = 0.0
    for width in column_widths:
        running += width
        boundaries.append(running)

    for boundary_x in boundaries:
        meshes.append(
            _box_mesh(
                layout_cfg.divider_wall_thickness,
                row_length,
                layout_cfg.divider_wall_height,
                (boundary_x, 0.0, 0.5 * layout_cfg.divider_wall_height),
            )
        )
    for boundary_y in (-0.5 * row_length, 0.5 * row_length):
        meshes.append(
            _box_mesh(
                total_width,
                layout_cfg.divider_wall_thickness,
                layout_cfg.divider_wall_height,
                (0.5 * total_width, boundary_y, 0.5 * layout_cfg.divider_wall_height),
            )
        )

    level_mesh = merge_meshes(meshes, False)
    level_mesh = _normalize_ground_center(level_mesh)

    if layout_cfg.center_rows_on_origin:
        offset_x = 0.5 * total_width
        for cell in terrain_cells:
            cell["x_min"] = _round_float(cell["x_min"] - offset_x)
            cell["x_max"] = _round_float(cell["x_max"] - offset_x)

    metadata = {
        "level": level,
        "terrain_count": len(terrains),
        "row_width": _round_float(total_width),
        "row_length": _round_float(row_length),
        "terrain_cells": terrain_cells,
        "mesh_extents": _mesh_extents(level_mesh),
    }
    return level_mesh, metadata


def build_curriculum_level(level: int, layout_cfg: Optional[CurriculumLayoutCfg] = None) -> CurriculumLevel:
    _validate_level(level)
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    terrains = _expand_terrains_to_layout(LEVEL_BUILDERS[level](), layout_cfg)
    level_mesh, level_layout = _assemble_level_mesh(level, terrains, layout_cfg)
    metadata = {
        "level": level,
        "terrain_labels": [terrain.label for terrain in terrains],
        "terrain_ids": [terrain.terrain_id for terrain in terrains],
        "level_layout": level_layout,
    }
    return CurriculumLevel(level=level, terrains=terrains, level_mesh=level_mesh, metadata=metadata)


def build_curriculum(
    levels: Optional[Iterable[int]] = None,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Tuple[CurriculumLevel, ...]:
    level_numbers = _normalize_levels(levels)
    return tuple(build_curriculum_level(level, layout_cfg=layout_cfg) for level in level_numbers)


def build_all_levels_mesh(
    levels: Optional[Iterable[int]] = None,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Tuple[trimesh.Trimesh, Dict[str, object], Tuple[CurriculumLevel, ...]]:
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    curriculum_levels = build_curriculum(levels, layout_cfg=layout_cfg)

    max_width = 0.0
    row_heights: List[float] = []
    for level_bundle in curriculum_levels:
        extents = level_bundle.level_mesh.bounds[1] - level_bundle.level_mesh.bounds[0]
        max_width = max(max_width, float(extents[0]))
        row_heights.append(float(extents[1]))

    meshes = []
    row_layouts = []
    current_top = 0.0
    for idx, level_bundle in enumerate(curriculum_levels):
        row_height = row_heights[idx]
        row_center_y = current_top - 0.5 * row_height
        row_mesh = level_bundle.level_mesh.copy()
        row_mesh.apply_translation([0.0, row_center_y, 0.0])
        meshes.append(row_mesh)
        row_layouts.append(
            {
                "level": level_bundle.level,
                "row_index": idx,
                "y_max": _round_float(current_top),
                "y_min": _round_float(current_top - row_height),
                "row_height": _round_float(row_height),
                "row_width": level_bundle.metadata["level_layout"]["row_width"],
            }
        )
        current_top = current_top - row_height - layout_cfg.row_gap

    all_levels_mesh = merge_meshes(meshes, False)
    if len(all_levels_mesh.vertices) > 0:
        bounds = all_levels_mesh.bounds
        x_offset = 0.5 * (bounds[0, 0] + bounds[1, 0])
        y_offset = 0.5 * (bounds[0, 1] + bounds[1, 1])
        all_levels_mesh.apply_translation(
            [
                -x_offset,
                -y_offset,
                -bounds[0, 2],
            ]
        )
        for row_layout in row_layouts:
            row_layout["y_min"] = _round_float(row_layout["y_min"] - y_offset)
            row_layout["y_max"] = _round_float(row_layout["y_max"] - y_offset)

    metadata = {
        "total_levels": len(curriculum_levels),
        "max_row_width": _round_float(max_width),
        "row_gap": _round_float(layout_cfg.row_gap),
        "rows": row_layouts,
        "mesh_extents": _mesh_extents(all_levels_mesh),
        "layout_cfg": asdict(layout_cfg),
    }
    return all_levels_mesh, metadata, curriculum_levels


def export_curriculum(
    output_dir: Path,
    levels: Optional[Iterable[int]] = None,
    mesh_extension: str = DEFAULT_MESH_EXTENSION,
    layout_cfg: Optional[CurriculumLayoutCfg] = None,
) -> Dict[str, object]:
    layout_cfg = CurriculumLayoutCfg() if layout_cfg is None else layout_cfg
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_levels_mesh, all_levels_layout, curriculum_levels = build_all_levels_mesh(levels, layout_cfg=layout_cfg)

    manifest = {
        "total_levels": len(curriculum_levels),
        "mesh_extension": mesh_extension,
        "all_levels_mesh_file": f"all_levels.{mesh_extension}",
        "all_levels_layout_file": "all_levels_layout.json",
        "levels": [],
    }

    all_levels_mesh.export(output_path / manifest["all_levels_mesh_file"])
    with open(output_path / manifest["all_levels_layout_file"], "w", encoding="utf-8") as f:
        json.dump(all_levels_layout, f, indent=2, sort_keys=True)

    for level_bundle in curriculum_levels:
        level_dir = output_path / f"level_{level_bundle.level:02d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        terrain_records = []
        for terrain in level_bundle.terrains:
            mesh_file = f"{terrain.terrain_id}.{mesh_extension}"
            terrain.mesh.export(level_dir / mesh_file)
            terrain_record = dict(terrain.metadata)
            terrain_record["mesh_file"] = mesh_file
            terrain_records.append(terrain_record)

        level_mesh_file = f"level_{level_bundle.level:02d}_mesh.{mesh_extension}"
        level_bundle.level_mesh.export(level_dir / level_mesh_file)
        level_record = {
            "level": level_bundle.level,
            "terrain_count": len(level_bundle.terrains),
            "terrains": terrain_records,
            "level_mesh_file": level_mesh_file,
            "level_layout": level_bundle.metadata["level_layout"],
        }
        with open(level_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(level_record, f, indent=2, sort_keys=True)

        manifest["levels"].append(
            {
                "level": level_bundle.level,
                "directory": level_dir.name,
                "terrain_count": len(level_bundle.terrains),
                "terrain_ids": [terrain.terrain_id for terrain in level_bundle.terrains],
                "level_mesh_file": level_mesh_file,
            }
        )

    with open(output_path / "curriculum_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
