from .common import TerrainScene, build_flat_terrain


def build_category_terrain(level: int) -> TerrainScene:
    return build_flat_terrain("flat", f"Level {level} flat terrain", width=14.0, length=14.0)
