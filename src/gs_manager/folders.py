"""フォルダ構造作成."""

from __future__ import annotations

from pathlib import Path

from .models import ProjectConfig

FOLDER_NAMES = [
    "images_S",
    "images_P",
    "Spherical",
    "tools",
    "output",
    "output/point_cloud",
    "output/cameras",
    "colmap",
    "colmap/sparse",
    "colmap/dense",
    "distorted",
    "undistorted",
]


def create_project_folders(config: ProjectConfig) -> list[Path]:
    """プロジェクトフォルダ構造を作成する（idempotent）."""
    created = []
    base = config.base
    base.mkdir(parents=True, exist_ok=True)
    for name in FOLDER_NAMES:
        folder = base / name
        folder.mkdir(parents=True, exist_ok=True)
        created.append(folder)
    return created
