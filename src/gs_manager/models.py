"""データクラス定義."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SplitParams:
    """FFmpeg分割パラメータ."""

    directions: int = 8
    fov: int = 90
    width: int = 2048
    height: int = 2048
    input_format: str = "equirect"


@dataclass
class ProjectConfig:
    """プロジェクト設定."""

    name: str
    base_path: str
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1
    split_params: SplitParams = field(default_factory=SplitParams)
    ffmpeg_path: str = "ffmpeg"
    route: str = ""
    metashape_xml: str = ""

    @property
    def base(self) -> Path:
        return Path(self.base_path)
