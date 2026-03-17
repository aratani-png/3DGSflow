"""YAML設定の読み書き."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .models import ProjectConfig, SplitParams

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def save_config(config: ProjectConfig) -> Path:
    """設定をYAMLファイルに保存する."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIGS_DIR / f"{config.name}.yaml"
    data = asdict(config)
    path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    return path


def load_config(name: str) -> ProjectConfig:
    """YAMLファイルから設定を読み込む."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    split_data = data.pop("split_params", {})
    split_params = SplitParams(**split_data)
    return ProjectConfig(split_params=split_params, **data)


def list_configs() -> list[str]:
    """保存済みプロジェクト名の一覧を返す."""
    if not CONFIGS_DIR.exists():
        return []
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
