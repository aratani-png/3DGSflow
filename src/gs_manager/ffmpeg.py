"""FFmpegコマンド生成."""

from __future__ import annotations

from pathlib import Path

from .models import ProjectConfig


def _yaw_angles(directions: int) -> list[float]:
    """方向数に基づいてyaw角の一覧を生成する."""
    step = 360.0 / directions
    return [step * i for i in range(directions)]


def generate_split_commands(config: ProjectConfig, input_video: str) -> list[str]:
    """v360フィルタを使ったFFmpegコマンド一覧を生成する."""
    sp = config.split_params
    yaws = _yaw_angles(sp.directions)
    commands = []

    for frame_idx in range(config.frame_start, config.frame_end + 1, config.frame_step):
        for dir_idx, yaw in enumerate(yaws):
            output_name = f"frame{frame_idx:06d}_dir{dir_idx:02d}.jpg"
            output_path = config.base / "images_P" / output_name

            v360_filter = (
                f"v360={sp.input_format}:flat:"
                f"h_fov={sp.fov}:v_fov={sp.fov}:"
                f"yaw={yaw}:pitch=0:roll=0"
            )

            cmd = (
                f'"{config.ffmpeg_path}" -y '
                f'-i "{input_video}" '
                f'-vf "select=eq(n\\,{frame_idx}),{v360_filter},'
                f'scale={sp.width}:{sp.height}" '
                f'-frames:v 1 -q:v 2 '
                f'"{output_path}"'
            )
            commands.append(cmd)

    return commands


def write_batch_script(commands: list[str], path: Path) -> Path:
    """.batファイルにコマンドを書き出す."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["@echo off", f"echo 全{len(commands)}コマンドを実行します...", ""]
    for i, cmd in enumerate(commands, 1):
        lines.append(f"echo [{i}/{len(commands)}] 処理中...")
        lines.append(cmd)
        lines.append("")
    lines.append("echo 完了しました。")
    lines.append("pause")
    path.write_text("\r\n".join(lines), encoding="utf-8")
    return path
