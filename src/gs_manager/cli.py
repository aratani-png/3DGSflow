"""Click CLIエントリポイント（日本語プロンプト）."""

from __future__ import annotations

import click

from .config import list_configs, load_config, save_config
from .converters.xml_to_xmp import XmpParams, generate_xmp
from .ffmpeg import generate_split_commands, write_batch_script
from .folders import FOLDER_NAMES, create_project_folders
from .models import ProjectConfig, SplitParams


@click.group()
def main():
    """3DGS ワークフローマネージャー"""


@main.command()
def new():
    """新規プロジェクトを対話式で作成する."""
    click.echo("\n=== 3DGS プロジェクトセットアップ ===\n")

    name = click.prompt("プロジェクト名")
    base_path = click.prompt("ベースパス")
    frame_start = click.prompt("フレーム開始", default=0, type=int)
    frame_end = click.prompt("フレーム終了", type=int)
    frame_step = click.prompt("フレームステップ", default=1, type=int)

    click.echo("\n-- FFmpeg分割設定 --")
    directions = click.prompt("方向数", default=8, type=int)
    fov = click.prompt("FOV (度)", default=90, type=int)
    resolution = click.prompt("出力解像度 (幅x高さ)", default="2048x2048")
    width, height = (int(x) for x in resolution.split("x"))
    ffmpeg_path = click.prompt("FFmpegパス", default="ffmpeg")

    config = ProjectConfig(
        name=name,
        base_path=base_path,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
        split_params=SplitParams(
            directions=directions,
            fov=fov,
            width=width,
            height=height,
        ),
        ffmpeg_path=ffmpeg_path,
    )

    click.echo("")
    if click.confirm("設定を保存しますか？", default=True):
        path = save_config(config)
        click.echo(f"→ 設定を保存しました: {path}")

    if click.confirm("フォルダ構造を作成しますか？", default=True):
        created = create_project_folders(config)
        click.echo(f"→ {len(created)}個のフォルダを作成しました")


@main.command()
@click.argument("name")
def prepare(name: str):
    """フォルダ構造を作成する."""
    config = _load_or_exit(name)
    created = create_project_folders(config)
    click.echo(f"→ {len(created)}個のフォルダを作成しました ({config.base_path})")


@main.command()
@click.argument("name")
@click.option("--input", "-i", "input_video", required=True, help="入力動画ファイルパス")
@click.option("--output", "-o", "output_bat", default=None, help="出力.batファイルパス")
def split(name: str, input_video: str, output_bat: str | None):
    """FFmpegコマンドを生成して.batファイルに書き出す."""
    config = _load_or_exit(name)
    commands = generate_split_commands(config, input_video)

    if output_bat is None:
        output_bat = str(config.base / "tools" / "split.bat")

    from pathlib import Path

    bat_path = write_batch_script(commands, Path(output_bat))
    click.echo(f"→ {len(commands)}個のコマンドを生成しました")
    click.echo(f"→ バッチファイル: {bat_path}")


@main.command()
def projects():
    """保存済みプロジェクト一覧を表示する."""
    names = list_configs()
    if not names:
        click.echo("プロジェクトがありません。`gs new` で作成してください。")
        return
    click.echo("=== プロジェクト一覧 ===")
    for name in names:
        click.echo(f"  - {name}")


@main.command()
@click.argument("name")
def info(name: str):
    """プロジェクト設定を表示する."""
    config = _load_or_exit(name)
    click.echo(f"\n=== {config.name} ===")
    click.echo(f"  ベースパス:     {config.base_path}")
    click.echo(f"  フレーム範囲:   {config.frame_start} - {config.frame_end} (ステップ: {config.frame_step})")
    sp = config.split_params
    click.echo(f"  分割方向数:     {sp.directions}")
    click.echo(f"  FOV:            {sp.fov}°")
    click.echo(f"  解像度:         {sp.width}x{sp.height}")
    click.echo(f"  FFmpegパス:     {config.ffmpeg_path}")
    if config.route:
        click.echo(f"  ルート:         {config.route}")
    if config.metashape_xml:
        click.echo(f"  Metashape XML:  {config.metashape_xml}")

    total_frames = len(range(config.frame_start, config.frame_end + 1, config.frame_step))
    total_images = total_frames * sp.directions
    click.echo(f"\n  推定画像数:     {total_images} ({total_frames}フレーム × {sp.directions}方向)")


@main.command()
@click.argument("name")
@click.option("--xml", "xml_path", default=None, help="Metashape XMLファイルパス（未指定ならconfig参照）")
@click.option("--images", "image_dir", default=None, help="分割画像フォルダパス（未指定ならconfig参照）")
@click.option("--fov", default=90.0, type=float, help="FOV (度)")
@click.option("--width", default=1920, type=int, help="画像幅")
@click.option("--height", default=1920, type=int, help="画像高さ")
@click.option("--dry-run", is_flag=True, help="実際のファイルを書き込まずにカウントのみ")
def route(name: str, xml_path: str | None, image_dir: str | None, fov: float,
          width: int, height: int, dry_run: bool):
    """Metashape XMLからRS用XMPファイルを生成する."""
    config = _load_or_exit(name)

    if xml_path is None:
        xml_path = config.metashape_xml
    if not xml_path:
        click.echo("エラー: XMLパスが指定されていません。--xml または config.metashape_xml を設定してください。", err=True)
        raise SystemExit(1)

    if image_dir is None:
        image_dir = str(config.base / "images_P")

    params = XmpParams(
        xml_path=xml_path,
        image_dir=image_dir,
        fov_deg=fov,
        img_width=width,
        img_height=height,
    )

    count = generate_xmp(params, dry_run=dry_run)
    if dry_run:
        click.echo(f"→ {count}個のXMPを生成予定 (dry-run)")
    else:
        click.echo(f"→ {count}個のXMPを生成しました ({image_dir})")


def _load_or_exit(name: str) -> ProjectConfig:
    """設定を読み込む。見つからない場合はエラーメッセージを表示して終了."""
    try:
        return load_config(name)
    except FileNotFoundError:
        click.echo(f"エラー: プロジェクト '{name}' が見つかりません。", err=True)
        click.echo("`gs projects` で一覧を確認してください。", err=True)
        raise SystemExit(1)
