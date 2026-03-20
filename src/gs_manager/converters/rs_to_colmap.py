"""RealityScan CSV + メッシュPLY + XMP → COLMAP形式変換.

ワークフロー:
  1. RSでAlign → メッシュ構築(High Detail) → CSV + メッシュPLY エクスポート
  2. 本スクリプトでCOLMAP形式に変換
  3. LichtFeld Studioで colmap/ フォルダを開いてトレーニング

XMPの回転行列:
  divisionのXMPは転置済み(w2c)なのでそのまま使用する。
  再転置すると二重転置になり、全カメラが同じ方向を向いてしまう。

COLMAP images.txt:
  - 回転: w2c クォータニオン (wxyz, スカラーファースト)
  - 並進: T = -R_w2c @ C (Cはカメラ中心 = CSVのx,y,alt)

フォルダ構成:
  colmap/
  ├── images/         ← 画像フォルダへのジャンクション
  ├── cameras.txt     ← PINHOLE 1920x1920 f=960
  ├── images.txt      ← カメラ姿勢
  ├── points3D.ply    ← メッシュPLYへのハードリンク
  └── points3D.txt    ← 空(fallback)
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil

import numpy as np


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3回転行列 → クォータニオン (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return (0.25 / s, (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def generate_colmap(
    csv_path: str,
    ply_path: str,
    xmp_dir: str,
    colmap_dir: str,
    img_width: int = 1920,
    img_height: int = 1920,
    focal: float = 960.0,
) -> int:
    """CSV + PLY + XMP から COLMAP フラット構造を生成する.

    Args:
        csv_path: RSエクスポートのCSVファイルパス.
        ply_path: RSエクスポートのPLYファイルパス (メッシュ推奨).
        xmp_dir: 転置済みXMP + JPEGが入ったフォルダパス.
        colmap_dir: 出力先COLMAPフォルダパス.
        img_width: 画像幅.
        img_height: 画像高さ.
        focal: 焦点距離 (px). FOV90°, 1920px → 960.

    Returns:
        COLMAPに書き出したカメラ数.
    """
    os.makedirs(colmap_dir, exist_ok=True)

    # Junction for images
    colmap_img_dir = os.path.join(colmap_dir, "images")
    if not os.path.exists(colmap_img_dir):
        os.system(f'mklink /J "{colmap_img_dir}" "{xmp_dir}"')

    # 1. Read CSV positions
    positions: dict[str, list[float]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["#name"]
            positions[name] = [float(row["x"]), float(row["y"]), float(row["alt"])]

    # 2. Read XMP rotations (already w2c transposed)
    cam_data: dict[str, dict] = {}
    for name in sorted(positions.keys()):
        xmp_path = os.path.join(xmp_dir, os.path.splitext(name)[0] + ".xmp")
        if not os.path.exists(xmp_path):
            continue
        with open(xmp_path, "r", encoding="utf-8") as f:
            content = f.read()
        rot_match = re.search(r'xcr:Rotation="([^"]+)"', content)
        if not rot_match:
            continue
        vals = list(map(float, rot_match.group(1).split()))
        R_w2c = np.array([
            [vals[0], vals[1], vals[2]],
            [vals[3], vals[4], vals[5]],
            [vals[6], vals[7], vals[8]],
        ])
        C = np.array(positions[name])
        T = -R_w2c @ C
        cam_data[name] = {"R_w2c": R_w2c, "T": T}

    # 3. cameras.txt
    cx = img_width / 2.0
    cy = img_height / 2.0
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {img_width} {img_height} {focal} {focal} {cx} {cy}\n")

    # 4. images.txt
    sorted_names = sorted(cam_data.keys())
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(sorted_names)}\n")
        for img_id, name in enumerate(sorted_names, 1):
            d = cam_data[name]
            qw, qx, qy, qz = _rot_to_quat(d["R_w2c"])
            t = d["T"]
            f.write(f"{img_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n")
            f.write("\n")

    # 5. PLY hardlink
    ply_dst = os.path.join(colmap_dir, "points3D.ply")
    if not os.path.exists(ply_dst):
        os.system(f'mklink /H "{ply_dst}" "{ply_path}"')

    # 6. Empty points3D.txt
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

    return len(sorted_names)
