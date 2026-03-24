"""Metashape全天球XMLからRealityScan用XMPファイルを生成する.

回転行列の座標変換:
  Metashape: Z-up, c2w, OpenCV camera (X-right, Y-down, Z-forward)
  RS XMP:    Y-up, w2c (転置して格納)

変換手順:
  1. c2w にカメラローカルyaw回転を適用: R = R_c2w @ R_yaw
  2. Z-up→Y-up (Pattern I): R_yup = [R[0], -R[2], R[1]]
  3. 転置してw2cに変換: R_xmp = R_yup^T

位置:
  (x, y, z) → (x, -z, y)

検証: project04で RS align 成功を確認済み (2026-03-23).
"""

from __future__ import annotations

import glob
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 8方向 yaw角マッピング
# ============================================================
ANGLE_MAP_8 = {
    "000": 0.0,
    "045": 45.0,
    "090": 90.0,
    "135": 135.0,
    "180": 180.0,
    "225": 225.0,
    "270": 270.0,
    "315": 315.0,
}


@dataclass
class XmpParams:
    """XMP生成パラメータ."""

    xml_path: str
    image_dir: str
    fov_deg: float = 90.0
    img_width: int = 1920
    img_height: int = 1920
    directions: int = 8


# ============================================================
# 行列演算
# ============================================================
def _mat_mul(A: list, B: list) -> list:
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def _rotation_yaw(yaw_deg: float) -> list:
    """Y軸回り回転行列 (camera local, OpenCV: Z-forward)."""
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return [[c, 0, s],
            [0, 1, 0],
            [-s, 0, c]]


# ============================================================
# 座標変換: Metashape Z-up c2w → RS XMP w2c (Y-up)
# ============================================================
def _convert_rotation(R_c2w: list, yaw_deg: float) -> list:
    """c2w回転行列をMetashape座標系からRS XMP形式(w2c, Y-up)に変換する.

    1. カメラローカルでyaw回転を適用
    2. Z-up→Y-up (Pattern I): row0=R[0], row1=-R[2], row2=R[1]
    3. 転置してw2cに変換（RS XMPはw2cを格納する）
    """
    R_split = _rotation_yaw(yaw_deg)
    R = _mat_mul(R_c2w, R_split)
    # Pattern I: c2w in Y-up
    R_yup = [
        R[0],
        [-R[2][0], -R[2][1], -R[2][2]],
        R[1],
    ]
    # Transpose to w2c
    return [
        [R_yup[0][0], R_yup[1][0], R_yup[2][0]],
        [R_yup[0][1], R_yup[1][1], R_yup[2][1]],
        [R_yup[0][2], R_yup[1][2], R_yup[2][2]],
    ]


def _convert_position(C_world: list) -> list:
    """位置をMetashape Z-up → RS Y-up に変換する.

    Same transform as rotation: S' = [[1,0,0],[0,0,-1],[0,1,0]]
    (x, y, z) → (x, -z, y)
    """
    return [C_world[0], -C_world[2], C_world[1]]


# ============================================================
# XML読み込み
# ============================================================
def _parse_xml_transforms(xml_path: str) -> dict[str, list[float]]:
    """MetashapeのXMLからカメラtransformを読み込む."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    transforms = {}
    for cam in root.findall(".//cameras/camera"):
        label = cam.get("label")
        t_elem = cam.find("transform")
        if t_elem is not None and t_elem.text:
            values = list(map(float, t_elem.text.strip().split()))
            transforms[label] = values
    return transforms


# ============================================================
# XMP生成
# ============================================================
_XMP_TEMPLATE = """\
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xcr:Version="3"
      xcr:PosePrior="exact"
      xcr:Rotation="{rot_str}"
      xcr:Coordinates="absolute"
      xcr:DistortionModel="brown3"
      xcr:DistortionCoeficients="0 0 0 0 0 0"
      xcr:FocalLength35mm="{focal35}"
      xcr:Skew="0"
      xcr:AspectRatio="1"
      xcr:PrincipalPointU="0"
      xcr:PrincipalPointV="0"
      xcr:CalibrationPrior="exact"
      xcr:CalibrationGroup="-1"
      xcr:DistortionGroup="-1"
      xcr:InTexturing="1"
      xcr:InMeshing="1">
      <xcr:Position>{pos_str}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def generate_xmp(params: XmpParams, *, dry_run: bool = False) -> int:
    """XMPファイルを生成する.

    Returns:
        生成したXMPファイル数.
    """
    f_pixels = (params.img_width / 2.0) / math.tan(math.radians(params.fov_deg / 2.0))
    focal35 = f_pixels * 36.0 / params.img_width

    # 方向マップ（8方向固定）
    angle_map = ANGLE_MAP_8

    # 画像ファイルをスキャン
    image_files: dict[str, tuple[str, str, str]] = {}
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(params.image_dir, ext)):
            fname = os.path.basename(path)
            name_no_ext = os.path.splitext(fname)[0]
            parts = name_no_ext.rsplit("_", 1)
            if len(parts) != 2:
                continue
            frame_label, angle_str = parts
            if angle_str in angle_map:
                image_files[fname] = (frame_label, angle_str, name_no_ext)

    # XML transform読み込み
    xml_transforms = _parse_xml_transforms(params.xml_path)

    # 既存XMP削除
    if not dry_run:
        for f in glob.glob(os.path.join(params.image_dir, "*.xmp")):
            os.remove(f)

    # XMP生成
    exported = 0
    for fname in sorted(image_files.keys()):
        frame_label, angle_str, name_no_ext = image_files[fname]
        yaw = angle_map[angle_str]

        if frame_label not in xml_transforms:
            continue

        vals = xml_transforms[frame_label]

        # c2w回転行列 (Metashape Z-up)
        R_c2w = [
            [vals[0], vals[1], vals[2]],
            [vals[4], vals[5], vals[6]],
            [vals[8], vals[9], vals[10]],
        ]
        C_world = [vals[3], vals[7], vals[11]]

        # 座標変換
        R_rs = _convert_rotation(R_c2w, yaw)
        P_rs = _convert_position(C_world)

        rot_str = " ".join(f"{R_rs[i][j]}" for i in range(3) for j in range(3))
        pos_str = f"{P_rs[0]} {P_rs[1]} {P_rs[2]}"

        if not dry_run:
            xmp_path = os.path.join(params.image_dir, name_no_ext + ".xmp")
            with open(xmp_path, "w", encoding="utf-8") as f:
                f.write(_XMP_TEMPLATE.format(
                    rot_str=rot_str,
                    focal35=focal35,
                    pos_str=pos_str,
                ))

        exported += 1

    return exported
