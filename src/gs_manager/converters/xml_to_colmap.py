"""Metashape透視投影XML + PLY → COLMAP形式変換 (LichtFeld Studio用).

Metashapeでカメラ最適化したXML (frame/pinhole, c2w, Z-up) と PLY を
LichtFeld Studioが読み込めるCOLMAP形式 (Y-up) に変換する。

座標変換:
  Metashape: Z-up, c2w
  COLMAP/LichtFeld: Y-up, w2c
  変換行列 S = [[1,0,0],[0,0,-1],[0,1,0]]
  位置: (x,y,z) → (x,-z,y)
  回転: R_c2w_yup = S @ R_c2w_zup → R_w2c = R_c2w_yup^T

使い方:
  python xml_to_colmap.py --xml camera.xml --ply points.ply --images ./perspective --output ./colmap
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np


# Z-up → Y-up 変換行列
S_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


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


def _convert_ply_zup_to_yup(src: str, dst: str) -> int:
    """PLYの全頂点をZ-up→Y-upに変換する.

    対応フォーマット: binary_little_endian, x y z nx ny nz r g b (27 bytes/vertex).

    Returns:
        変換した頂点数.
    """
    with open(src, "rb") as fin:
        header_bytes = b""
        header_lines = []
        while True:
            line = fin.readline()
            header_bytes += line
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        vertex_count = 0
        for l in header_lines:
            if l.startswith("element vertex"):
                vertex_count = int(l.split()[-1])

        # float x,y,z,nx,ny,nz + uchar r,g,b = 27 bytes
        props = [l for l in header_lines if l.startswith("property")]
        float_props = sum(1 for p in props if "float" in p)
        uchar_props = sum(1 for p in props if "uchar" in p)
        vertex_size = float_props * 4 + uchar_props * 1

        with open(dst, "wb") as fout:
            fout.write(header_bytes)
            for i in range(vertex_count):
                data = fin.read(vertex_size)
                vals = list(struct.unpack(f"<{float_props}f", data[: float_props * 4]))
                rgb = data[float_props * 4:]

                # Position: (x,y,z) → (x,-z,y)
                vals[0], vals[1], vals[2] = vals[0], -vals[2], vals[1]
                # Normals (if present): (nx,ny,nz) → (nx,-nz,ny)
                if float_props >= 6:
                    vals[3], vals[4], vals[5] = vals[3], -vals[5], vals[4]

                fout.write(struct.pack(f"<{float_props}f", *vals))
                fout.write(rgb)

    return vertex_count


def generate_colmap(
    xml_path: str,
    ply_path: str,
    image_dir: str,
    colmap_dir: str,
) -> int:
    """Metashape XML + PLY → COLMAP形式変換.

    Args:
        xml_path: MetashapeエクスポートのカメラXMLパス.
        ply_path: Metashapeエクスポートの点群PLYパス.
        image_dir: 透視投影画像のフォルダパス.
        colmap_dir: 出力先COLMAPフォルダパス.

    Returns:
        COLMAPに書き出したカメラ数.
    """
    os.makedirs(colmap_dir, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Calibration (adjusted優先、なければinitial)
    sensor = root.find(".//sensors/sensor")
    cal = sensor.find('calibration[@class="adjusted"]')
    if cal is None:
        cal = sensor.find('calibration[@class="initial"]')
    f = float(cal.find("f").text)
    cx_off = float(cal.find("cx").text) if cal.find("cx") is not None else 0.0
    cy_off = float(cal.find("cy").text) if cal.find("cy") is not None else 0.0
    res = cal.find("resolution")
    w = int(res.get("width"))
    h = int(res.get("height"))
    cx = w / 2.0 + cx_off
    cy = h / 2.0 + cy_off

    # Parse cameras
    cameras = root.findall(".//cameras/camera")
    cam_data = []

    for cam in cameras:
        t_elem = cam.find("transform")
        if t_elem is None or not t_elem.text:
            continue
        label = cam.get("label")
        fname = label + ".jpg"
        if not os.path.exists(os.path.join(image_dir, fname)):
            continue

        vals = list(map(float, t_elem.text.strip().split()))
        R_c2w_zup = np.array([
            [vals[0], vals[1], vals[2]],
            [vals[4], vals[5], vals[6]],
            [vals[8], vals[9], vals[10]],
        ])
        C_zup = np.array([vals[3], vals[7], vals[11]])

        # Z-up → Y-up
        R_c2w_yup = S_ZUP_TO_YUP @ R_c2w_zup
        C_yup = S_ZUP_TO_YUP @ C_zup

        # w2c
        R_w2c = R_c2w_yup.T
        T = -R_w2c @ C_yup

        qw, qx, qy, qz = _rot_to_quat(R_w2c)
        cam_data.append((fname, qw, qx, qy, qz, T[0], T[1], T[2]))

    cam_data.sort(key=lambda x: x[0])

    # cameras.txt
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as fout:
        fout.write("# Camera list with one line of data per camera:\n")
        fout.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fout.write(f"# Number of cameras: 1\n")
        fout.write(f"1 PINHOLE {w} {h} {f} {f} {cx} {cy}\n")

    # images.txt
    with open(os.path.join(colmap_dir, "images.txt"), "w") as fout:
        fout.write("# Image list with two lines of data per image:\n")
        fout.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fout.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fout.write(f"# Number of images: {len(cam_data)}\n")
        for img_id, (fname, qw, qx, qy, qz, tx, ty, tz) in enumerate(cam_data, 1):
            fout.write(f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {fname}\n")
            fout.write("\n")

    # points3D.txt (empty)
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as fout:
        fout.write("# 3D point list with one line of data per point:\n")
        fout.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fout.write("# Number of points: 0\n")

    # PLY: Z-up → Y-up 変換
    ply_dst = os.path.join(colmap_dir, "points3D.ply")
    print("PLYをY-upに変換中...")
    vtx_count = _convert_ply_zup_to_yup(ply_path, ply_dst)
    print(f"  {vtx_count} 頂点変換完了")

    # Junction for images
    colmap_img = os.path.join(colmap_dir, "images")
    if not os.path.exists(colmap_img):
        os.system(f'mklink /J "{colmap_img}" "{image_dir}"')

    return len(cam_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metashape XML + PLY → COLMAP変換 (LichtFeld Studio用)")
    parser.add_argument("--xml", required=True, help="Metashape XMLファイルパス")
    parser.add_argument("--ply", required=True, help="Metashape PLYファイルパス")
    parser.add_argument("--images", required=True, help="透視投影画像フォルダパス")
    parser.add_argument("--output", required=True, help="出力COLMAPフォルダパス")
    args = parser.parse_args()

    count = generate_colmap(args.xml, args.ply, args.images, args.output)
    print(f"Done! {count} cameras written")
