"""Metashape透視投影XML → Postshot用CSV変換.

MetashapeでカメラAlignした結果のXML (frame/pinhole, c2w, Z-up) を
PostshotのCSV形式に変換する。

座標系: Metashape Z-upのまま（PLYと同じ座標系で統一）。

CSV列:
  #name, x, y, alt, heading, pitch, roll, f, px, py, k1, k2, k3, k4, t1, t2

使い方:
  python xml_to_csv.py --xml camera.xml --output cameras.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET

import numpy as np


def _rotation_to_hpr(R: np.ndarray) -> tuple[float, float, float]:
    """c2w回転行列 → heading, pitch, roll (ZYX Euler分解)."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    if abs(R[2, 0]) < 0.99999:
        heading = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        heading = math.degrees(math.atan2(-R[0, 1], R[1, 1]))
        roll = 0.0
    return heading, pitch, roll


def generate_csv(xml_path: str, output_path: str) -> int:
    """Metashape XML → Postshot CSV変換.

    Args:
        xml_path: MetashapeエクスポートのカメラXMLパス.
        output_path: 出力CSVファイルパス.

    Returns:
        書き出したカメラ数.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Calibration (adjusted優先)
    sensor = root.find(".//sensors/sensor")
    cal = sensor.find('calibration[@class="adjusted"]')
    if cal is None:
        cal = sensor.find('calibration[@class="initial"]')

    f_px = float(cal.find("f").text)
    cx_off = float(cal.find("cx").text) if cal.find("cx") is not None else 0.0
    cy_off = float(cal.find("cy").text) if cal.find("cy") is not None else 0.0
    k1 = float(cal.find("k1").text) if cal.find("k1") is not None else 0.0
    k2 = float(cal.find("k2").text) if cal.find("k2") is not None else 0.0
    k3 = float(cal.find("k3").text) if cal.find("k3") is not None else 0.0
    p1 = float(cal.find("p1").text) if cal.find("p1") is not None else 0.0
    p2 = float(cal.find("p2").text) if cal.find("p2") is not None else 0.0

    res = cal.find("resolution")
    w = int(res.get("width"))
    h = int(res.get("height"))

    # 35mm換算焦点距離
    f_35mm = f_px * 36.0 / w
    # 主点オフセット (正規化)
    px = cx_off / w
    py = cy_off / h

    # Parse cameras
    cameras = root.findall(".//cameras/camera")
    rows = []

    for cam in cameras:
        t_elem = cam.find("transform")
        if t_elem is None or not t_elem.text:
            continue
        label = cam.get("label")
        fname = label + ".jpg"

        vals = list(map(float, t_elem.text.strip().split()))
        R_c2w = np.array([
            [vals[0], vals[1], vals[2]],
            [vals[4], vals[5], vals[6]],
            [vals[8], vals[9], vals[10]],
        ])
        C = np.array([vals[3], vals[7], vals[11]])

        heading, pitch, roll = _rotation_to_hpr(R_c2w)

        rows.append([fname, C[0], C[1], C[2], heading, pitch, roll,
                     f_35mm, px, py, k1, k2, k3, 0.0, p1, p2])

    rows.sort(key=lambda r: r[0])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "#name", "x", "y", "alt", "heading", "pitch", "roll",
            "f", "px", "py", "k1", "k2", "k3", "k4", "t1", "t2",
        ])
        for row in rows:
            writer.writerow(row)

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metashape XML → Postshot CSV変換")
    parser.add_argument("--xml", required=True, help="Metashape XMLファイルパス")
    parser.add_argument("--output", required=True, help="出力CSVファイルパス")
    args = parser.parse_args()

    count = generate_csv(args.xml, args.output)
    print(f"Done! {count} cameras written to {args.output}")
