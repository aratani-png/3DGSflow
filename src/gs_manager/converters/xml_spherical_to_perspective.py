"""全天球Metashape XML → 透視投影Metashape XML 変換.

全天球（spherical）カメラのアライメント結果XMLから、
FFmpegで分割した透視投影画像用のXMLを生成する。

パターンC（16方向）: 水平8枚(h) + 上4枚(u,pitch+30) + 下4枚(d,pitch-30)
パターンB（12方向）: 水平8枚(h) + 下2枚(d) + 上2枚(u)
パターンA（8方向）:  水平8枚(h)

命名: {フレーム}_{h/u/d}{角度}.jpg
  h=水平(pitch=0), u=上(pitch=+30), d=下(pitch=-30)

使い方:
  python xml_spherical_to_perspective.py \\
    --input XML.xml \\
    --output perspective.xml \\
    --pattern C
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np


# ============================================================
# 分割パターン定義
# ============================================================
PATTERNS = {
    "A": [
        (0, 0, "h000"), (45, 0, "h045"), (90, 0, "h090"), (135, 0, "h135"),
        (180, 0, "h180"), (225, 0, "h225"), (270, 0, "h270"), (315, 0, "h315"),
    ],
    "B": [
        (0, 0, "h000"), (45, 0, "h045"), (90, 0, "h090"), (135, 0, "h135"),
        (180, 0, "h180"), (225, 0, "h225"), (270, 0, "h270"), (315, 0, "h315"),
        (0, -30, "d000"), (180, -30, "d180"),
        (90, 30, "u090"), (270, 30, "u270"),
    ],
    "C": [
        (0, 0, "h000"), (45, 0, "h045"), (90, 0, "h090"), (135, 0, "h135"),
        (180, 0, "h180"), (225, 0, "h225"), (270, 0, "h270"), (315, 0, "h315"),
        (0, 30, "u000"), (90, 30, "u090"), (180, 30, "u180"), (270, 30, "u270"),
        (0, -30, "d000"), (90, -30, "d090"), (180, -30, "d180"), (270, -30, "d270"),
    ],
}


def _rotation_yaw(yaw_deg: float) -> np.ndarray:
    """Y軸周り回転行列（カメラローカル、OpenCV: Z前方）."""
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_pitch(pitch_deg: float) -> np.ndarray:
    """X軸周り回転行列（カメラローカル、OpenCV: Y下方）."""
    t = math.radians(pitch_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def convert_xml(input_path: str, output_path: str, pattern: str = "C",
                crop_size: int = 1920, fov_deg: float = 90.0) -> int:
    """全天球XML → 透視投影XML変換.

    Args:
        input_path: 入力XMLパス（Metashape spherical）
        output_path: 出力XMLパス（Metashape frame/pinhole）
        pattern: 分割パターン ("A", "B", "C")
        crop_size: 出力画像サイズ (px)
        fov_deg: FOV (度)

    Returns:
        生成したカメラ数
    """
    directions = PATTERNS[pattern]

    tree = ET.parse(input_path)
    root = tree.getroot()
    chunk = root.find(".//chunk")

    # 焦点距離計算 (pixels)
    f_pixels = (crop_size / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    # 元のセンサーを透視投影に変更
    sensor = chunk.find(".//sensors/sensor")
    sensor.set("type", "frame")
    resolution = sensor.find("resolution")
    resolution.set("width", str(crop_size))
    resolution.set("height", str(crop_size))

    # キャリブレーション追加
    calib = sensor.find("calibration")
    if calib is not None:
        sensor.remove(calib)
    calib = ET.SubElement(sensor, "calibration", {
        "type": "frame", "class": "initial"
    })
    ET.SubElement(calib, "resolution", {
        "width": str(crop_size), "height": str(crop_size)
    })
    fx_elem = ET.SubElement(calib, "f")
    fx_elem.text = str(f_pixels)
    cx_elem = ET.SubElement(calib, "cx")
    cx_elem.text = "0"
    cy_elem = ET.SubElement(calib, "cy")
    cy_elem.text = "0"

    # 元のカメラ一覧を取得
    cameras_elem = chunk.find("cameras")
    original_cameras = []
    for cam in cameras_elem.findall("camera"):
        t_elem = cam.find("transform")
        if t_elem is not None and t_elem.text:
            vals = list(map(float, t_elem.text.strip().split()))
            R_c2w = np.array([
                [vals[0], vals[1], vals[2]],
                [vals[4], vals[5], vals[6]],
                [vals[8], vals[9], vals[10]],
            ])
            C = np.array([vals[3], vals[7], vals[11]])
            original_cameras.append({
                "label": cam.get("label"),
                "sensor_id": cam.get("sensor_id"),
                "component_id": cam.get("component_id"),
                "R_c2w": R_c2w,
                "C": C,
            })

    # 元のカメラを削除
    for cam in cameras_elem.findall("camera"):
        cameras_elem.remove(cam)

    # 新しいカメラを生成
    cam_id = 0
    for orig in original_cameras:
        R_c2w = orig["R_c2w"]
        C = orig["C"]
        label = orig["label"]

        for yaw_deg, pitch_deg, suffix in directions:
            # 方向回転: まずpitch、次にyaw（カメラローカル座標系）
            R_dir = _rotation_yaw(yaw_deg) @ _rotation_pitch(pitch_deg)
            R_new = R_c2w @ R_dir

            # 4x4変換行列を構成
            transform_vals = [
                R_new[0, 0], R_new[0, 1], R_new[0, 2], C[0],
                R_new[1, 0], R_new[1, 1], R_new[1, 2], C[1],
                R_new[2, 0], R_new[2, 1], R_new[2, 2], C[2],
                0, 0, 0, 1,
            ]
            transform_str = " ".join(f"{v}" for v in transform_vals)

            new_cam = ET.SubElement(cameras_elem, "camera", {
                "id": str(cam_id),
                "sensor_id": orig["sensor_id"],
                "component_id": orig["component_id"],
                "label": f"{label}_{suffix}",
            })
            t_elem = ET.SubElement(new_cam, "transform")
            t_elem.text = transform_str

            cam_id += 1

    # next_id更新
    cameras_elem.set("next_id", str(cam_id))

    # partition（カメラIDリスト）を削除（新しいIDと合わないため）
    for comp in chunk.findall(".//component"):
        partition = comp.find("partition")
        if partition is not None:
            comp.remove(partition)

    # 書き出し
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    return cam_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全天球XML → 透視投影XML変換")
    parser.add_argument("--input", required=True, help="入力XML")
    parser.add_argument("--output", required=True, help="出力XML")
    parser.add_argument("--pattern", default="C", choices=["A", "B", "C"])
    parser.add_argument("--crop-size", type=int, default=1920)
    parser.add_argument("--fov", type=float, default=90.0)
    args = parser.parse_args()

    count = convert_xml(args.input, args.output, args.pattern, args.crop_size, args.fov)
    print(f"Generated {count} cameras (pattern {args.pattern})")
