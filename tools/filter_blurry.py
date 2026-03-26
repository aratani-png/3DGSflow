"""透視投影画像のぶれフィルタリング.

FFmpegで分割した透視投影画像のシャープネスを計測し、
ぶれている画像を間引く（削除または移動）。

同一フレーム内で閾値以下の画像のみ除去し、
対応するXMPファイルも同時に処理する。

使い方:
  # シャープネス解析のみ（削除しない）
  python filter_blurry.py --input ./images_P --analyze

  # 閾値以下を削除
  python filter_blurry.py --input ./images_P --threshold 100

  # 閾値以下をbackupフォルダに移動（安全）
  python filter_blurry.py --input ./images_P --threshold 100 --move ./images_P_backup
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def calc_sharpness(image_path: str, resize_width: int = 960) -> float:
    """Laplacianの分散でシャープネスを計算する.

    Args:
        image_path: 画像ファイルパス.
        resize_width: 計算速度のためのリサイズ幅.

    Returns:
        シャープネススコア（高い=シャープ、低い=ぶれ）.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape
    scale = resize_width / w
    img = cv2.resize(img, (resize_width, int(h * scale)))
    return cv2.Laplacian(img, cv2.CV_64F).var()


def analyze_images(image_dir: str, output_csv: str | None = None) -> list[dict]:
    """全画像のシャープネスを計測する.

    Returns:
        [{"name": str, "frame": str, "direction": str, "sharpness": float}, ...]
    """
    files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )

    results = []
    for path in tqdm(files, desc="Analyzing sharpness"):
        fname = os.path.basename(path)
        name_no_ext = os.path.splitext(fname)[0]
        parts = name_no_ext.rsplit("_", 1)
        if len(parts) == 2:
            frame, direction = parts
        else:
            frame, direction = name_no_ext, ""

        score = calc_sharpness(path)
        results.append({
            "name": fname,
            "frame": frame,
            "direction": direction,
            "sharpness": score,
        })

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "frame", "direction", "sharpness"])
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV saved: {output_csv}")

    return results


def filter_blurry(
    image_dir: str,
    threshold: float,
    move_dir: str | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """閾値以下のぶれ画像を間引く.

    Args:
        image_dir: 画像フォルダパス.
        threshold: シャープネス閾値（これ以下を除去）.
        move_dir: 移動先フォルダ（Noneなら削除）.
        dry_run: Trueなら実際に削除/移動しない.

    Returns:
        (除去数, 残り数)
    """
    results = analyze_images(image_dir)
    total = len(results)
    remove_count = 0

    if move_dir and not dry_run:
        os.makedirs(move_dir, exist_ok=True)

    for r in results:
        if r["sharpness"] < threshold:
            img_path = os.path.join(image_dir, r["name"])
            xmp_path = os.path.join(
                image_dir, os.path.splitext(r["name"])[0] + ".xmp"
            )

            if dry_run:
                print(f"  [REMOVE] {r['name']} (score={r['sharpness']:.1f})")
            elif move_dir:
                shutil.move(img_path, os.path.join(move_dir, r["name"]))
                if os.path.exists(xmp_path):
                    shutil.move(
                        xmp_path,
                        os.path.join(
                            move_dir, os.path.splitext(r["name"])[0] + ".xmp"
                        ),
                    )
            else:
                os.remove(img_path)
                if os.path.exists(xmp_path):
                    os.remove(xmp_path)

            remove_count += 1

    remaining = total - remove_count
    return remove_count, remaining


def print_stats(results: list[dict], threshold: float | None = None) -> None:
    """シャープネス統計を表示する."""
    scores = [r["sharpness"] for r in results]
    scores_arr = np.array(scores)

    print(f"\n=== Sharpness Statistics ===")
    print(f"  Total images: {len(scores)}")
    print(f"  Min:    {scores_arr.min():.1f}")
    print(f"  Max:    {scores_arr.max():.1f}")
    print(f"  Mean:   {scores_arr.mean():.1f}")
    print(f"  Median: {np.median(scores_arr):.1f}")
    print(f"  Std:    {scores_arr.std():.1f}")

    # Distribution
    percentiles = [10, 25, 50, 75, 90]
    print(f"\n  Percentiles:")
    for p in percentiles:
        print(f"    {p}%: {np.percentile(scores_arr, p):.1f}")

    if threshold is not None:
        below = sum(1 for s in scores if s < threshold)
        print(f"\n  Threshold {threshold:.1f}: {below} images would be removed ({below/len(scores)*100:.1f}%)")

    # Bottom 20 worst
    results_sorted = sorted(results, key=lambda r: r["sharpness"])
    print(f"\n  Worst 20 images:")
    for r in results_sorted[:20]:
        print(f"    {r['name']}: {r['sharpness']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="透視投影画像ぶれフィルタリング")
    parser.add_argument("--input", required=True, help="画像フォルダパス")
    parser.add_argument("--analyze", action="store_true", help="解析のみ（削除しない）")
    parser.add_argument("--threshold", type=float, default=None, help="シャープネス閾値")
    parser.add_argument("--move", default=None, help="移動先フォルダ（削除の代わりに移動）")
    parser.add_argument("--dry-run", action="store_true", help="実際に削除/移動しない")
    parser.add_argument("--csv", default=None, help="CSV出力パス")
    args = parser.parse_args()

    if args.analyze or args.threshold is None:
        results = analyze_images(args.input, output_csv=args.csv)
        print_stats(results, threshold=args.threshold)
    else:
        removed, remaining = filter_blurry(
            args.input, args.threshold, move_dir=args.move, dry_run=args.dry_run
        )
        print(f"\nRemoved: {removed}, Remaining: {remaining}")
