"""
diagnose_rotation.py

フレーム0753について、4パターンの回転行列を計算し、
RSの正解データ（all.csv）と比較して正しい座標変換を特定する。

パターン:
  A: 現在のスクリプト（c2w + OpenCV camera）
  B: c2w + OpenGL camera（列1,2の符号反転）
  C: w2c（Aの転置）
  D: w2c + OpenGL camera（Bの転置）
"""

import math
import sys

# ============================================================
# フレーム0753のMetashape transform (Z-up, c2w)
# ============================================================
TRANSFORM_0753 = [
    0.34515965388208453, -0.0074331999538787413, 0.93851455016448515, 6.3815265844018425,
    -0.007050733218758487, -0.99996095540905128, -0.0053268019013124196, -8.9760621937098914,
    0.93851750143142132, -0.0047786186145774567, -0.34519858677442283, -58.231274899939308,
    0, 0, 0, 1,
]

# ============================================================
# RS正解データ (all.csv) — フレーム0753
# name: (heading, pitch, roll)
# ============================================================
RS_GROUND_TRUTH = {
    "000": (0.6977253839174097, -1.115906068628737, -69.13622425473217),
    "045": (0.6901970833013632, -1.117849665906873, -114.1391304086573),
    "090": (0.7105126574656648, -1.085199380147509, -158.9925937105814),
    "135": (0.7305767077837578, -1.135212004927059, 155.9322525575753),
    "180": (0.7166977985142575, -1.150782920035236, 110.934931365521),
    "225": (0.6919432712210341, -1.119514805813922, 65.92234271236092),
    "270": (0.6366165024346812, -1.18824879098783, 20.98250381257844),
    "315": (0.678692147162091, -1.08500000301872, -24.07336515953038),
}

ANGLE_MAP = {
    "000": 0.0, "045": 45.0, "090": 90.0, "135": 135.0,
    "180": 180.0, "225": 225.0, "270": 270.0, "315": 315.0,
}


# ============================================================
# 行列演算
# ============================================================
def mat_mul(A, B):
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]


def rotation_yaw(yaw_deg):
    """Y軸回り回転（Metashape Z-up空間ではZ軸回り）."""
    t = math.radians(yaw_deg)
    c, s = math.cos(t), math.sin(t)
    return [[c, 0, s],
            [0, 1, 0],
            [-s, 0, c]]


def opencv_to_opengl(R):
    """OpenCV camera → OpenGL camera: 列1,2を符号反転."""
    return [[R[i][0], -R[i][1], -R[i][2]] for i in range(3)]


def z_up_to_y_up(R):
    """Z-up → Y-up座標変換: S = [[1,0,0],[0,0,1],[0,-1,0]]."""
    return [
        R[0],                                              # row0 そのまま
        R[2],                                              # row1 = 旧row2
        [-R[1][0], -R[1][1], -R[1][2]],                   # row2 = -旧row1
    ]


# ============================================================
# 回転行列 → heading/pitch/roll (RS convention)
#
# RSのCSVがどの回転順序を使っているか不明なので、
# 複数の一般的なEuler分解を試す。
# ============================================================
def rotation_to_euler_xyz(R):
    """R = Rx(pitch) * Ry(heading) * Rz(roll) — extrinsic XYZ."""
    sy = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.degrees(math.atan2(R[1][0], R[0][0]))
        heading = math.degrees(math.atan2(-R[2][0], sy))
        pitch = math.degrees(math.atan2(R[2][1], R[2][2]))
    else:
        roll = math.degrees(math.atan2(-R[0][1], R[1][1]))
        heading = math.degrees(math.atan2(-R[2][0], sy))
        pitch = 0.0
    return heading, pitch, roll


def rotation_to_euler_zyx(R):
    """R = Rz(heading) * Ry(pitch) * Rx(roll) — extrinsic ZYX / intrinsic XYZ."""
    sy = math.sqrt(R[0][0] ** 2 + R[0][1] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.degrees(math.atan2(R[1][2], R[2][2]))
        pitch = math.degrees(math.atan2(-R[0][2], sy))
        heading = math.degrees(math.atan2(R[0][1], R[0][0]))
    else:
        roll = math.degrees(math.atan2(-R[2][1], R[1][1]))
        pitch = math.degrees(math.atan2(-R[0][2], sy))
        heading = 0.0
    return heading, pitch, roll


def rotation_to_euler_yxz(R):
    """R = Ry(heading) * Rx(pitch) * Rz(roll) — common in games/3D."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[1][2]))))
    if abs(R[1][2]) < 0.99999:
        heading = math.degrees(math.atan2(R[0][2], R[2][2]))
        roll = math.degrees(math.atan2(R[1][0], R[1][1]))
    else:
        heading = math.degrees(math.atan2(-R[2][0], R[0][0]))
        roll = 0.0
    return heading, pitch, roll


EULER_FUNCS = {
    "XYZ": rotation_to_euler_xyz,
    "ZYX": rotation_to_euler_zyx,
    "YXZ": rotation_to_euler_yxz,
}


def normalize_angle(a):
    """角度を -180..180 に正規化."""
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


# ============================================================
# パターン生成
# ============================================================
def negate_cols_12(R):
    """列1,2を符号反転 (OpenCV→OpenGL)."""
    return [[R[i][0], -R[i][1], -R[i][2]] for i in range(3)]


def negate_rows_12(R):
    """行1,2を符号反転."""
    return [R[0],
            [-R[1][0], -R[1][1], -R[1][2]],
            [-R[2][0], -R[2][1], -R[2][2]]]


PATTERNS = {
    "A":  "c2w + OpenCV (現行)",
    "B":  "c2w + OpenGL (列1,2反転)",
    "C":  "w2c (Aの転置)",
    "D":  "w2c + OpenGL (Bの転置)",
    "E":  "c2w + OpenCV + 負yaw",
    "F":  "c2w + OpenGL + 負yaw",
    "G":  "w2c + 負yaw",
    "H":  "w2c + OpenGL + 負yaw",
    "I":  "c2w + 行1,2反転 (pre-flip)",
    "J":  "c2w + 行1,2反転 + 負yaw",
}


def compute_pattern(pattern_name, vals, yaw_deg):
    """指定パターンで回転行列を生成して返す."""
    R_c2w = [
        [vals[0], vals[1], vals[2]],
        [vals[4], vals[5], vals[6]],
        [vals[8], vals[9], vals[10]],
    ]

    neg_yaw = pattern_name in ("E", "F", "G", "H", "J")
    yaw = -yaw_deg if neg_yaw else yaw_deg

    R_split = rotation_yaw(yaw)
    R_new_c2w = mat_mul(R_c2w, R_split)

    # Z-up → Y-up
    R_yup = z_up_to_y_up(R_new_c2w)

    if pattern_name in ("A", "E"):
        return R_yup
    elif pattern_name in ("B", "F"):
        return negate_cols_12(R_yup)
    elif pattern_name in ("C", "G"):
        return transpose(R_yup)
    elif pattern_name in ("D", "H"):
        return transpose(negate_cols_12(R_yup))
    elif pattern_name == "I":
        return negate_rows_12(R_yup)
    elif pattern_name == "J":
        return negate_rows_12(R_yup)
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")


# ============================================================
# メイン
# ============================================================
def main():
    vals = TRANSFORM_0753
    directions = sorted(ANGLE_MAP.keys())

    # RS正解のroll差分
    rs_roll_000 = RS_GROUND_TRUTH["000"][2]
    print("=" * 80)
    print("RS正解データ (フレーム0753)")
    print("=" * 80)
    print(f"{'方向':>6}  {'heading':>10}  {'pitch':>10}  {'roll':>10}  {'Δroll':>10}")
    for d in directions:
        h, p, r = RS_GROUND_TRUTH[d]
        dr = normalize_angle(r - rs_roll_000)
        print(f"{d:>6}  {h:10.4f}  {p:10.4f}  {r:10.4f}  {dr:10.4f}")
    print()

    # 各パターン × 各Euler分解
    best_score = float("inf")
    best_combo = None

    for pattern in PATTERNS:
        for euler_name, euler_func in EULER_FUNCS.items():
            print("=" * 80)
            print(f"パターン {pattern} ({PATTERNS[pattern]}) + Euler {euler_name}")
            print("=" * 80)

            results = {}
            for d in directions:
                yaw = ANGLE_MAP[d]
                R = compute_pattern(pattern, vals, yaw)
                h, p, r = euler_func(R)
                results[d] = (h, p, r)

            roll_000 = results["000"][2]
            print(f"{'方向':>6}  {'heading':>10}  {'pitch':>10}  {'roll':>10}  {'Δroll':>10}  {'RS Δroll':>10}  {'誤差':>10}")

            total_error = 0.0
            for d in directions:
                h, p, r = results[d]
                dr = normalize_angle(r - roll_000)
                rs_dr = normalize_angle(RS_GROUND_TRUTH[d][2] - rs_roll_000)
                err = abs(normalize_angle(dr - rs_dr))
                total_error += err
                marker = " <<<" if err > 5 else ""
                print(f"{d:>6}  {h:10.4f}  {p:10.4f}  {r:10.4f}  {dr:10.4f}  {rs_dr:10.4f}  {err:10.4f}{marker}")

            avg_error = total_error / len(directions)
            print(f"\n  → roll差分の平均誤差: {avg_error:.4f}°")

            # heading/pitchの安定性もチェック
            headings = [results[d][0] for d in directions]
            pitches = [results[d][1] for d in directions]
            h_range = max(headings) - min(headings)
            p_range = max(pitches) - min(pitches)
            print(f"  → heading範囲: {h_range:.4f}°, pitch範囲: {p_range:.4f}°")

            # RS正解はheading範囲≈0.09°, pitch範囲≈0.10°
            rs_headings = [RS_GROUND_TRUTH[d][0] for d in directions]
            rs_pitches = [RS_GROUND_TRUTH[d][1] for d in directions]
            print(f"  → RS正解: heading範囲={max(rs_headings)-min(rs_headings):.4f}°, "
                  f"pitch範囲={max(rs_pitches)-min(rs_pitches):.4f}°")

            score = avg_error + h_range + p_range
            if score < best_score:
                best_score = score
                best_combo = (pattern, euler_name, avg_error, h_range, p_range)
            print()

    print("=" * 80)
    print("結果サマリー")
    print("=" * 80)
    if best_combo:
        pat, eul, aerr, hr, pr = best_combo
        print(f"  最良の組み合わせ: パターン {pat} + Euler {eul}")
        print(f"  roll差分平均誤差: {aerr:.4f}°")
        print(f"  heading範囲: {hr:.4f}°, pitch範囲: {pr:.4f}°")
        print(f"  総合スコア(低い方が良い): {best_score:.4f}")
    print()
    print("パターン説明:")
    for k, v in PATTERNS.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
