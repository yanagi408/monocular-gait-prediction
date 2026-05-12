#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zcap.py（段階別デバッグ付き・ロール除去版・膝向き固定 IK） + contactチューニング版 + GZ再推定（自然化）

機能概要
- MediaPipe Pose の world_landmarks から下肢 6 点 (L/R HIP, KNEE, ANKLE) を取得
- 床整形（床を y=0 にそろえる）
    - ロール成分を切り、Z–Y 平面内のピッチ（前後傾き）のみを補正
- 骨長一定の IK（膝位置補正）
    - 元の MediaPipe 座標から膝の「前後向き」を推定し、
      IK 解のうち同じ向きのものを優先して選択（逆折れ防止）
- 骨盤 Z アライン（HIP と ANKLE 中点の Z 差を一定に保つ）
- 接地判定（stance + contact_L/R）
    - score_w_v / score_w_y / min_stance_frames / contact_v_enter_q のチューニング
    - 0,0 を作らない（必ずどちらかは contact=1）
    - 切り替わり近傍だけ短い二重接地を作る（長すぎるDSを削る）
- 足首ロック・toe-off リベース（ここは次段でさらに改善予定）
- GZ（累積前進）推定を「support足のZがほぼ一定」になる方向で再推定
    - ただし推定が極端に小さい場合は従来HSベースへフォールバック
- 1 本の動画につき
    {stem}_raw.csv
    {stem}_processed.csv
  を出力

デバッグ用オプション（--debug_stages）
- 以下の CSV を出力:
    {stem}_stage0.csv
    {stem}_stage1.csv
    {stem}_stage2.csv
    {stem}_stage3.csv
    {stem}_stage4.csv
    {stem}_stage5.csv
    {stem}_stage6.csv
    {stem}_stage7.csv
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# -----------------------
# パラメータ
# -----------------------
DEFAULT_SCORE_W_V = 1.0
DEFAULT_SCORE_W_Y = 0.3
DEFAULT_MIN_STANCE_FRAMES = 5
DEFAULT_MIN_CONTACT_RUN = 2  # contact_L/R の短いラン除去（0で無効）
DEFAULT_MIN_SUPPORT_RUN = 3  # support(支持脚) の短いラン除去（0で無効）
DEFAULT_CONTACT_V_ENTER_Q = 0.60
DEFAULT_CONTACT_MODE = "thr"
DEFAULT_H_THR_M = 0.03
DEFAULT_U_THR_MPS = 0.25
DEFAULT_ON_N = 2
DEFAULT_OFF_N = 3
DEFAULT_EVENT_WIN = 2
DEFAULT_USE_EXTRA_FOOT_INTERNAL = True

DEFAULT_DS_ALLOW_RADIUS = 1  # 二重接地“許可”の半径（フレーム）
DEFAULT_DS_FORCE_RADIUS = 0  # 二重接地“強制”の半径（フレーム）
DEFAULT_MAX_DOUBLE_FRAMES = 2  # 二重接地(1,1)の最大連続フレーム
DEFAULT_MIN_SWITCH_GAP = 6  # stance切替の最小間隔（ジッタ抑制）
DEFAULT_HS_SEARCH_MARGIN = 4

DEFAULT_STEP_GAIN_Z = 1.0
DEFAULT_STEP_CLIP_MIN = 0.0
DEFAULT_STEP_CLIP_MAX_M = 2.0
DEFAULT_STEP_MEDIAN_WIN = 5

DEFAULT_PELVIS_ALIGN_USE_DS_ONLY = False

# 足首ロックは前回版と同じ（やや強め）
DEFAULT_LOCK_TAIL_FRAMES = 6
DEFAULT_LOCK_TOL_Z_M = 0.004
DEFAULT_LOCK_TOL_X_M = 0.004

DEFAULT_REBASE_BLEND_FRAMES = 5
DEFAULT_REBASE_ON_X = True
DEFAULT_REBASE_ON_Z = True

# -----------------------
# Joint 設定（下肢 6 点）
# -----------------------
mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

POINTS = {
    "L_HIP":   LM.LEFT_HIP.value,
    "R_HIP":   LM.RIGHT_HIP.value,
    "L_KNEE":  LM.LEFT_KNEE.value,
    "R_KNEE":  LM.RIGHT_KNEE.value,
    "L_ANKLE": LM.LEFT_ANKLE.value,
    "R_ANKLE": LM.RIGHT_ANKLE.value,
    "L_HEEL": LM.LEFT_HEEL.value,
    "R_HEEL": LM.RIGHT_HEEL.value,
    "L_FOOT_INDEX": LM.LEFT_FOOT_INDEX.value,
    "R_FOOT_INDEX": LM.RIGHT_FOOT_INDEX.value,
}

ORDER = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]

BONES = [
    ("L_HIP", "L_KNEE"),
    ("L_KNEE", "L_ANKLE"),
    ("R_HIP", "R_KNEE"),
    ("R_KNEE", "R_ANKLE"),
    ("L_HIP", "R_HIP"),
]

# -----------------------
# データクラス / ヘルパー
# -----------------------
@dataclass
class JointData:
    name: str
    x: float
    y: float
    z_raw: float
    vis: float


def plusZ(z_raw: float) -> float:
    """MediaPipe world の Z を「前方 +」にそろえるための符号反転ヘルパー。"""
    return -float(z_raw)


def open_video(path: str):
    """動画ファイルを開く。fps が取れない場合は 30fps とみなす。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {path}")
    return cap, float(fps)


def collect_lower(results, include_extra_foot: bool = False) -> Optional[List[JointData]]:
    """
    MediaPipe の結果から下肢 6 点（ORDER）の world_landmarks を取り出す。
    オプションで HEEL / FOOT_INDEX も取得する（内部処理用。CSV 出力の対象外にする想定）。

    仕様:
    - 位置: pose_world_landmarks（メートル）
    - visibility: pose_landmarks（2D）から取得
    """
    if results.pose_landmarks is None or results.pose_world_landmarks is None:
        return None
    norm = results.pose_landmarks.landmark
    world = results.pose_world_landmarks.landmark

    out: List[JointData] = []
    # 必須 6 点
    for nm in ORDER:
        idx = POINTS[nm]
        ln = norm[idx]
        lw = world[idx]
        out.append(JointData(nm, lw.x, lw.y, lw.z, getattr(ln, "visibility", 1.0)))

    if include_extra_foot:
        extra = ["L_HEEL", "R_HEEL", "L_FOOT_INDEX", "R_FOOT_INDEX"]
        for nm in extra:
            if nm not in POINTS:
                continue
            idx = POINTS[nm]
            ln = norm[idx]
            lw = world[idx]
            out.append(JointData(nm, lw.x, lw.y, lw.z, getattr(ln, "visibility", 1.0)))

    return out

def valid(joints: List[JointData], th: float = 0.5) -> bool:
    """
    必須 6 点（ORDER）の visibility が閾値以上かどうかを判定。
    joints に余分な点が混ざっていても、ORDER のみで判定する。
    """
    req = set(ORDER)
    got = {j.name for j in joints}
    if not req.issubset(got):
        return False
    for j in joints:
        if j.name in req:
            if j.vis < th:
                return False
    return True

def normalize(v: np.ndarray) -> np.ndarray:
    """ゼロ割を避けつつ正規化。"""
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def build_upright_basis(
    n: np.ndarray,
    forward_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    床法線 n を「上向き Y 軸」とみなし、
    それと直交する X/Z を構成して回転行列を作る。

    戻り値:
        Q: shape=(3,3), 列に (Xp, Yp, Zp) を持つ直交基底行列
        新座標 = Q^T * 旧座標 で「床が水平・上が +Y」な座標系へ変換できる。
    """
    Yp = normalize(n)
    f = forward_hint - np.dot(forward_hint, Yp) * Yp
    if np.linalg.norm(f) < 1e-6:
        f = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), Yp) * Yp
    Zp = normalize(f)
    Xp = normalize(np.cross(Yp, Zp))
    Zp = normalize(np.cross(Xp, Yp))
    return np.stack([Xp, Yp, Zp], axis=1)


# -----------------------
# キャプチャ（stage0）
# -----------------------
def capture_to_world(
    video_path: str,
    vis_threshold: float = 0.5,
    include_extra_foot: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """
    動画から world_landmarks を取得し、原点と床高を決めて相対座標に変換する。

    仕様:
    - 原点: 初回有効フレームの両足首中点 (x,z) を (0,0) に平行移動。
    - Z:    MediaPipe world の z_raw から原点 z を引き、符号反転して「前方 +Z」へ。
    - Y:    初回有効フレームの両足首の min(y) を床高さ floor_y0 とし、
            腰高さ hip_y0 と比較して「腰が足より上 (+Y)」になるよう y_sign を決定。
            y = y_sign * (world_y - floor_y0) とする。
    - include_extra_foot=True のとき、HEEL / FOOT_INDEX も内部列として取り込む
      （ただし processed.csv / raw.csv には出力しない前提で後段で列を落とす）。
    """
    cap, fps = open_video(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: List[dict] = []
    frame = 0
    origin = None  # (ox, oz)
    y_sign = 1.0
    floor_y0 = None
    basename = os.path.splitext(os.path.basename(video_path))[0]

    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            js_all = collect_lower(res, include_extra_foot=include_extra_foot) if res else None
            if js_all:
                js_req = [j for j in js_all if j.name in set(ORDER)]
            else:
                js_req = None

            if js_all and js_req and valid(js_req, vis_threshold):
                if origin is None:
                    # 初回フレームで原点と Y 符号を決定
                    L = next(j for j in js_req if j.name == "L_ANKLE")
                    R = next(j for j in js_req if j.name == "R_ANKLE")
                    LH = next(j for j in js_req if j.name == "L_HIP")
                    RH = next(j for j in js_req if j.name == "R_HIP")
                    ox = 0.5 * (L.x + R.x)
                    oz = 0.5 * (L.z_raw + R.z_raw)
                    origin = (ox, oz)

                    floor_y0 = min(L.y, R.y)
                    hip_y0 = 0.5 * (LH.y + RH.y)
                    y_sign = 1.0 if (hip_y0 - floor_y0) >= 0 else -1.0
                    print(
                        f"[Diag:{basename}] init: y_sign={y_sign}, "
                        f"hip_y0={hip_y0:.4f}, floor_y0={floor_y0:.4f}"
                    )

                ox, oz = origin
                row = {
                    "frame": frame,
                    "time_sec": round(frame / (fps if fps > 0 else 30.0), 6),
                }
                for j in js_all:
                    x = j.x - ox
                    y = y_sign * (j.y - floor_y0)
                    z = plusZ(j.z_raw - oz)
                    row[f"{j.name}_x_m"] = x
                    row[f"{j.name}_y_m"] = y
                    row[f"{j.name}_z_m"] = z
                rows.append(row)
            frame += 1
    finally:
        cap.release()
        try:
            pose.close()
        except Exception:
            pass

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    return df, float(fps if fps and fps > 0 else 30.0)

def level_floor(df: pd.DataFrame) -> pd.DataFrame:
    """
    「ロールを切った床ピッチ補正」を常に適用する。

    手順:
      1. 足首点群の (y, z) から y ≈ b*z + c を最小二乗フィット（x は使わない）。
      2. 法線ベクトル n = (0, -1, b) を Y–Z 平面内で構成（ロール成分なし）。
      3. 「足首→腰」の平均ベクトルを up_hint として計算し、
         n と up_hint の内積が正になる向きに n を反転。
      4. n を上方向とする直交基底 (Xp, Yp, Zp) を構成し、
         新座標 = Q^T * 旧座標 により全関節を回転。
      5. 回転後の足首 y の 5% 分位を床高さとみなし、
         全関節の y から引いて床を y=0 にそろえる。
    """
    out = df.copy()

    # 足首点群（L/R の y,z を縦結合）
    z_ank = np.r_[out["L_ANKLE_z_m"].to_numpy(), out["R_ANKLE_z_m"].to_numpy()]
    y_ank = np.r_[out["L_ANKLE_y_m"].to_numpy(), out["R_ANKLE_y_m"].to_numpy()]

    # y ≈ b*z + c を最小二乗でフィット
    A = np.c_[z_ank, np.ones_like(z_ank)]
    coef, _, _, _ = np.linalg.lstsq(A, y_ank, rcond=None)
    b, c = coef
    b = float(b)
    c = float(c)

    # ロール成分を切る: a = 0 とみなし、法線を n = (0, -1, b) とする。
    n = np.array([0.0, -1.0, b], dtype=float)

    # 「足首→腰」の平均ベクトルを up_hint として計算
    hip_xyz = 0.5 * np.c_[
        out["L_HIP_x_m"].to_numpy() + out["R_HIP_x_m"].to_numpy(),
        out["L_HIP_y_m"].to_numpy() + out["R_HIP_y_m"].to_numpy(),
        out["L_HIP_z_m"].to_numpy() + out["R_HIP_z_m"].to_numpy(),
    ]
    ankle_xyz = 0.5 * np.c_[
        out["L_ANKLE_x_m"].to_numpy() + out["R_ANKLE_x_m"].to_numpy(),
        out["L_ANKLE_y_m"].to_numpy() + out["R_ANKLE_y_m"].to_numpy(),
        out["L_ANKLE_z_m"].to_numpy() + out["R_ANKLE_z_m"].to_numpy(),
    ]
    up_hint = normalize(np.nanmean(hip_xyz - ankle_xyz, axis=0))

    # 法線の向きを「腰が足より上になる」側にそろえる
    if np.dot(n, up_hint) < 0:
        n = -n

    # 床法線と現在の +Y 軸とのなす角（傾き）を診断用に出しておく
    tilt_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(normalize(n), np.array([0.0, 1.0, 0.0], dtype=float)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    print(
        f"[Diag] level_floor(pitch-only): "
        f"y ≈ {b:.6f} z + {c:.6f}, tilt vs +Y ≈ {tilt_deg:.2f} deg"
    )

    # 床法線 n を Y軸とする直交基底を構成し、回転行列を作成
    Q = build_upright_basis(n, forward_hint=np.array([0.0, 0.0, 1.0], dtype=float))

    # 新座標 = Q^T * 旧座標 で「床が水平」な座標系に移る。
    for nm in ORDER:
        P = np.c_[
            out[f"{nm}_x_m"].to_numpy(),
            out[f"{nm}_y_m"].to_numpy(),
            out[f"{nm}_z_m"].to_numpy(),
        ].T  # shape=(3, N)
        R = (Q.T @ P).T  # shape=(N, 3)
        out[f"{nm}_x_m"] = R[:, 0]
        out[f"{nm}_y_m"] = R[:, 1]
        out[f"{nm}_z_m"] = R[:, 2]

    # 回転後の足首高さから床 y=0 を決める（下位 5% 分位）
    ay = np.r_[out["L_ANKLE_y_m"].to_numpy(), out["R_ANKLE_y_m"].to_numpy()]
    y_floor = float(np.quantile(ay, 0.05))
    for nm in ORDER:
        out[f"{nm}_y_m"] = out[f"{nm}_y_m"] - y_floor

    print(
        "[Diag] level_floor(pitch-only): "
        f"y_floor(p05 ankle)={y_floor:.5f} → "
        f"shift 後 min(L_ANKLE_y_m,R_ANKLE_y_m) = "
        f"{min(out['L_ANKLE_y_m'].min(), out['R_ANKLE_y_m'].min()):.5f}"
    )

    return out


# -----------------------
# IK / 骨長推定
# -----------------------
def seg_len(df: pd.DataFrame, A: str, B: str) -> np.ndarray:
    v = np.c_[
        df[f"{A}_x_m"],
        df[f"{A}_y_m"],
        df[f"{A}_z_m"],
    ] - np.c_[
        df[f"{B}_x_m"],
        df[f"{B}_y_m"],
        df[f"{B}_z_m"],
    ]
    return np.linalg.norm(v, axis=1)


def estimate_bone_lengths(df: pd.DataFrame, stance: List[str]) -> dict:
    """
    骨長は主に stance フレームからロバストに推定される。
    Z 方向の「前進量」には影響せず、骨長のみを安定化する。
    """
    idx_all = np.arange(len(df))
    if any(s == "L" for s in stance) and any(s == "R" for s in stance):
        maskL = np.array(stance) == "L"
        maskR = np.array(stance) == "R"
        idxL = idx_all[maskL] if maskL.any() else idx_all
        idxR = idx_all[maskR] if maskR.any() else idx_all
    else:
        idxL = idxR = idx_all
    return {
        "L_thigh": float(np.nanmedian(seg_len(df, "L_HIP", "L_KNEE")[idxL])),
        "L_shank": float(np.nanmedian(seg_len(df, "L_KNEE", "L_ANKLE")[idxL])),
        "R_thigh": float(np.nanmedian(seg_len(df, "R_HIP", "R_KNEE")[idxR])),
        "R_shank": float(np.nanmedian(seg_len(df, "R_KNEE", "R_ANKLE")[idxR])),
    }


def estimate_knee_direction(df: pd.DataFrame) -> dict:
    """
    各脚ごとに「膝が前後どちら側にあるか」の符号を推定する。

    - 股関節 H と足首 A の中点 mid を取り、
      dz = K_z - mid_z の符号をフレームごとに計算。
    - その中央値の符号を +1/-1 として、その脚の「標準的な曲がり方向」とみなす。
      （絶対値が極端に小さい場合は +1 をデフォルトとする）

    返り値:
        {"L": +1 or -1, "R": +1 or -1}
    """
    dirs = {}
    for side in ("L", "R"):
        H = np.c_[
            df[f"{side}_HIP_x_m"].to_numpy(),
            df[f"{side}_HIP_y_m"].to_numpy(),
            df[f"{side}_HIP_z_m"].to_numpy(),
        ]
        A = np.c_[
            df[f"{side}_ANKLE_x_m"].to_numpy(),
            df[f"{side}_ANKLE_y_m"].to_numpy(),
            df[f"{side}_ANKLE_z_m"].to_numpy(),
        ]
        K = np.c_[
            df[f"{side}_KNEE_x_m"].to_numpy(),
            df[f"{side}_KNEE_y_m"].to_numpy(),
            df[f"{side}_KNEE_z_m"].to_numpy(),
        ]
        mid_z = 0.5 * (H[:, 2] + A[:, 2])
        dz = K[:, 2] - mid_z
        dz = dz[np.isfinite(dz)]
        if dz.size == 0:
            dirs[side] = 1.0
            continue
        med = float(np.median(dz))
        if abs(med) < 1e-4:
            dirs[side] = 1.0
        else:
            dirs[side] = 1.0 if med > 0 else -1.0
    print(f"[Diag] knee_direction (L,R) = ({dirs.get('L', 1.0):+.0f}, {dirs.get('R', 1.0):+.0f})")
    return dirs


def ik_knee(
    H: np.ndarray,
    A: np.ndarray,
    K0: np.ndarray,
    l1: float,
    l2: float,
    ex_hint: np.ndarray,
    knee_dir: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    股関節 H と足首 A の間に長さ l1,l2 の 2 関節リンクを置き、
    元の膝 K0 から近く、かつ「指定された前後方向 knee_dir」に曲がる解を求める。

    knee_dir:
        +1: mid.z より Z プラス側に膝が位置することを優先
        -1: mid.z より Z マイナス側に膝が位置することを優先
    """
    H = np.asarray(H, float)
    A = np.asarray(A, float)
    K0 = np.asarray(K0, float)
    ex_hint = normalize(np.asarray(ex_hint, float))

    d = A - H
    L = np.linalg.norm(d) + eps
    w = d / L  # 股→足首 方向

    # 伸び切り / 折りたたみ端の処理
    if L >= l1 + l2 - 1e-6:
        return H + w * l1
    if L <= abs(l1 - l2) + 1e-6:
        sgn = 1.0 if l1 >= l2 else -1.0
        return H + w * (l1 * sgn)

    # 2 関節リンクの幾何パラメータ
    a = (l1 * l1 - l2 * l2 + L * L) / (2.0 * L)
    r2 = max(l1 * l1 - a * a, 0.0)
    r = np.sqrt(r2)
    P = H + a * w  # 股→足首の線分上の点

    # 膝が回転する平面の法線方向 u を決める
    u = np.cross(w, ex_hint)
    if np.linalg.norm(u) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], float)
        u = np.cross(w, up)
        if np.linalg.norm(u) < 1e-6:
            up = np.array([0.0, 0.0, 1.0], float)
            u = np.cross(w, up)
    u = u / (np.linalg.norm(u) + eps)

    # 候補 2 点
    Kc1 = P + r * u
    Kc2 = P - r * u

    # 股と足首の中点
    mid = 0.5 * (H + A)
    sign_pref = 1.0 if knee_dir >= 0 else -1.0

    # 各候補の「前後向き」の符号
    sign1 = np.sign(Kc1[2] - mid[2])
    sign2 = np.sign(Kc2[2] - mid[2])

    cond1 = (sign1 == sign_pref)
    cond2 = (sign2 == sign_pref)

    # 片方だけ向きが合うならそちらを採用
    if cond1 and not cond2:
        return Kc1
    if cond2 and not cond1:
        return Kc2

    # どちらも同じ / 両方合わない場合は元の膝に近い方
    return Kc1 if np.linalg.norm(Kc1 - K0) <= np.linalg.norm(Kc2 - K0) else Kc2


def apply_ik(df: pd.DataFrame, lens: dict, knee_dirs: dict) -> pd.DataFrame:
    """
    各フレームで膝位置を IK により補正。

    - 左右で膝の曲がり方向（前後）は knee_dirs["L"], knee_dirs["R"] に固定。
    - 左右で膝の外側方向が安定するよう、ex_hint を L/R で変える。
    """
    out = df.copy()
    for side in ("L", "R"):
        l1 = lens[f"{side}_thigh"]
        l2 = lens[f"{side}_shank"]
        H = np.c_[
            out[f"{side}_HIP_x_m"],
            out[f"{side}_HIP_y_m"],
            out[f"{side}_HIP_z_m"],
        ]
        A = np.c_[
            out[f"{side}_ANKLE_x_m"],
            out[f"{side}_ANKLE_y_m"],
            out[f"{side}_ANKLE_z_m"],
        ]
        K = np.c_[
            out[f"{side}_KNEE_x_m"],
            out[f"{side}_KNEE_y_m"],
            out[f"{side}_KNEE_z_m"],
        ]
        # 左右で外側方向を変える
        ex_hint = np.array([1.0, 0.0, 0.0]) if side == "L" else np.array([-1.0, 0.0, 0.0])
        knee_dir = float(knee_dirs.get(side, 1.0))
        K_new = np.zeros_like(K)
        for i in range(len(out)):
            K_new[i] = ik_knee(H[i], A[i], K[i], l1, l2, ex_hint=ex_hint, knee_dir=knee_dir)
        out[f"{side}_KNEE_x_m"] = K_new[:, 0]
        out[f"{side}_KNEE_y_m"] = K_new[:, 1]
        out[f"{side}_KNEE_z_m"] = K_new[:, 2]
    return out


# -----------------------
# pelvis Z align
# -----------------------
def pelvis_z_align(
    df: pd.DataFrame,
    stance: List[str],
    use_ds_only: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """
    HIP と足首中点の Z 差が一定になるように HIP を前後方向に平行移動。
    足首座標は変えないので、前進 GZ の推定には影響しない。
    """
    out = df.copy()
    mid_ankle_z = 0.5 * (
        out["L_ANKLE_z_m"].to_numpy() + out["R_ANKLE_z_m"].to_numpy()
    )
    hip_avg_z = 0.5 * (
        out["L_HIP_z_m"].to_numpy() + out["R_HIP_z_m"].to_numpy()
    )
    if use_ds_only:
        s = np.array(stance)
        ds_mask = np.r_[s[1:] != s[:-1], True]
        base = np.median((hip_avg_z - mid_ankle_z)[ds_mask])
    else:
        base = np.median(hip_avg_z - mid_ankle_z)
    delta = (mid_ankle_z + base) - hip_avg_z
    out["L_HIP_z_m"] = out["L_HIP_z_m"] + delta
    out["R_HIP_z_m"] = out["R_HIP_z_m"] + delta
    return out, float(base)


# -----------------------
# stance/contact 判定
# -----------------------
def _safe_fps(fps: Optional[float], fallback: float = 30.0) -> float:
    """fps が不正（0/NaN/None）なら fallback を返す。"""
    try:
        v = float(fps) if fps is not None else float(fallback)
    except Exception:
        return float(fallback)
    if not np.isfinite(v) or v <= 1e-6:
        return float(fallback)
    return float(v)


def vel_norm(df: pd.DataFrame, side: str, fps: Optional[float] = None) -> np.ndarray:
    """足首の速度ノルム（m/s）。"""
    fpsv = _safe_fps(fps)
    vx = df[f"{side}_ANKLE_x_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
    vy = df[f"{side}_ANKLE_y_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
    vz = df[f"{side}_ANKLE_z_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
    return np.sqrt(vx * vx + vy * vy + vz * vz)


def _foot_xyz(df: pd.DataFrame, side: str, use_extra_foot: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    接地判定のための代表点を作る。
    - x,z: (ANKLE, HEEL, FOOT_INDEX) の平均（存在する列のみ）
    - y  : (ANKLE, HEEL, FOOT_INDEX) の最小（存在する列のみ）
    """
    xs, ys, zs = [], [], []

    def add_point(prefix: str):
        cx = f"{prefix}_x_m"
        cy = f"{prefix}_y_m"
        cz = f"{prefix}_z_m"
        if cx in df.columns and cy in df.columns and cz in df.columns:
            xs.append(df[cx].to_numpy(dtype=float))
            ys.append(df[cy].to_numpy(dtype=float))
            zs.append(df[cz].to_numpy(dtype=float))

    add_point(f"{side}_ANKLE")
    if use_extra_foot:
        add_point(f"{side}_HEEL")
        add_point(f"{side}_FOOT_INDEX")

    if len(xs) == 0:
        # 必須列が無い場合（通常は起きない）
        n = len(df)
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    Z = np.stack(zs, axis=0)
    x = np.nanmean(X, axis=0)
    z = np.nanmean(Z, axis=0)
    y = np.nanmin(Y, axis=0)
    return x, y, z


def _hysteresis_binary(cond: np.ndarray, on_n: int, off_n: int) -> np.ndarray:
    """
    cond（bool）から 0/1 を生成するヒステリシス。
    - 0->1: cond=True が on_n 連続
    - 1->0: cond=False が off_n 連続
    """
    on_n = max(1, int(on_n))
    off_n = max(1, int(off_n))
    out = np.zeros(len(cond), dtype=int)
    state = 0
    t_run = 0
    f_run = 0
    for i, c in enumerate(cond.astype(bool)):
        if state == 0:
            if c:
                t_run += 1
            else:
                t_run = 0
            if t_run >= on_n:
                state = 1
                f_run = 0
        else:
            if not c:
                f_run += 1
            else:
                f_run = 0
            if f_run >= off_n:
                state = 0
                t_run = 0
        out[i] = state
    return out


def _debounce_events(ev: np.ndarray, event_win: int) -> np.ndarray:
    """1 が連発しないように間隔を空ける（最低 event_win フレーム）。"""
    w = max(0, int(event_win))
    if w <= 0:
        return ev.astype(int)
    out = np.zeros_like(ev, dtype=int)
    last = -10**9
    for i, v in enumerate(ev.astype(int)):
        if v == 1 and (i - last) > w:
            out[i] = 1
            last = i
    return out

def choose_stance(
    df: pd.DataFrame,
    *,
    fps: Optional[float] = None,
    use_extra_foot_internal: bool = False,
    contact_mode: str = DEFAULT_CONTACT_MODE,
    # score mode parameters
    score_w_v: float = DEFAULT_SCORE_W_V,
    score_w_y: float = DEFAULT_SCORE_W_Y,
    min_stance_frames: int = DEFAULT_MIN_STANCE_FRAMES,
    contact_v_enter_q: float = DEFAULT_CONTACT_V_ENTER_Q,
    # thr mode parameters
    h_thr_m: float = DEFAULT_H_THR_M,
    u_thr_mps: float = DEFAULT_U_THR_MPS,
    on_n: int = DEFAULT_ON_N,
    off_n: int = DEFAULT_OFF_N,
    # ds control
    ds_allow_radius: int = DEFAULT_DS_ALLOW_RADIUS,
    ds_force_radius: int = DEFAULT_DS_FORCE_RADIUS,
    max_double_frames: int = DEFAULT_MAX_DOUBLE_FRAMES,
    min_switch_gap: int = DEFAULT_MIN_SWITCH_GAP,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    stance と contact_L/R を決める。

    contact_mode:
      - "thr"  : (足高さ<=h_thr_m) & (水平速度<=u_thr_mps) をヒステリシスで平滑化して contact を作る
      - "score": 従来の速度ノルムの quantile による contact を作る（互換用）

    共通の制約（設計意図）:
      - 0,0（空中）は作らない（必ず片足以上を contact=1 にする）
      - 二重接地(1,1)は切替近傍で短く発生させる（長い DS は削る）
      - stance 切替ジッタは min_switch_gap で抑える
    """
    n = len(df)
    if n == 0:
        return [], np.array([], dtype=int), np.array([], dtype=int)

    fpsv = _safe_fps(fps)

    # --- stance 推定（代表点の速度と高さでスコア化） ---
    xL, yL, zL = _foot_xyz(df, "L", use_extra_foot=use_extra_foot_internal)
    xR, yR, zR = _foot_xyz(df, "R", use_extra_foot=use_extra_foot_internal)

    dxL = np.diff(xL, prepend=xL[0]) * fpsv
    dyL = np.diff(yL, prepend=yL[0]) * fpsv
    dzL = np.diff(zL, prepend=zL[0]) * fpsv
    dxR = np.diff(xR, prepend=xR[0]) * fpsv
    dyR = np.diff(yR, prepend=yR[0]) * fpsv
    dzR = np.diff(zR, prepend=zR[0]) * fpsv

    vL = np.sqrt(dxL * dxL + dyL * dyL + dzL * dzL)
    vR = np.sqrt(dxR * dxR + dyR * dyR + dzR * dzR)

    sL = float(score_w_v) * vL + float(score_w_y) * yL
    sR = float(score_w_v) * vR + float(score_w_y) * yR

    stance: List[str] = []
    st = "L" if sL[0] <= sR[0] else "R"
    stance.append(st)
    run = 1
    for i in range(1, n):
        pick = "L" if sL[i] <= sR[i] else "R"
        if pick == st:
            run += 1
        else:
            if run < int(min_stance_frames):
                pick = st
                run += 1
            else:
                st = pick
                run = 1
        stance.append(pick)

    # stance の切替点
    switches = [i for i in range(1, n) if stance[i] != stance[i - 1]]

    # 切替ジッタ抑制：短すぎる間隔の切替は捨てる
    if int(min_switch_gap) > 0 and len(switches) >= 2:
        filtered = [switches[0]]
        for i in switches[1:]:
            if i - filtered[-1] >= int(min_switch_gap):
                filtered.append(i)
        switches = filtered

    # --- contact の初期推定 ---
    cmode = str(contact_mode).strip().lower()
    if cmode not in ("thr", "score"):
        cmode = DEFAULT_CONTACT_MODE

    if cmode == "score":
        # 従来互換: 速度ノルムの quantile で接地候補
        vLa = vel_norm(df, "L", fps=fpsv)
        vRa = vel_norm(df, "R", fps=fpsv)
        thrL = np.nanquantile(vLa, float(contact_v_enter_q))
        thrR = np.nanquantile(vRa, float(contact_v_enter_q))
        contact_L = (vLa <= thrL).astype(int)
        contact_R = (vRa <= thrR).astype(int)
    else:
        # 新方式: 高さ + 水平速度
        uL = np.sqrt(dxL * dxL + dzL * dzL)
        uR = np.sqrt(dxR * dxR + dzR * dzR)
        condL = (yL <= float(h_thr_m)) & (uL <= float(u_thr_mps))
        condR = (yR <= float(h_thr_m)) & (uR <= float(u_thr_mps))
        contact_L = _hysteresis_binary(condL, int(on_n), int(off_n))
        contact_R = _hysteresis_binary(condR, int(on_n), int(off_n))

    # --- 切替近傍のみ二重接地を許可/強制 ---
    ds_allow = np.zeros(n, dtype=bool)
    ds_force = np.zeros(n, dtype=bool)
    ar = max(0, int(ds_allow_radius))
    fr = max(0, int(ds_force_radius))
    for i in switches:
        if ar > 0:
            ds_allow[max(0, i - ar) : min(n, i + ar + 1)] = True
        if fr > 0:
            ds_force[max(0, i - fr) : min(n, i + fr + 1)] = True

    # ds_allow 以外で (1,1) は stance 側のみに戻す
    both1 = (contact_L == 1) & (contact_R == 1)
    for i in np.where(both1 & (~ds_allow))[0]:
        if stance[i] == "L":
            contact_L[i] = 1
            contact_R[i] = 0
        else:
            contact_L[i] = 0
            contact_R[i] = 1

    # ds_force は必ず (1,1)
    for i in np.where(ds_force)[0]:
        contact_L[i] = 1
        contact_R[i] = 1

    # (0,0) を禁止（stance 側を 1）
    both0 = (contact_L == 0) & (contact_R == 0)
    if np.any(both0):
        mask = np.array(stance)[both0]
        contact_L[both0] = (mask == "L").astype(int)
        contact_R[both0] = (mask == "R").astype(int)

    # --- 二重接地の連続長を上限制御 ---
    maxd = int(max_double_frames)
    if maxd <= 0:
        # 二重接地を完全禁止
        for i in range(n):
            if stance[i] == "L":
                contact_L[i] = 1
                contact_R[i] = 0
            else:
                contact_L[i] = 0
                contact_R[i] = 1
    else:
        i = 0
        while i < n:
            if contact_L[i] == 1 and contact_R[i] == 1:
                j = i
                while j + 1 < n and (contact_L[j + 1] == 1 and contact_R[j + 1] == 1):
                    j += 1
                run_len = j - i + 1
                if run_len > maxd:
                    for t in range(i + maxd, j + 1):
                        if stance[t] == "L":
                            contact_L[t] = 1
                            contact_R[t] = 0
                        else:
                            contact_L[t] = 0
                            contact_R[t] = 1
                i = j + 1
            else:
                i += 1

    # 念のため 0,0 を再チェック
    both0 = (contact_L == 0) & (contact_R == 0)
    if np.any(both0):
        mask = np.array(stance)[both0]
        contact_L[both0] = (mask == "L").astype(int)
        contact_R[both0] = (mask == "R").astype(int)

    return stance, contact_L.astype(int), contact_R.astype(int)


# -------------------------
# stance → support(支持脚) 変換
# -------------------------
def _forward_fill_int(arr, fill_value=0):
    """Forward-fill -1 in a 1D int array."""
    out = arr.copy()
    last = None
    for i in range(len(out)):
        if out[i] == -1:
            if last is not None:
                out[i] = last
        else:
            last = int(out[i])
    # back-fill head if still -1
    if len(out) > 0 and out[0] == -1:
        first = None
        for v in out:
            if v != -1:
                first = int(v)
                break
        if first is None:
            first = int(fill_value)
        for i in range(len(out)):
            if out[i] != -1:
                break
            out[i] = first
    # if all -1 (rare)
    out[out == -1] = int(fill_value)
    return out


def _smooth_short_runs_binary(arr01, min_run=3):
    """Merge short runs in a binary 1D array (values 0/1)."""
    if min_run is None or min_run <= 1:
        return arr01
    out = arr01.copy()
    n = len(out)
    i = 0
    while i < n:
        v = int(out[i])
        j = i + 1
        while j < n and int(out[j]) == v:
            j += 1
        run_len = j - i
        if run_len < min_run:
            left = int(out[i - 1]) if i > 0 else None
            right = int(out[j]) if j < n else None

            if left is None and right is None:
                fill = v
            elif left is None:
                fill = right
            elif right is None:
                fill = left
            else:
                if left == right:
                    fill = left
                else:
                    # choose side with longer adjacent run
                    l = i - 1
                    while l >= 0 and int(out[l]) == left:
                        l -= 1
                    left_len = (i - 1) - l
                    r = j
                    while r < n and int(out[r]) == right:
                        r += 1
                    right_len = r - j
                    fill = left if left_len >= right_len else right

            out[i:j] = fill
        i = j
    return out


def support_from_stance_list(stance_list, fallback=0, min_run=3):
    """
    stance_list (例: ['L','R','DS','NONE']) を 0/1 の支持脚ラベルへ変換。
    - L -> 0, R -> 1
    - DS/NONE は -1 として扱い、前方補間で直近の支持脚を維持
    - 先頭の -1 は最初の有効値で埋める（無い場合は fallback）
    - 連続長が短い区間は _smooth_short_runs_binary で抑制
    """
    import numpy as np

    raw = np.full((len(stance_list),), -1, dtype=np.int32)
    for i, s in enumerate(stance_list):
        if s == 'L':
            raw[i] = 0
        elif s == 'R':
            raw[i] = 1

    sup = _forward_fill_int(raw, fill_value=fallback)
    sup = _smooth_short_runs_binary(sup, min_run=min_run)
    return sup.astype(np.int32)


def support_switch_events(support01, event_win: int = 0):
    """支持脚ラベル(support: 0=Left, 1=Right)の切替イベント抽出。

    返り値（全て shape=(T,) の 0/1）:
      - sw   : support 切替（任意方向）
      - hsL  : Right→Left（左が支持脚になった）
      - hsR  : Left→Right（右が支持脚になった）
      - toL  : Left→Right（左が支持脚でなくなった）
      - toR  : Right→Left（右が支持脚でなくなった）

    event_win>0 の場合、近接イベントをデバウンスする。
    """
    s = np.asarray(support01, dtype=np.int32).reshape(-1)
    T = int(s.shape[0])
    if T == 0:
        z = np.zeros((0,), dtype=np.int32)
        return z, z, z, z, z
    if T == 1:
        z = np.zeros((1,), dtype=np.int32)
        return z, z, z, z, z

    sw = np.zeros((T,), dtype=np.int32)
    hsL = np.zeros((T,), dtype=np.int32)
    hsR = np.zeros((T,), dtype=np.int32)
    toL = np.zeros((T,), dtype=np.int32)
    toR = np.zeros((T,), dtype=np.int32)

    for t in range(1, T):
        if s[t] != s[t - 1]:
            sw[t] = 1
            if (s[t - 1] == 1) and (s[t] == 0):
                hsL[t] = 1
                toR[t] = 1
            elif (s[t - 1] == 0) and (s[t] == 1):
                hsR[t] = 1
                toL[t] = 1

    if event_win and int(event_win) > 0:
        w = int(event_win)
        sw = _debounce_events(sw, w)
        hsL = _debounce_events(hsL, w)
        hsR = _debounce_events(hsR, w)
        toL = _debounce_events(toL, w)
        toR = _debounce_events(toR, w)

    return sw, hsL, hsR, toL, toR

def find_switches(stance: List[str]) -> List[int]:
    return [i for i in range(1, len(stance)) if stance[i] != stance[i - 1]]


def refine_hs_index(
    df: pd.DataFrame,
    rough_i: int,
    leg: str,
    margin: int = DEFAULT_HS_SEARCH_MARGIN,
) -> int:
    """
    stance 切り替え rough_i の近傍から、
    足首高さ y が最小のフレームを HS として取り直す。
    """
    a = max(0, rough_i - margin)
    b = min(len(df) - 1, rough_i + margin)
    y = df[f"{leg}_ANKLE_y_m"].iloc[a : b + 1].to_numpy()
    return a + int(np.argmin(y))


def build_forward_from_hs_robust_legacy(
    df: pd.DataFrame,
    stance: List[str],
    step_gain_z: float = DEFAULT_STEP_GAIN_Z,
    step_clip_min: float = DEFAULT_STEP_CLIP_MIN,
    step_clip_max_m: float = DEFAULT_STEP_CLIP_MAX_M,
    step_median_win: int = DEFAULT_STEP_MEDIAN_WIN,
):
    """
    HS 切り替え点から一歩ごとの前進量を推定し、累積前進量 GZ を返す。
    （従来実装：zcap4 までの方式）
    """
    switches = find_switches(stance)
    if not switches:
        print("[Diag:GZ] stance 切り替えが検出されず、前進 GZ は 0 になります。")
        return np.zeros(len(df), dtype=float), []

    events = []
    for i in switches:
        leg = stance[i]
        if leg in ("L", "R"):
            events.append((refine_hs_index(df, i, leg), leg))

    hs_L = sorted([i for i, l in events if l == "L"])
    hs_R = sorted([i for i, l in events if l == "R"])

    steps = []

    def push(hs_list, leg):
        other = "L" if leg == "R" else "R"
        for k in range(1, len(hs_list)):
            i0, i1 = hs_list[k - 1], hs_list[k]
            z0l = float(df.at[i0, f"{leg}_ANKLE_z_m"])
            z1l = float(df.at[i1, f"{leg}_ANKLE_z_m"])
            z0o = float(df.at[i0, f"{other}_ANKLE_z_m"])
            z1o = float(df.at[i1, f"{other}_ANKLE_z_m"])
            dz_abs = z1l - z0l
            dz_rel = (z1l - z1o) - (z0l - z0o)
            dz = max(dz_abs, dz_rel, 0.0) * step_gain_z
            dz = max(step_clip_min, min(step_clip_max_m, dz))
            steps.append((i0, i1, dz))

    push(hs_L, "L")
    push(hs_R, "R")
    steps.sort(key=lambda t: t[0])

    if not steps:
        print("[Diag:GZ] HS はあるが step が作成されず、GZ は 0 になります。")
        return np.zeros(len(df), dtype=float), []

    arr = np.array([s[2] for s in steps], float)
    if len(arr) >= max(3, step_median_win):
        try:
            from scipy.signal import medfilt

            k = step_median_win if step_median_win % 2 == 1 else step_median_win + 1
            arr_f = medfilt(arr, kernel_size=k)
        except Exception:
            arr_f = arr
    else:
        arr_f = arr
    for j, (i0, i1, _) in enumerate(steps):
        steps[j] = (i0, i1, float(arr_f[j]))

    GZ = np.zeros(len(df), float)
    cur = 0.0
    last = 0
    for i0, i1, step in steps:
        if i0 > last:
            GZ[last:i0] = cur
        if i1 > i0:
            GZ[i0:i1] = np.linspace(cur, cur + step, i1 - i0, endpoint=False)
            cur += step
            last = i1
    GZ[last:] = cur
    GZ = np.maximum.accumulate(GZ)

    print(f"[Diag:GZ] legacy steps={len(steps)}, GZ range=({GZ.min():.3f} .. {GZ.max():.3f})")
    return GZ, steps


def build_forward_from_hs_robust(
    df: pd.DataFrame,
    stance: List[str],
    contact_L: np.ndarray,
    contact_R: np.ndarray,
    fps: Optional[float] = None,
    step_gain_z: float = DEFAULT_STEP_GAIN_Z,
    step_clip_min: float = DEFAULT_STEP_CLIP_MIN,
    step_clip_max_m: float = DEFAULT_STEP_CLIP_MAX_M,
    step_median_win: int = DEFAULT_STEP_MEDIAN_WIN,
):
    """\
    HS（立脚切り替え）由来の GZ（累積前進）を推定し、processed.csv でのみ Z に加算する。

    目的:
      - processed の Z を「世界座標で前進する」系列にする（raw は前進なし）。
      - 接地足（support）をできるだけ“床上で滑らない（Zがほぼ一定）”ようにする。

    実装方針（ハイブリッド）:
      A) contact（接地フラグ）から「support 足」を決め、
         support 足の Z 変化（support_z の減少）を打ち消すように GZ を推定する
         （= stance 足 Z を“ほぼ一定”に保つ方向の推定）。
      B) もし A の結果が極端に小さい（前進がほぼ 0）場合は、
         従来の HS ベース（legacy）にフォールバックする。

    補足:
      - world_z = measured_z + GZ
      - support 足の world_z が一定 ⇒ ΔGZ ≈ -Δ(measured_z_support)

    返り値:
      - GZ: shape=(N,), 非減少の累積前進量 [m]
      - steps: デバッグ用のステップ区間リスト（(i0,i1,step_m)）
    """

    n = len(df)
    if n <= 1:
        return np.zeros(n, dtype=float), []

    # ---------- A) contact ベース（support 足 Z を一定に寄せる） ----------
    vL = vel_norm(df, "L", fps=fps)
    vR = vel_norm(df, "R", fps=fps)
    yL = df["L_ANKLE_y_m"].to_numpy()
    yR = df["R_ANKLE_y_m"].to_numpy()
    zL = df["L_ANKLE_z_m"].to_numpy()
    zR = df["R_ANKLE_z_m"].to_numpy()

    # support 足の決定（各フレーム）
    # - 両足接地: “より床に近く(低y) / 遅い”方を support とみなす
    # - 片足接地: その足
    # - 接地なし: stance を採用（contact 補正で原則発生しない想定）
    support = []
    for i in range(n):
        cL = int(contact_L[i]) == 1
        cR = int(contact_R[i]) == 1
        if cL and not cR:
            support.append("L")
        elif cR and not cL:
            support.append("R")
        elif cL and cR:
            # 速度と高さの“合成スコア”が小さい方を support
            sL = 0.7 * float(vL[i]) + 0.3 * float(max(yL[i], 0.0))
            sR = 0.7 * float(vR[i]) + 0.3 * float(max(yR[i], 0.0))
            support.append("L" if sL <= sR else "R")
        else:
            support.append(stance[i] if i < len(stance) else "L")

    # support が連続する区間ごとに “support_z の後退量” を step として積分
    steps_A = []
    GZ_A = np.zeros(n, dtype=float)
    cur = 0.0
    i = 0
    while i < n - 1:
        side = support[i]
        j = i + 1
        while j < n and support[j] == side:
            j += 1
        # 区間 [i, j)（j は次の side 先頭 or n）
        if j - i >= 2:
            z = zL if side == "L" else zR
            # start/end はノイズ耐性のため“窓中央値”
            k = int(min(3, j - i))
            z0 = float(np.median(z[i : i + k]))
            z1 = float(np.median(z[j - k : j]))
            dz = z1 - z0
            # support 足が“後退（dz<0）”した分だけ前進として加算
            step = max(-dz * step_gain_z, 0.0)
            step = max(step_clip_min, min(step_clip_max_m, step))
            if step > 0:
                steps_A.append((i, j, float(step)))
                GZ_A[i:j] = np.linspace(cur, cur + step, j - i, endpoint=False)
                cur += step
            else:
                GZ_A[i:j] = cur
        else:
            GZ_A[i:j] = cur
        i = j

    GZ_A[i:] = cur
    GZ_A = np.maximum.accumulate(GZ_A)

    # 区間 step のメディアンフィルタ（歩幅の外れ値対策）
    if steps_A:
        arr = np.array([s[2] for s in steps_A], float)
        if len(arr) >= max(3, step_median_win):
            try:
                from scipy.signal import medfilt
                kf = step_median_win if step_median_win % 2 == 1 else step_median_win + 1
                arr_f = medfilt(arr, kernel_size=kf)
            except Exception:
                arr_f = arr
        else:
            arr_f = arr
        steps_A = [(i0, i1, float(arr_f[t])) for t, (i0, i1, _) in enumerate(steps_A)]
        # フィルタ後に GZ を再構築
        GZ_A = np.zeros(n, float)
        cur = 0.0
        last = 0
        for i0, i1, step in steps_A:
            if i0 > last:
                GZ_A[last:i0] = cur
            if i1 > i0:
                GZ_A[i0:i1] = np.linspace(cur, cur + step, i1 - i0, endpoint=False)
                cur += step
                last = i1
        GZ_A[last:] = cur
        GZ_A = np.maximum.accumulate(GZ_A)

    total_A = float(GZ_A[-1]) if n > 0 else 0.0

    # ---------- B) legacy（従来の HS ベース） ----------
    GZ_L, steps_L = build_forward_from_hs_robust_legacy(
        df,
        stance,
        step_gain_z=step_gain_z,
        step_clip_min=step_clip_min,
        step_clip_max_m=step_clip_max_m,
        step_median_win=step_median_win,
    )
    total_L = float(GZ_L[-1]) if len(GZ_L) > 0 else 0.0

    # ---------- 選択ロジック ----------
    # A が「その場足踏み」レベルに小さいときは legacy を採用
    # 目安: legacy の 35% 未満、かつ 0.05m 未満
    if (total_A < 0.05) or (total_L > 0.0 and total_A < 0.35 * total_L):
        print(
            f"[Diag:GZ] choose legacy: total_A={total_A:.3f}m, total_L={total_L:.3f}m"
        )
        return GZ_L, steps_L

    print(f"[Diag:GZ] choose contact-support: total_A={total_A:.3f}m, total_L={total_L:.3f}m")
    return GZ_A, steps_A


# -----------------------
# 足首ロック / リベース
# -----------------------
def extend_contact_mask(contact: np.ndarray, tail: int) -> np.ndarray:
    """contact=1 の区間を tail フレームぶん後ろに延長する。"""
    m = contact.astype(int).copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i] == 1:
            j = i
            while j + 1 < n and m[j + 1] == 1:
                j += 1
            end = min(n, j + 1 + tail)
            m[j + 1 : end] = 1
            i = end
        else:
            i += 1
    return m


def ankle_lock(
    df: pd.DataFrame,
    stance: List[str],
    cL: np.ndarray,
    cR: np.ndarray,
    lock_tail_frames: int = DEFAULT_LOCK_TAIL_FRAMES,
    lock_tol_z: float = DEFAULT_LOCK_TOL_Z_M,
    lock_tol_x: float = DEFAULT_LOCK_TOL_X_M,
) -> pd.DataFrame:
    """接地中の足首が大きく滑らないように補正する。"""
    out = df.copy()
    cLx = extend_contact_mask(cL, lock_tail_frames)
    cRx = extend_contact_mask(cR, lock_tail_frames)

    def lock_side(side: str, mask: np.ndarray):
        x = out[f"{side}_ANKLE_x_m"].to_numpy()
        z = out[f"{side}_ANKLE_z_m"].to_numpy()
        n = len(x)
        i = 0
        while i < n:
            if mask[i] == 1:
                j = i
                while j + 1 < n and mask[j + 1] == 1:
                    j += 1
                x0, z0 = x[i], z[i]
                dx = x[i : j + 1] - x0
                dz = z[i : j + 1] - z0
                x[i : j + 1] -= np.clip(dx, -lock_tol_x, lock_tol_x)
                z[i : j + 1] -= np.clip(dz, -lock_tol_z, lock_tol_z)
                i = j + 1
            else:
                i += 1
        out[f"{side}_ANKLE_x_m"] = x
        out[f"{side}_ANKLE_z_m"] = z

    lock_side("L", cLx)
    lock_side("R", cRx)
    return out


def rebase_swing_from_toeoff(
    df: pd.DataFrame,
    contact_L: np.ndarray,
    contact_R: np.ndarray,
    rebase_blend_frames: int = DEFAULT_REBASE_BLEND_FRAMES,
    rebase_on_x: bool = DEFAULT_REBASE_ON_X,
    rebase_on_z: bool = DEFAULT_REBASE_ON_Z,
) -> pd.DataFrame:
    """
    toe-off（contact 1->0）を基準に、スイング中の足の軌跡をずらして
    連続性を保つ。
    """
    out = df.copy()

    def rebase_one(side: str, contact: np.ndarray):
        x = out[f"{side}_ANKLE_x_m"].to_numpy()
        z = out[f"{side}_ANKLE_z_m"].to_numpy()
        n = len(x)
        toeoffs = np.where((contact[:-1] == 1) & (contact[1:] == 0))[0]
        for j in toeoffs:
            t1 = j + 1
            k = t1
            while k < n and contact[k] == 0:
                k += 1
            tend = k - 1 if k > t1 else t1
            if t1 >= n:
                continue
            off_x = (x[j] - x[t1]) if rebase_on_x else 0.0
            off_z = (z[j] - z[t1]) if rebase_on_z else 0.0
            if off_x == 0.0 and off_z == 0.0:
                continue
            seg_len = tend - t1 + 1
            if seg_len <= 0:
                continue
            if rebase_blend_frames > 0:
                fade_n = min(rebase_blend_frames, seg_len)
                alpha = np.ones(seg_len, dtype=float)
                ramp = np.linspace(0.0, 1.0, fade_n, endpoint=True)
                alpha[:fade_n] = ramp
            else:
                alpha = np.ones(seg_len, dtype=float)
            x[t1 : tend + 1] = x[t1 : tend + 1] + alpha * off_x
            z[t1 : tend + 1] = z[t1 : tend + 1] + alpha * off_z
        out[f"{side}_ANKLE_x_m"] = x
        out[f"{side}_ANKLE_z_m"] = z

    rebase_one("L", contact_L.astype(int))
    rebase_one("R", contact_R.astype(int))
    return out


# -----------------------
# 速度特徴量追加
# -----------------------
def add_ankle_features(df: pd.DataFrame, fps: Optional[float] = None) -> pd.DataFrame:
    """L/R 足首の速度ノルム（m/s）と dy（m/s）を特徴量として追加。"""
    fpsv = _safe_fps(fps)
    out = df.copy()
    for side in ("L", "R"):
        vx = out[f"{side}_ANKLE_x_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
        vy = out[f"{side}_ANKLE_y_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
        vz = out[f"{side}_ANKLE_z_m"].astype(float).diff().fillna(0).to_numpy() * fpsv
        vnorm = np.sqrt(vx * vx + vy * vy + vz * vz)
        out[f"{side}_ANKLE_vnorm"] = vnorm
        out[f"{side}_ANKLE_dy"] = vy
    return out

def write_columns_and_config(out_dir: str, seq_len: int, horizon: int):
    """columns.json / config.json を out_dir に保存。"""
    pos_cols = []
    for nm in ORDER:
        pos_cols += [f"{nm}_x_m", f"{nm}_y_m", f"{nm}_z_m"]

    vel_cols = [
        "L_ANKLE_vnorm",
        "R_ANKLE_vnorm",
        "L_ANKLE_dy",
        "R_ANKLE_dy",
    ]

    cols = {
        "pos_cols": pos_cols,
        "vel_cols": vel_cols,
        "input_cols": pos_cols + vel_cols,
        "contact_cols": ["contact_L", "contact_R"],
        "event_cols": ["HS_L", "TO_L", "HS_R", "TO_R"],
        "target_cols": pos_cols,
        "X_cols": pos_cols + vel_cols,
        "y_cols": pos_cols,
    }

    cfg = {
        "seq_len": int(seq_len),
        "horizon": int(horizon),
        "use_vel": True,
        "use_contact": True,
        "use_stance": False,
        "use_events": True,
        "Dpos": len(pos_cols),
        "Din": len(pos_cols) + len(vel_cols),
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "columns.json"), "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] columns.json / config.json saved in {out_dir}")


# -----------------------
# 1 動画分パイプライン
# -----------------------
def process_one(
    video_path: str,
    out_dir: str,
    score_w_v: float = DEFAULT_SCORE_W_V,
    score_w_y: float = DEFAULT_SCORE_W_Y,
    min_stance_frames: int = DEFAULT_MIN_STANCE_FRAMES,
    contact_v_enter_q: float = DEFAULT_CONTACT_V_ENTER_Q,
    contact_mode: str = DEFAULT_CONTACT_MODE,
    h_thr_m: float = DEFAULT_H_THR_M,
    u_thr_mps: float = DEFAULT_U_THR_MPS,
    on_n: int = DEFAULT_ON_N,
    off_n: int = DEFAULT_OFF_N,
    event_win: int = DEFAULT_EVENT_WIN,
    use_extra_foot_internal: bool = DEFAULT_USE_EXTRA_FOOT_INTERNAL,
    ds_allow_radius: int = DEFAULT_DS_ALLOW_RADIUS,
    ds_force_radius: int = DEFAULT_DS_FORCE_RADIUS,
    max_double_frames: int = DEFAULT_MAX_DOUBLE_FRAMES,
    min_switch_gap: int = DEFAULT_MIN_SWITCH_GAP,
    min_contact_run: int = DEFAULT_MIN_CONTACT_RUN,
    min_support_run: int = DEFAULT_MIN_SUPPORT_RUN,
    lock_tail_frames: int = DEFAULT_LOCK_TAIL_FRAMES,
    lock_tol_z: float = DEFAULT_LOCK_TOL_Z_M,
    lock_tol_x: float = DEFAULT_LOCK_TOL_X_M,
    rebase_blend_frames: int = DEFAULT_REBASE_BLEND_FRAMES,
    rebase_on_x: bool = DEFAULT_REBASE_ON_X,
    rebase_on_z: bool = DEFAULT_REBASE_ON_Z,
    pelvis_align_use_ds_only: bool = DEFAULT_PELVIS_ALIGN_USE_DS_ONLY,
    debug_stages: bool = False,
):
    """指定動画を処理し、raw/processed CSV を出力する。"""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    raw_csv = os.path.join(out_dir, f"{stem}_raw.csv")
    proc_csv = os.path.join(out_dir, f"{stem}_processed.csv")

    print(f"\n=== Processing: {video_path} -> {proc_csv} ===")

    # --- stage0: capture ---
    df0, fps = capture_to_world(video_path, include_extra_foot=bool(use_extra_foot_internal))
    if df0.empty:
        print(f"[Warn] {video_path}: 有効フレームなしのためスキップ")
        return None
    if debug_stages:
        df0.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage0.csv"),
            index=False,
            encoding="utf-8",
        )

    # --- stage1: floor level ---
    df1 = level_floor(df0)
    if debug_stages:
        df1.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage1.csv"),
            index=False,
            encoding="utf-8",
        )

    # stance + contact（df1 基準）
    stance_list, cL, cR = choose_stance(
        df1,
        fps=fps,
        use_extra_foot_internal=bool(use_extra_foot_internal),
        contact_mode=str(contact_mode),
        h_thr_m=float(h_thr_m),
        u_thr_mps=float(u_thr_mps),
        on_n=int(on_n),
        off_n=int(off_n),
        score_w_v=score_w_v,
        score_w_y=score_w_y,
        min_stance_frames=min_stance_frames,
        contact_v_enter_q=contact_v_enter_q,
        ds_allow_radius=ds_allow_radius,
        ds_force_radius=ds_force_radius,
        max_double_frames=max_double_frames,
        min_switch_gap=min_switch_gap,
    )

    # ---- contact チラつき抑制（短いランの除去）----
    # choose_stance のヒステリシス(on_n/off_n)に加えて、1〜2フレーム程度の孤立した反転を除去する。
    # 0で無効。
    if int(min_contact_run) and int(min_contact_run) > 0:
        cL = _smooth_short_runs_binary(np.asarray(cL, dtype=np.int32), min_run=int(min_contact_run)).astype(np.int32)
        cR = _smooth_short_runs_binary(np.asarray(cR, dtype=np.int32), min_run=int(min_contact_run)).astype(np.int32)

        # (0,0) を許さない方針のため、stance を使って補正する（飛行相が本当にある場合はここを見直す）
        st_lr = np.asarray([0 if s == "L" else 1 for s in stance_list], dtype=np.int32).reshape(-1)
        both0 = (cL == 0) & (cR == 0)
        if np.any(both0):
            cL[both0 & (st_lr == 0)] = 1
            cR[both0 & (st_lr == 1)] = 1


    # 膝の前後向きを推定
    knee_dirs = estimate_knee_direction(df1)

    # --- stage2: IK1 ---
    lens = estimate_bone_lengths(df1, stance_list)
    df_ik1 = apply_ik(df1, lens, knee_dirs)
    if debug_stages:
        df_ik1.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage2.csv"),
            index=False,
            encoding="utf-8",
        )

    # --- stage3: pelvis Z align ---
    df_pelvis, base = pelvis_z_align(
        df_ik1,
        stance_list,
        use_ds_only=pelvis_align_use_ds_only,
    )
    print(f"[Diag:{stem}] pelvis Z align base = {base:.3f} m")
    if debug_stages:
        df_pelvis.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage3.csv"),
            index=False,
            encoding="utf-8",
        )

    # --- stage4: IK2（pelvis 調整後に再 IK） ---
    df_ik2 = apply_ik(df_pelvis, lens, knee_dirs)
    if debug_stages:
        df_ik2.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage4.csv"),
            index=False,
            encoding="utf-8",
        )

    # --- stage5: ankle lock ---
    df_locked = ankle_lock(
        df_ik2,
        stance_list,
        cL,
        cR,
        lock_tail_frames=lock_tail_frames,
        lock_tol_z=lock_tol_z,
        lock_tol_x=lock_tol_x,
    )
    if debug_stages:
        df_locked.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage5.csv"),
            index=False,
            encoding="utf-8",
        )

    # --- stage6: rebase swing ---
    df_rebased = rebase_swing_from_toeoff(
        df_locked,
        cL,
        cR,
        rebase_blend_frames=rebase_blend_frames,
        rebase_on_x=rebase_on_x,
        rebase_on_z=rebase_on_z,
    )
    if debug_stages:
        df_rebased.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage6.csv"),
            index=False,
            encoding="utf-8",
        )

    # raw 出力（前進 GZ 未適用）
    df_out_raw = df_rebased.copy()
    df_out_raw["video_id"] = stem

    # 出力列は 6 点（ANKLEまで）だけに限定（内部用の HEEL/FOOT_INDEX は落とす）
    pos_cols = []
    for nm in ORDER:
        pos_cols += [f"{nm}_x_m", f"{nm}_y_m", f"{nm}_z_m"]
    raw_keep = ["frame", "time_sec"] + pos_cols + ["video_id"]
    df_out_raw = df_out_raw[[c for c in raw_keep if c in df_out_raw.columns]]

    df_out_raw.to_csv(raw_csv, index=False, encoding="utf-8")
    print(f"[OK] raw CSV: {raw_csv}")

    # --- GZ 推定と適用（processed 用） ---
    GZ, step_list = build_forward_from_hs_robust(df_rebased, stance_list, cL, cR, fps=fps)

    if debug_stages:
        df_forward = df_rebased.copy()
        for nm in ORDER:
            df_forward[f"{nm}_z_m"] = df_forward[f"{nm}_z_m"].to_numpy() + GZ
        df_forward.assign(video_id=stem).to_csv(
            os.path.join(out_dir, f"{stem}_stage7.csv"),
            index=False,
            encoding="utf-8",
        )
        print(f"[OK] stage7_forward CSV: {stem}_stage7.csv")

    proc_df = df_rebased.copy()
    for nm in ORDER:
        proc_df[f"{nm}_z_m"] = proc_df[f"{nm}_z_m"].to_numpy() + GZ

    proc_df["contact_L"] = cL.astype(int)
    proc_df["contact_R"] = cR.astype(int)


    # --- 支持脚(stances) のラベル化 ---
    # stance_list は choose_stance の出力（"L"/"R"/"DS"/"NONE"）。
    # 学習ラベルとして扱いやすいよう、ここでは 0/1 の二値 support に落とし込み、
    # - "DS"/"NONE" は直前の support を維持（先頭は最初に出現する L/R で埋める）
    # - 短いチラつきは min_stance_frames を閾値として統合
    # を行う。
    support_lr = support_from_stance_list(
        stance_list=stance_list,
        fallback=0,  # どちらも決められない場合は左を既定にする（必要なら変更）
        min_run=int(min_support_run),
    )
    proc_df["support"] = support_lr.astype(int)
    proc_df["stance_L"] = (support_lr == 0).astype(int)
    proc_df["stance_R"] = (support_lr == 1).astype(int)

    # 支持脚の切替タイミング（予測 support から接地イベントを抽出する際の基準として使える）
    sw, HSsL, HSsR, TOsL, TOsR = support_switch_events(support_lr, event_win=int(event_win))
    proc_df["support_switch"] = sw.astype(int)
    proc_df["HS_support_L"] = HSsL
    proc_df["HS_support_R"] = HSsR
    proc_df["TO_support_L"] = TOsL
    proc_df["TO_support_R"] = TOsR
    # ------------------------------------------------------------
    # Contact event columns (HS/TO)
    #  - IMPORTANT:
    #    This project uses 6 lower-limb joints only. Therefore we treat the ankle/foot joint
    #    as a representative point. HS/TO here means "contact on/off events" derived from
    #    contact_L/contact_R transitions (not true heel/toe markers).
    # ------------------------------------------------------------
    prev_L = np.concatenate([[0], cL[:-1].astype(int)])
    prev_R = np.concatenate([[0], cR[:-1].astype(int)])
    HS_L = ((cL.astype(int) == 1) & (prev_L == 0)).astype(int)
    TO_L = ((cL.astype(int) == 0) & (prev_L == 1)).astype(int)
    HS_R = ((cR.astype(int) == 1) & (prev_R == 0)).astype(int)
    TO_R = ((cR.astype(int) == 0) & (prev_R == 1)).astype(int)
    # イベントのジッタ抑制（連続イベントを抑える）
    HS_L = _debounce_events(HS_L, event_win)
    TO_L = _debounce_events(TO_L, event_win)
    HS_R = _debounce_events(HS_R, event_win)
    TO_R = _debounce_events(TO_R, event_win)
    proc_df["HS_L"] = HS_L
    proc_df["TO_L"] = TO_L
    proc_df["HS_R"] = HS_R
    proc_df["TO_R"] = TO_R

    proc_df = add_ankle_features(proc_df, fps=fps)
    proc_df["video_id"] = stem

    # processed 出力は 6 点 + vel + contact/events + stance/support のみに限定（内部列を落とす）
    vel_cols = ["L_ANKLE_vnorm", "R_ANKLE_vnorm", "L_ANKLE_dy", "R_ANKLE_dy"]
    keep = ["frame", "time_sec"] + pos_cols + vel_cols + [
        "contact_L", "contact_R", "HS_L", "TO_L", "HS_R", "TO_R",
        "support", "support_switch", "stance_L", "stance_R",
        "HS_support_L", "TO_support_L", "HS_support_R", "TO_support_R",
        "video_id"
    ]
    proc_df = proc_df[[c for c in keep if c in proc_df.columns]]

    proc_df.to_csv(proc_csv, index=False, encoding="utf-8")
    print(f"[OK] processed CSV: {proc_csv}")

    return {
        "stem": stem,
        "raw_csv": raw_csv,
        "proc_csv": proc_csv,
        "steps": step_list,
    }


# -----------------------
# メイン
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="zcap: capture + preprocess (lower-body)")
    parser.add_argument(
        "--input_glob",
        type=str,
        default=None,
        help='入力動画の glob。例: "C:/data/*.mp4"',
    )
    parser.add_argument(
        "--input_list",
        type=str,
        default=None,
        help="動画パスを 1 行 1 本で列挙したテキストファイル。input_list が優先。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="CSV 出力先ディレクトリ",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=50,
        help="config.json に書き出す seq_len",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="config.json に書き出す horizon",
    )
    parser.add_argument(
        "--contact_mode",
        type=str,
        default=DEFAULT_CONTACT_MODE,
        choices=["thr", "score"],
        help="contact 推定モード: thr(高さ+速度) / score(速度quantile)",
    )
    parser.add_argument(
        "--h_thr_m",
        type=float,
        default=DEFAULT_H_THR_M,
        help="contact_mode=thr の高さ閾値（m）",
    )
    parser.add_argument(
        "--u_thr_mps",
        type=float,
        default=DEFAULT_U_THR_MPS,
        help="contact_mode=thr の水平速度閾値（m/s）",
    )
    parser.add_argument(
        "--on_n",
        type=int,
        default=DEFAULT_ON_N,
        help="contact_mode=thr の 0->1 に必要な連続フレーム数",
    )
    parser.add_argument(
        "--off_n",
        type=int,
        default=DEFAULT_OFF_N,
        help="contact_mode=thr の 1->0 に必要な連続フレーム数",
    )
    parser.add_argument(
        "--event_win",
        type=int,
        default=DEFAULT_EVENT_WIN,
        help="HS/TO の連続イベント抑制（最低間隔フレーム）",
    )
    parser.add_argument(
        "--use_extra_foot_internal",
        type=int,
        choices=[0, 1],
        default=1 if DEFAULT_USE_EXTRA_FOOT_INTERNAL else 0,
        help="1: HEEL/FOOT_INDEX を内部処理に使用（CSVには出力しない）",
    )

    parser.add_argument(
        "--contact_v_enter_q",
        type=float,
        default=DEFAULT_CONTACT_V_ENTER_Q,
        help="接地速度閾値として使う quantile (0〜1)",
    )
    parser.add_argument(
    "--ds_allow_radius",
    type=int,
    default=DEFAULT_DS_ALLOW_RADIUS,
    help="二重接地(1,1)を“許可”する切替近傍の半径（フレーム）",
    )
    parser.add_argument(
        "--ds_force_radius",
        type=int,
        default=DEFAULT_DS_FORCE_RADIUS,
        help="二重接地(1,1)を“強制”する切替中心の半径（フレーム）",
    )
    parser.add_argument(
        "--max_double_frames",
        type=int,
        default=DEFAULT_MAX_DOUBLE_FRAMES,
        help="二重接地(1,1)の最大連続フレーム（超過分は単脚支持へ戻す）",
    )
    parser.add_argument(
        "--min_switch_gap",
        type=int,
        default=DEFAULT_MIN_SWITCH_GAP,
        help="stance 切替の最小間隔（これ未満の連続切替は捨てる）",
    )

    parser.add_argument(
        "--min_contact_run",
        type=int,
        default=DEFAULT_MIN_CONTACT_RUN,
        help="contact_L/R の短いラン（0/1どちらも）を除去してチラつきを抑える最小長。0で無効。",
    )
    parser.add_argument(
        "--min_support_run",
        type=int,
        default=DEFAULT_MIN_SUPPORT_RUN,
        help="support(支持脚:0=L,1=R) の短いランを除去してチラつきを抑える最小長。0で無効。",
    )
    parser.add_argument(
        "--score_w_v",
        type=float,
        default=DEFAULT_SCORE_W_V,
        help="stance/contact スコアの速度重み（大きいほど“動いていない足”優先）",
    )
    parser.add_argument(
        "--score_w_y",
        type=float,
        default=DEFAULT_SCORE_W_Y,
        help="stance/contact スコアの高さ重み（大きいほど“低い足”優先）",
    )
    parser.add_argument(
        "--min_stance_frames",
        type=int,
        default=DEFAULT_MIN_STANCE_FRAMES,
        help="stance の最低継続フレーム（小さいほど切替が増える）",
    )
    parser.add_argument(
        "--debug_stages",
        action="store_true",
        help="各ステージの中間 CSV を出力する",
    )

    args = parser.parse_args()

    # 入力動画リスト
    videos: List[str] = []
    if args.input_list:
        if not os.path.exists(args.input_list):
            print(f"[Error] input_list が見つかりません: {args.input_list}", file=sys.stderr)
            sys.exit(1)
        with open(args.input_list, "r", encoding="utf-8") as f:
            videos = [line.strip() for line in f if line.strip()]
    elif args.input_glob:
        videos = sorted(glob.glob(args.input_glob))
    else:
        print("[Error] --input_glob または --input_list を指定してください", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print("[Error] 指定パターンに一致する動画が 0 件です", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for v in videos:
        try:
            res = process_one(
                v,
                out_dir,
                score_w_v=args.score_w_v,
                score_w_y=args.score_w_y,
                min_stance_frames=args.min_stance_frames,
                contact_v_enter_q=args.contact_v_enter_q,
                contact_mode=args.contact_mode,
                h_thr_m=args.h_thr_m,
                u_thr_mps=args.u_thr_mps,
                on_n=args.on_n,
                off_n=args.off_n,
                event_win=args.event_win,
                use_extra_foot_internal=bool(args.use_extra_foot_internal),

                ds_allow_radius=args.ds_allow_radius,
                ds_force_radius=args.ds_force_radius,
                max_double_frames=args.max_double_frames,
                min_switch_gap=args.min_switch_gap,
                min_contact_run=args.min_contact_run,
                min_support_run=args.min_support_run,
                debug_stages=args.debug_stages,
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"[Skip] {v}: {e}")

    if results:
        write_columns_and_config(out_dir, seq_len=args.seq_len, horizon=args.horizon)

    print(f"\n[Done] processed {len(results)} / {len(videos)} videos. outputs -> {out_dir}")


if __name__ == "__main__":
    main()
