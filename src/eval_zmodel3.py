# -*- coding: utf-8 -*-
"""
eval_zmodel3_INTEGRATED.py

目的:
- zmodel4/zmodel5 系で学習した out_dir を入力し、論文貼り付け用の表(CSV)と図(PNG)を自動生成する統合評価スクリプト。
- eval_zmodel2.py への import 依存を持たない（単体で動く）ため、モジュール関係の AttributeError を回避する。

想定 out_dir の構造（zmodel4系）:
- out_dir/
    - zmodel_lstm.pt
    - scaler.pkl
    - config.json
    - columns.json
    - train_log.csv（任意）
    - train_val_test_split.json（任意; which=val/test を使う場合に参照）

主な出力（out_dir/eval_thesis/<which>/<run_tag>[_ABS|_ROOT]/）:
- eval_summary.csv / eval_summary.json: 主要指標まとめ
- eval_per_window.csv: window 単位の指標（分布/外れの解析用）
- per_horizon_scores.csv: 未来 t=1..H での誤差推移（位置・Z・骨格・膝角）
- per_horizon_support.csv: 未来 t=1..H での支持脚予測（acc/f1等）
- per_horizon_scores.png / per_horizon_support.png: 上記の図
- sample_<idx>_ts.png: サンプル時系列図（hipZ、膝角、supportなど）
- sample_<idx>_<view>.gif: サンプルの骨格 overlay GIF（--gif 指定時）
- losses.png: train/val の loss 推移（--plot_losses 指定時）

注意:
- 位置誤差は ABS（前進を含む絶対座標）と ROOT（hip中心のx,z平行移動を除去）を切り替え可能。
- 支持脚(support)は 2クラス固定（0=left, 1=right）。両足非接地や両足接地は評価側で丸める前提。

CLI 例:
python eval_zmodel3_INTEGRATED.py --out_dir outputs_hitobetu --which test --run_tag A_test --plot_losses --plot_sample_ts --gif --view yz --gif_fps 5 --support_min_run 3 --event_win 2 --z_tol 0.05 --coord both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Matplotlib backend: GIF生成時に TkAgg になると tostring_rgb が無い等で落ちることがあるため固定
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__version__ = "eval_zmodel3_integrated_fixed_v8"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ------------------------------------------------------------
# I/O utilities
# ------------------------------------------------------------

def filter_state_dict_by_shape(model: torch.nn.Module, sd: dict):
    """Filter checkpoint state_dict by matching keys and tensor shapes.

    Some of your eval scripts load checkpoints with slightly different heads/LayerNorm shapes.
    This helper keeps only parameters that exist in the current model AND have identical shapes.

    Returns:
        sd_filtered: dict suitable for model.load_state_dict(sd_filtered, strict=False)
        skipped: list of keys skipped due to missing key or shape mismatch
    """
    if sd is None:
        return {}, []

    # If checkpoint was saved as {'state_dict': ...} style, unwrap.
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    model_sd = model.state_dict()
    sd_filtered = {}
    skipped = []

    for k, v in sd.items():
        # Strip a common DistributedDataParallel prefix if present
        k2 = k[7:] if k.startswith("module.") else k

        if k2 not in model_sd:
            skipped.append(k)
            continue
        try:
            # Compare tensor shapes (ignore non-tensors)
            if hasattr(v, "shape") and hasattr(model_sd[k2], "shape"):
                if tuple(v.shape) != tuple(model_sd[k2].shape):
                    skipped.append(k)
                    continue
        except Exception:
            # If anything odd happens, skip to be safe
            skipped.append(k)
            continue

        sd_filtered[k2] = v

    return sd_filtered, skipped

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_float_str(x: float) -> str:
    s = f"{x:.4f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")

def read_csv_smart(path: Path) -> pd.DataFrame:
    # encodingの揺れに耐える（Windows環境対策）
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ------------------------------------------------------------
# Scaler utilities（contact/support をスケーリングしない）
# ------------------------------------------------------------

def load_scaler_robust(path: Path):
    """
    scaler.pkl の読み込み。joblib→pickle の順で試す。
    """
    import pickle
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)

def get_scaler_dim(scaler) -> int:
    if hasattr(scaler, "mean_"):
        return int(len(scaler.mean_))
    if hasattr(scaler, "scale_"):
        return int(len(scaler.scale_))
    raise ValueError("Scaler object does not have mean_ / scale_ attributes.")

def _nan_fill_cols(x: np.ndarray) -> np.ndarray:
    """
    StandardScaler が NaN を扱えないため、列ごとに NaN を埋める。
    - まず列平均(nanmean)で埋める
    - 全部 NaN の列は 0
    """
    x = x.copy()
    col_mean = np.nanmean(x, axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    inds = np.where(~np.isfinite(x))
    x[inds] = np.take(col_mean, inds[1])
    return x

def scale_inputs(
    x_all: np.ndarray,
    scaler,
    input_cols: List[str],
    flag_cols: List[str],
    numeric_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    x_all: (T, Din) raw
    returns:
      x_scaled: (T, Din)
      scaler_on_cols: scaler.fit の列順（input_cols か numeric_cols）
    """
    Din = x_all.shape[1]
    if Din != len(input_cols):
        raise ValueError(f"Input dim mismatch: x_all has {Din}, input_cols has {len(input_cols)}")

    if numeric_cols is None:
        numeric_cols = [c for c in input_cols if c not in set(flag_cols)]
    sdim = get_scaler_dim(scaler)

    x_all = _nan_fill_cols(x_all)

    if sdim == len(input_cols):
        xs = scaler.transform(x_all)
        return xs.astype(np.float32), list(input_cols)

    if sdim == len(numeric_cols):
        # scale numeric only, keep flags as-is
        num_set = set(numeric_cols)
        num_idx = [i for i, c in enumerate(input_cols) if c in num_set]
        xs = x_all.astype(np.float32).copy()
        xs[:, num_idx] = scaler.transform(x_all[:, num_idx]).astype(np.float32)
        return xs, list(numeric_cols)

    raise ValueError(
        f"Scaler dimension mismatch: scaler_dim={sdim}, input_dim={len(input_cols)}, numeric_dim={len(numeric_cols)}"
    )

def inverse_pos_only(
    pos_scaled: np.ndarray,  # (..., Dpos) scaled
    scaler,
    scaler_on_cols: List[str],
    pos_cols: List[str],
) -> np.ndarray:
    """
    posのみ逆変換。scaler が numeric だけでfitされている場合でも対応するため、
    scaler_on_cols 上で pos_cols を探して逆変換する。
    """
    if scaler is None:
        return pos_scaled

    mean = np.asarray(getattr(scaler, "mean_", None), dtype=np.float32)
    scale = np.asarray(getattr(scaler, "scale_", None), dtype=np.float32)
    if mean is None or scale is None:
        raise ValueError("Scaler must have mean_ and scale_.")

    idx = []
    col_to_i = {c: i for i, c in enumerate(scaler_on_cols)}
    for c in pos_cols:
        if c not in col_to_i:
            raise ValueError(f"pos_col '{c}' is not in scaler_on_cols. scaler_on_cols={scaler_on_cols[:5]}..")
        idx.append(col_to_i[c])
    idx = np.array(idx, dtype=np.int64)

    # broadcasting: (..., Dpos)
    return pos_scaled * scale[idx] + mean[idx]

# ------------------------------------------------------------
# Support label utilities（2クラス固定: 0=left, 1=right）
# ------------------------------------------------------------

def build_support_from_contacts(
    contact_L: np.ndarray, contact_R: np.ndarray
) -> np.ndarray:
    """
    contact_L/R: (T,) 0/1
    returns support: (T,) 0/1 (left/right)
    - 両足接地(1,1) は「直前の support を維持」
    - 両足非接地(0,0) は「直前の support を維持」
    先頭が曖昧な場合は left(0) とする。
    """
    T = len(contact_L)
    sup = np.zeros((T,), dtype=np.int64)
    cur = 0
    for t in range(T):
        l = int(contact_L[t] > 0.5)
        r = int(contact_R[t] > 0.5)
        if l == 1 and r == 0:
            cur = 0
        elif l == 0 and r == 1:
            cur = 1
        # else: keep cur
        sup[t] = cur
    return sup

def smooth_min_run_labels(labels: np.ndarray, min_run: int) -> np.ndarray:
    """
    連続長が min_run 未満の島を近傍に吸収。
    labels: (T,) int
    """
    if min_run <= 1:
        return labels.copy()
    y = labels.copy()
    T = len(y)
    # run-length encoding
    runs = []
    s = 0
    while s < T:
        e = s + 1
        while e < T and y[e] == y[s]:
            e += 1
        runs.append((s, e))
        s = e

    for (s, e) in runs:
        if e - s >= min_run:
            continue
        # find left/right label
        left_lab = y[s - 1] if s - 1 >= 0 else None
        right_lab = y[e] if e < T else None
        # choose neighbor label if exists, prioritize longer run
        if left_lab is None and right_lab is None:
            continue
        if left_lab is None:
            y[s:e] = right_lab
        elif right_lab is None:
            y[s:e] = left_lab
        else:
            # decide by neighbor run lengths
            left_len = 0
            i = s - 1
            while i >= 0 and y[i] == left_lab:
                left_len += 1
                i -= 1
            right_len = 0
            i = e
            while i < T and y[i] == right_lab:
                right_len += 1
                i += 1
            y[s:e] = left_lab if left_len >= right_len else right_lab
    return y

def switch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_win: int = 2,
) -> Dict[str, float]:
    """
    switch: support が切り替わるフレーム（t-1 -> t でラベルが変化）
    event_win: 予測切替が正解切替の ±event_win フレーム内ならTPとする

    returns: prec/rec/f1/acc
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    T = len(y_true)
    if T <= 1:
        return {"switch_prec": float("nan"), "switch_rec": float("nan"), "switch_f1": float("nan"), "switch_acc": float("nan")}

    true_sw = np.where(y_true[1:] != y_true[:-1])[0] + 1
    pred_sw = np.where(y_pred[1:] != y_pred[:-1])[0] + 1

    true_set = set(true_sw.tolist())
    pred_set = set(pred_sw.tolist())

    # match within window
    matched_true = set()
    tp = 0
    for p in pred_sw:
        ok = False
        for dt in range(-event_win, event_win + 1):
            t = int(p + dt)
            if t in true_set and t not in matched_true:
                matched_true.add(t)
                ok = True
                break
        if ok:
            tp += 1
    fp = len(pred_set) - tp
    fn = len(true_set) - len(matched_true)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # switch_acc: 各フレームで「切替/非切替」が一致する割合
    true_sw_bin = np.zeros((T,), dtype=np.int64)
    pred_sw_bin = np.zeros((T,), dtype=np.int64)
    true_sw_bin[true_sw] = 1
    pred_sw_bin[pred_sw] = 1
    acc = float(np.mean((true_sw_bin == pred_sw_bin).astype(np.float32)))

    return {"switch_prec": float(prec), "switch_rec": float(rec), "switch_f1": float(f1), "switch_acc": acc}

def cls_report_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    2クラス classification の簡易指標。macro F1 を出す。
    """
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    acc = float(np.mean((y_true == y_pred).astype(np.float32)))

    def f1_for_cls(c: int) -> float:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    f1_0 = f1_for_cls(0)
    f1_1 = f1_for_cls(1)
    f1_macro = float((f1_0 + f1_1) / 2.0)
    return {"sup_acc": acc, "sup_f1_macro": f1_macro}

# ------------------------------------------------------------
# Geometry utilities（骨・膝角・MPJPE）
# ------------------------------------------------------------

@dataclass
class JointMap:
    # pos_cols を joint -> (x_col,y_col,z_col) に写像する
    joint_to_cols: Dict[str, Tuple[str, str, str]]
    joint_list: List[str]

def build_joint_map(pos_cols: List[str]) -> JointMap:
    """
    例: L_HIP_x_m, L_HIP_y_m, L_HIP_z_m 形式を想定
    """
    # 末尾を _x_m/_y_m/_z_m と仮定
    joint_names = set()
    for c in pos_cols:
        if c.endswith("_x_m"):
            joint_names.add(c[:-4])
        elif c.endswith("_y_m"):
            joint_names.add(c[:-4])
        elif c.endswith("_z_m"):
            joint_names.add(c[:-4])
    joint_list = sorted(list(joint_names))

    joint_to_cols = {}
    for j in joint_list:
        cx, cy, cz = f"{j}_x_m", f"{j}_y_m", f"{j}_z_m"
        if cx in pos_cols and cy in pos_cols and cz in pos_cols:
            joint_to_cols[j] = (cx, cy, cz)
    if len(joint_to_cols) == 0:
        raise ValueError("Failed to parse pos_cols naming. Expected *_x_m/*_y_m/*_z_m.")
    return JointMap(joint_to_cols=joint_to_cols, joint_list=joint_list)

def get_joint_xyz(arr: np.ndarray, jm: JointMap, joint: str) -> np.ndarray:
    """
    arr: (..., Dpos) with columns order = pos_cols
    returns (..., 3)
    """
    cx, cy, cz = jm.joint_to_cols[joint]
    # We cannot index by name directly; build indices once outside for speed
    raise RuntimeError("get_joint_xyz should not be called directly; use precomputed indices.")

def precompute_pos_indices(pos_cols: List[str], jm: JointMap) -> Dict[str, Tuple[int, int, int]]:
    idx = {}
    col_to_i = {c: i for i, c in enumerate(pos_cols)}
    for j, (cx, cy, cz) in jm.joint_to_cols.items():
        idx[j] = (col_to_i[cx], col_to_i[cy], col_to_i[cz])
    return idx

def mpjpe_m(gt: np.ndarray, pr: np.ndarray, pos_dim: int) -> np.ndarray:
    """
    gt/pr: (N, H, Dpos)
    returns mpjpe per horizon: (H,)
    """
    N, H, D = gt.shape
    assert D == pos_dim
    # reshape to (N,H,J,3)
    J = D // 3
    g = gt.reshape(N, H, J, 3)
    p = pr.reshape(N, H, J, 3)
    d = np.linalg.norm(p - g, axis=-1)  # (N,H,J)
    return np.mean(d, axis=(0, 2)).astype(np.float32)  # (H,)

def per_horizon_mae_rmse(gt: np.ndarray, pr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    gt/pr: (N,H,D)
    returns mae(H,), rmse(H,)
    """
    e = pr - gt
    mae = np.mean(np.abs(e), axis=(0, 2)).astype(np.float32)
    rmse = np.sqrt(np.mean(e * e, axis=(0, 2))).astype(np.float32)
    return mae, rmse

def per_horizon_z_mae(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """
    Z成分のみ（全関節の z をまとめた MAE）: (H,)
    """
    N, H, D = gt.shape
    J = D // 3
    g = gt.reshape(N, H, J, 3)[..., 2]
    p = pr.reshape(N, H, J, 3)[..., 2]
    return np.mean(np.abs(p - g), axis=(0, 2)).astype(np.float32)

def per_horizon_z_acc(gt: np.ndarray, pr: np.ndarray, z_tol: float) -> np.ndarray:
    N, H, D = gt.shape
    J = D // 3
    g = gt.reshape(N, H, J, 3)[..., 2]
    p = pr.reshape(N, H, J, 3)[..., 2]
    ok = (np.abs(p - g) <= float(z_tol)).astype(np.float32)
    return np.mean(ok, axis=(0, 2)).astype(np.float32)

def hip_center_z(arr: np.ndarray, pos_cols: List[str], jm: JointMap, pos_idx: Dict[str, Tuple[int,int,int]]) -> np.ndarray:
    """
    arr: (N,H,Dpos)
    returns hip_center_z: (N,H)
    """
    if "L_HIP" not in pos_idx or "R_HIP" not in pos_idx:
        # fallback: first two joints
        joints = list(pos_idx.keys())
        j0, j1 = joints[0], joints[1]
        li = pos_idx[j0][2]
        ri = pos_idx[j1][2]
    else:
        li = pos_idx["L_HIP"][2]
        ri = pos_idx["R_HIP"][2]
    return (arr[..., li] + arr[..., ri]) / 2.0

def root_xz(arr: np.ndarray, pos_idx: Dict[str, Tuple[int,int,int]]) -> np.ndarray:
    """
    hip中心の x,z を除去（yは保持）。
    arr: (N,H,Dpos)
    """
    if "L_HIP" not in pos_idx or "R_HIP" not in pos_idx:
        return arr
    lx, ly, lz = pos_idx["L_HIP"]
    rx, ry, rz = pos_idx["R_HIP"]
    cx = (arr[..., lx] + arr[..., rx]) / 2.0
    cz = (arr[..., lz] + arr[..., rz]) / 2.0
    out = arr.copy()
    # x: every 3rd starting 0
    out[..., 0::3] = out[..., 0::3] - cx[..., None]
    out[..., 2::3] = out[..., 2::3] - cz[..., None]
    return out

def bone_and_knee_by_t(
    gt: np.ndarray, pr: np.ndarray, pos_cols: List[str], bone_cos_thr: float, knee_deg_tol: float
) -> pd.DataFrame:
    """
    gt/pr: (N,H,Dpos) in meters
    returns DataFrame with columns:
      t, bone_len_mae_m, bone_len_rel_mae, bone_cos_mean, bone_cos_acc, knee_mae_deg, knee_acc
    """
    jm = build_joint_map(pos_cols)
    pos_idx = precompute_pos_indices(pos_cols, jm)

    # bones used (5 bones): L_HIP-L_KNEE, L_KNEE-L_ANKLE, R_HIP-R_KNEE, R_KNEE-R_ANKLE, L_HIP-R_HIP
    bones = []
    def add_bone(a: str, b: str):
        if a in pos_idx and b in pos_idx:
            bones.append((a,b))
    add_bone("L_HIP","L_KNEE")
    add_bone("L_KNEE","L_ANKLE")
    add_bone("R_HIP","R_KNEE")
    add_bone("R_KNEE","R_ANKLE")
    add_bone("L_HIP","R_HIP")
    if len(bones) == 0:
        # fallback: first 5 pairs
        jlist = list(pos_idx.keys())
        for i in range(min(5, len(jlist)-1)):
            bones.append((jlist[i], jlist[i+1]))

    N,H,D = gt.shape
    # compute bone vectors
    def vec(arr, a, b):
        ax, ay, az = pos_idx[a]
        bx, by, bz = pos_idx[b]
        va = arr[..., [ax, ay, az]]
        vb = arr[..., [bx, by, bz]]
        return vb - va

    rows = []
    for t in range(H):
        len_errs = []
        len_rel_errs = []
        cos_vals = []
        for (a,b) in bones:
            vg = vec(gt[:,t,:], a,b)
            vp = vec(pr[:,t,:], a,b)
            lg = np.linalg.norm(vg, axis=-1) + 1e-8
            lp = np.linalg.norm(vp, axis=-1)
            len_errs.append(np.abs(lp - lg))
            len_rel_errs.append(np.abs(lp - lg) / lg)

            # cosine similarity
            dot = np.sum(vg * vp, axis=-1)
            ng = np.linalg.norm(vg, axis=-1) + 1e-8
            npv = np.linalg.norm(vp, axis=-1) + 1e-8
            cos = dot / (ng * npv)
            cos = np.clip(cos, -1.0, 1.0)
            cos_vals.append(cos)

        len_errs = np.concatenate([x.reshape(-1,1) for x in len_errs], axis=1)  # (N,B)
        len_rel_errs = np.concatenate([x.reshape(-1,1) for x in len_rel_errs], axis=1)
        cos_vals = np.concatenate([x.reshape(-1,1) for x in cos_vals], axis=1)
        bone_len_mae = float(np.mean(len_errs))
        bone_len_rel_mae = float(np.mean(len_rel_errs))
        bone_cos_mean = float(np.mean(cos_vals))
        bone_cos_acc = float(np.mean((cos_vals >= float(bone_cos_thr)).astype(np.float32)))

        # knee angle: angle between (HIP-KNEE) and (ANKLE-KNEE)
        def knee_angle(arr, side: str) -> np.ndarray:
            hip = f"{side}_HIP"
            knee = f"{side}_KNEE"
            ankle = f"{side}_ANKLE"
            if hip not in pos_idx or knee not in pos_idx or ankle not in pos_idx:
                return np.full((N,), np.nan, dtype=np.float32)
            hx, hy, hz = pos_idx[hip]
            kx, ky, kz = pos_idx[knee]
            ax, ay, az = pos_idx[ankle]
            v1 = arr[:, [hx,hy,hz]] - arr[:, [kx,ky,kz]]
            v2 = arr[:, [ax,ay,az]] - arr[:, [kx,ky,kz]]
            n1 = np.linalg.norm(v1, axis=-1) + 1e-8
            n2 = np.linalg.norm(v2, axis=-1) + 1e-8
            cos = np.sum(v1*v2, axis=-1) / (n1*n2)
            cos = np.clip(cos, -1.0, 1.0)
            ang = np.degrees(np.arccos(cos))
            return ang.astype(np.float32)

        gt_L = knee_angle(gt[:,t,:], "L")
        pr_L = knee_angle(pr[:,t,:], "L")
        gt_R = knee_angle(gt[:,t,:], "R")
        pr_R = knee_angle(pr[:,t,:], "R")
        err = np.nanmean(np.stack([np.abs(pr_L-gt_L), np.abs(pr_R-gt_R)], axis=1), axis=1)
        knee_mae_deg = float(np.nanmean(err))
        knee_acc = float(np.nanmean((err <= float(knee_deg_tol)).astype(np.float32)))

        rows.append({
            "t": int(t+1),
            "bone_len_mae_m": bone_len_mae,
            "bone_len_rel_mae": bone_len_rel_mae,
            "bone_cos_mean": bone_cos_mean,
            "bone_cos_acc": bone_cos_acc,
            "knee_mae_deg": knee_mae_deg,
            "knee_acc": knee_acc,
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Model definition（zmodel4 と整合）
# ------------------------------------------------------------

class GaitLSTMSupport(nn.Module):
    """
    Training-side model compatible with zmodel4/5 checkpoints.

    Checkpoint key patterns (examples):
      - lstm.*
      - ln_pos.*, ln_sup.* (LayerNorm over hidden)
      - pos_head.0/2.*, sup_head.0/2.* (2-layer MLP heads)
    """
    def __init__(
        self,
        din: int,
        hidden: int,
        layers: int,
        dropout: float,
        horizon: int,
        dpos: int,
        support_classes: int = 2,
        residual: bool = True,
        use_layernorm: bool = True,
        head_mlp: bool = True,
    ):
        super().__init__()
        self.din = int(din)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.horizon = int(horizon)
        self.dpos = int(dpos)
        self.support_classes = int(support_classes)
        self.residual = bool(residual)
        self.use_layernorm = bool(use_layernorm)
        self.head_mlp = bool(head_mlp)

        self.lstm = nn.LSTM(
            input_size=self.din,
            hidden_size=self.hidden,
            num_layers=self.layers,
            dropout=self.dropout if self.layers > 1 else 0.0,
            batch_first=True,
        )

        # Match checkpoints: LayerNorm(hidden)
        if self.use_layernorm:
            self.ln_pos = nn.LayerNorm(self.hidden)
            self.ln_sup = nn.LayerNorm(self.hidden)

        # Match checkpoints: 2-layer MLP heads by default (pos_head / sup_head)
        out_pos = self.horizon * self.dpos
        out_sup = self.horizon * self.support_classes

        if self.head_mlp:
            self.pos_head = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, out_pos),
            )
            self.sup_head = nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, out_sup),
            )
        else:
            # Fallback (older checkpoints)
            self.fc_pos = nn.Linear(self.hidden, out_pos)
            self.fc_sup = nn.Linear(self.hidden, out_sup)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, Din)
        returns:
          pos_out: (B, H, Dpos)
          sup_logits: (B, H, C)
        """
        # last hidden output
        y, _ = self.lstm(x)               # (B, T, hidden)
        h = y[:, -1, :]                   # (B, hidden)

        # pos branch
        h_pos = self.ln_pos(h) if self.use_layernorm else h
        if self.head_mlp:
            pos = self.pos_head(h_pos)    # (B, H*Dpos)
        else:
            pos = self.fc_pos(h_pos)
        pos = pos.view(-1, self.horizon, self.dpos)

        # support branch
        h_sup = self.ln_sup(h) if self.use_layernorm else h
        if self.head_mlp:
            sup = self.sup_head(h_sup)    # (B, H*C)
        else:
            sup = self.fc_sup(h_sup)
        sup = sup.view(-1, self.horizon, self.support_classes)

        return pos, sup
def load_model(
    model_pt: Path,
    cfg: dict,
    cols: dict,
    device: torch.device,
) -> nn.Module:
    input_cols = cols["input_cols"]
    pos_cols = cols["pos_cols"]
    din = len(input_cols)
    dpos = len(pos_cols)
    hidden = int(cfg.get("hidden", 256))
    layers = int(cfg.get("layers", 2))
    dropout = float(cfg.get("dropout", 0.3))
    horizon = int(cfg.get("horizon", 30))
    support_classes = int(cfg.get("support_classes", 2))
    residual = bool(cfg.get("residual", True))

    try:
        sd = torch.load(str(model_pt), map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(str(model_pt), map_location=device)
    use_layernorm = ("ln_pos.weight" in sd) or ("ln_sup.weight" in sd)
    # Detect head style from checkpoint keys
    head_mlp = ("pos_head.0.weight" in sd) and ("sup_head.0.weight" in sd)

    model = GaitLSTMSupport(
        din=din, hidden=hidden, layers=layers, dropout=dropout,
        horizon=horizon, dpos=dpos, support_classes=support_classes,
        residual=residual, use_layernorm=use_layernorm, head_mlp=head_mlp,
    ).to(device)

    sd_f, skipped = filter_state_dict_by_shape(model, sd)
    if len(skipped) > 0:
        print(f"[Warn] skipped keys (shape mismatch or missing): {len(skipped)}")
    missing, unexpected = model.load_state_dict(sd_f, strict=False)
    if len(unexpected) > 0:
        print(f"[Warn] unexpected keys: {unexpected}")
    if len(missing) > 0:
        print(f"[Warn] missing keys: {missing}")
    model.eval()
    return model

# ------------------------------------------------------------
# Data windowing
# ------------------------------------------------------------

@dataclass
class EvalData:
    X: np.ndarray            # (N, seq_len, Din) scaled
    Ypos_m: np.ndarray       # (N, H, Dpos) meters (GT)
    Ysup: np.ndarray         # (N, H) int (GT support 0/1)
    meta: pd.DataFrame       # per-window meta (csv_path, video_id, start, etc.)
    scaler_on_cols: List[str]
    input_cols: List[str]
    pos_cols: List[str]
    contact_cols: List[str]
    flag_cols: List[str]

def iter_sequences(df: pd.DataFrame) -> List[pd.DataFrame]:
    if "video_id" in df.columns:
        seqs = []
        for _, g in df.groupby("video_id", sort=False):
            gg = g.copy()
            if "frame_idx" in gg.columns:
                gg = gg.sort_values("frame_idx")
            seqs.append(gg.reset_index(drop=True))
        return seqs
    gg = df.copy()
    if "frame_idx" in gg.columns:
        gg = gg.sort_values("frame_idx")
    return [gg.reset_index(drop=True)]

def ensure_columns(df: pd.DataFrame, need_cols: List[str]) -> pd.DataFrame:
    miss = [c for c in need_cols if c not in df.columns]
    if len(miss) == 0:
        return df
    # support を contact から補完
    if "support" in miss and ("contact_L" in df.columns and "contact_R" in df.columns):
        sup = build_support_from_contacts(df["contact_L"].to_numpy(), df["contact_R"].to_numpy())
        df = df.copy()
        df["support"] = sup.astype(np.int64)
        miss = [c for c in need_cols if c not in df.columns]
    if len(miss) > 0:
        raise ValueError(f"CSV is missing required columns: {miss}")
    return df
def parse_video_id(v):
    """
    video_id は数値とは限らない（例: 'data_riku_test0'）。
    - 数値に変換できる場合は int
    - それ以外は文字列のまま返す
    """
    if v is None:
        return -1
    # pandas may give numpy scalar
    try:
        vv = v.item()
    except Exception:
        vv = v
    s = str(vv)
    # allow simple integer strings
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    return s

def load_eval_data(
    csvs: List[Path],
    cfg: dict,
    cols: dict,
    scaler,
    support_min_run: int,
) -> EvalData:
    seq_len = int(cfg.get("seq_len", 50))
    horizon = int(cfg.get("horizon", 30))
    stride = int(cfg.get("stride", 1))

    input_cols = list(cols["input_cols"])
    pos_cols = list(cols["pos_cols"])
    contact_cols = list(cols.get("contact_cols", ["contact_L", "contact_R"]))
    flag_cols = list(cols.get("flag_cols", []))
    if "support" in input_cols and "support" not in flag_cols:
        flag_cols = flag_cols + ["support"]
    numeric_cols = list(cols.get("numeric_cols", [c for c in input_cols if c not in set(flag_cols)]))

    X_list = []
    Ypos_list = []
    Ysup_list = []
    meta_rows = []
    scaler_on_cols = None

    for csv_path in csvs:
        df = read_csv_smart(csv_path)
        df = ensure_columns(df, list(set(input_cols + pos_cols + contact_cols)))
        # support GT from contacts (2クラス丸め) -> future側に使う
        sup_gt = build_support_from_contacts(df[contact_cols[0]].to_numpy(), df[contact_cols[1]].to_numpy())
        if support_min_run > 1:
            sup_gt = smooth_min_run_labels(sup_gt, support_min_run)

        # raw inputs and pos in meters
        x_all = df[input_cols].to_numpy(dtype=np.float32)
        pos_all_m = df[pos_cols].to_numpy(dtype=np.float32)

        x_scaled, s_cols = scale_inputs(x_all, scaler, input_cols, flag_cols, numeric_cols=numeric_cols)
        if scaler_on_cols is None:
            scaler_on_cols = s_cols

        # sequences (video_id split)
        for seq_df in iter_sequences(df):
            idx0 = seq_df.index.to_numpy()
            T = len(idx0)
            if T < seq_len + horizon:
                continue

            # map back to arrays for this sequence
            x_seq = x_scaled[idx0, :]
            pos_seq = pos_all_m[idx0, :]
            sup_seq = sup_gt[idx0]

            for st in range(0, T - (seq_len + horizon) + 1, stride):
                x_win = x_seq[st:st + seq_len, :]
                y_pos = pos_seq[st + seq_len: st + seq_len + horizon, :]
                y_sup = sup_seq[st + seq_len: st + seq_len + horizon]

                X_list.append(x_win[None, ...])
                Ypos_list.append(y_pos[None, ...])
                Ysup_list.append(y_sup[None, ...])

                meta_rows.append({
                    "csv": str(csv_path),
                    "video_id": parse_video_id(seq_df["video_id"].iloc[0]) if "video_id" in seq_df.columns else -1,
                    "start": int(st),
                    "seq_len": int(seq_len),
                    "horizon": int(horizon),
                })

    if len(X_list) == 0:
        raise RuntimeError("No windows were constructed. Check seq_len/horizon and CSV lengths.")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Ypos_m = np.concatenate(Ypos_list, axis=0).astype(np.float32)
    Ysup = np.concatenate(Ysup_list, axis=0).astype(np.int64)
    meta = pd.DataFrame(meta_rows)

    return EvalData(
        X=X, Ypos_m=Ypos_m, Ysup=Ysup, meta=meta,
        scaler_on_cols=scaler_on_cols or [],
        input_cols=input_cols, pos_cols=pos_cols, contact_cols=contact_cols, flag_cols=flag_cols
    )

# ------------------------------------------------------------
# Inference + Metrics
# ------------------------------------------------------------

@dataclass
class PredOut:
    pred_pos_m: np.ndarray          # (N,H,Dpos)
    sup_logits: np.ndarray          # (N,H,C)
    pred_sup: np.ndarray            # (N,H)
    pred_sup_smooth: np.ndarray     # (N,H)

def run_infer(
    model: nn.Module,
    data: EvalData,
    cfg: dict,
    scaler,
    support_min_run: int,
    device: torch.device,
    batch_size: int = 256,
) -> PredOut:
    horizon = int(cfg.get("horizon", 30))
    pos_cols = data.pos_cols
    scaler_on_cols = data.scaler_on_cols

    N = data.X.shape[0]
    pred_pos_m = np.zeros((N, horizon, len(pos_cols)), dtype=np.float32)
    sup_logits = np.zeros((N, horizon, int(cfg.get("support_classes", 2))), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(data.X[i:i+batch_size]).to(device)
            pos_s, sup_l = model(xb)
            pos_s = pos_s.detach().cpu().numpy().astype(np.float32)
            sup_l = sup_l.detach().cpu().numpy().astype(np.float32)

            # inverse transform only for pos
            pos_m = inverse_pos_only(pos_s, scaler, scaler_on_cols, pos_cols)
            pred_pos_m[i:i+pos_m.shape[0]] = pos_m
            sup_logits[i:i+sup_l.shape[0]] = sup_l

    pred_sup = np.argmax(sup_logits, axis=-1).astype(np.int64)

    # smooth predicted support per-window
    if support_min_run > 1:
        pred_sup_smooth = np.zeros_like(pred_sup)
        for n in range(N):
            pred_sup_smooth[n] = smooth_min_run_labels(pred_sup[n], support_min_run)
    else:
        pred_sup_smooth = pred_sup.copy()

    return PredOut(pred_pos_m=pred_pos_m, sup_logits=sup_logits, pred_sup=pred_sup, pred_sup_smooth=pred_sup_smooth)

def summarize_metrics(
    gt_pos: np.ndarray,
    pr_pos: np.ndarray,
    gt_sup: np.ndarray,
    pr_sup: np.ndarray,
    pr_sup_smooth: np.ndarray,
    pos_cols: List[str],
    z_tol: float,
    bone_cos_thr: float,
    knee_deg_tol: float,
    event_win: int,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    returns:
      summary_dict
      per_horizon_scores_df
      per_horizon_support_df
      per_window_df（骨/膝/支持脚など）
    """
    N, H, Dpos = gt_pos.shape

    # per-horizon position metrics
    pos_mae, pos_rmse = per_horizon_mae_rmse(gt_pos, pr_pos)
    z_mae = per_horizon_z_mae(gt_pos, pr_pos)
    z_acc = per_horizon_z_acc(gt_pos, pr_pos, z_tol=float(z_tol))
    mpjpe = mpjpe_m(gt_pos, pr_pos, Dpos)

    bone_df = bone_and_knee_by_t(gt_pos, pr_pos, pos_cols, bone_cos_thr=float(bone_cos_thr), knee_deg_tol=float(knee_deg_tol))

    # support per-horizon
    sup_acc_t = []
    sup_f1_t = []
    for t in range(H):
        rep = cls_report_binary(gt_sup[:, t], pr_sup_smooth[:, t])
        sup_acc_t.append(rep["sup_acc"])
        sup_f1_t.append(rep["sup_f1_macro"])
    sup_acc_t = np.array(sup_acc_t, dtype=np.float32)
    sup_f1_t = np.array(sup_f1_t, dtype=np.float32)

    per_h_scores = pd.DataFrame({
        "t": np.arange(1, H+1, dtype=np.int64),
        "pos_mae_m": pos_mae,
        "pos_rmse_m": pos_rmse,
        "mpjpe_m": mpjpe,
        "z_mae_m": z_mae,
        "z_acc": z_acc,
    })
    per_h_scores = per_h_scores.merge(bone_df, on="t", how="left")

    per_h_sup = pd.DataFrame({
        "t": np.arange(1, H+1, dtype=np.int64),
        "sup_acc": sup_acc_t,
        "sup_f1_macro": sup_f1_t,
    })

    # overall summary: mean over horizons
    rep_all = cls_report_binary(gt_sup.reshape(-1), pr_sup_smooth.reshape(-1))
    sw = switch_metrics(gt_sup.reshape(-1), pr_sup_smooth.reshape(-1), event_win=int(event_win))

    summary = {
        "N_windows": int(N),
        "pos_mae_m": float(np.mean(pos_mae)),
        "pos_rmse_m": float(np.mean(pos_rmse)),
        "mpjpe_m": float(np.mean(mpjpe)),
        "z_mae_m": float(np.mean(z_mae)),
        "z_acc": float(np.mean(z_acc)),
        "bone_len_mae_m": float(per_h_scores["bone_len_mae_m"].mean()),
        "bone_len_rel_mae": float(per_h_scores["bone_len_rel_mae"].mean()),
        "bone_cos_mean": float(per_h_scores["bone_cos_mean"].mean()),
        "bone_cos_acc": float(per_h_scores["bone_cos_acc"].mean()),
        "knee_mae_deg": float(per_h_scores["knee_mae_deg"].mean()),
        "knee_acc": float(per_h_scores["knee_acc"].mean()),
        "sup_acc": float(rep_all["sup_acc"]),
        "sup_f1_macro": float(rep_all["sup_f1_macro"]),
        "switch_prec": float(sw["switch_prec"]),
        "switch_rec": float(sw["switch_rec"]),
        "switch_f1": float(sw["switch_f1"]),
        "switch_acc": float(sw["switch_acc"]),
    }

    # per-window metrics（分布を見る）
    # ここでは horizon平均の膝角誤差等を window単位で計算する
    jm = build_joint_map(pos_cols)
    pos_idx = precompute_pos_indices(pos_cols, jm)
    hipz_gt = hip_center_z(gt_pos, pos_cols, jm, pos_idx)
    hipz_pr = hip_center_z(pr_pos, pos_cols, jm, pos_idx)
    hipz_mae_t = np.mean(np.abs(hipz_pr - hipz_gt), axis=1)  # (N,)

    # knee mae per-window
    def knee_angle_seq(arr: np.ndarray, side: str) -> np.ndarray:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        if hip not in pos_idx or knee not in pos_idx or ankle not in pos_idx:
            return np.full((arr.shape[0], arr.shape[1]), np.nan, dtype=np.float32)
        hx, hy, hz = pos_idx[hip]; kx, ky, kz = pos_idx[knee]; ax, ay, az = pos_idx[ankle]
        v1 = arr[..., [hx,hy,hz]] - arr[..., [kx,ky,kz]]
        v2 = arr[..., [ax,ay,az]] - arr[..., [kx,ky,kz]]
        n1 = np.linalg.norm(v1, axis=-1) + 1e-8
        n2 = np.linalg.norm(v2, axis=-1) + 1e-8
        cos = np.sum(v1*v2, axis=-1) / (n1*n2)
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos)).astype(np.float32)

    gtL = knee_angle_seq(gt_pos, "L"); prL = knee_angle_seq(pr_pos, "L")
    gtR = knee_angle_seq(gt_pos, "R"); prR = knee_angle_seq(pr_pos, "R")
    knee_err = np.nanmean(np.stack([np.abs(prL-gtL), np.abs(prR-gtR)], axis=2), axis=2)  # (N,H)
    knee_mae_win = np.nanmean(knee_err, axis=1)

    # support acc per-window
    sup_acc_win = np.mean((gt_sup == pr_sup_smooth).astype(np.float32), axis=1)
    sw_win = []
    for n in range(N):
        sw_n = switch_metrics(gt_sup[n], pr_sup_smooth[n], event_win=int(event_win))
        sw_win.append(sw_n["switch_f1"])
    sw_win = np.array(sw_win, dtype=np.float32)

    per_win = pd.DataFrame({
        "hipz_mae_m": hipz_mae_t.astype(np.float32),
        "knee_mae_deg": knee_mae_win.astype(np.float32),
        "sup_acc": sup_acc_win.astype(np.float32),
        "switch_f1": sw_win.astype(np.float32),
    })

    return summary, per_h_scores, per_h_sup, per_win

# ------------------------------------------------------------
# Plotting utilities
# ------------------------------------------------------------

def plot_losses(train_log_csv: Path, out_png: Path) -> None:
    df = read_csv_smart(train_log_csv)
    # expected columns: epoch, train_total, val_total, train_pos, val_pos, train_support, val_support ...
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df)+1)
    plt.figure()
    for col in ["train_total", "val_total", "train_pos", "val_pos", "train_support", "val_support"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_per_horizon_scores(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure()
    if "pos_mae_m" in df.columns:
        plt.plot(df["t"], df["pos_mae_m"], label="pos_mae_m")
    if "z_mae_m" in df.columns:
        plt.plot(df["t"], df["z_mae_m"], label="z_mae_m")
    if "mpjpe_m" in df.columns:
        plt.plot(df["t"], df["mpjpe_m"], label="mpjpe_m")
    if "knee_mae_deg" in df.columns:
        plt.plot(df["t"], df["knee_mae_deg"], label="knee_mae_deg")
    if "bone_cos_mean" in df.columns:
        plt.plot(df["t"], df["bone_cos_mean"], label="bone_cos_mean")
    plt.xlabel("horizon t (frame)")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_per_horizon_support(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure()
    plt.plot(df["t"], df["sup_acc"], label="sup_acc")
    plt.plot(df["t"], df["sup_f1_macro"], label="sup_f1_macro")
    plt.xlabel("horizon t (frame)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

# ------------------------------------------------------------
# Shape (骨格構造) 分布評価: 骨方向cos類似度 / 骨長相対誤差 / 膝角度誤差
# ------------------------------------------------------------

def _safe_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2 norm with epsilon to avoid divide-by-zero."""
    return np.sqrt(np.sum(v * v, axis=axis) + eps)

def compute_shape_distributions(
    gt_pos: np.ndarray,
    pr_pos: np.ndarray,
    pos_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    gt_pos/pr_pos: (N, H, Dpos) in meters.
    returns:
      df_bone_cos: columns=[bone, t, cos]
      df_bone_lenrel: columns=[bone, t, len_rel_err]
      df_knee_err: columns=[knee, t, ang_err_deg]
    """
    assert gt_pos.shape == pr_pos.shape
    N, H, D = gt_pos.shape

    jm = build_joint_map(pos_cols)
    idx = precompute_pos_indices(pos_cols, jm)

    # bones to evaluate (names are for plots/tables)
    bones = [
        ("L_thigh", "L_HIP", "L_KNEE"),
        ("L_shank", "L_KNEE", "L_ANKLE"),
        ("R_thigh", "R_HIP", "R_KNEE"),
        ("R_shank", "R_KNEE", "R_ANKLE"),
        ("Hip_width", "L_HIP", "R_HIP"),
    ]

    rows_cos = []
    rows_len = []

    # per-horizon index
    t_idx = np.arange(1, H + 1, dtype=np.int64)

    for bname, a, b in bones:
        a_gt = gt_pos[..., idx[a]]  # (N,H,3)
        b_gt = gt_pos[..., idx[b]]
        a_pr = pr_pos[..., idx[a]]
        b_pr = pr_pos[..., idx[b]]

        v_gt = b_gt - a_gt
        v_pr = b_pr - a_pr

        # cosine similarity
        dot = np.sum(v_gt * v_pr, axis=-1)
        ng = _safe_norm(v_gt, axis=-1)
        np_ = _safe_norm(v_pr, axis=-1)
        cos = dot / (ng * np_ + 1e-9)
        cos = np.clip(cos, -1.0, 1.0)

        # bone length relative error
        lg = ng
        lp = np_
        len_rel = np.abs(lp - lg) / (lg + 1e-9)

        # flatten with t
        for ti in range(H):
            c = cos[:, ti]
            l = len_rel[:, ti]
            # keep as list of python floats (for pandas performance and compatibility)
            rows_cos.extend([(bname, int(t_idx[ti]), float(x)) for x in c])
            rows_len.extend([(bname, int(t_idx[ti]), float(x)) for x in l])

    df_bone_cos = pd.DataFrame(rows_cos, columns=["bone", "t", "cos"])
    df_bone_len = pd.DataFrame(rows_len, columns=["bone", "t", "len_rel_err"])

    # knee angle errors (degrees)
    # angle at knee: (hip-knee)-(ankle-knee)
    def knee_angle_deg(arr: np.ndarray, side: str) -> np.ndarray:
        hip = arr[..., idx[f"{side}_HIP"]]
        knee = arr[..., idx[f"{side}_KNEE"]]
        ank = arr[..., idx[f"{side}_ANKLE"]]
        u = hip - knee
        v = ank - knee
        dot = np.sum(u * v, axis=-1)
        nu = _safe_norm(u, axis=-1)
        nv = _safe_norm(v, axis=-1)
        c = np.clip(dot / (nu * nv + 1e-9), -1.0, 1.0)
        ang = np.degrees(np.arccos(c))
        return ang  # (N,H)

    kL_gt = knee_angle_deg(gt_pos, "L")
    kL_pr = knee_angle_deg(pr_pos, "L")
    kR_gt = knee_angle_deg(gt_pos, "R")
    kR_pr = knee_angle_deg(pr_pos, "R")

    errL = np.abs(kL_pr - kL_gt)
    errR = np.abs(kR_pr - kR_gt)

    rows_knee = []
    for ti in range(H):
        rows_knee.extend([("L_knee", int(t_idx[ti]), float(x)) for x in errL[:, ti]])
        rows_knee.extend([("R_knee", int(t_idx[ti]), float(x)) for x in errR[:, ti]])
    df_knee = pd.DataFrame(rows_knee, columns=["knee", "t", "ang_err_deg"])

    return df_bone_cos, df_bone_len, df_knee

def _plot_violin(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    out_png: Path,
    order: Optional[List[str]] = None,
    hline: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    groups = order if order is not None else list(pd.unique(df[group_col]))
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    vp = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.9)

    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=1.2, alpha=0.8)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_shape_bone_cos_violin(df_bone_cos: pd.DataFrame, out_png: Path, thr: float = 0.95) -> None:
    order = ["L_thigh", "L_shank", "R_thigh", "R_shank", "Hip_width"]
    _plot_violin(
        df=df_bone_cos,
        group_col="bone",
        value_col="cos",
        title=f"Bone direction cosine similarity (higher is better). Dashed line: {thr:.2f}",
        ylabel="cosine similarity",
        out_png=out_png,
        order=order,
        hline=thr,
        ylim=(-1.0, 1.0),
    )

def plot_shape_bone_lenrel_violin(df_bone_len: pd.DataFrame, out_png: Path, thr: float = 0.05) -> None:
    order = ["L_thigh", "L_shank", "R_thigh", "R_shank", "Hip_width"]
    _plot_violin(
        df=df_bone_len,
        group_col="bone",
        value_col="len_rel_err",
        title=f"Bone length relative error |L_pred-L_gt|/L_gt (lower is better). Dashed line: {thr:.2f}",
        ylabel="relative error",
        out_png=out_png,
        order=order,
        hline=thr,
        ylim=(0.0, max(0.15, float(df_bone_len["len_rel_err"].quantile(0.99)))),
    )

def plot_shape_knee_err_violin(df_knee: pd.DataFrame, out_png: Path, tol_deg: float = 10.0) -> None:
    order = ["L_knee", "R_knee"]
    _plot_violin(
        df=df_knee,
        group_col="knee",
        value_col="ang_err_deg",
        title=f"Knee angle absolute error (deg, lower is better). Dashed line: {tol_deg:.1f}°",
        ylabel="absolute error (deg)",
        out_png=out_png,
        order=order,
        hline=tol_deg,
        ylim=(0.0, max(30.0, float(df_knee["ang_err_deg"].quantile(0.99)))),
    )

def summarize_shape_tables(
    df_bone_cos: pd.DataFrame,
    df_bone_len: pd.DataFrame,
    df_knee: pd.DataFrame,
    bone_cos_thr: float,
    bone_lenrel_thr: float,
    knee_deg_tol: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-group summary tables for thesis."""
    bone_cos_sum = (
        df_bone_cos.groupby("bone")["cos"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    bone_cos_sum["acc_ge_thr"] = df_bone_cos.groupby("bone")["cos"].apply(lambda s: float((s >= bone_cos_thr).mean())).values

    bone_len_sum = (
        df_bone_len.groupby("bone")["len_rel_err"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    bone_len_sum["acc_le_thr"] = df_bone_len.groupby("bone")["len_rel_err"].apply(lambda s: float((s <= bone_lenrel_thr).mean())).values

    knee_sum = (
        df_knee.groupby("knee")["ang_err_deg"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    knee_sum["acc_le_tol"] = df_knee.groupby("knee")["ang_err_deg"].apply(lambda s: float((s <= knee_deg_tol).mean())).values

    return bone_cos_sum, bone_len_sum, knee_sum

def plot_per_horizon_shape(per_h_scores: pd.DataFrame, out_png: Path) -> None:
    """
    per_h_scores contains columns:
      t, bone_cos_mean, bone_cos_acc, bone_len_rel_mae, knee_mae_deg, knee_acc
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = per_h_scores["t"].values
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if "bone_cos_mean" in per_h_scores.columns:
        ax.plot(t, per_h_scores["bone_cos_mean"].values, label="bone_cos_mean (↑)")
    if "bone_len_rel_mae" in per_h_scores.columns:
        ax.plot(t, per_h_scores["bone_len_rel_mae"].values, label="bone_len_rel_mae (↓)")
    if "knee_mae_deg" in per_h_scores.columns:
        ax.plot(t, per_h_scores["knee_mae_deg"].values, label="knee_mae_deg (↓)")
    ax.set_xlabel("horizon step t (1..H)")
    ax.set_title("Shape metrics per horizon (mean across windows)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_sample_ts(
    gt_pos: np.ndarray, pr_pos: np.ndarray,
    gt_sup: np.ndarray, pr_sup: np.ndarray,
    pos_cols: List[str],
    out_png: Path,
) -> None:
    # use hipZ and knee angles
    jm = build_joint_map(pos_cols)
    pos_idx = precompute_pos_indices(pos_cols, jm)
    hipz_gt = hip_center_z(gt_pos[None, ...], pos_cols, jm, pos_idx)[0]
    hipz_pr = hip_center_z(pr_pos[None, ...], pos_cols, jm, pos_idx)[0]

    # knee angle
    def knee_angle(arr: np.ndarray, side: str) -> np.ndarray:
        hip = f"{side}_HIP"; knee = f"{side}_KNEE"; ankle = f"{side}_ANKLE"
        if hip not in pos_idx or knee not in pos_idx or ankle not in pos_idx:
            return np.full((arr.shape[0],), np.nan, dtype=np.float32)
        hx, hy, hz = pos_idx[hip]; kx, ky, kz = pos_idx[knee]; ax, ay, az = pos_idx[ankle]
        v1 = arr[:, [hx,hy,hz]] - arr[:, [kx,ky,kz]]
        v2 = arr[:, [ax,ay,az]] - arr[:, [kx,ky,kz]]
        n1 = np.linalg.norm(v1, axis=-1) + 1e-8
        n2 = np.linalg.norm(v2, axis=-1) + 1e-8
        cos = np.sum(v1*v2, axis=-1) / (n1*n2)
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos)).astype(np.float32)

    kneeL_gt = knee_angle(gt_pos, "L")
    kneeL_pr = knee_angle(pr_pos, "L")
    kneeR_gt = knee_angle(gt_pos, "R")
    kneeR_pr = knee_angle(pr_pos, "R")

    t = np.arange(1, gt_pos.shape[0]+1)
    plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, hipz_gt, label="hipZ_gt")
    ax1.plot(t, hipz_pr, label="hipZ_pred")
    ax1.set_ylabel("hipZ (m)")
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t, kneeL_gt, label="kneeL_gt")
    ax2.plot(t, kneeL_pr, label="kneeL_pred")
    ax2.plot(t, kneeR_gt, label="kneeR_gt")
    ax2.plot(t, kneeR_pr, label="kneeR_pred")
    ax2.set_ylabel("knee angle (deg)")
    ax2.legend(ncol=2)

    ax3 = plt.subplot(3, 1, 3)
    ax3.step(t, gt_sup, where="mid", label="sup_gt")
    ax3.step(t, pr_sup, where="mid", label="sup_pred_smooth")
    ax3.set_ylabel("support")
    ax3.set_xlabel("future t (frame)")
    ax3.set_yticks([0, 1])
    ax3.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

# ------------------------------------------------------------
# GIF rendering（simple skeleton）
# ------------------------------------------------------------

def render_skeleton_frame(ax, pts: np.ndarray, pos_cols: List[str], view: str, title: str, alpha: float, color: str):
    jm = build_joint_map(pos_cols)
    idx = precompute_pos_indices(pos_cols, jm)

    def p(j):
        ix, iy, iz = idx[j]
        return np.array([pts[ix], pts[iy], pts[iz]], dtype=np.float32)

    # axis mapping by view
    # yz: (z,y) or (y,z)? user uses yz view (y vertical, z horizontal)
    if view == "yz":
        def proj(v): return np.array([v[2], v[1]], dtype=np.float32)  # (z,y)
        xlab, ylab = "z (m)", "y (m)"
    elif view == "xz":
        def proj(v): return np.array([v[0], v[2]], dtype=np.float32)  # (x,z)
        xlab, ylab = "x (m)", "z (m)"
    else:  # "xy"
        def proj(v): return np.array([v[0], v[1]], dtype=np.float32)
        xlab, ylab = "x (m)", "y (m)"

    bones = []
    for a, b in [("L_HIP","L_KNEE"), ("L_KNEE","L_ANKLE"), ("R_HIP","R_KNEE"), ("R_KNEE","R_ANKLE"), ("L_HIP","R_HIP")]:
        if a in idx and b in idx:
            bones.append((a,b))
    if len(bones) == 0:
        joints = list(idx.keys())
        for i in range(min(4, len(joints)-1)):
            bones.append((joints[i], joints[i+1]))

    for a, b in bones:
        va = proj(p(a)); vb = proj(p(b))
        ax.plot([va[0], vb[0]], [va[1], vb[1]], alpha=alpha, color=color, linewidth=2)

    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title)

def save_gif_sample(
    gt_pos: np.ndarray, pr_pos: np.ndarray, pos_cols: List[str],
    out_gif: Optional[Path], view: str, fps: int = 5,
    out_frames_dir: Optional[Path] = None, frame_ext: str = "png", frame_prefix: str = "frame", zero_pad: int = 4
) -> None:
    import imageio.v2 as imageio
    frames = []

    # Save per-frame images if requested
    if out_frames_dir is not None:
        out_frames_dir = Path(out_frames_dir)
        out_frames_dir.mkdir(parents=True, exist_ok=True)
        ext = frame_ext.lower().lstrip('.').strip()
        if ext not in ("png", "jpg", "jpeg"): 
            raise ValueError(f"Unsupported frame_ext: {frame_ext} (use png/jpg)")
        frame_ext = ext

    H = gt_pos.shape[0]
    # fixed limits
    # compute ranges from gt and pr
    pts_all = np.concatenate([gt_pos.reshape(H, -1), pr_pos.reshape(H, -1)], axis=0)
    # for view mapping, compute projected min/max
    jm = build_joint_map(pos_cols)
    idx = precompute_pos_indices(pos_cols, jm)

    def proj_all(arr):
        # arr (H,Dpos)
        if view == "yz":
            xs = arr[:, 2::3]; ys = arr[:, 1::3]
        elif view == "xz":
            xs = arr[:, 0::3]; ys = arr[:, 2::3]
        else:
            xs = arr[:, 0::3]; ys = arr[:, 1::3]
        return xs, ys

    xs, ys = proj_all(np.concatenate([gt_pos, pr_pos], axis=0))
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    pad_x = (xmax - xmin) * 0.1 + 1e-6
    pad_y = (ymax - ymin) * 0.1 + 1e-6

    for t in range(H):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        render_skeleton_frame(ax, gt_pos[t], pos_cols, view=view, title=f"t={t+1} GT", alpha=0.8, color="C0")
        render_skeleton_frame(ax, pr_pos[t], pos_cols, view=view, title=f"t={t+1} GT/Pred", alpha=0.8, color="C1")
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        fig.canvas.draw()
        canvas = fig.canvas
        # Matplotlibのバージョン差を吸収してRGB画像を取得
        img = None
        # Matplotlibのバージョン/バックエンド差を吸収してRGB画像を取得（例外は握りつぶして順に試す）
        # 1) buffer_rgba (推奨; 多くの環境で利用可)
        try:
            buf = np.asarray(canvas.buffer_rgba())  # (h,w,4) RGBA
            img = buf[..., :3].copy()
        except Exception:
            img = None
        # 2) tostring_rgb (古いMatplotlibで利用可)
        if img is None:
            try:
                w, h = canvas.get_width_height()
                rgb = canvas.tostring_rgb()
                img = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
            except Exception:
                img = None
        # 3) tostring_argb (Matplotlib 3.8+などで残りやすい)
        if img is None:
            try:
                w, h = canvas.get_width_height()
                argb = canvas.tostring_argb()
                argb = np.frombuffer(argb, dtype=np.uint8).reshape(h, w, 4)
                img = argb[..., [1, 2, 3]].copy()  # ARGB -> RGB
            except Exception as e:
                raise RuntimeError("Canvas does not support RGB extraction. Tried buffer_rgba/tostring_rgb/tostring_argb.") from e
        frames.append(img)

        # write each frame as an image (optional)
        if out_frames_dir is not None:
            fn = f"{frame_prefix}_{t+1:0{zero_pad}d}.{frame_ext}"
            imageio.imwrite(out_frames_dir / fn, img)

        plt.close(fig)

    if out_gif is not None:
        out_gif = Path(out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_gif, frames, fps=fps)

# ------------------------------------------------------------
# Split file selection
# ------------------------------------------------------------

def load_split_csvs(out_dir: Path, which: str) -> List[Path]:
    split_path = out_dir / "train_val_test_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"train_val_test_split.json not found in {out_dir}")
    split = load_json(split_path)

    key = f"{which}_csvs"
    if key in split:
        csvs = split[key]
    else:
        # legacy
        key2 = f"{which}_csv"
        if key2 in split:
            csvs = split[key2]
        else:
            raise KeyError(f"split json has no '{key}' or '{key2}'. keys={list(split.keys())}")

    if isinstance(csvs, str):
        csvs = [csvs]
    csv_paths = [Path(p) for p in csvs]
    return csv_paths

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="学習出力ディレクトリ")
    ap.add_argument("--which", type=str, default="test", choices=["train", "val", "test"], help="評価対象 split")
    ap.add_argument("--run_tag", type=str, default="run", help="出力名の識別子")
    ap.add_argument("--coord", type=str, default="both", choices=["abs", "root", "both"], help="ABS/ROOT の評価モード")
    ap.add_argument("--plot_losses", action="store_true")
    ap.add_argument("--plot_sample_ts", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--view", type=str, default="yz", choices=["yz", "xz", "xy"])
    ap.add_argument("--gif_fps", type=int, default=5)
    ap.add_argument("--save_frames", action="store_true", help="GIFの各フレーム画像を保存する（--gifなしでも可）")
    ap.add_argument("--frames_dir", type=str, default="", help="フレーム画像の出力先ディレクトリ（空なら out_eval_dir に自動生成）")
    ap.add_argument("--frame_ext", type=str, default="png", help="フレーム画像の拡張子（png/jpg）")
    ap.add_argument("--frame_prefix", type=str, default="frame", help="フレーム画像ファイル名の接頭辞")
    ap.add_argument("--frame_zeropad", type=int, default=4, help="フレーム番号の0埋め桁数")

    ap.add_argument("--sample_idx", type=int, default=0, help="可視化する window index")
    ap.add_argument("--support_min_run", type=int, default=3, help="support smoothing 用の最小連続長")
    ap.add_argument("--event_win", type=int, default=2, help="switch の許容誤差（フレーム）")
    ap.add_argument("--z_tol", type=float, default=0.05, help="Z acc の許容誤差（m）")
    ap.add_argument("--bone_cos_thr", type=float, default=0.95, help="bone_cos_acc の閾値")
    ap.add_argument("--bone_lenrel_thr", type=float, default=0.05, help="bone length relative error threshold for accuracy")
    ap.add_argument("--knee_deg_tol", type=float, default=10.0, help="knee_acc の許容角度（deg）")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cfg = load_json(out_dir / "config.json")
    cols = load_json(out_dir / "columns.json")

    # use config values for seq_len/horizon
    cfg = cfg.copy()
    if "seq_len" not in cfg:
        cfg["seq_len"] = 50
    if "horizon" not in cfg:
        cfg["horizon"] = int(cfg.get("H", 30))

    # decide csvs
    csvs = load_split_csvs(out_dir, args.which)

    print(f"[Diag] out_dir={out_dir}")
    print(f"[Diag] which={args.which} csvs={len(csvs)}")
    for p in csvs[:10]:
        print(f"  - {p}")

    # scaler
    scaler_path = out_dir / "scaler.pkl"
    scaler = load_scaler_robust(scaler_path)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model_pt = out_dir / "zmodel_lstm.pt"
    if not model_pt.exists():
        # fallback
        cand = list(out_dir.glob("*.pt"))
        if len(cand) > 0:
            model_pt = cand[0]
        else:
            raise FileNotFoundError(f"model pt not found in {out_dir}")
    model = load_model(model_pt, cfg, cols, device)

    # load windows
    data = load_eval_data(csvs, cfg, cols, scaler, support_min_run=int(args.support_min_run))

    # infer
    pred = run_infer(
        model=model, data=data, cfg=cfg, scaler=scaler,
        support_min_run=int(args.support_min_run), device=device,
        batch_size=int(args.batch_size),
    )

    # choose ABS/ROOT/both
    coord_modes = ["abs", "root"] if args.coord == "both" else [args.coord]
    for mode in coord_modes:
        suffix = "_ABS" if mode == "abs" else "_ROOT"
        out_eval_dir = out_dir / "eval_thesis" / args.which / f"{args.run_tag}{suffix}"
        out_eval_dir.mkdir(parents=True, exist_ok=True)

        # coordinate transform for evaluation
        gt_pos = data.Ypos_m
        pr_pos = pred.pred_pos_m
        jm = build_joint_map(data.pos_cols)
        pos_idx = precompute_pos_indices(data.pos_cols, jm)
        if mode == "root":
            gt_pos_e = root_xz(gt_pos, pos_idx)
            pr_pos_e = root_xz(pr_pos, pos_idx)
        else:
            gt_pos_e = gt_pos
            pr_pos_e = pr_pos

        # metrics
        summary, per_h_scores, per_h_sup, per_win = summarize_metrics(
            gt_pos=gt_pos_e,
            pr_pos=pr_pos_e,
            gt_sup=data.Ysup,
            pr_sup=pred.pred_sup,
            pr_sup_smooth=pred.pred_sup_smooth,
            pos_cols=data.pos_cols,
            z_tol=float(args.z_tol),
            bone_cos_thr=float(args.bone_cos_thr),
            knee_deg_tol=float(args.knee_deg_tol),
            event_win=int(args.event_win),
        )
        summary["which"] = args.which
        summary["coord"] = mode
        summary["seq_len"] = int(cfg.get("seq_len", 50))
        summary["horizon"] = int(cfg.get("horizon", 30))
        summary["support_min_run"] = int(args.support_min_run)
        summary["event_win"] = int(args.event_win)
        summary["z_tol"] = float(args.z_tol)

        # save
        sum_csv = out_eval_dir / "eval_summary.csv"
        pd.DataFrame([summary]).to_csv(sum_csv, index=False, encoding="utf-8-sig")
        save_json(out_eval_dir / "eval_summary.json", summary)
        per_h_scores.to_csv(out_eval_dir / "per_horizon_scores.csv", index=False, encoding="utf-8-sig")
        per_h_sup.to_csv(out_eval_dir / "per_horizon_support.csv", index=False, encoding="utf-8-sig")

        # per-window with meta
        per_win_full = pd.concat([data.meta.reset_index(drop=True), per_win.reset_index(drop=True)], axis=1)
        per_win_full.to_csv(out_eval_dir / "eval_per_window.csv", index=False, encoding="utf-8-sig")

        # plots
        plot_per_horizon_scores(per_h_scores, out_eval_dir / "per_horizon_scores.png")
        plot_per_horizon_support(per_h_sup, out_eval_dir / "per_horizon_support.png")

        # --- shape (骨格構造) tables & figures for thesis ---
        # per-horizon shape curves (already contained in per_h_scores)
        shape_cols = [c for c in ["t", "bone_cos_mean", "bone_cos_acc", "bone_len_rel_mae", "knee_mae_deg", "knee_acc"] if c in per_h_scores.columns]
        if len(shape_cols) >= 2:
            per_h_scores.loc[:, shape_cols].to_csv(out_eval_dir / "per_horizon_shape.csv", index=False, encoding="utf-8-sig")
            plot_per_horizon_shape(per_h_scores.loc[:, shape_cols], out_eval_dir / "per_horizon_shape.png")

        # distributions (violin plots) across all windows & horizon steps
        df_bone_cos, df_bone_len, df_knee = compute_shape_distributions(gt_pos_e, pr_pos_e, data.pos_cols)
        df_bone_cos.to_csv(out_eval_dir / "shape_bone_cos_values.csv", index=False, encoding="utf-8-sig")
        df_bone_len.to_csv(out_eval_dir / "shape_bone_lenrel_values.csv", index=False, encoding="utf-8-sig")
        df_knee.to_csv(out_eval_dir / "shape_knee_err_values.csv", index=False, encoding="utf-8-sig")

        plot_shape_bone_cos_violin(df_bone_cos, out_eval_dir / "shape_bone_cos_violin.png", thr=float(args.bone_cos_thr))
        plot_shape_bone_lenrel_violin(df_bone_len, out_eval_dir / "shape_bone_lenrel_violin.png", thr=float(args.bone_lenrel_thr))
        plot_shape_knee_err_violin(df_knee, out_eval_dir / "shape_knee_err_violin.png", tol_deg=float(args.knee_deg_tol))

        bone_cos_sum, bone_len_sum, knee_sum = summarize_shape_tables(
            df_bone_cos, df_bone_len, df_knee,
            bone_cos_thr=float(args.bone_cos_thr),
            bone_lenrel_thr=float(args.bone_lenrel_thr),
            knee_deg_tol=float(args.knee_deg_tol),
        )
        bone_cos_sum.to_csv(out_eval_dir / "shape_bone_cos_summary.csv", index=False, encoding="utf-8-sig")
        bone_len_sum.to_csv(out_eval_dir / "shape_bone_lenrel_summary.csv", index=False, encoding="utf-8-sig")
        knee_sum.to_csv(out_eval_dir / "shape_knee_err_summary.csv", index=False, encoding="utf-8-sig")


        # sample plots/gif
        si = int(args.sample_idx)
        si = max(0, min(si, gt_pos.shape[0]-1))
        if args.plot_sample_ts:
            plot_sample_ts(
                gt_pos=gt_pos_e[si], pr_pos=pr_pos_e[si],
                gt_sup=data.Ysup[si], pr_sup=pred.pred_sup_smooth[si],
                pos_cols=data.pos_cols,
                out_png=out_eval_dir / f"sample_{si:03d}_ts.png",
            )
        if args.gif or args.save_frames:
            # frames output directory
            if str(args.frames_dir).strip() != "":
                frames_dir = Path(args.frames_dir)
            else:
                frames_dir = out_eval_dir / f"sample_{si:03d}_{args.view}_frames"

            out_gif = (out_eval_dir / f"sample_{si:03d}_{args.view}.gif") if args.gif else None

            save_gif_sample(
                gt_pos=gt_pos_e[si], pr_pos=pr_pos_e[si],
                pos_cols=data.pos_cols,
                out_gif=out_gif,
                view=args.view,
                fps=int(args.gif_fps),
                out_frames_dir=frames_dir if args.save_frames else None,
                frame_ext=str(args.frame_ext),
                frame_prefix=str(args.frame_prefix),
                zero_pad=int(args.frame_zeropad),
            )

        # losses plot
        if args.plot_losses:
            log_csv = out_dir / "train_log.csv"
            if log_csv.exists():
                plot_losses(log_csv, out_eval_dir / "losses.png")

        # run info
        run_info = {
            "out_dir": str(out_dir),
            "which": args.which,
            "run_tag": args.run_tag,
            "coord": mode,
            "csvs": [str(p) for p in csvs],
            "args": vars(args),
            "cfg": cfg,
            "columns": {
                "pos_cols": data.pos_cols,
                "input_cols": data.input_cols,
                "contact_cols": data.contact_cols,
                "flag_cols": data.flag_cols,
                "scaler_on_cols": data.scaler_on_cols,
            },
        }
        save_json(out_eval_dir / "run_info.json", run_info)

        print(f"[OK] saved: {sum_csv}")

if __name__ == "__main__":
    main()
