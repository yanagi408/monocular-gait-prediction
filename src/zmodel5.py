#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zmodel4_thesis.py

目的
- *_processed.csv（zcap後のprocessed）群から、LSTMで
  1) 未来horizonのpos（6点×xyz=18成分）を回帰（scaled空間で学習）
  2) 未来horizonの「支持脚（L/R）」を分類（2クラス）し、支持脚の切替点から接地イベントを抽出できるようにする
を同時に行うマルチタスク学習を行う。

本ファイルは zmodel4_*（multi-val対応）をベースに、contact(BCE)→支持脚(CE)へ置換した版です。

重要仕様
- scaler(StandardScaler) は TRAIN の numeric(pos+vel) のみでfit（contactは除外）
- posは scaler で正規化した空間（scaled）で学習・予測
- 支持脚ラベルは contact_L/contact_R（0/1）から内部で生成する
  - 1フレームで必ず片足のみ接地（either=100%）が理想だが、両方/両方0が混じる場合も想定し補完する
- チラつき抑制:
  - 支持脚系列に対して「短すぎる区間を潰す」run-lengthベースの平滑化を実装（--support_smooth_min_run）
  - 評価時の切替イベント抽出も同じ平滑化を通す（誤検出を抑える）

入出力
- out_dir に以下を保存
  - zmodel_lstm.pt, scaler.pkl, columns.json, config.json, train_log.csv
  - per_horizon_val.csv, （任意）per_horizon_test.csv
  - split: train_val_test_split.json

分割（multi-val/test）
- --val_csv と --test_csv は以下を受け付ける（カンマ区切りで複数指定可）
  1) 単一CSVパス
  2) glob（例: outputs_e/val/*.csv）
  3) 上記の混在（例: "a.csv,b*.csv,dir/*.csv"）
- --reuse_split を付けると、out_dir/train_val_test_split.json を再利用する

注意
- 本スクリプトは「支持脚（L/R）」を学習します。接地（HS/TO）そのものは、
  推論後に支持脚が切り替わる瞬間として抽出してください（eval側に簡易実装あり）。

"""

import os
import json
import glob
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# -------------------------
# 乱数固定（再現性）
# -------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    再現性のためのseed固定。
    deterministic=True は CUDA 環境で速度低下やエラーになる場合があるため任意。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# -------------------------
# Scaler（sklearnが無い環境のフォールバック込み）
# -------------------------
class SimpleStandardScaler:
    """
    sklearn.preprocessing.StandardScaler の最小互換
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_seen_: int = 0

    def partial_fit(self, X: np.ndarray) -> "SimpleStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            var = X.var(axis=0)
            self.scale_ = np.sqrt(np.maximum(var, 1e-12))
            self.n_seen_ = X.shape[0]
        else:
            n0 = self.n_seen_
            n1 = X.shape[0]
            m0 = self.mean_
            s0 = self.scale_
            m1 = X.mean(axis=0)
            v1 = X.var(axis=0)

            v0 = s0 * s0
            n = n0 + n1
            delta = m1 - m0
            m = m0 + delta * (n1 / max(n, 1))
            v = (n0 * v0 + n1 * v1 + (delta * delta) * (n0 * n1 / max(n, 1))) / max(n, 1)

            self.mean_ = m
            self.scale_ = np.sqrt(np.maximum(v, 1e-12))
            self.n_seen_ = n

        return self

    def fit(self, X: np.ndarray) -> "SimpleStandardScaler":
        self.mean_ = None
        self.scale_ = None
        self.n_seen_ = 0
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_.astype(np.float32)) / self.scale_.astype(np.float32)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        X = np.asarray(X, dtype=np.float32)
        return X * self.scale_.astype(np.float32) + self.mean_.astype(np.float32)


def make_scaler():
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    except Exception:
        return SimpleStandardScaler()


def save_scaler(scaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(scaler, str(path))
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(scaler, f)


# -------------------------
# 列推定（columns.json が無い場合）
# -------------------------
DEFAULT_VEL_COLS = ["ankle_speed_L", "ankle_speed_R", "ankle_dy_L", "ankle_dy_R"]
DEFAULT_CONTACT_COLS = ["contact_L", "contact_R"]

ORDER = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]
AXES = ["x", "y", "z"]


def infer_pos_cols(df_cols: List[str]) -> List[str]:
    cols_set = set(df_cols)

    # Support either *_ANKLE_* or *_FOOT_* naming for the distal joint.
    order = ORDER
    if ("L_ANKLE_x_m" not in cols_set) and ("L_FOOT_x_m" in cols_set):
        order = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_FOOT", "R_FOOT"]

    pos_cols: List[str] = []
    for j in order:
        for a in AXES:
            c = f"{j}_{a}_m"
            if c in cols_set:
                pos_cols.append(c)

    if len(pos_cols) == len(order) * len(AXES):
        return pos_cols

    # fallback: endswith _x_m/_y_m/_z_m
    pos_cols2 = []
    for c in df_cols:
        cl = c.lower()
        if cl.endswith("_x_m") or cl.endswith("_y_m") or cl.endswith("_z_m"):
            pos_cols2.append(c)
    return pos_cols2


def build_columns(out_dir: Path, sample_csv: Path, use_vel: bool = True, use_support_input: bool = False) -> Dict[str, List[str]]:
    """
    columns.json があればそれを使用。
    無ければ sample_csv の列から推定し columns.json を新規作成。
    """
    cols_path = out_dir / "columns.json"
    if cols_path.exists():
        with open(cols_path, "r", encoding="utf-8") as f:
            cols = json.load(f)
        for k in ["pos_cols", "input_cols"]:
            if k not in cols or not isinstance(cols[k], list) or len(cols[k]) == 0:
                raise ValueError(f"columns.json is invalid: missing {k}")
        cols.setdefault("vel_cols", [])
        cols.setdefault("contact_cols", [])
        return cols

    df = pd.read_csv(sample_csv, nrows=5)
    df_cols = list(df.columns)

    pos_cols = infer_pos_cols(df_cols)
    vel_cols = [c for c in DEFAULT_VEL_COLS if c in df_cols] if use_vel else []
    contact_cols = [c for c in DEFAULT_CONTACT_COLS if c in df_cols]
    support_col = 'support' if 'support' in df_cols else ''
    numeric_cols = pos_cols + vel_cols  # スケーリング対象（連続値）
    flag_cols = []
    if use_support_input and support_col:
        flag_cols = [support_col]

    cols = {
        "pos_cols": pos_cols,
        "vel_cols": vel_cols,
        "contact_cols": contact_cols,
        "support_col": support_col,
        "numeric_cols": numeric_cols,
        "flag_cols": flag_cols,
        "input_cols": pos_cols + vel_cols + flag_cols,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(cols_path, "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)

    return cols


# -------------------------
# window生成（video_id対応）
# -------------------------
def make_windows(n: int, seq_len: int, horizon: int, stride: int) -> List[int]:
    max_start = n - (seq_len + horizon) + 1
    if max_start <= 0:
        return []
    return list(range(0, max_start, stride))


def iter_sequences(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    video_id列があれば動画単位に分割してwindow生成。
    """
    if "video_id" in df.columns:
        return [g for _, g in df.groupby("video_id")]
    return [df]


def sort_sequence(df: pd.DataFrame) -> pd.DataFrame:
    if "frame" in df.columns:
        return df.sort_values("frame").reset_index(drop=True)
    return df.reset_index(drop=True)


# -------------------------
# posだけ scaler を使って正規化/復元
# -------------------------
def get_pos_indices(input_cols: List[str], pos_cols: List[str]) -> List[int]:
    idx = []
    m = {c: i for i, c in enumerate(input_cols)}
    for c in pos_cols:
        if c not in m:
            raise ValueError(f"pos col {c} not in input_cols")
        idx.append(m[c])
    return idx


def scale_pos_only(pos_m: np.ndarray, scaler, pos_idx: List[int]) -> np.ndarray:
    """
    scalerでfitしたDin次元のうち、pos_idxに対応する成分だけをスケーリングする。
    """
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float32)[pos_idx]
        scale = np.asarray(scaler.scale_, dtype=np.float32)[pos_idx]
        return (pos_m - mean) / scale

    # fallback: ダミーDinを作ってtransform
    Din = len(getattr(scaler, "mean_", []))
    if Din <= 0:
        Din = max(pos_idx) + 1
    dummy = np.zeros((pos_m.shape[0], Din), dtype=np.float32)
    dummy[:, pos_idx] = pos_m
    tr = scaler.transform(dummy)
    return tr[:, pos_idx]


def inverse_pos_only(pos_scaled: np.ndarray, scaler, pos_idx: List[int]) -> np.ndarray:
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.asarray(scaler.mean_, dtype=np.float32)[pos_idx]
        scale = np.asarray(scaler.scale_, dtype=np.float32)[pos_idx]
        return pos_scaled * scale + mean

    Din = len(getattr(scaler, "mean_", []))
    if Din <= 0:
        Din = max(pos_idx) + 1
    dummy = np.zeros((pos_scaled.shape[0], Din), dtype=np.float32)
    dummy[:, pos_idx] = pos_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, pos_idx]


def transform_numeric_and_assemble(
    df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    scaler,
) -> np.ndarray:
    """input_cols順に (N, Din) を作る。numeric_colsはscalerで標準化し、flagはそのまま。"""
    if scaler is None:
        return df[input_cols].to_numpy(dtype=np.float32)

    Xnum = df[numeric_cols].to_numpy(dtype=np.float32)
    Xnum_s = scaler.transform(Xnum)

    idx_num = {c: i for i, c in enumerate(numeric_cols)}
    out = np.zeros((len(df), len(input_cols)), dtype=np.float32)
    for j, c in enumerate(input_cols):
        if c in idx_num:
            out[:, j] = Xnum_s[:, idx_num[c]]
        else:
            out[:, j] = df[c].to_numpy(dtype=np.float32)
    return out


def scale_pos_by_numeric_scaler(
    pos_m: np.ndarray,
    scaler,
    pos_cols: List[str],
    numeric_cols: List[str],
) -> np.ndarray:
    """pos_m(N,Dpos) を numeric_cols にfitしたscalerで標準化して返す。"""
    if scaler is None:
        return pos_m.astype(np.float32)
    idx_num = {c: i for i, c in enumerate(numeric_cols)}
    pos_idx = [idx_num[c] for c in pos_cols]
    dummy = np.zeros((pos_m.shape[0], len(numeric_cols)), dtype=np.float32)
    dummy[:, pos_idx] = pos_m.astype(np.float32)
    tr = scaler.transform(dummy)
    return tr[:, pos_idx].astype(np.float32)


# -------------------------
# 支持脚ラベル生成 + チラつき抑制
# -------------------------
def _forward_fill_unknown(x: np.ndarray) -> np.ndarray:
    """
    x: (T,) with values -1/0/1
    -1 を直前値で埋める（先頭が-1の場合は後方から埋める）
    """
    x = x.copy()
    # forward fill
    last = -1
    for i in range(len(x)):
        if x[i] != -1:
            last = int(x[i])
        elif last != -1:
            x[i] = last
    # backward fill for leading -1
    if len(x) > 0 and x[0] == -1:
        last = -1
        for i in range(len(x) - 1, -1, -1):
            if x[i] != -1:
                last = int(x[i])
            elif last != -1:
                x[i] = last
    # if still -1 (all unknown), set to 0
    x[x == -1] = 0
    return x.astype(np.int64)


def _merge_short_runs(y: np.ndarray, min_run: int) -> np.ndarray:
    """
    y: (T,) int {0,1}
    min_run以下の短区間を潰して、前後どちらかに吸収する。
    - 前後両方ある場合は「前の区間」を優先（時系列の安定性優先）。
    """
    if min_run <= 1 or len(y) == 0:
        return y

    y = y.copy().astype(np.int64)

    # run-length encode
    runs = []
    s = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[s]:
            runs.append((s, i, int(y[s])))  # [start,end), value
            s = i

    # iterative merging (1回で十分なことが多いが、安全のため複数回)
    changed = True
    it = 0
    while changed and it < 5:
        it += 1
        changed = False
        # rebuild runs each pass
        runs = []
        s = 0
        for i in range(1, len(y) + 1):
            if i == len(y) or y[i] != y[s]:
                runs.append((s, i, int(y[s])))
                s = i

        for idx, (a, b, v) in enumerate(runs):
            L = b - a
            if L >= min_run:
                continue
            # decide replacement label
            left_exists = idx - 1 >= 0
            right_exists = idx + 1 < len(runs)
            if left_exists:
                newv = runs[idx - 1][2]
            elif right_exists:
                newv = runs[idx + 1][2]
            else:
                newv = v
            if newv != v:
                y[a:b] = newv
                changed = True

    return y




def smooth_min_run_labels(y: np.ndarray, min_run: int) -> np.ndarray:
    """
    短いラベル区間（run）を潰して時系列ラベルを安定化する簡易平滑化。

    - y: (T,) の 0/1 ラベル（-1 を含む場合は unknown とみなし前後で補完）
    - min_run: この長さ未満の連続区間を前後の区間に吸収する（<=1 ならそのまま）

    目的（論文で説明できる形）:
    - support は接地判定や可視化に使う「歩行状態量」で、フレーム単位のノイズがあると
      switch（切替）評価が不安定になる。そこで教師ラベル（またはGT列）の短い揺れを抑える。

    注意:
    - これは学習用教師（GT）を安定化するための前処理です。
      推論後の平滑化（eval側）とは別に扱います。
    """
    y = np.asarray(y).reshape(-1).astype(np.int64)
    if min_run is None or int(min_run) <= 1 or y.size == 0:
        # 0/1に丸めて返す
        return np.clip(y, 0, 1).astype(np.int64)

    # unknown(-1) が混ざる場合は前後で補完
    if np.any(y < 0):
        y = _forward_fill_unknown(y)

    # 0/1 に丸める
    y = np.clip(y, 0, 1).astype(np.int64)

    return _merge_short_runs(y, int(min_run)).astype(np.int64)

def contact_to_support(
    contact_lr01: np.ndarray,
    smooth_min_run: int = 0,
) -> np.ndarray:
    """
    contact_lr01: (T,2) 0/1
    戻り値 support: (T,) 0=L支持, 1=R支持

    ルール:
    - (1,0) -> 0
    - (0,1) -> 1
    - (1,1) または (0,0) -> unknown(-1) として前後で補完
    """
    if contact_lr01.ndim != 2 or contact_lr01.shape[1] != 2:
        raise ValueError("contact_lr01 must be (T,2)")
    L = contact_lr01[:, 0].astype(np.int64)
    R = contact_lr01[:, 1].astype(np.int64)

    sup = np.where(L >= R, 0, 1).astype(np.int64)
    amb = ((L == 0) & (R == 0)) | ((L == 1) & (R == 1))
    sup[amb] = -1
    sup = _forward_fill_unknown(sup)

    if smooth_min_run and smooth_min_run > 1:
        sup = _merge_short_runs(sup, smooth_min_run)

    return sup.astype(np.int64)


def support_to_contact(support01: np.ndarray) -> np.ndarray:
    """
    support01: (T,) 0/1
    片足支持を contact_L/contact_R に戻す
    """
    support01 = support01.astype(np.int64).reshape(-1)
    con = np.zeros((len(support01), 2), dtype=np.float32)
    con[:, 0] = (support01 == 0).astype(np.float32)
    con[:, 1] = (support01 == 1).astype(np.float32)
    return con


def support_switch_events(support01: np.ndarray) -> np.ndarray:
    """
    support01: (T,) 0/1
    戻り値: (T,) 0/1。t>=1で support[t]!=support[t-1] を1とする（切替点）
    """
    s = support01.astype(np.int64).reshape(-1)
    ev = np.zeros_like(s, dtype=np.int64)
    if len(s) >= 2:
        ev[1:] = (s[1:] != s[:-1]).astype(np.int64)
    return ev


# -------------------------
# データ読み込み→window化
# -------------------------
def load_split_windows(
    csv_paths: List[Path],
    cols: Dict[str, List[str]],
    seq_len: int,
    horizon: int,
    stride: int,
    scaler,
    support_smooth_min_run: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : (N, seq_len, Din) scaled
    Ypos_s : (N, horizon, Dpos) scaled
    Ypos_m : (N, horizon, Dpos) meter
    Ysup : (N, horizon) int64, 0=L support, 1=R support
    """
    input_cols = cols["input_cols"]
    pos_cols = cols["pos_cols"]
    contact_cols = cols.get("contact_cols", [])

    if len(contact_cols) < 2:
        raise ValueError("contact_cols must contain 2 columns (contact_L, contact_R) to derive support label.")

    numeric_cols = cols.get('numeric_cols', pos_cols)
    support_col = cols.get('support_col', '')

    Din = len(input_cols)
    Dpos = len(pos_cols)

    X_list = []
    Yps_list = []
    Ypm_list = []
    Ysup_list = []

    for p in csv_paths:
        df = pd.read_csv(p)

        need = list(dict.fromkeys(input_cols + pos_cols + contact_cols))
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {p} -> {missing}")

        for seq in iter_sequences(df):
            seq = sort_sequence(seq)
            n = len(seq)
            starts = make_windows(n, seq_len, horizon, stride)
            if not starts:
                continue

            x_all = seq[input_cols].to_numpy(dtype=np.float32)
            pos_all_m = seq[pos_cols].to_numpy(dtype=np.float32)

            con_all = seq[contact_cols].fillna(0.0).to_numpy(dtype=np.float32)
            con_all = (con_all >= 0.5).astype(np.float32)  # 0/1固定

            # support教師（優先：CSVのsupport列、無ければcontactから推定）
            if support_col and support_col in df.columns:
                sup_all = df[support_col].to_numpy(dtype=np.int64).reshape(-1)
                sup_all = np.clip(sup_all, 0, 1)
                if support_smooth_min_run > 0:
                    sup_all = smooth_min_run_labels(sup_all, min_run=support_smooth_min_run)
            else:
                sup_all = contact_to_support(con_all, smooth_min_run=support_smooth_min_run)  # (n,)

            x_all_s = transform_numeric_and_assemble(df, input_cols=input_cols, numeric_cols=(pos_cols + cols.get('vel_cols', [])), scaler=scaler)
            pos_all_s = scale_pos_by_numeric_scaler(pos_all_m, scaler=scaler, pos_cols=pos_cols, numeric_cols=(pos_cols + cols.get('vel_cols', [])))

            for st in starts:
                x_seq = x_all_s[st:st + seq_len]
                y_pos_s = pos_all_s[st + seq_len:st + seq_len + horizon]
                y_pos_m = pos_all_m[st + seq_len:st + seq_len + horizon]
                y_sup = sup_all[st + seq_len:st + seq_len + horizon]

                X_list.append(x_seq)
                Yps_list.append(y_pos_s)
                Ypm_list.append(y_pos_m)
                Ysup_list.append(y_sup)

    if not X_list:
        return (
            np.zeros((0, seq_len, Din), dtype=np.float32),
            np.zeros((0, horizon, Dpos), dtype=np.float32),
            np.zeros((0, horizon, Dpos), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.int64),
        )

    X = np.stack(X_list, axis=0)
    Ypos_s = np.stack(Yps_list, axis=0)
    Ypos_m = np.stack(Ypm_list, axis=0)
    Ysup = np.stack(Ysup_list, axis=0).astype(np.int64)

    return X, Ypos_s, Ypos_m, Ysup


def fit_scaler_on_train(train_paths: List[Path], numeric_cols: List[str]) -> object:
    """
    TRAINの numeric(pos+vel) のみで scaler を fit
    """
    scaler = make_scaler()
    has_partial = hasattr(scaler, "partial_fit")

    if has_partial:
        for p in train_paths:
            df = pd.read_csv(p, usecols=numeric_cols)
            X = df.to_numpy(dtype=np.float32)
            scaler.partial_fit(X)
    else:
        Xs = []
        for p in train_paths:
            df = pd.read_csv(p, usecols=numeric_cols)
            Xs.append(df.to_numpy(dtype=np.float32))
        Xall = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(numeric_cols)), dtype=np.float32)
        scaler.fit(Xall)

    return scaler


# -------------------------
# split（multi CSV spec）
# -------------------------
def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def _expand_csv_specs(spec: Optional[str]) -> List[str]:
    """
    spec: None or string like "a.csv,b*.csv,dir/*.csv"
    returns: list of resolved file paths (unique, sorted)
    """
    if spec is None:
        return []
    parts = [s.strip() for s in str(spec).split(",") if s.strip()]
    out: List[str] = []
    for it in parts:
        if any(ch in it for ch in ["*", "?", "[", "]"]):
            out.extend(sorted(glob.glob(it)))
        else:
            out.append(it)
    # resolve, unique
    out_abs = sorted({resolve_path(p) for p in out})
    return out_abs


def choose_split(
    all_csvs: List[Path],
    out_dir: Path,
    seed: int,
    val_csv: Optional[str],
    test_csv: Optional[str],
    reuse_split: bool = True,
) -> Dict[str, object]:
    """
    Returns dict:
      {seed, train_csvs: [..], val_csvs: [..], test_csvs: [..]}
    """
    split_path = out_dir / "train_val_test_split.json"

    if reuse_split and split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        if "train_csvs" in sp and ("val_csvs" in sp or "val_csv" in sp) and ("test_csvs" in sp or "test_csv" in sp):
            # backward compatibility: accept val_csv/test_csv single
            if "val_csvs" not in sp and "val_csv" in sp:
                sp["val_csvs"] = [sp["val_csv"]]
            if "test_csvs" not in sp and "test_csv" in sp:
                sp["test_csvs"] = [sp["test_csv"]]
            return sp

    all_abs = [resolve_path(str(p)) for p in all_csvs]
    if len(all_abs) < 3:
        raise ValueError("csvが少なすぎます（train/val/testに最低3ファイル必要）")

    # user specified val/test
    val_abs_list = _expand_csv_specs(val_csv)
    test_abs_list = _expand_csv_specs(test_csv)
    if val_abs_list and test_abs_list:
        used = set(val_abs_list + test_abs_list)
        train_abs = [p for p in all_abs if p not in used]
        if len(train_abs) == 0:
            raise ValueError("train_csvsが0です。val/test指定が広すぎます。")
        return {"seed": seed, "train_csvs": train_abs, "val_csvs": val_abs_list, "test_csvs": test_abs_list}

    # otherwise random selection (deterministic by seed)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(all_abs))
    rng.shuffle(idx)
    val_abs = all_abs[idx[0]]
    test_abs = all_abs[idx[1]]
    train_abs = [all_abs[i] for i in idx[2:]]
    return {"seed": seed, "train_csvs": train_abs, "val_csvs": [val_abs], "test_csvs": [test_abs]}


# -------------------------
# モデル
# -------------------------
class GaitLSTM(nn.Module):
    """
    2-head:
      pos: (B, H*Dpos)  ※scaled空間
      support_logits: (B, H*2)  ※各tで2クラス（0=L支持, 1=R支持）
    """
    def __init__(self, din: int, hidden: int, layers: int, dropout: float,
                 horizon: int, dpos: int,
                 use_layernorm: bool = False):
        super().__init__()
        self.horizon = int(horizon)
        self.dpos = int(dpos)
        self.use_layernorm = bool(use_layernorm)

        # LayerNormは有無にかかわらず作っておく（state_dict互換性）
        self.ln_pos = nn.LayerNorm(hidden)
        self.ln_sup = nn.LayerNorm(hidden)

        self.lstm = nn.LSTM(
            input_size=din,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * dpos),
        )
        self.sup_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_pos = self.ln_pos(h_last) if self.use_layernorm else h_last
        h_sup = self.ln_sup(h_last) if self.use_layernorm else h_last
        pos = self.pos_head(h_pos)
        sup_logits = self.sup_head(h_sup)
        return pos, sup_logits


# -------------------------
# 指標
# -------------------------
def mae_rmse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = a - b
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, rmse


def binary_f1_acc(gt01: np.ndarray, pr01: np.ndarray) -> Tuple[float, float]:
    gt = gt01.reshape(-1).astype(np.int32)
    pr = pr01.reshape(-1).astype(np.int32)

    tp = int(np.sum((pr == 1) & (gt == 1)))
    fp = int(np.sum((pr == 1) & (gt == 0)))
    fn = int(np.sum((pr == 0) & (gt == 1)))
    tn = int(np.sum((pr == 0) & (gt == 0)))

    denom = (2 * tp + fp + fn)
    f1 = 1.0 if denom == 0 else float(2 * tp / denom)
    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    return f1, acc


def eval_epoch(
    model: nn.Module,
    device: torch.device,
    X: np.ndarray,
    Ypos_s: np.ndarray,
    Ysup: np.ndarray,
    horizon: int,
    dpos: int,
    batch_size: int,
    lambda_support: float,
    class_weight: Optional[np.ndarray],
    support_smooth_min_run: int = 0,
) -> Dict[str, float]:
    """
    val/test 用: total/pos/support と support acc、switch-event F1/Acc
    """
    model.eval()

    mse = nn.MSELoss(reduction="mean")

    # class_weight: (2,)
    w = None
    if class_weight is not None:
        cw = np.asarray(class_weight, dtype=np.float32).reshape(-1)
        if cw.shape[0] != 2:
            raise ValueError("class_weight must be (2,)")
        w = torch.tensor(cw, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w, reduction="mean")

    n = X.shape[0]
    if n == 0:
        return {"total": float("nan"), "pos": float("nan"), "support": float("nan"),
                "sup_acc": float("nan"), "sw_f1": float("nan"), "sw_acc": float("nan")}

    total_sum = 0.0
    pos_sum = 0.0
    sup_sum = 0.0

    sup_correct = 0
    sup_count = 0

    sw_pr_all = []
    sw_gt_all = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            bs = xb.shape[0]

            yps = torch.from_numpy(Ypos_s[i:i + batch_size]).to(device).reshape(-1, horizon * dpos)
            ysup = torch.from_numpy(Ysup[i:i + batch_size]).to(device).reshape(-1, horizon)  # (B,H) int64

            pos_pred, sup_logits = model(xb)
            pos_loss = mse(pos_pred, yps)

            # CE: (B,H,2) -> (B*H,2)
            sup_logits2 = sup_logits.view(-1, horizon, 2).reshape(-1, 2)
            ysup_flat = ysup.reshape(-1)
            sup_loss = ce(sup_logits2, ysup_flat)

            total_loss = pos_loss + float(lambda_support) * sup_loss

            total_sum += float(total_loss.item()) * bs
            pos_sum += float(pos_loss.item()) * bs
            sup_sum += float(sup_loss.item()) * bs

            # support accuracy
            pred_sup = torch.argmax(sup_logits.view(-1, horizon, 2), dim=-1)  # (B,H)
            sup_correct += int((pred_sup == ysup).sum().item())
            sup_count += int(pred_sup.numel())

            # switch events (after smoothing, optional)
            pred_sup_np = pred_sup.detach().cpu().numpy().astype(np.int64)
            gt_sup_np = ysup.detach().cpu().numpy().astype(np.int64)

            if support_smooth_min_run and support_smooth_min_run > 1:
                for k in range(pred_sup_np.shape[0]):
                    pred_sup_np[k] = _merge_short_runs(pred_sup_np[k], support_smooth_min_run)
                    gt_sup_np[k] = _merge_short_runs(gt_sup_np[k], support_smooth_min_run)

            pr_sw = np.stack([support_switch_events(pred_sup_np[k]) for k in range(pred_sup_np.shape[0])], axis=0)
            gt_sw = np.stack([support_switch_events(gt_sup_np[k]) for k in range(gt_sup_np.shape[0])], axis=0)

            sw_pr_all.append(pr_sw)
            sw_gt_all.append(gt_sw)

    sup_acc = float(sup_correct / max(sup_count, 1))

    sw_f1 = float("nan")
    sw_acc = float("nan")
    if sw_pr_all:
        pr = np.concatenate(sw_pr_all, axis=0).reshape(-1)
        gt = np.concatenate(sw_gt_all, axis=0).reshape(-1)
        sw_f1, sw_acc = binary_f1_acc(gt, pr)

    return {
        "total": total_sum / n,
        "pos": pos_sum / n,
        "support": sup_sum / n,
        "sup_acc": sup_acc,
        "sw_f1": float(sw_f1),
        "sw_acc": float(sw_acc),
    }


def per_horizon_metrics(
    model: nn.Module,
    device: torch.device,
    X: np.ndarray,
    Ypos_s: np.ndarray,
    Ypos_m: np.ndarray,
    scaler,
    pos_idx: List[int],
    horizon: int,
    dpos: int,
    batch_size: int,
) -> pd.DataFrame:
    model.eval()
    n = X.shape[0]
    if n == 0:
        return pd.DataFrame({"t": np.arange(horizon)})

    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            pos_pred, _ = model(xb)
            preds.append(pos_pred.detach().cpu().numpy().astype(np.float32))

    pred_pos_s = np.concatenate(preds, axis=0).reshape(-1, horizon, dpos)

    rows = []
    for t in range(horizon):
        a_s = pred_pos_s[:, t, :]
        b_s = Ypos_s[:, t, :]
        mae_s, rmse_s = mae_rmse(a_s, b_s)

        a_m = inverse_pos_only(a_s, scaler, pos_idx)
        b_m = Ypos_m[:, t, :]
        mae_m, rmse_m = mae_rmse(a_m, b_m)

        rows.append({
            "t": int(t),
            "mae_scaled": float(mae_s),
            "rmse_scaled": float(rmse_s),
            "mae_m": float(mae_m),
            "rmse_m": float(rmse_m),
        })

    return pd.DataFrame(rows)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_glob", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--val_csv", default=None,
                        help='Validation CSV spec. Comma-separated list and/or glob patterns.')
    parser.add_argument("--test_csv", default=None,
                        help='Test CSV spec. Comma-separated list and/or glob patterns.')
    parser.add_argument("--reuse_split", action="store_true",
                        help="Reuse out_dir/train_val_test_split.json if exists.")

    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=20)

    # classification loss weight (keep name for backward familiarity)
    parser.add_argument("--lambda_contact", type=float, default=1.0,
                        help="Weight for support classification loss (historical name: lambda_contact).")
    parser.add_argument("--support_class_weight_auto", action="store_true",
                        help="Auto class weight for CrossEntropyLoss based on TRAIN label distribution.")

    parser.add_argument("--use_support_input", action="store_true",
                    help="入力に過去のsupport(0/1)を追加します（観測可能な状態量として扱う）。比較実験用。")
    parser.add_argument("--lambda_switch", type=float, default=0.0,
                    help="supportの切替(switch)を別タスクで学習する場合の重み（0で無効）")
    parser.add_argument("--switch_pos_weight_auto", action="store_true",
                    help="switch(正例が希少)のpos_weightをtrainから自動推定します")
    parser.add_argument("--support_smooth_min_run", type=int, default=3,
                        help="Run-length smoothing for support labels (>=2 recommended). 0 disables.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--eval_test_end", action="store_true")

    parser.add_argument("--use_layernorm", action="store_true",
                        help="Enable LayerNorm on the LSTM last hidden state before each head.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)

    csvs = [Path(p) for p in sorted(glob.glob(args.csv_glob))]
    if len(csvs) == 0:
        raise FileNotFoundError(f"no csv matched: {args.csv_glob}")

    sp = choose_split(
        all_csvs=csvs,
        out_dir=out_dir,
        seed=args.seed,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        reuse_split=bool(args.reuse_split),
    )

    split_path = out_dir / "train_val_test_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=2)

    train_paths = [Path(p) for p in sp["train_csvs"]]
    val_paths = [Path(p) for p in sp["val_csvs"]]
    test_paths = [Path(p) for p in sp["test_csvs"]]

    print(f"[Diag] seq_len={args.seq_len}, horizon={args.horizon}, stride={args.stride}")
    print(f"[Diag] seed={args.seed}")
    print(f"[OK] split saved: {split_path}")
    print(f"[Diag] train_csvs: {len(train_paths)}")
    print(f"[Diag] val_csvs  : {len(val_paths)}")
    print(f"[Diag] test_csvs : {len(test_paths)}")

    cols = build_columns(out_dir, sample_csv=train_paths[0], use_vel=True, use_support_input=bool(args.use_support_input))
    input_cols = cols["input_cols"]
    pos_cols = cols["pos_cols"]
    vel_cols = cols.get("vel_cols", [])
    contact_cols = cols.get("contact_cols", [])

    Din = len(input_cols)
    Dpos = len(pos_cols)
    Dvel = len(vel_cols)
    if len(contact_cols) < 2:
        raise RuntimeError("contact_cols must contain contact_L/contact_R in columns.json or CSV.")

    print(f"[Diag] Din={Din}, Dpos={Dpos}, Dvel={Dvel}, support_classes=2")

    scaler = fit_scaler_on_train(train_paths, numeric_cols=(pos_cols + vel_cols))
    save_scaler(scaler, out_dir / "scaler.pkl")
    print("[OK] scaler fitted on TRAIN numeric(pos+vel) and saved: scaler.pkl")

    Xtr, Ytr_pos_s, _, Ytr_sup = load_split_windows(
        train_paths, cols, args.seq_len, args.horizon, args.stride, scaler,
        support_smooth_min_run=int(args.support_smooth_min_run),
    )
    Xva, Yva_pos_s, Yva_pos_m, Yva_sup = load_split_windows(
        val_paths, cols, args.seq_len, args.horizon, args.stride, scaler,
        support_smooth_min_run=int(args.support_smooth_min_run),
    )
    Xte, Yte_pos_s, Yte_pos_m, Yte_sup = load_split_windows(
        test_paths, cols, args.seq_len, args.horizon, args.stride, scaler,
        support_smooth_min_run=int(args.support_smooth_min_run),
    )

    print(f"[Diag] windows: train={Xtr.shape[0]}, val={Xva.shape[0]}, test={Xte.shape[0]}")

    if Xtr.shape[0] == 0:
        raise RuntimeError("train windowsが0です。seq_len/horizon/strideやCSVの長さを確認してください。")
    if Xva.shape[0] == 0:
        raise RuntimeError("val windowsが0です。val_csvの長さ・seq_len/horizonを確認してください。")

    # class weight (2,) for CE
    class_weight = None
    if args.support_class_weight_auto:
        y = Ytr_sup.reshape(-1).astype(np.int64)
        c0 = int(np.sum(y == 0))
        c1 = int(np.sum(y == 1))
        tot = max(c0 + c1, 1)
        # weight inversely proportional to frequency
        w0 = tot / max(c0, 1)
        w1 = tot / max(c1, 1)
        class_weight = np.asarray([w0, w1], dtype=np.float32)
        print(f"[Diag] support class counts: L={c0}, R={c1}")
        print(f"[Diag] support class_weight (CE): {class_weight.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diag] device={device}")

    model = GaitLSTM(
        din=Din,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        horizon=args.horizon,
        dpos=Dpos,
        use_layernorm=bool(args.use_layernorm),
    ).to(device)

    mse = nn.MSELoss(reduction="mean")

    # CE loss for support
    w = None
    if class_weight is not None:
        w = torch.tensor(class_weight, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=w, reduction="mean")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    best_state = None
    log_rows = []

    Ntr = Xtr.shape[0]
    idx_all = np.arange(Ntr)

    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx_all)

        total_sum = 0.0
        pos_sum = 0.0
        sup_sum = 0.0
        seen = 0

        for i in range(0, Ntr, args.batch_size):
            bi = idx_all[i:i + args.batch_size]
            xb = torch.from_numpy(Xtr[bi]).to(device)
            yps = torch.from_numpy(Ytr_pos_s[bi]).to(device).reshape(-1, args.horizon * Dpos)
            ysup = torch.from_numpy(Ytr_sup[bi]).to(device).reshape(-1, args.horizon)  # (B,H)

            pos_pred, sup_logits = model(xb)
            pos_loss = mse(pos_pred, yps)

            sup_logits2 = sup_logits.view(-1, args.horizon, 2).reshape(-1, 2)
            ysup_flat = ysup.reshape(-1)
            sup_loss = ce(sup_logits2, ysup_flat)

            loss = pos_loss + float(args.lambda_contact) * sup_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()

            bs = xb.shape[0]
            total_sum += float(loss.item()) * bs
            pos_sum += float(pos_loss.item()) * bs
            sup_sum += float(sup_loss.item()) * bs
            seen += bs

        train_total = total_sum / max(seen, 1)
        train_pos = pos_sum / max(seen, 1)
        train_sup = sup_sum / max(seen, 1)

        va = eval_epoch(
            model, device,
            Xva, Yva_pos_s, Yva_sup,
            horizon=args.horizon, dpos=Dpos,
            batch_size=max(args.batch_size, 256),
            lambda_support=float(args.lambda_contact),
            class_weight=class_weight,
            support_smooth_min_run=int(args.support_smooth_min_run),
        )

        print(f"[Epoch {ep:04d}] train total={train_total:.6f} pos={train_pos:.6f} support={train_sup:.6f} | "
              f"val total={va['total']:.6f} pos={va['pos']:.6f} support={va['support']:.6f} sup_acc={va['sup_acc']:.3f} sw_f1={va['sw_f1']:.3f} sw_acc={va['sw_acc']:.3f}")

        log_rows.append({
            "epoch": ep,
            "train_total": train_total,
            "train_pos": train_pos,
            "train_support": train_sup,
            "val_total": va["total"],
            "val_pos": va["pos"],
            "val_support": va["support"],
            "val_sup_acc": va["sup_acc"],
            "val_sw_f1": va["sw_f1"],
            "val_sw_acc": va["sw_acc"],
        })

        if va["total"] < best_val - 1e-12:
            best_val = va["total"]
            best_epoch = ep
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EarlyStop] val total が {args.patience} epoch 改善なし → 停止")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    torch.save(model.state_dict(), str(out_dir / "zmodel_lstm.pt"))

    cfg = {
        "seed": int(args.seed),
        "seq_len": int(args.seq_len),
        "horizon": int(args.horizon),
        "stride": int(args.stride),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "dropout": float(args.dropout),
        "use_layernorm": bool(getattr(args, "use_layernorm", False)),
        "Din": int(Din),
        "Dpos": int(Dpos),
        "Dvel": int(Dvel),
        "support_classes": 2,
        "lambda_contact": float(args.lambda_contact),
        "support_class_weight_auto": bool(args.support_class_weight_auto),
        "use_support_input": bool(args.use_support_input),
        "lambda_switch": float(args.lambda_switch),
        "support_smooth_min_run": int(args.support_smooth_min_run),
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "train_csvs": [str(p) for p in train_paths],
        "val_csvs": [str(p) for p in val_paths],
        "test_csvs": [str(p) for p in test_paths],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

    pos_idx2 = get_pos_indices(input_cols, pos_cols)
    df_val = per_horizon_metrics(
        model, device,
        Xva, Yva_pos_s, Yva_pos_m,
        scaler=scaler,
        pos_idx=pos_idx2,
        horizon=args.horizon, dpos=Dpos,
        batch_size=max(args.batch_size, 256),
    )
    df_val.to_csv(out_dir / "per_horizon_val.csv", index=False)

    if args.eval_test_end:
        df_test = per_horizon_metrics(
            model, device,
            Xte, Yte_pos_s, Yte_pos_m,
            scaler=scaler,
            pos_idx=pos_idx2,
            horizon=args.horizon, dpos=Dpos,
            batch_size=max(args.batch_size, 256),
        )
        df_test.to_csv(out_dir / "per_horizon_test.csv", index=False)

    print("[OK] saved:")
    print(" -", out_dir / "zmodel_lstm.pt")
    print(" -", out_dir / "scaler.pkl")
    print(" -", out_dir / "columns.json")
    print(" -", out_dir / "config.json")
    print(" -", out_dir / "train_log.csv")
    print(" -", out_dir / "per_horizon_val.csv")
    if args.eval_test_end:
        print(" -", out_dir / "per_horizon_test.csv")


if __name__ == "__main__":
    main()
