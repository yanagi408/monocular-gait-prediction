#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zcap_draw3d.py (batch QA edition)

目的:
- zcap5-2.py が出力した *_raw.csv / *_processed.csv（下肢6点の3D座標）を対象に、
  前処理の整合性確認に必要な「GIF」「時系列図」「数値サマリ」を一括生成する。

入力:
- 複数CSVに対応（--csv, --csv_glob, --csv_list）
- 必須列（座標）:
    {NAME}_{x,y,z}_m  for NAME in ["L_HIP","R_HIP","L_KNEE","R_KNEE","L_ANKLE","R_ANKLE"]
  互換として {NAME}_{x,y,z} も許可。
- 任意列（ある場合に追加で解析・描画）:
    frame, time_sec
    contact_L, contact_R
    stance_L, stance_R
    support（0/1）, support_switch（0/1）

出力（各CSVごと）:
- <out_dir>/<stem>/
    - <stem>_3d.gif          （--views 3d）
    - <stem>_yz.gif          （--views yz）
    - <stem>_ts.png          （時系列：hipΔZ / hipY+ankleY / flags）
    - <stem>_bones.png       （骨長時系列）
    - <stem>_summary.json    （数値サマリ）
- <out_dir>/summary.csv      （全CSVの集計）
- <out_dir>/run_info.json    （実行条件）
- 追加（--compare_raw_processed 指定時）:
    - <out_dir>/_compare/<key>_compare_ts.png
      ※ stem の末尾が "_raw" と "_processed" のペアを自動対応付けし、hipΔZ/ankleY/flags を重ね描画。

使用例:
  # processed をまとめて確認（YZ+3D、GIF=5fps、最大500フレームに間引き）
  python zcap_draw3d.py --csv_glob "outputs_hitobetu_riku/*_processed.csv" --out_dir outputs_hitobetu_riku/qa --views 3d yz --fps 5 --max_frames 500

  # raw と processed を混ぜて確認し、比較図も作る
  python zcap_draw3d.py --csv_glob "outputs_final/.csv" --csv_glob --out_dir outputs_final/byouga --views 3d yz --fps 5 --compare_raw_processed
  python zcap_draw3d.py --csv_glob "outputs_hitobetu_riku/*_raw.csv" --csv_glob "outputs_hitobetu_riku/*_processed.csv" --out_dir outputs_hitobetu_riku/qa --views yz --fps 5 --compare_raw_processed
  # リストファイル（1行1パス or 1行1glob）
  python zcap_draw3d.py --csv_list filelist.txt --out_dir qa_out --views yz --fps 5

注意:
- Matplotlib の Agg バックエンドでバッチ実行（GUI不要）。
- GIF生成はフレーム数に比例して重くなるため、--max_frames / --frame_step を推奨。
"""

from __future__ import annotations

import argparse
import json
import glob as _glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------
# 下肢6点の定義（zcap系と揃える）
# ------------------------------------------------------------
ORDER: List[str] = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]

BONES: List[Tuple[int, int]] = [
    (0, 2),  # L_HIP   -> L_KNEE
    (2, 4),  # L_KNEE  -> L_ANKLE
    (1, 3),  # R_HIP   -> R_KNEE
    (3, 5),  # R_KNEE  -> R_ANKLE
    (0, 1),  # pelvis: L_HIP   -> R_HIP
]

BONE_LABELS: List[str] = ["L_HIP-L_KNEE", "L_KNEE-L_ANKLE", "R_HIP-R_KNEE", "R_KNEE-R_ANKLE", "L_HIP-R_HIP"]


# ------------------------------------------------------------
# 基本ユーティリティ
# ------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_list_file(list_path: Path) -> List[str]:
    items: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def collect_csv_paths(csvs: List[str], csv_glob: Optional[List[str]], csv_list: Optional[str]) -> List[Path]:
    """
    入力の集約:
    - --csv: そのまま
    - --csv_glob: glob展開（絶対/相対どちらも可、複数指定可）
    - --csv_list: 1行1パス もしくは 1行1glob を許可
    """
    raw: List[str] = []
    raw.extend(csvs)

    if csv_glob:
        for g in csv_glob:
            raw.extend(sorted(_glob.glob(g, recursive=True)))

    if csv_list:
        for item in read_list_file(Path(csv_list)):
            if any(ch in item for ch in ["*", "?", "[", "]"]):
                raw.extend(sorted(_glob.glob(item, recursive=True)))
            else:
                raw.append(item)

    out: List[Path] = []
    seen = set()
    for s in raw:
        p = Path(s).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.suffix.lower() != ".csv":
            continue
        if not p.exists():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return out


def expand_limits(lim: Tuple[float, float], margin_ratio: float = 0.1) -> Tuple[float, float]:
    lo, hi = lim
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -0.1, 0.1
    if hi == lo:
        return lo - 0.1, hi + 0.1
    span = hi - lo
    margin = span * margin_ratio
    return lo - margin, hi + margin


def choose_frame_indices(T: int, frame_step: int = 1, max_frames: int = 0) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=int)

    step = max(1, int(frame_step))
    idx = np.arange(0, T, step, dtype=int)

    if max_frames and len(idx) > max_frames:
        stride = int(np.ceil(len(idx) / max_frames))
        idx = idx[::max(1, stride)]

    return idx


# ------------------------------------------------------------
# CSV 読み込みと座標抽出
# ------------------------------------------------------------
def resolve_coord_cols(df: pd.DataFrame, nm: str) -> Tuple[str, str, str]:
    cand_sets = [
        (f"{nm}_x_m", f"{nm}_y_m", f"{nm}_z_m"),
        (f"{nm}_x", f"{nm}_y", f"{nm}_z"),
    ]
    for xcol, ycol, zcol in cand_sets:
        if xcol in df.columns and ycol in df.columns and zcol in df.columns:
            return xcol, ycol, zcol
    raise ValueError(f"座標列がありません: {nm}_x_m/{nm}_y_m/{nm}_z_m（または _x/_y/_z）")


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for nm in ORDER:
        resolve_coord_cols(df, nm)
    return df


def extract_xyz_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    T = len(df)
    J = len(ORDER)
    xs = np.zeros((T, J), dtype=float)
    ys = np.zeros((T, J), dtype=float)
    zs = np.zeros((T, J), dtype=float)

    for j, nm in enumerate(ORDER):
        xcol, ycol, zcol = resolve_coord_cols(df, nm)
        xs[:, j] = df[xcol].to_numpy(dtype=float)
        ys[:, j] = df[ycol].to_numpy(dtype=float)
        zs[:, j] = df[zcol].to_numpy(dtype=float)

    return {"x": xs, "y": ys, "z": zs}


def compute_bounds(xyz: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    xs, ys, zs = xyz["x"], xyz["y"], xyz["z"]
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    zmin, zmax = float(np.nanmin(zs)), float(np.nanmax(zs))
    return {"x": (xmin, xmax), "y": (ymin, ymax), "z": (zmin, zmax)}


# ------------------------------------------------------------
# 診断値（前処理確認）
# ------------------------------------------------------------
@dataclass
class Diagnostics:
    metrics: Dict[str, float]
    series: Dict[str, np.ndarray]
    flags_present: Dict[str, bool]


def compute_diagnostics(df: pd.DataFrame, xyz: Dict[str, np.ndarray]) -> Diagnostics:
    xs, ys, zs = xyz["x"], xyz["y"], xyz["z"]
    T = len(df)

    hip_x = 0.5 * (xs[:, 0] + xs[:, 1])
    hip_y = 0.5 * (ys[:, 0] + ys[:, 1])
    hip_z = 0.5 * (zs[:, 0] + zs[:, 1])
    hip_dz = hip_z - hip_z[0] if T > 0 else np.array([], dtype=float)

    lank_y = ys[:, 4]
    rank_y = ys[:, 5]

    bone_len = []
    for (i, j) in BONES:
        dx = xs[:, i] - xs[:, j]
        dy = ys[:, i] - ys[:, j]
        dzb = zs[:, i] - zs[:, j]
        bone_len.append(np.sqrt(dx * dx + dy * dy + dzb * dzb))
    bone_len = np.stack(bone_len, axis=1)

    bone_mean = np.nanmean(bone_len, axis=0)
    bone_std = np.nanstd(bone_len, axis=0)
    bone_cv = np.where(bone_mean > 1e-9, bone_std / bone_mean, np.nan)

    has_time = "time_sec" in df.columns
    has_contact = ("contact_L" in df.columns) and ("contact_R" in df.columns)
    has_support = "support" in df.columns
    has_support_switch = "support_switch" in df.columns
    has_stance = ("stance_L" in df.columns) and ("stance_R" in df.columns)

    if has_time:
        tsec = df["time_sec"].to_numpy(dtype=float)
        duration = float(tsec[-1] - tsec[0]) if T > 1 else 0.0
        dt_med = float(np.median(np.diff(tsec))) if T > 2 else float("nan")
    else:
        tsec = None
        duration = float("nan")
        dt_med = float("nan")

    if has_contact:
        cL = df["contact_L"].to_numpy(dtype=float)
        cR = df["contact_R"].to_numpy(dtype=float)
        both = (cL > 0.5) & (cR > 0.5)
        none = (cL <= 0.5) & (cR <= 0.5)
        double_rate = float(np.mean(both)) if T > 0 else 0.0
        none_rate = float(np.mean(none)) if T > 0 else 0.0
        cL_rate = float(np.mean(cL > 0.5)) if T > 0 else 0.0
        cR_rate = float(np.mean(cR > 0.5)) if T > 0 else 0.0
    else:
        cL = cR = None
        double_rate = none_rate = cL_rate = cR_rate = float("nan")

    if has_support:
        sup = df["support"].to_numpy(dtype=float)
        sup_r_rate = float(np.mean(sup > 0.5)) if T > 0 else 0.0
        if T > 1:
            sw_from_sup = (sup[1:] != sup[:-1]).astype(float)
            sw_rate = float(np.mean(sw_from_sup))
            sw_count = float(np.sum(sw_from_sup))
        else:
            sw_from_sup = np.zeros((0,), dtype=float)
            sw_rate = 0.0
            sw_count = 0.0
    else:
        sup = None
        sw_from_sup = None
        sup_r_rate = float("nan")
        sw_rate = float("nan")
        sw_count = float("nan")

    if has_support_switch:
        ssw = df["support_switch"].to_numpy(dtype=float)
        ssw_rate = float(np.mean(ssw > 0.5)) if T > 0 else 0.0
        ssw_count = float(np.sum(ssw > 0.5)) if T > 0 else 0.0
    else:
        ssw = None
        ssw_rate = float("nan")
        ssw_count = float("nan")

    metrics: Dict[str, float] = {
        "T": float(T),
        "duration_sec": duration,
        "dt_median_sec": dt_med,
        "hip_dz_total_m": float(hip_dz[-1]) if T > 0 else 0.0,
        "hip_z_range_m": float(np.nanmax(hip_z) - np.nanmin(hip_z)) if T > 0 else 0.0,
        "hip_y_range_m": float(np.nanmax(hip_y) - np.nanmin(hip_y)) if T > 0 else 0.0,
        "ankle_y_min_m": float(np.nanmin(np.minimum(lank_y, rank_y))) if T > 0 else float("nan"),
        "ankle_y_mean_m": float(np.nanmean(0.5 * (lank_y + rank_y))) if T > 0 else float("nan"),
        "double_contact_rate": double_rate,
        "no_contact_rate": none_rate,
        "contact_L_rate": cL_rate,
        "contact_R_rate": cR_rate,
        "support_R_rate": sup_r_rate,
        "switch_rate_from_support": sw_rate,
        "switch_count_from_support": sw_count,
        "switch_rate_from_support_switch": ssw_rate,
        "switch_count_from_support_switch": ssw_count,
    }

    for k, lbl in enumerate(BONE_LABELS):
        key_base = lbl.replace("-", "_")
        metrics[f"bone_mean_{key_base}"] = float(bone_mean[k])
        metrics[f"bone_cv_{key_base}"] = float(bone_cv[k])

    series: Dict[str, np.ndarray] = {
        "hip_x": hip_x,
        "hip_y": hip_y,
        "hip_z": hip_z,
        "hip_dz": hip_dz,
        "lank_y": lank_y,
        "rank_y": rank_y,
        "bone_len": bone_len,
        "bone_mean": bone_mean,
        "bone_cv": bone_cv,
    }
    if tsec is not None:
        series["time_sec"] = tsec
    if cL is not None:
        series["contact_L"] = cL
        series["contact_R"] = cR
    if sup is not None:
        series["support"] = sup
        series["switch_from_support"] = sw_from_sup
    if ssw is not None:
        series["support_switch"] = ssw

    flags_present = {
        "time_sec": has_time,
        "contact": has_contact,
        "stance": has_stance,
        "support": has_support,
        "support_switch": has_support_switch,
    }

    return Diagnostics(metrics=metrics, series=series, flags_present=flags_present)


# ------------------------------------------------------------
# 図: 時系列 / 骨長
# ------------------------------------------------------------
def plot_timeseries(diag: Diagnostics, out_png: Path, title: str = "") -> None:
    s = diag.series
    T = int(diag.metrics.get("T", 0))
    x = s["time_sec"] if "time_sec" in s else np.arange(T, dtype=float)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 1], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, s["hip_dz"], label="hip ΔZ (relative)")
    ax1.set_ylabel("ΔZ [m]")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, s["hip_y"], label="hip_y")
    ax2.plot(x, s["lank_y"], label="L_ANKLE_y")
    ax2.plot(x, s["rank_y"], label="R_ANKLE_y")
    ax2.set_ylabel("Y [m]")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(gs[2, 0])
    if "contact_L" in s:
        ax3.step(x, s["contact_L"], where="post", label="contact_L")
        ax3.step(x, s["contact_R"], where="post", label="contact_R")
    if "support" in s:
        ax3.step(x, s["support"], where="post", label="support (0=L,1=R)")
    if "support_switch" in s:
        ax3.step(x, s["support_switch"], where="post", label="support_switch")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel("flags")
    ax3.grid(True)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(gs[3, 0])
    txt = [
        f"T={int(diag.metrics['T'])}",
        f"ΔZ_total={diag.metrics['hip_dz_total_m']:.3f} m",
        f"double_contact={diag.metrics.get('double_contact_rate', float('nan')):.3f}",
        f"no_contact={diag.metrics.get('no_contact_rate', float('nan')):.3f}",
        f"switch_rate={diag.metrics.get('switch_rate_from_support', float('nan')):.3f}",
    ]
    ax4.axis("off")
    ax4.text(0.01, 0.5, " | ".join(txt), va="center", ha="left", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bones(diag: Diagnostics, out_png: Path, title: str = "") -> None:
    s = diag.series
    T = int(diag.metrics.get("T", 0))
    x = s["time_sec"] if "time_sec" in s else np.arange(T, dtype=float)

    bone = s["bone_len"]
    cv = s["bone_cv"]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for k in range(bone.shape[1]):
        lbl = BONE_LABELS[k]
        if np.isfinite(cv[k]):
            lbl = f"{lbl} (cv={cv[k]:.3f})"
        ax.plot(x, bone[:, k], label=lbl)

    ax.set_ylabel("bone length [m]")
    ax.grid(True)
    ax.legend(loc="best", ncol=2, fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_raw_processed(diag_raw: Diagnostics, diag_proc: Diagnostics, out_png: Path, title: str = "") -> None:
    """
    raw と processed の主要系列を同一図に重ね描画する（前処理の効果確認用）。
    """
    s0, s1 = diag_raw.series, diag_proc.series
    T0 = int(diag_raw.metrics.get("T", 0))
    T1 = int(diag_proc.metrics.get("T", 0))

    x0 = s0["time_sec"] if "time_sec" in s0 else np.arange(T0, dtype=float)
    x1 = s1["time_sec"] if "time_sec" in s1 else np.arange(T1, dtype=float)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x0, s0["hip_dz"], label="raw hip ΔZ")
    ax1.plot(x1, s1["hip_dz"], label="processed hip ΔZ")
    ax1.set_ylabel("ΔZ [m]")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x0, s0["lank_y"], label="raw L_ANKLE_y")
    ax2.plot(x0, s0["rank_y"], label="raw R_ANKLE_y")
    ax2.plot(x1, s1["lank_y"], label="processed L_ANKLE_y")
    ax2.plot(x1, s1["rank_y"], label="processed R_ANKLE_y")
    ax2.set_ylabel("Y [m]")
    ax2.grid(True)
    ax2.legend(loc="best", ncol=2, fontsize=8)

    ax3 = fig.add_subplot(gs[2, 0])
    if "contact_L" in s0 and "contact_L" in s1:
        ax3.step(x0, s0["contact_L"], where="post", label="raw contact_L")
        ax3.step(x0, s0["contact_R"], where="post", label="raw contact_R")
        ax3.step(x1, s1["contact_L"], where="post", label="processed contact_L")
        ax3.step(x1, s1["contact_R"], where="post", label="processed contact_R")
    if "support" in s0 and "support" in s1:
        ax3.step(x0, s0["support"], where="post", label="raw support")
        ax3.step(x1, s1["support"], where="post", label="processed support")
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel("flags")
    ax3.grid(True)
    ax3.legend(loc="best", ncol=2, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# アニメーション描画 (3D / 2D-YZ)
# ------------------------------------------------------------
def make_animation_3d(
    xyz: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    out_gif: Path,
    fps: float,
    elev: float,
    azim: float,
    frame_idx: np.ndarray,
) -> None:
    xs, ys, zs = xyz["x"], xyz["y"], xyz["z"]
    T, _ = xs.shape

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    xlim = expand_limits(bounds["x"])
    ylim = expand_limits(bounds["y"])
    zlim = expand_limits(bounds["z"])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    mid_x = 0.5 * (xlim[0] + xlim[1])
    mid_y = 0.5 * (ylim[0] + ylim[1])
    mid_z = 0.5 * (zlim[0] + zlim[1])
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.view_init(elev=elev, azim=azim)

    lines = []
    for _ in BONES:
        line, = ax.plot([], [], [], linewidth=2)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(k):
        t = int(frame_idx[k])
        if t < 0 or t >= T:
            return lines
        x_t = xs[t]
        y_t = ys[t]
        z_t = zs[t]
        for line, (i, j) in zip(lines, BONES):
            line.set_data([x_t[i], x_t[j]], [y_t[i], y_t[j]])
            line.set_3d_properties([z_t[i], z_t[j]])
        return lines

    interval = 1000.0 / fps if fps > 0 else 33.0
    anim = FuncAnimation(fig, update, init_func=init, frames=len(frame_idx), interval=interval, blit=True)

    ensure_dir(out_gif.parent)
    anim.save(str(out_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_animation_yz(
    xyz: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[float, float]],
    out_gif: Path,
    fps: float,
    frame_idx: np.ndarray,
) -> None:
    ys, zs = xyz["y"], xyz["z"]
    T, _ = ys.shape

    fig, ax = plt.subplots(figsize=(4, 4))
    zlim = expand_limits(bounds["z"])
    ylim = expand_limits(bounds["y"])
    ax.set_xlim(*zlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")

    lines = []
    for _ in BONES:
        line, = ax.plot([], [], linewidth=2)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(k):
        t = int(frame_idx[k])
        if t < 0 or t >= T:
            return lines
        y_t = ys[t]
        z_t = zs[t]
        for line, (i, j) in zip(lines, BONES):
            line.set_data([z_t[i], z_t[j]], [y_t[i], y_t[j]])
        return lines

    interval = 1000.0 / fps if fps > 0 else 33.0
    anim = FuncAnimation(fig, update, init_func=init, frames=len(frame_idx), interval=interval, blit=True)

    ensure_dir(out_gif.parent)
    anim.save(str(out_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)


# ------------------------------------------------------------
# raw/processed対応付け（比較図用）
# ------------------------------------------------------------
def stem_key_and_kind(stem: str) -> Tuple[str, str]:
    """
    stem 末尾が _raw / _processed の場合にペアリングする。
    kind: "raw" | "processed" | "other"
    """
    if stem.endswith("_raw"):
        return stem[:-4], "raw"
    if stem.endswith("_processed"):
        return stem[:-10], "processed"
    return stem, "other"


# ------------------------------------------------------------
# メイン（バッチ処理）
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="zcap CSV batch QA drawer (GIF+plots+summary)")
    parser.add_argument("--csv", type=str, action="append", default=[], help="入力CSV（複数可）")
    parser.add_argument("--csv_glob", type=str, action="append", default=None, help="入力CSVのglob（複数可）")
    parser.add_argument("--csv_list", type=str, default=None, help="入力CSVのリストファイル（1行1パス or 1行1glob）")
    parser.add_argument("--out_dir", type=str, required=True, help="出力ディレクトリ（各CSVごとのサブフォルダを作る）")

    parser.add_argument("--views", type=str, nargs="+", default=["yz"], choices=["3d", "yz"], help="生成するGIFの種類")
    parser.add_argument("--fps", type=float, default=5.0, help="GIFのfps")
    parser.add_argument("--elev", type=float, default=20.0, help="3D viewの仰角（view=3d用）")
    parser.add_argument("--azim", type=float, default=-70.0, help="3D viewの方位角（view=3d用）")

    parser.add_argument("--frame_step", type=int, default=1, help="フレーム間引き（この間隔で描画）")
    parser.add_argument("--max_frames", type=int, default=0, help="描画に使う最大フレーム数（0で無制限）")
    parser.add_argument("--overwrite", action="store_true", help="既存出力があっても上書きする")
    parser.add_argument("--compare_raw_processed", action="store_true", help="raw/processedの比較図を追加で出力する")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    csv_paths = collect_csv_paths(args.csv, args.csv_glob, args.csv_list)
    if not csv_paths:
        raise SystemExit("入力CSVが見つかりません。--csv / --csv_glob / --csv_list を確認してください。")

    run_info = {
        "out_dir": str(out_dir),
        "n_csv": len(csv_paths),
        "views": args.views,
        "fps": args.fps,
        "elev": args.elev,
        "azim": args.azim,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "compare_raw_processed": bool(args.compare_raw_processed),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: List[Dict[str, object]] = []
    diag_map: Dict[str, Dict[str, Diagnostics]] = {}  # key -> kind -> diag

    for i, csv_path in enumerate(csv_paths):
        stem = csv_path.stem
        sub = out_dir / stem
        ensure_dir(sub)

        print(f"[{i+1}/{len(csv_paths)}] {csv_path}")

        try:
            df = load_csv(csv_path)
            xyz = extract_xyz_arrays(df)
            bounds = compute_bounds(xyz)
            diag = compute_diagnostics(df, xyz)

            frame_idx = choose_frame_indices(int(diag.metrics["T"]), frame_step=args.frame_step, max_frames=args.max_frames)

            ts_png = sub / f"{stem}_ts.png"
            bones_png = sub / f"{stem}_bones.png"
            js_path = sub / f"{stem}_summary.json"

            if args.overwrite or (not ts_png.exists()):
                plot_timeseries(diag, ts_png, title=stem)

            if args.overwrite or (not bones_png.exists()):
                plot_bones(diag, bones_png, title=f"{stem} bone lengths")

            js_path.write_text(
                json.dumps(
                    {"csv": str(csv_path), "metrics": diag.metrics, "flags_present": diag.flags_present},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            if "3d" in args.views:
                out_gif = sub / f"{stem}_3d.gif"
                if args.overwrite or (not out_gif.exists()):
                    make_animation_3d(xyz, bounds, out_gif, fps=args.fps, elev=args.elev, azim=args.azim, frame_idx=frame_idx)

            if "yz" in args.views:
                out_gif = sub / f"{stem}_yz.gif"
                if args.overwrite or (not out_gif.exists()):
                    make_animation_yz(xyz, bounds, out_gif, fps=args.fps, frame_idx=frame_idx)

            # store diagnostics for compare
            key, kind = stem_key_and_kind(stem)
            diag_map.setdefault(key, {})[kind] = diag

            row: Dict[str, object] = {"csv": str(csv_path), "out_dir": str(sub)}
            row.update({k: float(v) for k, v in diag.metrics.items()})
            row.update({f"has_{k}": bool(v) for k, v in diag.flags_present.items()})
            rows.append(row)

            print(f"  [OK] wrote: {sub}")

        except Exception as e:
            rows.append({"csv": str(csv_path), "out_dir": str(sub), "error": str(e)})
            print(f"  [ERR] {e}")

    # compare plots
    if args.compare_raw_processed:
        cmp_dir = out_dir / "_compare"
        ensure_dir(cmp_dir)
        for key, d in diag_map.items():
            if ("raw" in d) and ("processed" in d):
                out_png = cmp_dir / f"{key}_compare_ts.png"
                if args.overwrite or (not out_png.exists()):
                    plot_compare_raw_processed(d["raw"], d["processed"], out_png, title=f"{key}: raw vs processed")

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
