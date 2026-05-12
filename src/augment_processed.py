# augment_processed.py
# ------------------------------------------------------------
# 目的:
#   zcap5-2.py が出力した processed.csv（下肢6点×xyz=18次元 + contact/support 等）を入力として、
#   学習データ量を増やすためのデータ拡張を行う。
#
# 重要（骨長を壊さない拡張に修正済み）:
#   processed.csv はIK等により骨長が強く安定している。
#   関節ごとに独立ノイズを入れると骨長が崩れ、zcapの整形思想と矛盾する。
#   そこで A/B は「骨格全体への剛体並進（同じΔを全関節へ加算）」として実装する。
#   これにより、骨長・関節相対配置は保持される（数値誤差を除き）。
#
# 拡張方式:
#   A: 微小ノイズ（剛体並進の微小ジッタ）
#   B: 低周波ドリフト（剛体並進のゆっくりしたずれ）
#   D: 単調時間伸縮（順序を保った速度差の模擬）
#
# contact / support の扱い（比較条件統一のため）:
#   - A/B（時間軸不変）: contact_L/R と support は保持（値を変えない）
#   - D（時間軸変更） : contact_L/R と support を最近傍でリサンプルして追従
#   - その後、派生列（stance, HS/TO, support系イベント）を再計算して内部整合を取る
#
# 前提（本プロジェクトの processed.csv）:
#   - pos列は 6点（L/R HIP, KNEE, ANKLE）の *_x_m,*_y_m,*_z_m の計18列
#   - support は2クラス（0/1）
# ------------------------------------------------------------

import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 列定義（pos18）
# -----------------------------
def get_pos_cols_or_raise(df: pd.DataFrame) -> List[str]:
    joints = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]
    pos_cols: List[str] = []
    for j in joints:
        for ax in ["x", "y", "z"]:
            col = f"{j}_{ax}_m"
            if col not in df.columns:
                raise ValueError(f"必要な列がありません: {col}")
            pos_cols.append(col)
    return pos_cols


def get_known_discrete_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "contact_L", "contact_R",
        "support",
        "stance_L", "stance_R",
        "HS_L", "TO_L", "HS_R", "TO_R",
        "HS_support_L", "TO_support_L", "HS_support_R", "TO_support_R",
        "support_switch",
        "video_id",
    ]
    return [c for c in candidates if c in df.columns]


# -----------------------------
# time_sec 推定
# -----------------------------
def infer_dt_time_sec(df: pd.DataFrame) -> Optional[float]:
    if "time_sec" not in df.columns:
        return None
    ts = df["time_sec"].to_numpy(dtype=np.float64)
    if ts.size < 3:
        return None
    d = np.diff(ts)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    dt = float(np.median(d))
    if dt <= 0 or (not np.isfinite(dt)):
        return None
    return dt


# -----------------------------
# 平滑
# -----------------------------
def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    反射パディング + 移動平均
    x: (T, D)
    """
    if win <= 1:
        return x.copy()
    win = int(win)
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(win, dtype=np.float64) / win
    y = np.zeros_like(xp, dtype=np.float64)
    for d in range(x.shape[1]):
        y[:, d] = np.convolve(xp[:, d], kernel, mode="same")
    return y[pad:-pad, :]


def estimate_highfreq_sigma(x: np.ndarray, smooth_win: int) -> np.ndarray:
    """
    高周波成分（x - MA(x)）の標準偏差を軸ごとに推定する。
    """
    sm = moving_average(x, int(smooth_win))
    hf = x - sm
    sigma = np.std(hf, axis=0)
    sigma = np.maximum(sigma, 1e-9)
    return sigma


def make_lowfreq_drift(T: int, dims: int, drift_win: int, rng: np.random.Generator) -> np.ndarray:
    """
    ランダム系列を移動平均して低周波ドリフトを作る（平均0に調整）
    """
    raw = rng.normal(0.0, 1.0, size=(T, dims))
    drift = moving_average(raw, int(drift_win))
    drift -= drift.mean(axis=0, keepdims=True)
    return drift


# -----------------------------
# 時間ワープ（単調）
# -----------------------------
def monotonic_time_warp(T: int, rate_range: float, rng: np.random.Generator) -> np.ndarray:
    rate_range = float(rate_range)
    if rate_range <= 0.0:
        return np.arange(T, dtype=np.float64)

    eps = rng.uniform(-rate_range, rate_range, size=T)
    v = 1.0 + eps
    tau = np.cumsum(v)
    tau = (tau - tau[0]) / (tau[-1] - tau[0]) * (T - 1)
    return tau.astype(np.float64)


def resample_continuous(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    連続値: 線形補間
    x: (T, D)
    """
    T = x.shape[0]
    t = np.arange(T, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(t, tau, x[:, d])
    return out


def resample_discrete(arr: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    離散値: 最近傍（round）
    """
    T = arr.shape[0]
    t = np.arange(T, dtype=np.float64)
    idx = np.interp(t, tau, np.arange(T, dtype=np.float64))
    idx = np.clip(np.rint(idx).astype(int), 0, T - 1)
    return arr[idx]


# -----------------------------
# contact/support から派生列を再計算
# -----------------------------
def transitions_01(x01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    0->1: HS, 1->0: TO（同長T, 0/1）
    """
    x = x01.astype(int)
    d = np.diff(x, prepend=x[0])
    hs = (d == 1).astype(int)
    to = (d == -1).astype(int)
    return hs, to


def recompute_from_contact_support(df: pd.DataFrame, *, overwrite: bool = True) -> pd.DataFrame:
    out = df.copy()

    has_cL = "contact_L" in out.columns
    has_cR = "contact_R" in out.columns
    has_sup = "support" in out.columns

    # contact を int(0/1) に正規化
    if has_cL:
        out["contact_L"] = out["contact_L"].fillna(0).astype(int)
    if has_cR:
        out["contact_R"] = out["contact_R"].fillna(0).astype(int)

    # support を int(0/1) に正規化（ffill/bfill で FutureWarning 回避）
    if has_sup:
        sup_s = out["support"]
        sup_s = sup_s.ffill().bfill().fillna(0)
        sup = sup_s.astype(int).to_numpy()
        sup = (sup != 0).astype(int)  # 0/1以外が混ざっても2クラスへ丸める
        out["support"] = sup.astype(int)

        # stance: support=0→左支持, support=1→右支持
        stance_L = (sup == 0).astype(int)
        stance_R = (sup == 1).astype(int)
        if overwrite or ("stance_L" not in out.columns):
            out["stance_L"] = stance_L
        if overwrite or ("stance_R" not in out.columns):
            out["stance_R"] = stance_R

        # support切替イベント
        ds = np.diff(sup, prepend=sup[0])
        support_switch = (ds != 0).astype(int)

        # supportが 0へ切替 → 左支持開始
        # supportが 1へ切替 → 右支持開始
        hs_sup_L = ((ds != 0) & (sup == 0)).astype(int)
        hs_sup_R = ((ds != 0) & (sup == 1)).astype(int)

        # 支持終了（切替の逆向き）
        to_sup_L = ((ds != 0) & (sup == 1)).astype(int)  # 左→右へ
        to_sup_R = ((ds != 0) & (sup == 0)).astype(int)  # 右→左へ

        if overwrite or ("support_switch" not in out.columns):
            out["support_switch"] = support_switch
        if overwrite or ("HS_support_L" not in out.columns):
            out["HS_support_L"] = hs_sup_L
        if overwrite or ("HS_support_R" not in out.columns):
            out["HS_support_R"] = hs_sup_R
        if overwrite or ("TO_support_L" not in out.columns):
            out["TO_support_L"] = to_sup_L
        if overwrite or ("TO_support_R" not in out.columns):
            out["TO_support_R"] = to_sup_R

    # contact のHS/TO
    if has_cL:
        hsL, toL = transitions_01(out["contact_L"].to_numpy(dtype=int))
        if overwrite or ("HS_L" not in out.columns):
            out["HS_L"] = hsL
        if overwrite or ("TO_L" not in out.columns):
            out["TO_L"] = toL
    if has_cR:
        hsR, toR = transitions_01(out["contact_R"].to_numpy(dtype=int))
        if overwrite or ("HS_R" not in out.columns):
            out["HS_R"] = hsR
        if overwrite or ("TO_R" not in out.columns):
            out["TO_R"] = toR

    return out


# -----------------------------
# モード解析
# -----------------------------
def parse_modes(modes_str: str) -> Tuple[bool, bool, bool]:
    s = modes_str.replace(" ", "")
    parts = [p for p in s.split(",") if p]
    allowed = {"A", "B", "D"}
    for p in parts:
        if p not in allowed:
            raise ValueError(f"--modes は A,B,D のみ指定可能です。無効: {p}")
    do_A = "A" in parts
    do_B = "B" in parts
    do_D = "D" in parts
    if not (do_A or do_B or do_D):
        raise ValueError("--modes に少なくとも1つ指定してください（例: A または A,B）")
    return do_A, do_B, do_D


def modes_tag(modes: Tuple[bool, bool, bool]) -> str:
    do_A, do_B, do_D = modes
    tag = []
    if do_A:
        tag.append("A")
    if do_B:
        tag.append("B")
    if do_D:
        tag.append("D")
    return "".join(tag)


# -----------------------------
# 剛体並進のΔを作る（A/Bで使用）
# -----------------------------
def make_rigid_delta_A(T: int, sigma_xyz: np.ndarray, rng: np.random.Generator,
                       noise_scale: float, x_scale: float, y_scale: float, z_scale: float) -> np.ndarray:
    """
    微小ジッタ（剛体並進）
    delta: (T, 3)
    """
    axis = np.array([x_scale, y_scale, z_scale], dtype=np.float64)
    delta = rng.normal(0.0, 1.0, size=(T, 3)) * (sigma_xyz[None, :] * float(noise_scale)) * axis[None, :]
    return delta


def make_rigid_delta_B(T: int, sigma_xyz: np.ndarray, rng: np.random.Generator,
                       drift_scale: float, drift_win: int, drift_cap_k: float,
                       x_scale: float, y_scale: float, z_scale: float) -> np.ndarray:
    """
    低周波ドリフト（剛体並進）
    """
    axis = np.array([x_scale, y_scale, z_scale], dtype=np.float64)
    drift = make_lowfreq_drift(T, 3, int(drift_win), rng=rng)
    drift = drift * (sigma_xyz[None, :] * float(drift_scale)) * axis[None, :]

    cap = float(drift_cap_k) * sigma_xyz
    drift = np.clip(drift, -cap[None, :], cap[None, :])
    return drift


def poscols_to_pos6x3(pos: np.ndarray) -> np.ndarray:
    """
    pos: (T, 18) -> (T, 6, 3)
    pos_cols は [L_HIP(x,y,z), R_HIP(x,y,z), L_KNEE(x,y,z), ...] の順を想定
    """
    T = pos.shape[0]
    return pos.reshape(T, 6, 3)


def pos6x3_to_poscols(pos6: np.ndarray) -> np.ndarray:
    """
    (T,6,3) -> (T,18)
    """
    T = pos6.shape[0]
    return pos6.reshape(T, 18)


# -----------------------------
# 拡張本体（A/B/D）
# -----------------------------
def apply_augment_one(
    df: pd.DataFrame,
    pos_cols: List[str],
    discrete_cols: List[str],
    modes: Tuple[bool, bool, bool],
    rng: np.random.Generator,
    *,
    smooth_win: int,
    noise_scale: float,
    drift_scale: float,
    drift_win: int,
    drift_cap_k: float,
    time_warp_rate: float,
    x_scale: float,
    y_scale: float,
    z_scale: float,
    recompute_derived: bool,
    overwrite_derived: bool,
) -> pd.DataFrame:
    """
    1本の系列に対して拡張を適用する。

    A/Bは「剛体並進」で、骨長を壊さない。
    Dは「単調時間伸縮」で、posは補間、離散列は最近傍。
    """
    do_A, do_B, do_D = modes
    out = df.copy()

    pos = out[pos_cols].to_numpy(dtype=np.float64)  # (T,18)
    T = pos.shape[0]
    pos6 = poscols_to_pos6x3(pos)  # (T,6,3)

    # 剛体並進の尺度は「骨盤中心（左右HIPの平均）」の高周波成分から推定する
    hip_center = (pos6[:, 0, :] + pos6[:, 1, :]) * 0.5  # (T,3)
    sigma_xyz = estimate_highfreq_sigma(hip_center, int(smooth_win))  # (3,)

    # A: 微小ジッタ（剛体並進）
    if do_A:
        dA = make_rigid_delta_A(T, sigma_xyz, rng,
                                noise_scale=float(noise_scale),
                                x_scale=float(x_scale), y_scale=float(y_scale), z_scale=float(z_scale))
        pos6 = pos6 + dA[:, None, :]  # 全関節に同じΔを加算

    # B: 低周波ドリフト（剛体並進）
    if do_B:
        dB = make_rigid_delta_B(T, sigma_xyz, rng,
                                drift_scale=float(drift_scale),
                                drift_win=int(drift_win),
                                drift_cap_k=float(drift_cap_k),
                                x_scale=float(x_scale), y_scale=float(y_scale), z_scale=float(z_scale))
        pos6 = pos6 + dB[:, None, :]

    out[pos_cols] = pos6x3_to_poscols(pos6)

    # D: 単調時間伸縮（posは補間、離散列は最近傍）
    if do_D and float(time_warp_rate) > 0.0:
        tau = monotonic_time_warp(T, rate_range=float(time_warp_rate), rng=rng)

        out[pos_cols] = resample_continuous(out[pos_cols].to_numpy(dtype=np.float64), tau)
        for dc in discrete_cols:
            out[dc] = resample_discrete(out[dc].to_numpy(), tau)

        if "frame" in out.columns:
            out["frame"] = np.arange(T, dtype=int)

        dt = infer_dt_time_sec(df)
        if dt is not None and "time_sec" in out.columns:
            out["time_sec"] = out["frame"].to_numpy(dtype=np.float64) * float(dt)

    # 派生列（stance/events）を再計算して内部整合を取る
    if recompute_derived:
        out = recompute_from_contact_support(out, overwrite=overwrite_derived)

    return out


# -----------------------------
# 入力列挙
# -----------------------------
def collect_input_paths(in_csvs: List[str], in_glob: Optional[str]) -> List[str]:
    paths: List[str] = []
    for p in in_csvs:
        if p:
            paths.append(p)
    if in_glob:
        paths.extend(sorted(glob.glob(in_glob)))

    # 重複排除（絶対パスで一意化）
    seen = set()
    uniq = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            uniq.append(p)
            seen.add(ap)
    return uniq


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_csvs", nargs="*", default=[], help="入力CSVを複数指定（スペース区切り）")
    ap.add_argument("--in_glob", default=None, help='入力CSVのglob（例: outputs_jinbutu/*_processed.csv）')

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=4, help="各入力CSVから生成する拡張系列数")
    ap.add_argument("--modes", type=str, default="A", help="A,B,D をカンマ区切り。例: A / A,B / A,B,D / D")
    ap.add_argument("--seed", type=int, default=1234)

    # A/B 強度
    ap.add_argument("--noise_scale", type=float, default=0.5, help="A: 微小ジッタ強度")
    ap.add_argument("--drift_scale", type=float, default=1.0, help="B: 低周波ドリフト強度")

    # D 強度
    ap.add_argument("--time_warp_rate", type=float, default=0.03, help="D: 時間伸縮率（例: 0.03=±3%程度）")

    # 高周波/低周波分離
    ap.add_argument("--smooth_win", type=int, default=9, help="尺度推定用の平滑窓（骨盤中心）")
    ap.add_argument("--drift_win", type=int, default=81, help="ドリフト生成用の平滑窓")
    ap.add_argument("--drift_cap_k", type=float, default=2.5, help="ドリフト上限（sigmaのk倍）")

    # 軸別倍率（xを小さく、zを大きめにしやすい）
    ap.add_argument("--x_scale", type=float, default=0.1)
    ap.add_argument("--y_scale", type=float, default=0.5)
    ap.add_argument("--z_scale", type=float, default=1.0)

    # 既定で派生列を再計算（比較条件統一・内部整合のため）
    ap.add_argument("--no_recompute_derived", action="store_true", help="派生列（stance/events）を再計算しない")
    ap.add_argument("--no_overwrite_derived", action="store_true", help="派生列が既にある場合は上書きしない")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    modes = parse_modes(args.modes)
    tag = modes_tag(modes)

    input_paths = collect_input_paths(args.in_csvs, args.in_glob)
    if len(input_paths) == 0:
        raise ValueError("入力CSVが0件です。--in_csvs または --in_glob を指定してください。")

    recompute_derived = (not bool(args.no_recompute_derived))
    overwrite_derived = (not bool(args.no_overwrite_derived))

    rng0 = np.random.default_rng(int(args.seed))
    total_written = 0

    for in_csv in input_paths:
        df = pd.read_csv(in_csv)

        pos_cols = get_pos_cols_or_raise(df)
        discrete_cols = get_known_discrete_cols(df)

        base_name = os.path.splitext(os.path.basename(in_csv))[0]

        for k in range(int(args.n)):
            seed_try = int(rng0.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed_try)

            df_aug = apply_augment_one(
                df=df,
                pos_cols=pos_cols,
                discrete_cols=discrete_cols,
                modes=modes,
                rng=rng,
                smooth_win=int(args.smooth_win),
                noise_scale=float(args.noise_scale),
                drift_scale=float(args.drift_scale),
                drift_win=int(args.drift_win),
                drift_cap_k=float(args.drift_cap_k),
                time_warp_rate=float(args.time_warp_rate),
                x_scale=float(args.x_scale),
                y_scale=float(args.y_scale),
                z_scale=float(args.z_scale),
                recompute_derived=recompute_derived,
                overwrite_derived=overwrite_derived,
            )

            out_path = os.path.join(args.out_dir, f"{base_name}_aug{k:02d}_{tag}.csv")
            df_aug.to_csv(out_path, index=False)
            total_written += 1

    print(f"done: out_dir={args.out_dir} written={total_written}")


if __name__ == "__main__":
    main()
