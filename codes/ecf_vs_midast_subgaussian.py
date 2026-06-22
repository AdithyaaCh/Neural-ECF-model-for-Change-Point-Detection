

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

NO_MMD = bool(os.getenv("NO_MMD"))
if not NO_MMD:
    os.environ["USE_R_TESTS"] = "True"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)  
sys.path.insert(0, HERE)   

from trajectory_utils import stblrnd
from vendored_midast.multivariate_statistical_test_method import ChangeDetector
from midast_runner import run_midast_ks, midast_mmd_dispatch
from common.data_simulators import generate_subgaussian_segment

SMOKE = bool(os.getenv("SMOKE"))


DIM = int(os.getenv("DIM", 10))  
N = 1000                 # series length
WINDOW = 150           
SHIFT = 10               # MIDAST sliding shift
GAP = 10                 # ECF past/future gap
SCAN_STEP = 5            # ECF scan stride
SMOOTH = 5               # moving-average width on the ECF score curve
ALPHA_TEST = 0.05        # MIDAST significance level
M_FREQ = 256             # number of ECF frequencies (variance of the noise floor ~1/M)
SCALES = (0.5, 1.0, 2.0)  # multi-scale ECF bandwidths
SEED = 0

OUTDIR = os.path.join(ROOT, f"results_subg_d{DIM}")
os.makedirs(OUTDIR, exist_ok=True)

if SMOKE:
    N_TRIALS_A = 2
    N_TRIALS_B = 2
    N_TIMING = 2
    RHO2_GRID = np.array([-0.9, 0.0, 0.9])
    ALPHA2_GRID = [1.5, 1.95]
    RATIO_GRID = np.array([0.3, 0.5, 0.7])
else:
    N_TRIALS_A = int(os.getenv("TRIALS_A", 30))
    N_TRIALS_B = int(os.getenv("TRIALS_B", 50))
    N_TIMING = int(os.getenv("TIMING", 10))
    RHO2_GRID = np.round(np.arange(-0.9, 1.0, 0.3), 1)
    ALPHA2_GRID = [1.5, 1.7, 1.85, 1.95, 1.98]
    RATIO_GRID = np.round(np.arange(0.1, 1.0, 0.2), 1)

RHO1 = 0.5
ALPHA1 = 1.9
B_ALPHA2 = 1.5
B_RHO2 = -0.5

METHODS = ["MIDAST[KS]", "Neural-ECF[argmax]", "Neural-ECF[peaks]"]

N_WORKERS = int(os.getenv("WORKERS", min(os.cpu_count() or 4, 8)))

HAVE_MMD = False
if not NO_MMD:
    try:
        from vendored_midast.multivariate_tests_from_R import MMDTest  
        HAVE_MMD = True
    except Exception as e: 
        print(f"[!] MMD/R unavailable ({str(e)[:80]}); skipping MMD timing.")


def make_subgaussian(n, n_star, rho1, rho2, alpha1, alpha2, d=DIM):
    seg1 = generate_subgaussian_segment(alpha=alpha1, rho=rho1, n=n_star, p=d)
    seg2 = generate_subgaussian_segment(alpha=alpha2, rho=rho2, n=n - n_star, p=d)
    return np.vstack([seg1, seg2]), n_star


class FixedECF:
    def __init__(self, d, M=M_FREQ, scales=SCALES, seed=SEED):
        rng = np.random.default_rng(seed)
        self.U = rng.standard_normal((M, d)) / np.sqrt(d)
        self.scales = scales

    @staticmethod
    def _robust_standardize(X):
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        return (X - med) / scale

    def fingerprint(self, win):
        """win: (L, d) -> L2-normalised multi-scale ECF vector."""
        feats = []
        for s in self.scales:
            S = win @ (self.U * s).T            
            feats.append(np.cos(S).mean(axis=0))
            feats.append(np.sin(S).mean(axis=0))
        z = np.concatenate(feats)
        return z / (np.linalg.norm(z) + 1e-12)

    def score_series(self, X, L=WINDOW, gap=GAP, step=SCAN_STEP, smooth=SMOOTH):
        Xs = self._robust_standardize(X)
        n = len(Xs)
        idx = np.arange(L, n - L - gap, step)
        scores = np.empty(len(idx))
        for i, t in enumerate(idx):
            zp = self.fingerprint(Xs[t - L:t])
            zf = self.fingerprint(Xs[t + gap:t + L + gap])
            scores[i] = 1.0 - float(zp @ zf)
        if smooth > 1 and len(scores) >= smooth:
            scores = np.convolve(scores, np.ones(smooth) / smooth, mode="same")
        return idx, scores

    @staticmethod
    def _extract(idx, scores, mode):
        if len(scores) == 0:
            return None
        if mode == "argmax":
            return int(idx[int(np.argmax(scores))])
        peaks, props = find_peaks(scores, prominence=1e-6, distance=WINDOW // SCAN_STEP)
        if len(peaks) == 0:
            return int(idx[int(np.argmax(scores))])
        best = peaks[int(np.argmax(props["prominences"]))]
        return int(idx[best])

    def detect(self, X, mode="argmax"):
        idx, scores = self.score_series(X)
        return self._extract(idx, scores, mode)



def run_midast(X, test_name):

    if test_name == "KSTest":
        cps = run_midast_ks(X, dim=DIM, target_cps=1, dist="sub_gaussian",
                            fast=SMOKE)
    else:
        cps = midast_mmd_dispatch(X, dim=DIM, target_cps=1,
                                  dist="sub_gaussian", fast=True)
    cps = np.asarray(cps) if cps is not None else np.array([])
    return None if cps.size == 0 else int(cps[0])


def mae(true_cp, pred):
    return np.nan if pred is None else abs(pred - true_cp)

def evaluate_all(X, true_cp, ecf):
    out = {}
    if any(m.startswith("Neural-ECF") for m in METHODS):
        t0 = time.time()
        try:
            idx, scores = ecf.score_series(X)
        except Exception:
            idx, scores = np.array([]), np.array([])
        scan_t = time.time() - t0
        if "Neural-ECF[argmax]" in METHODS:
            out["Neural-ECF[argmax]"] = (
                mae(true_cp, ecf._extract(idx, scores, "argmax")), scan_t)
        if "Neural-ECF[peaks]" in METHODS:
            out["Neural-ECF[peaks]"] = (
                mae(true_cp, ecf._extract(idx, scores, "peaks")), scan_t)

    for name in METHODS:
        if name.startswith("Neural-ECF"):
            continue
        t0 = time.time()
        try:
            pred = run_midast(X, "KSTest" if name == "MIDAST[KS]" else "MMDTest")
        except Exception:
            pred = None
        out[name] = (mae(true_cp, pred), time.time() - t0)
    return out

def _cell_worker_A(args):
    a2, r2, n_trials, base_seed = args
    ecf = FixedECF(d=DIM)
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_subgaussian(N, N // 2, RHO1, r2, ALPHA1, a2)
        res = evaluate_all(X, cp, ecf)
        for m, (e, t) in res.items():
            rows.append({"Method": m, "rho2": r2, "alpha2": a2, "MAE": e, "time_s": t})
    return rows


def _cell_worker_B(args):
    ratio, n_trials, base_seed = args
    ecf = FixedECF(d=DIM)
    n_star = int(N * ratio)
    rows = []
    for trial in range(n_trials):
        np.random.seed(base_seed + trial)
        X, cp = make_subgaussian(N, n_star, RHO1, B_RHO2, ALPHA1, B_ALPHA2)
        res = evaluate_all(X, cp, ecf)
        for m, (e, t) in res.items():
            rows.append({"Method": m, "n_star_ratio": ratio, "MAE": e, "time_s": t})
    return rows

def _append_cell(path, rows):
    """Append one cell's rows to a partial CSV (write header only once). This is
    the crash-checkpoint: if the run is killed mid-grid we keep every finished
    cell and resume the rest on the next launch."""
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def experiment_A(ecf):
    total_cells = len(ALPHA2_GRID) * len(RHO2_GRID)
    total_series = total_cells * N_TRIALS_A
    n_workers = min(N_WORKERS, total_cells)
    partial = os.path.join(OUTDIR, "raw_A_partial.csv")

    cells = [
        (a2, r2, N_TRIALS_A, idx * N_TRIALS_A)
        for idx, (a2, r2) in enumerate(
            (a, r) for a in ALPHA2_GRID for r in RHO2_GRID)
    ]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(map(tuple, prev[["alpha2", "rho2"]].drop_duplicates().values))
    todo = [c for c in cells if (c[0], c[1]) not in done_keys]

    print(f"\n=== Experiment A : {len(ALPHA2_GRID)}x{len(RHO2_GRID)} grid, "
          f"{N_TRIALS_A} trials/cell ({total_series} series), {n_workers} workers"
          f"  [resume: {len(done_keys)} cells done, {len(todo)} to go] ===",
          flush=True)

    completed = len(done_keys)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cell_worker_A, c): c for c in todo}
        for fut in as_completed(futures):
            rows = fut.result()
            _append_cell(partial, rows)       
            all_rows.extend(rows)
            completed += 1
            a2, r2 = futures[fut][0], futures[fut][1]
            print(f"  alpha2={a2:<5} rho2={r2:<5}  [{completed}/{total_cells} cells done]",
                  flush=True)
    return pd.DataFrame(all_rows)


def experiment_B(ecf):
    n_workers = min(N_WORKERS, len(RATIO_GRID))
    total_series = len(RATIO_GRID) * N_TRIALS_B
    partial = os.path.join(OUTDIR, "raw_B_partial.csv")

    cells = [
        (ratio, N_TRIALS_B, idx * N_TRIALS_B)
        for idx, ratio in enumerate(RATIO_GRID)
    ]

    all_rows, done_keys = [], set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        all_rows = prev.to_dict("records")
        done_keys = set(prev["n_star_ratio"].unique().tolist())
    todo = [c for c in cells if c[0] not in done_keys]

    print(f"\n=== Experiment B : {len(RATIO_GRID)} positions, {N_TRIALS_B} trials each "
          f"({total_series} series), {n_workers} workers"
          f"  [resume: {len(done_keys)} done, {len(todo)} to go] ===", flush=True)

    completed = len(done_keys)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cell_worker_B, c): c for c in todo}
        for fut in as_completed(futures):
            rows = fut.result()
            _append_cell(partial, rows)       
            all_rows.extend(rows)
            completed += 1
            ratio = futures[fut][0]
            print(f"  ratio={ratio:<4}  [{completed}/{len(RATIO_GRID)} positions done]",
                  flush=True)
    return pd.DataFrame(all_rows)


def timing_benchmark(ecf, n=N_TIMING):
    methods = ["Neural-ECF", "MIDAST[KS]"] + (["MIDAST[MMD]"] if HAVE_MMD else [])
    print(f"\n=== Runtime benchmark : {n} series, methods={methods} ===")
    rows = []
    for k in range(n):
        X, _ = make_subgaussian(N, N // 2, RHO1, -0.5, ALPHA1, 1.5)
        t0 = time.time(); ecf.score_series(X); rows.append(("Neural-ECF", time.time() - t0))
        t0 = time.time(); run_midast(X, "KSTest"); rows.append(("MIDAST[KS]", time.time() - t0))
        if HAVE_MMD:
            t0 = time.time(); run_midast(X, "MMDTest"); rows.append(("MIDAST[MMD]", time.time() - t0))
        print(f"  series {k + 1}/{n} timed")
    df = pd.DataFrame(rows, columns=["Method", "time_s"])
    df.to_csv(os.path.join(OUTDIR, "timing_benchmark.csv"), index=False)

    agg = df.groupby("Method")["time_s"].mean().reindex(methods)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(agg.index, agg.values,
                  color=["#2a9d8f", "#e76f51", "#6a4c93"][:len(agg)],
                  edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime per series (s, log)", fontweight="bold")
    ax.set_title(f"Runtime per series, Sub-Gaussian d={DIM}", fontweight="bold")
    for b, v in zip(bars, agg.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s",
                ha="center", va="bottom", fontweight="bold")
    base = agg.get("Neural-ECF", np.nan)
    speedups = "  ".join(f"{m}: {agg[m] / base:.0f}x" for m in agg.index if m != "Neural-ECF")
    ax.text(0.5, 0.02, f"ECF speed-up  ->  {speedups}", transform=ax.transAxes,
            ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    p = os.path.join(OUTDIR, "timing_benchmark.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  [plot] {p}")
    print("  mean runtimes (s):", {m: round(float(agg[m]), 3) for m in agg.index})


def _heatmaps(df, value, cmap, fname, title, fmt=".0f"):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 4),
                             squeeze=False)
    vmax = np.nanquantile(df[value], 0.95)
    for ax, m in zip(axes[0], METHODS):
        piv = (df[df.Method == m].groupby(["alpha2", "rho2"])[value]
               .mean().unstack().sort_index(ascending=False))
        sns.heatmap(piv, ax=ax, cmap=cmap, vmin=0, vmax=vmax, annot=True,
                    fmt=fmt, annot_kws={"size": 7}, cbar=True)
        ax.set_title(m, fontweight="bold")
        ax.set_xlabel(r"$\rho_2$")
        ax.set_ylabel(r"$\alpha_2$")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTDIR, fname)
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p}")


def plot_A(df):
    _heatmaps(df, "MAE", "rocket_r", "expA_mae_heatmaps.png",
              f"Exp A  -  Localisation Error (MAE), Sub-Gaussian d={DIM}", fmt=".0f")
    _heatmaps(df, "time_s", "mako_r", "expA_time_heatmaps.png",
              "Exp A  -  Runtime per series (s)", fmt=".2f")


def plot_B(df):
    sns.set_theme(style="whitegrid")
    # MAE vs position
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="n_star_ratio", y="MAE", hue="Method",
                 marker="o", errorbar="se", ax=ax, linewidth=2.2)
    ax.set_xlabel(r"Change-point position $n^*/N$", fontweight="bold")
    ax.set_ylabel(r"MAE $\pm$ SE", fontweight="bold")
    ax.set_title(f"Exp B  -  MAE vs change-point position (Sub-Gaussian d={DIM})",
                 fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTDIR, "expB_mae.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  [plot] {p}")
    # Runtime vs position
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="n_star_ratio", y="time_s", hue="Method",
                 marker="s", errorbar="se", ax=ax, linewidth=2.2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Change-point position $n^*/N$", fontweight="bold")
    ax.set_ylabel("Runtime per series (s, log)", fontweight="bold")
    ax.set_title("Exp B  -  Runtime vs change-point position",
                 fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUTDIR, "expB_time.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  [plot] {p}")

def summarize(df_A, df_B):
    print("\n=================  SUMMARY  =================")
    for label, df in [("Exp A", df_A), ("Exp B", df_B)]:

        g = df.groupby("Method").agg(
            MAE_mean=("MAE", "mean"),
            MAE_median=("MAE", "median"),
            detect_rate=("MAE", lambda s: 100.0 * s.notna().mean()),
            time_mean=("time_s", "mean"),
        ).reindex(METHODS)
        print(f"\n{label}")
        print(g.round(3).to_string())

    pooled = pd.concat([df_A, df_B], ignore_index=True)
    am = pooled[pooled.Method == "Neural-ECF[argmax]"]["MAE"].mean()
    pk = pooled[pooled.Method == "Neural-ECF[peaks]"]["MAE"].mean()
    winner = "argmax" if am <= pk else "peaks"
    print(f"\n[ECF cp=1 extraction]  argmax MAE={am:.2f}  vs  "
          f"peaks MAE={pk:.2f}  ->  use **{winner}** for single CP.")
    return winner


def prewarm_calibration():

    print("\n=== Pre-warming MIDAST[KS] Algorithm 1/2 calibration (one-time, "
          "cached) ===", flush=True)
    t0 = time.time()
    X, _ = make_subgaussian(N, N // 2, RHO1, -0.5, ALPHA1, 1.5)
    run_midast(X, "KSTest")
    print(f"  calibration ready in {time.time() - t0:.1f}s "
          f"(cached in checkpoints/calib/)", flush=True)


def main():
    np.random.seed(SEED)
    print(f"Mode: {'SMOKE' if SMOKE else 'FULL'} | d={DIM} | methods: {METHODS} | "
          f"workers={N_WORKERS}", flush=True)
    ecf = FixedECF(d=DIM)

    prewarm_calibration()
    timing_benchmark(ecf)

    skip_A = bool(os.getenv("SKIP_A"))
    raw_A_path = os.path.join(OUTDIR, "raw_A.csv")
    if skip_A and os.path.exists(raw_A_path):
        df_A = pd.read_csv(raw_A_path)
        print(f"Skipping Exp A — loaded {len(df_A)} rows from {raw_A_path}", flush=True)
    else:
        df_A = experiment_A(ecf)
        df_A.to_csv(raw_A_path, index=False)

    df_B = experiment_B(ecf)
    df_B.to_csv(os.path.join(OUTDIR, "raw_B.csv"), index=False)

    plot_A(df_A)
    plot_B(df_B)
    summarize(df_A, df_B)
    print(f"\nDone. Artifacts in {OUTDIR}")


if __name__ == "__main__":
    main()
