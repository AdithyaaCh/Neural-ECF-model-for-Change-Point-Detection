"""
ECF vs MIDAST  --  Sub-Gaussian, d=10, single change point.   [ s = 1 variant ]

Same experiment as `ecf_vs_midast_subgaussian.py`, but MIDAST is run with the
paper's *recommended* shift and grouping (Appendix A):

    * shift   s = 1            -- "using s = 1 is generally recommended ... for
                                  window lengths w <= 500, using s = 1 is
                                  typically feasible."
    * grouping k = w/(100*s)   -- the paper's rule-of-thumb (NOT Algorithm 2).

"""

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
from common.algorithm12 import algorithm_1_window, k_rule_of_thumb
from common.data_simulators import generate_subgaussian_segment


SMOKE = bool(os.getenv("SMOKE"))
DIM = int(os.getenv("DIM", 10))
DIST = "sub_gaussian"
N = 1000
WINDOW = 150            
GAP = 10
SCAN_STEP = 5           
SMOOTH = 5
M_FREQ = 256
SCALES = (0.5, 1.0, 2.0)
SEED = 0

MIDAST_SHIFT = int(os.getenv("MIDAST_SHIFT", 1))   # s = 1 per Appendix A

OUTDIR = os.path.join(ROOT, f"results_s1_subg_d{DIM}")
os.makedirs(OUTDIR, exist_ok=True)

if SMOKE:
    N_TRIALS_A = 2
    N_TIMING = 2
    RHO2_GRID = np.array([-0.9, 0.0, 0.9])
    ALPHA2_GRID = [1.5, 1.95]
else:
    N_TRIALS_A = int(os.getenv("TRIALS_A", 30))
    N_TIMING = int(os.getenv("TIMING", 10))
    RHO2_GRID = np.round(np.arange(-0.9, 1.0, 0.3), 1)
    ALPHA2_GRID = [1.5, 1.7, 1.85, 1.95, 1.98]

RHO1 = 0.5
ALPHA1 = 1.9

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


def _select_w_k_s1(test_name):
    """w via Algorithm 1 (cached; same regime as the s=10 baseline -> reuses the
    alg1 cache, w=50 for sub_gaussian d=10). k via the paper rule of thumb
    k = w/(100*s) with s = MIDAST_SHIFT. Algorithm 2 is intentionally NOT used."""
    rho_pre, rho_post = 0.5, 0.0
    tail_pre, tail_post = (1.95, 1.6) if DIST == "sub_gaussian" else (3.0, 8.0)

    calib_test = "KSTest"
    w = algorithm_1_window(
        test_name=calib_test, dist=DIST, dim=DIM,
        rho_pre=rho_pre, rho_post=rho_post,
        tail_pre=tail_pre, tail_post=tail_post,
        M=200, target_power=0.9, c=0.05,
    )
    k = k_rule_of_thumb(w, MIDAST_SHIFT)
    return int(w), int(k)


def run_midast(X, test_name):
    """MIDAST via the vendored ChangeDetector with shift s=1 and rule-of-thumb k.
    Single change point -> max_no_changes=1, decision based_on='statistic'."""
    w, k = _select_w_k_s1("KSTest" if test_name == "KSTest" else "MMDTest")
    s = MIDAST_SHIFT
    shift_group = max(1, int(k * w / 100 * s))

    det = ChangeDetector(test_name=test_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_df = det.fit(X, window_size=w, shift=s)
    cps = det.analyze_results(
        res_df, output_type="np.array", alpha=0.05,
        shift_group=shift_group, max_no_changes=1, based_on="statistic",
    )
    if cps is None:
        return None
    cps = np.asarray(cps)
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



def _append_cell(path, rows):
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

    print(f"\n=== Experiment A [s={MIDAST_SHIFT}] : {len(ALPHA2_GRID)}x{len(RHO2_GRID)} grid, "
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



def timing_benchmark(ecf, n=N_TIMING):
    methods = ["Neural-ECF", "MIDAST[KS]"] + (["MIDAST[MMD]"] if HAVE_MMD else [])
    print(f"\n=== Runtime benchmark [s={MIDAST_SHIFT}] : {n} series, methods={methods} ===")
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
    ax.set_title(f"Runtime per series, Sub-Gaussian d={DIM}  (MIDAST s={MIDAST_SHIFT})",
                 fontweight="bold")
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
              f"Exp A  -  Localisation Error (MAE), Sub-Gaussian d={DIM}  (MIDAST s={MIDAST_SHIFT})",
              fmt=".0f")
    _heatmaps(df, "time_s", "mako_r", "expA_time_heatmaps.png",
              f"Exp A  -  Runtime per series (s)  (MIDAST s={MIDAST_SHIFT})", fmt=".2f")

def summarize(df_A):
    print("\n=================  SUMMARY  (Exp A, s=%d)  =================" % MIDAST_SHIFT)
    g = df_A.groupby("Method").agg(
        MAE_mean=("MAE", "mean"),
        MAE_median=("MAE", "median"),
        detect_rate=("MAE", lambda s: 100.0 * s.notna().mean()),
        time_mean=("time_s", "mean"),
    ).reindex(METHODS)
    print(g.round(3).to_string())

    am = df_A[df_A.Method == "Neural-ECF[argmax]"]["MAE"].mean()
    pk = df_A[df_A.Method == "Neural-ECF[peaks]"]["MAE"].mean()
    winner = "argmax" if am <= pk else "peaks"
    print(f"\n[ECF cp=1 extraction]  argmax MAE={am:.2f}  vs  "
          f"peaks MAE={pk:.2f}  ->  use **{winner}** for single CP.")
    return winner



def prewarm_calibration():
    """Read the Algorithm-1 window cache once in the parent (warm -> instant) so
    workers don't race on it. With the rule-of-thumb k there is NO Algorithm 2."""
    print("\n=== Pre-warming MIDAST[KS] Algorithm 1 window cache (one-time) ===",
          flush=True)
    t0 = time.time()
    w, k = _select_w_k_s1("KSTest")
    s = MIDAST_SHIFT
    print(f"  w={w}, k=w/(100*s)={k}, s={s}, shift_group={max(1, int(k * w / 100 * s))}  "
          f"(ready in {time.time() - t0:.1f}s)", flush=True)


def main():
    np.random.seed(SEED)
    print(f"Mode: {'SMOKE' if SMOKE else 'FULL'} | d={DIM} | MIDAST shift s={MIDAST_SHIFT} | "
          f"methods: {METHODS} | workers={N_WORKERS}", flush=True)
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

    plot_A(df_A)
    summarize(df_A)
    print(f"\nDone. Artifacts in {OUTDIR}")


if __name__ == "__main__":
    main()
