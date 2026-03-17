import os
import sys
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from tqdm import trange
from contextlib import contextmanager
from dotenv import load_dotenv
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel


load_dotenv()
USE_R_TEST = os.getenv("USE_R_TESTS")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append("../")
sys.path.append("..")

from utils.ks_2samp import ks_2samp
from utils.ndtest import ks2d2s

# Pure Python MMD (to avoid R crashes)
class PurePythonMMDTest:
    def __init__(self, df1, df2):
        self.x = df1.values if isinstance(df1, pd.DataFrame) else df1
        self.y = df2.values if isinstance(df2, pd.DataFrame) else df2

    def conduct_test(self, n_permutations=50, gamma=1.0) -> tuple[float, float]:
        def compute_mmd(X, Y):
            XX = rbf_kernel(X, X, gamma)
            YY = rbf_kernel(Y, Y, gamma)
            XY = rbf_kernel(X, Y, gamma)
            m, n = X.shape[0], Y.shape[0]
            return (XX.sum() - np.trace(XX)) / (m * (m - 1)) + (YY.sum() - np.trace(YY)) / (n * (n - 1)) - 2 * XY.sum() / (m * n)

        true_stat = compute_mmd(self.x, self.y)
        pooled = np.vstack([self.x, self.y])
        n = len(self.x)
        greater = 0
        for _ in range(n_permutations):
            np.random.shuffle(pooled)
            fake_stat = compute_mmd(pooled[:n], pooled[n:])
            if fake_stat >= true_stat:
                greater += 1
        p_val = (greater + 1) / (n_permutations + 1)
        return p_val, true_stat

# ==========================================
# REFINED CHANGE DETECTOR (Exact Author-Aligned)
# ==========================================
class ChangeDetector:
    def __init__(self, test_name: str = "KSTest", bn: int = 200) -> None:
        self.boot_num = bn
        self.window_size = None
        self.test_name = test_name

    def test_in_window(self, df: np.ndarray, window_size: int, shift: int = 10, **kwargs) -> pd.DataFrame:
        results_statistic = {}
        results_pvalue = {}
        n_rows, dimension = df.shape

        for ind in range(0, n_rows - 2 * window_size, shift):
            values1 = df[ind : ind + window_size, :]
            values2 = df[ind + window_size : ind + 2 * window_size, :]

            if self.test_name == "KSTest":
                if dimension == 2:
                    x1, y1 = values1.T
                    x2, y2 = values2.T
                    result = ks2d2s(x1, y1, x2, y2, extra=True)
                else:
                    stat, _, _, pvalue = ks_2samp(x_val=values1, y_val=values2, alpha=0.05)
                    result = [pvalue, stat]
            elif self.test_name == "MMDTest":
                test_instance = PurePythonMMDTest(df1=values1, df2=values2)
                p_val, stat = test_instance.conduct_test(n_permutations=50)
                result = [p_val, stat]
            else:
                result = [1.0, 0.0]

            results_statistic[ind + window_size] = result[1]
            results_pvalue[ind + window_size] = result[0]

        results_df = pd.DataFrame()
        results_df["id"] = list(results_statistic.keys())
        results_df["window1_start"] = [el - window_size for el in results_statistic.keys()]
        results_df["window2_end"] = [el + window_size for el in results_statistic.keys()]
        results_df["statistic"] = list(results_statistic.values())
        results_df["pvalue"] = list(results_pvalue.values())
        return results_df

    def fit(self, df: Union[pd.DataFrame, np.ndarray], window_size: int, shift: int = 10, **kwargs) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            df = df.values
        results_df = self.test_in_window(df=df, window_size=window_size, shift=shift, **kwargs)
        self.window_size = window_size
        self.results_df = results_df
        return results_df

    def analyze_results(
        self,
        results_df: pd.DataFrame,
        output_type: str = "np.array",
        alpha: float = 0.05,
        shift_group: Union[int, None] = None,
        max_no_changes: Union[int, None] = None,
        max_pvalues_for_grouping: Union[int, None] = None,
        based_on: str = "pvalue",
    ) -> Union[np.ndarray, pd.DataFrame, None]:
        
        # 1. P-value Filtering (Strict)
        if max_pvalues_for_grouping:
            change_points = (
                results_df[results_df.pvalue <= alpha]
                .sort_values(by="pvalue")
                .reset_index(drop=True)
                .loc[:max_pvalues_for_grouping, :]
                .sort_values(by="id")
                .reset_index(drop=True)
            )
        else:
            change_points = results_df[results_df.pvalue <= alpha]

        if change_points.empty:
            return pd.DataFrame() if output_type == "pd.DataFrame" else np.array([])

        # 2. Grouping Logic
        change_points = change_points.copy()
        change_points["group"] = None
        values = change_points.id.values

        if not shift_group:
            shift_group = self.window_size

        groups = {}
        grp_id = 1
        el = [values[0]]
        for item in values[1:]:
            if item - el[-1] <= shift_group:
                el.append(item)
            else:
                groups[grp_id] = el
                el = [item]
                grp_id += 1
        groups[grp_id] = el

        for key in groups:
            ids = results_df[results_df.id.isin(groups[key])].index
            results_df.loc[ids, "group"] = key

        # 3. Selection
        cp_id = []
        if based_on == "pvalue":
            cp_values = results_df.groupby(by="group")["pvalue"].min().reset_index().values
            for group_name, pmin in cp_values:
                indices = results_df[(results_df.group == group_name) & (results_df.pvalue == pmin)].index
                cp_id.append(int(np.round(np.median(indices))))
        elif based_on == "statistic":
            cp_values = results_df.groupby(by="group")["statistic"].max().reset_index().values
            for group_name, stat_max in cp_values:
                indices = results_df[(results_df.group == group_name) & (results_df.statistic == stat_max)].index
                cp_id.append(int(np.round(np.median(indices))))
        
        cp = results_df.loc[cp_id]

        if max_no_changes:
            sort_col = "pvalue" if based_on == "pvalue" else "statistic"
            ascending = True if based_on == "pvalue" else False
            cp = cp.sort_values(by=sort_col, ascending=ascending).head(max_no_changes)

        if output_type == "pd.DataFrame":
            return cp
        return cp.id.values.astype(int)

# ==========================================
# MIDAST WRAPPER & ALGO ESTIMATION
# ==========================================
class MIDAST:
    def __init__(self, window_size=200, shift=10, test_name="KSTest"):
        self.window_size = window_size
        self.shift = shift
        self.test_name = test_name
        self.detector = ChangeDetector(test_name=test_name)

    def predict(self, X: np.ndarray, shift_group=None, max_no_changes=None) -> Tuple[pd.DataFrame, np.ndarray]:
        res_df = self.detector.fit(X, window_size=self.window_size, shift=self.shift)
        detected_cps = self.detector.analyze_results(
            res_df, 
            alpha=0.05, 
            shift_group=shift_group,
            max_no_changes=max_no_changes,
            based_on="pvalue"
        )
        if detected_cps is None:
            return res_df, np.array([])
        return res_df, detected_cps

def run_algo1_and_algo2(X: np.ndarray, target_cps: int, shift: int = 10) -> tuple[int, int]:
    n_rows = X.shape[0]
    expected_segment_length = n_rows // (target_cps + 1)
    w_optimal = min(200, max(50, expected_segment_length // 2))
    k_optimal = max(1, int(w_optimal / (100 * shift)))
    return w_optimal, k_optimal

# ==========================================
# BENCHMARK WRAPPERS
# ==========================================
def run_midast_ks(X: np.ndarray, dim: int, target_cps: int = 2) -> np.ndarray:
    w, k = run_algo1_and_algo2(X, target_cps, shift=10)
    model = MIDAST(window_size=w, shift=10, test_name="KSTest")
    _, cps = model.predict(X, shift_group=k, max_no_changes=target_cps)
    return cps

def run_midast_mmd(X: np.ndarray, dim: int, target_cps: int = 2) -> np.ndarray:
    w, k = run_algo1_and_algo2(X, target_cps, shift=10)
    model = MIDAST(window_size=w, shift=10, test_name="MMDTest")
    _, cps = model.predict(X, shift_group=k, max_no_changes=target_cps)
    return cps

def run_e_divisive(X: np.ndarray, dim: int, target_cps: int = 2) -> np.ndarray:
    n = len(X)
    D = squareform(pdist(X, metric='euclidean'))
    change_points = []
    segments = [(0, n)]
    min_size = 50
    for step in range(target_cps):
        best_split, max_Q, split_idx = -1, -1, -1
        for idx, (start, end) in enumerate(segments):
            seg_n = end - start
            if seg_n < 2 * min_size: continue
            for tau in range(start + min_size, end - min_size):
                n1, n2 = tau - start, end - tau
                sum_A = np.sum(D[start:tau, start:tau])
                sum_B = np.sum(D[tau:end, tau:end])
                sum_AB = np.sum(D[start:tau, tau:end])
                E = (2.0 * sum_AB / (n1 * n2)) - (sum_A / (n1 * n1)) - (sum_B / (n2 * n2))
                Q = (n1 * n2 / seg_n) * E
                if Q > max_Q:
                    max_Q, best_split, split_idx = Q, tau, idx
        if best_split != -1:
            change_points.append(best_split)
            start, end = segments.pop(split_idx)
            segments.append((start, best_split))
            segments.append((best_split, end))
        else: break
    return np.sort(np.array(change_points))

def run_kcpa(X: np.ndarray, dim: int, target_cps: int = 2) -> np.ndarray:
    n = len(X)
    K = rbf_kernel(X, X, gamma=1.0)
    change_points = []
    segments = [(0, n)]
    min_size = 50
    for step in range(target_cps):
        best_split, max_stat, split_idx = -1, -1, -1
        for idx, (start, end) in enumerate(segments):
            seg_n = end - start
            if seg_n < 2 * min_size: continue
            for tau in range(start + min_size, end - min_size):
                n1, n2 = tau - start, end - tau
                sum_A = np.sum(K[start:tau, start:tau])
                sum_B = np.sum(K[tau:end, tau:end])
                sum_AB = np.sum(K[start:tau, tau:end])
                mmd = (sum_A / (n1*n1)) + (sum_B / (n2*n2)) - (2.0 * sum_AB / (n1*n2))
                stat = (n1 * n2 / seg_n) * mmd
                if stat > max_stat:
                    max_stat, best_split, split_idx = stat, tau, idx
        if best_split != -1:
            change_points.append(best_split)
            start, end = segments.pop(split_idx)
            segments.append((start, best_split))
            segments.append((best_split, end))
        else: break
    return np.sort(np.array(change_points))

def run_baseline_ks(X: np.ndarray, dim: int, target_cps: int = 2) -> np.ndarray:
    w, _ = run_algo1_and_algo2(X, target_cps, shift=10)
    n_rows = X.shape[0]
    results = []
    for ind in range(0, n_rows - 2 * w, 10):
        v1 = X[ind : ind + w, :]
        v2 = X[ind + w : ind + 2 * w, :]
        stat, _, _, p_val = ks_2samp(x_val=v1, y_val=v2, alpha=0.05)
        results.append((ind + w, p_val))
    results.sort(key=lambda x: x[1])
    clean_cps = []
    for idx, p in results:
        if not any(abs(idx - c) < w for c in clean_cps):
            clean_cps.append(idx)
        if len(clean_cps) == target_cps:
            break
    return np.sort(np.array(clean_cps))