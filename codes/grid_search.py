import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")

# --- Import your modules ---
from models import SpectralECFDetector
from statistical_testing import run_detection_pipeline
from baselines import run_midast_ks, run_midast_mmd, run_e_divisive, run_kcpa, run_baseline_ks
from student_t import StudentTConfig, sample_student_t_series
from sub_gaussian import SubGaussianConfig, sample_subgaussian_series

# ==========================================
# 1. EVALUATION HELPERS
# ==========================================
def calc_mae(true_cp, detected_cps):
    """Calculates Mean Absolute Error for the closest detection."""
    if len(detected_cps) == 0:
        return np.nan # Treat missed detections as NaN for clean plotting
    return np.min(np.abs(np.array(detected_cps) - true_cp))

def run_all_models(models_dict, X, dim, true_cp, L=200):
    """Runs all models on a single time series and returns MAEs and execution times."""
    results = {'mae': {}, 'time': {}}
    
    for name, model_info in models_dict.items():
        start_t = time.time()
        try:
            if name == "Neural-ECF":
                # Ensure correct parameters for ECF pipeline
                verified_cps = run_detection_pipeline(
                    model=model_info['model'], X_series=X, L=L, gap=10, 
                    scan_step=1, alpha=0.01, device=model_info['device'], expected_cps=1
                )
                detected = [v[0] for v in verified_cps]
            else:
                detected = model_info['func'](X, dim, target_cps=1)
                
            results['mae'][name] = calc_mae(true_cp, detected)
        except Exception as e:
            results['mae'][name] = np.nan
            
        results['time'][name] = time.time() - start_t
        
    return results

# ==========================================
# 2. EXPERIMENT RUNNERS
# ==========================================
def run_experiment_A_grid(models_dict, dim, dist_type, num_trials, L=200):
    """Generates data for Panel A: Heatmaps (Varying rho2 and alpha2/nu2)."""
    

    rhos = np.arange(-0.9, 1.0, 0.3) 
    
    if dist_type == 'sub_gaussian':
        params_y = [1.5, 1.7, 1.85, 1.95, 1.98] 
    else:
        params_y = [1.5, 3.0, 5.0, 10.0, 15.0]

    results_list = []
    
    total_iters = len(rhos) * len(params_y) * num_trials
    pbar = tqdm(total=total_iters, desc=f"Exp A ({dist_type.upper()}, d={dim})")
    
    for py in params_y:
        for rho in rhos:
            for trial in range(num_trials):
                if dist_type == 'sub_gaussian':
                    config = SubGaussianConfig(n_samples=1000, n_star=500, p=dim, 
                                               rho_pre=0.5, rho_post=rho, 
                                               alpha_pre=1.9, alpha_post=py)
                    X, true_cp = sample_subgaussian_series(config)
                else:
                    config = StudentTConfig(n_samples=1000, n_star=500, p=dim, 
                                            rho_pre=0.5, rho_post=rho, 
                                            nu_pre=2.0, nu_post=py)
                    X, true_cp = sample_student_t_series(config)
                
                res = run_all_models(models_dict, X, dim, true_cp, L)
                
                for m_name in models_dict.keys():
                    results_list.append({
                        'Method': m_name, 'rho2': np.round(rho, 1), 
                        'param_y': py, 'MAE': res['mae'][m_name]
                    })
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(results_list)


def run_experiment_B_position(models_dict, dim, dist_type, num_trials, L=200):
    """Generates data for Panel B: Position of CP vs MAE."""

    n_star_ratios = np.arange(0.1, 1.0, 0.2) 
    
    results_list = []
    
    total_iters = len(n_star_ratios) * num_trials
    pbar = tqdm(total=total_iters, desc=f"Exp B ({dist_type.upper()}, d={dim})")
    
    for ratio in n_star_ratios:
        n_star = int(1000 * ratio)
        for trial in range(num_trials):
            if dist_type == 'sub_gaussian':
                config = SubGaussianConfig(n_samples=1000, n_star=n_star, p=dim)
                X, true_cp = sample_subgaussian_series(config)
            else:
                config = StudentTConfig(n_samples=1000, n_star=n_star, p=dim)
                X, true_cp = sample_student_t_series(config)
            
            res = run_all_models(models_dict, X, dim, true_cp, L)
            
            for m_name in models_dict.keys():
                results_list.append({
                    'Method': m_name, 'n_star_ratio': np.round(ratio, 1), 
                    'MAE': res['mae'][m_name]
                })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(results_list)


def run_experiment_C_time(models_dict, dim, dist_type, num_trials=10, L=200):
    """Generates data for Panel C: Average execution time."""
    time_results = {m: [] for m in models_dict.keys()}
    
    for _ in tqdm(range(num_trials), desc=f"Exp C (Time, {dist_type.upper()}, d={dim})"):
        if dist_type == 'sub_gaussian':
            X, true_cp = sample_subgaussian_series(SubGaussianConfig(p=dim))
        else:
            X, true_cp = sample_student_t_series(StudentTConfig(p=dim))
            
        res = run_all_models(models_dict, X, dim, true_cp, L)
        for m_name, t in res['time'].items():
            time_results[m_name].append(t)
            
    avg_times = {m: np.mean(t_list) for m, t_list in time_results.items()}
    return pd.DataFrame(list(avg_times.items()), columns=['Method', 'Time_s'])

# ==========================================
# 3. PLOTTING ENGINE (Matching Target Image)
# ==========================================
def generate_master_plot(df_A, df_B, df_C, title_prefix, save_path):
    """Constructs the exact layout from the provided image."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1], figure=fig)
    
    # ---------------------------------------------------------
    # PANEL A: Heatmaps Grid (Top row spans entire width)
    # ---------------------------------------------------------
    gs_A = gs[0, :].subgridspec(2, 3, wspace=0.1, hspace=0.3)
    methods_A = ["MIDAST[KS]", "MIDAST[MMD]", "Neural-ECF", "e-Divisive", "KCPA"]
    
    # Fix axes limits based on data
    vmax = df_A['MAE'].quantile(0.95) # Cap at 95th percentile to prevent outlier washout
    
    for idx, method in enumerate(methods_A):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs_A[row, col])
        
        subset = df_A[df_A['Method'] == method].groupby(['param_y', 'rho2'])['MAE'].mean().unstack()
        subset = subset.sort_index(ascending=False) # High values at top
        
        sns.heatmap(subset, ax=ax, cmap="rocket", vmin=0, vmax=vmax, cbar=(col==2 or idx==4))
        ax.set_title(method)
        ax.set_ylabel(r"$\alpha_2$" if "sub_gaussian" in save_path else r"$\nu_2$")
        ax.set_xlabel(r"$\rho_2$")

    # ---------------------------------------------------------
    # PANEL B: MAE vs Position (Bottom Left)
    # ---------------------------------------------------------
    ax_B = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=df_B, x='n_star_ratio', y='MAE', hue='Method', 
                 errorbar='se', ax=ax_B, marker="o")
    ax_B.set_ylabel("MAE $\pm$ SE")
    ax_B.set_xlabel("$n^* / N$")
    ax_B.legend(loc='upper right')
    ax_B.grid(True, alpha=0.3)
    ax_B.set_title("B", loc='left', fontweight='bold', fontsize=14, bbox=dict(facecolor='white', edgecolor='black'))

    # ---------------------------------------------------------
    # PANEL C: Log Computation Time (Bottom Right)
    # ---------------------------------------------------------
    ax_C = fig.add_subplot(gs[1, 1])
    ax_C.plot(df_C['Method'], df_C['Time_s'], marker='o', linewidth=2)
    ax_C.set_yscale('log')
    ax_C.set_ylabel("time (s)")
    ax_C.grid(True, alpha=0.3, axis='y')
    ax_C.set_title("C", loc='left', fontweight='bold', fontsize=14, bbox=dict(facecolor='white', edgecolor='black'))
    plt.xticks(rotation=15)

    # Final touches
    plt.suptitle(title_prefix, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_mc_trials = 100
    L_fixed = 200

    # Ensure output directory exists
    os.makedirs("paper_plots", exist_ok=True)

    for dim in [2, 10]:
        print(f"\n=========================================")
        print(f" INITIALIZING PIPELINE FOR DIMENSION: {dim}")
        print(f"=========================================")
        
        # Load Neural-ECF specifically for this dimension
        ecf_model = SpectralECFDetector(in_channels=dim, M=128).to(device)
        try:
            ecf_model.load_state_dict(torch.load(f"spectral_cnn_d{dim}.pt", map_location=device))
            ecf_model.eval()
        except FileNotFoundError:
            print(f"WARNING: spectral_cnn_d{dim}.pt not found. Neural-ECF will produce random results.")

        # Dictionary of methods to evaluate
        models = {
            "Neural-ECF": {'func': None, 'model': ecf_model, 'device': device},
            "MIDAST[KS]": {'func': run_midast_ks},
            "MIDAST[MMD]": {'func': run_midast_mmd},
            "e-Divisive": {'func': run_e_divisive},
            "KCPA": {'func': run_kcpa}
        }

        for dist in ['student_t', 'sub_gaussian']:
            print(f"\n---> Running suite for {dist.upper()} Data (d={dim})")
            
            # 1. Run Experiments
            df_A = run_experiment_A_grid(models, dim, dist, num_trials=num_mc_trials, L=L_fixed)
            df_B = run_experiment_B_position(models, dim, dist, num_trials=num_mc_trials, L=L_fixed)
            df_C = run_experiment_C_time(models, dim, dist, num_trials=10, L=L_fixed) # Only need ~10 trials for avg time
            
            # 2. Save raw data just in case the plot crashes
            df_A.to_csv(f"paper_plots/raw_data_A_{dist}_d{dim}.csv", index=False)
            df_B.to_csv(f"paper_plots/raw_data_B_{dist}_d{dim}.csv", index=False)
            df_C.to_csv(f"paper_plots/raw_data_C_{dist}_d{dim}.csv", index=False)

            # 3. Generate and save the composite image
            generate_master_plot(
                df_A, df_B, df_C, 
                title_prefix=f"Performance Benchmark: {dist.replace('_', ' ').title()} Distribution (d={dim})",
                save_path=f"paper_plots/benchmark_{dist}_d{dim}.png"
            )
    print("\nAll experiments finished and plots generated successfully.")