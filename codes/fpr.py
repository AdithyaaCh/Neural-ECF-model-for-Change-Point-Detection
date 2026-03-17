import torch
import numpy as np
import pandas as pd  # Explicitly here so we don't forget!
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
# Import your model and generators
from models import SpectralECFDetector
from student_t import generate_student_t_segment
from sub_gaussian import generate_subgaussian_segment
from statistical_testing import run_detection_pipeline

def run_comprehensive_fpr_experiment():
    """
    Evaluates Empirical FPR across Student-t and Sub-Gaussian distributions.
    Saves raw data to CSV immediately to prevent loss from plotting crashes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    dimensions = [2, 10]
    distributions = ['student_t', 'sub_gaussian']
    n_trials = 1000 
    alpha_level = 0.01 
    
    L, gap, scan_step = 200, 10, 1
    
    all_results = []
    all_fp_locations = {}

    print(f"--- Starting Comprehensive Null Hypothesis Analysis ---")
    os.makedirs("paper_plots", exist_ok=True)

    for dist in distributions:
        print(f"\n>>> Analyzing Distribution: {dist.upper()}")
        for d in dimensions:
            print(f"Testing Dimension d={d}...")
            
            model = SpectralECFDetector(in_channels=d, M=128).to(device)
            try:
                # Assuming weights were saved from your previous training runs
                model.load_state_dict(torch.load(f"spectral_cnn_d{d}.pt", map_location=device))
                model.eval()
            except FileNotFoundError:
                print(f"  [!] Weights for d={d} not found. Using random init.")

            trials_with_no_cps = 0
            fp_locs = []
            
            for trial in range(n_trials):
                # Generate Stationary Null Data
                if dist == 'student_t':
                    X_null = generate_student_t_segment(nu=3.0, rho=0.5, n=1000, p=d)
                else:
                    X_null = generate_subgaussian_segment(alpha=1.8, rho=0.5, n=1000, p=d)
                
                detections = run_detection_pipeline(
                    model=model, X_series=X_null, L=L, gap=gap, 
                    scan_step=scan_step, alpha=alpha_level, 
                    device=device, expected_cps=None 
                )
                
                if len(detections) == 0:
                    trials_with_no_cps += 1
                else:
                    fp_locs.extend([det[0] for det in detections])
                
                if (trial + 1) % 100 == 0:
                    print(f"    - Trial {trial+1}/{n_trials} complete...")
            
            success_rate = (trials_with_no_cps / n_trials) * 100
            fpr = 100.0 - success_rate
            
            all_results.append({
                'Distribution': dist,
                'Dimension': f"d={d}",
                'FPR': fpr
            })
            all_fp_locations[f"{dist}_d{d}"] = fp_locs

    # --- CRITICAL SAFETY: Save CSV before plotting ---
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("paper_plots/comprehensive_fpr_results.csv", index=False)
    print("\n[✓] Raw results saved to paper_plots/comprehensive_fpr_results.csv")

    plot_dual_dist_fpr(summary_df, all_fp_locations, n_trials)

def plot_dual_dist_fpr(df, fp_locations, n_trials):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    sns.set_theme(style="whitegrid")

    # --- PANEL A: FPR Comparison ---
    ax1 = fig.add_subplot(gs[0])
    sns.barplot(data=df, x='Dimension', y='FPR', hue='Distribution', 
                palette="viridis", ax=ax1, edgecolor='black', alpha=0.8)
    
    ax1.set_title(f"Empirical False Positive Rate (FPR) Comparison\n$N={n_trials}$ Monte Carlo Trials", 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel("False Positive Rate (%)", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Data Dimension", fontweight='bold', fontsize=14)
    
    # Text annotation for the bars
    for p in ax1.patches:
        if p.get_height() > 0:
            ax1.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 9), 
                         textcoords='offset points', fontweight='bold')

    # --- PANEL B: Spatio-Temporal FP Density ---
    ax2 = fig.add_subplot(gs[1])
    # Filter out empty lists to avoid KDE errors
    plot_labels = [k for k, v in fp_locations.items() if len(v) > 0]
    palette = sns.color_palette("husl", len(plot_labels))
    
    if not plot_labels:
        ax2.text(0.5, 0.5, "No False Positives Detected across all trials.", 
                 ha='center', va='center', fontsize=14, color='gray')
    else:
        for i, label in enumerate(plot_labels):
            locs = fp_locations[label]
            sns.kdeplot(locs, label=label.replace('_', ' ').title(), 
                        ax=ax2, color=palette[i], fill=True, alpha=0.1)
            ax2.scatter(locs, [0]*len(locs), marker='|', color=palette[i], alpha=0.4)

    ax2.set_title("Temporal Distribution of Spurious Detections (Type I Errors)", 
                  fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel("Time Index ($t$)", fontweight='bold', fontsize=14)
    ax2.set_ylabel("Kernel Density Estimate", fontweight='bold', fontsize=14)
    ax2.set_xlim(0, 1000)
    if plot_labels:
        ax2.legend(title="Config", frameon=True, shadow=True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35)
    
    save_path = "paper_plots/comprehensive_fpr_diagnostic.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[✓] Diagnostic plot saved: {save_path}")

if __name__ == "__main__":
    run_comprehensive_fpr_experiment()