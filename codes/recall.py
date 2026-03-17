import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_recall_benchmarks():
    # Define the datasets and dimensions based on your generated files
    distributions = ['student_t', 'sub_gaussian']
    dimensions = [2, 10]
    
    # We use a different colormap (e.g., 'mako' or 'viridis') to visually distinguish 
    # the Recall plots from your earlier MAE 'rocket' plots.
    cmap_choice = "mako"

    for dist in distributions:
        for dim in dimensions:
            print(f"Processing Recall for {dist.upper()} (d={dim})...")
            
            # 1. Load the raw data from your 2-day run
            file_A = f"paper_plots/raw_data_A_{dist}_d{dim}.csv"
            file_B = f"paper_plots/raw_data_B_{dist}_d{dim}.csv"
            
            if not os.path.exists(file_A) or not os.path.exists(file_B):
                print(f"  [!] Missing data files for {dist} d={dim}. Skipping.")
                continue
                
            df_A = pd.read_csv(file_A)
            df_B = pd.read_csv(file_B)
            
            # 2. Calculate Detection Success (1 if MAE is not NaN, 0 if NaN)
            # Multiplying by 100 converts it to a clean 0-100% percentage
            df_A['Recall_%'] = df_A['MAE'].notna().astype(float) * 100
            df_B['Recall_%'] = df_B['MAE'].notna().astype(float) * 100

            # 3. Initialize the Figure (Panel A on top, Panel B on bottom)
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(2, 1, height_ratios=[1.5, 1], figure=fig)
            
            # ---------------------------------------------------------
            # PANEL A: Recall Heatmaps Grid
            # ---------------------------------------------------------
            gs_A = gs[0].subgridspec(2, 3, wspace=0.15, hspace=0.3)
            methods = ["MIDAST[KS]", "MIDAST[MMD]", "Neural-ECF", "e-Divisive", "KCPA"]
            
            for idx, method in enumerate(methods):
                row = idx // 3
                col = idx % 3
                ax = fig.add_subplot(gs_A[row, col])
                
                # Filter for the specific method and calculate mean recall per grid point
                subset = df_A[df_A['Method'] == method].groupby(['param_y', 'rho2'])['Recall_%'].mean().unstack()
                subset = subset.sort_index(ascending=False)
                
                # Plot heatmap (fixed 0 to 100 range)
                sns.heatmap(subset, ax=ax, cmap=cmap_choice, vmin=0, vmax=100, 
                            cbar=(col==2 or idx==4), cbar_kws={'label': 'Recall %'})
                ax.set_title(method, fontweight='bold')
                ax.set_ylabel(r"$\alpha_2$" if dist == "sub_gaussian" else r"$\nu_2$")
                ax.set_xlabel(r"$\rho_2$")

            # ---------------------------------------------------------
            # PANEL B: Recall vs Position (Spanning the bottom)
            # ---------------------------------------------------------
            ax_B = fig.add_subplot(gs[1])
            
            # Seaborn's lineplot automatically calculates the mean and confidence intervals 
            # for the binary 0/100 data, giving us exactly what we need.
            sns.lineplot(data=df_B, x='n_star_ratio', y='Recall_%', hue='Method', 
                         errorbar=('ci', 95), ax=ax_B, marker="o", linewidth=2.5, markersize=8)
            
            ax_B.set_ylabel("Detection Recall (%)", fontweight='bold')
            ax_B.set_xlabel("True Change Point Position ($n^* / N$)", fontweight='bold')
            ax_B.set_ylim(-5, 105) # Add a tiny bit of padding so 0 and 100 dots don't get cut off
            ax_B.grid(True, alpha=0.3, linestyle='--')
            
            # Move legend outside the plot to prevent covering data
            ax_B.legend(title='Method', loc='upper right', bbox_to_anchor=(1.1, 1))
            
            # Formatting
            title_name = dist.replace('_', ' ').title()
            plt.suptitle(f"Recall Benchmark: {title_name} Distribution (d={dim})", fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            # Save the new plot
            save_path = f"paper_plots/recall_benchmark_{dist}_d{dim}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [✓] Saved plot to {save_path}")

if __name__ == "__main__":
    plot_recall_benchmarks()