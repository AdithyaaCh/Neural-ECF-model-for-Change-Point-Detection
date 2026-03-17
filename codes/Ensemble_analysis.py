import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from models import SpectralECFDetector
from student_t import StudentTConfig, sample_student_t_series
from data_generator import TripletDataGenerator

# ========================================================
# 1. FAST ENSEMBLE TRAINING
# ========================================================
def train_quick_model(dim, device, steps=200, L=100):
    """Trains a single instance of Neural-ECF from scratch."""
    model = SpectralECFDetector(in_channels=dim, M=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    generator = TripletDataGenerator(p=dim, L=L)
    triplet_loss = nn.TripletMarginLoss(margin=0.4, p=2)
    
    model.train()
    for _ in range(steps):
        batch = generator.generate_triplet_batch(batch_size=32)
        anc, pos, neg = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        za, zp, zn = model(anc, pos, neg)
        
        # Combined Triplet + Sharpness Loss
        loss_t = triplet_loss(za, zp, zn)
        loss_s = 1.0 - torch.sum(za * zp, dim=-1).mean()
        loss = loss_t + 0.5 * loss_s
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

# ========================================================
# 2. ENSEMBLE ANALYSIS FUNCTIONS
# ========================================================
def analyze_ensemble_U_matrix(models):
    """Analyzes spectral spread and rank collapse across ALL models."""
    all_s_learn = []
    all_v_learn = []
    
    for model in models:
        U_learned = model.U.detach().cpu().numpy()
        
        # Singular Values
        _, S_learned, _ = svd(U_learned)
        all_s_learn.append(S_learned / S_learned[0]) # Normalize
        
        # Cosine Similarities
        sim_learned = cosine_similarity(U_learned)
        triu_idx = np.triu_indices_from(sim_learned, k=1)
        all_v_learn.append(np.abs(sim_learned[triu_idx]))
        
    # Generate a random baseline for comparison
    U_rand = np.random.randn(*U_learned.shape)
    _, S_rand, _ = svd(U_rand)
    S_rand = S_rand / S_rand[0]
    sim_rand = np.abs(cosine_similarity(U_rand)[triu_idx])
        
    return np.array(all_v_learn), np.array(all_s_learn), sim_rand, S_rand

def analyze_ensemble_margin(models, dim, L=200, test_pairs=1000):
    """Calculates the empirical margin distribution across ALL models."""
    device = next(models[0].parameters()).device
    margins = []
    
    for model in tqdm(models, desc="Analyzing Ensemble Margins"):
        sim_AP, sim_AN = [], []
        with torch.no_grad():
            for _ in range(test_pairs):
                config = StudentTConfig(n_samples=int(L*3), n_star=int(L*1.5), p=dim, nu_pre=3.0, nu_post=7.0)
                X, _ = sample_student_t_series(config)
                
                anc = torch.tensor(X[0:L], dtype=torch.float32).unsqueeze(0).to(device)
                pos = torch.tensor(X[L:2*L], dtype=torch.float32).unsqueeze(0).to(device)
                neg = torch.tensor(X[-L:], dtype=torch.float32).unsqueeze(0).to(device)
                
                z_a = model.get_fingerprint(anc)
                z_p = model.get_fingerprint(pos)
                z_n = model.get_fingerprint(neg)
                
                sim_AP.append(torch.sum(z_a * z_p, dim=-1).item())
                sim_AN.append(torch.sum(z_a * z_n, dim=-1).item())
                
        # Margin = Mean(Anchor-Positive) - Mean(Anchor-Negative)
        model_margin = np.mean(sim_AP) - np.mean(sim_AN)
        margins.append(model_margin)
        
    return margins, sim_AP, sim_AN # Return the last model's distributions for plotting

def analyze_ensemble_horizon(models, dim, max_L=300, trials=100):
    """Evaluates the SNR convergence consistency across ALL models."""
    device = next(models[0].parameters()).device
    L_range = np.arange(20, max_L + 1, 20)
    
    # Store the mean gap curve for each model
    ensemble_curves = [] 
    
    for model in tqdm(models, desc="Analyzing Ensemble Horizons"):
        model_curve = []
        with torch.no_grad():
            for L in L_range:
                gaps = []
                for _ in range(trials): 
                    config = StudentTConfig(n_samples=int(max_L*3), n_star=int(max_L*1.5), p=dim, nu_pre=3.0, nu_post=10.0)
                    X, _ = sample_student_t_series(config)
                    
                    z1 = model.get_fingerprint(torch.tensor(X[:L], dtype=torch.float32).unsqueeze(0).to(device))
                    z2 = model.get_fingerprint(torch.tensor(X[L:2*L], dtype=torch.float32).unsqueeze(0).to(device))
                    z3 = model.get_fingerprint(torch.tensor(X[-L:], dtype=torch.float32).unsqueeze(0).to(device))
                    
                    noise = 1.0 - torch.sum(z1 * z2, dim=-1).item()
                    signal = 1.0 - torch.sum(z1 * z3, dim=-1).item()
                    gaps.append(max(0, signal - noise))
                    
                model_curve.append(np.mean(gaps))
        ensemble_curves.append(model_curve)
        
    return L_range, np.array(ensemble_curves)

# ========================================================
# 3. MAIN EXECUTION & PLOTTING
# ========================================================
def run_ensemble_analysis(dim=10, num_models=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training {num_models} independent Neural-ECF models (d={dim})...")
    models = [train_quick_model(dim, device) for _ in tqdm(range(num_models), desc="Training Models")]

    print("\nExtracting Ensemble Metrics...")
    v_learn_all, s_learn_all, v_rand, s_rand = analyze_ensemble_U_matrix(models)
    margins, sim_AP, sim_AN = analyze_ensemble_margin(models, dim)
    L_vals, ensemble_horizons = analyze_ensemble_horizon(models, dim)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig)
    sns.set_theme(style="whitegrid")

    # Panel A: Orthogonality Consistency
    ax1 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(v_rand, fill=True, label="Random Init", ax=ax1, color="gray", alpha=0.3)
    # Plot individual models as faint lines
    for i in range(num_models):
        sns.kdeplot(v_learn_all[i], fill=False, color="crimson", alpha=0.3, linewidth=1)
    # Plot ensemble mean
    sns.kdeplot(np.concatenate(v_learn_all), fill=False, label="Ensemble Mean", ax=ax1, color="crimson", linewidth=3)
    ax1.set_title(f"Spectral Dispersion (Across {num_models} Models)", fontweight="bold")
    ax1.set_xlabel("Absolute Cosine Similarity between Frequencies")
    ax1.legend()

    # Panel B: SVD Decay Consistency
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(s_rand, label="Random Init", color="gray", linestyle="--", linewidth=2)
    
    s_mean = np.mean(s_learn_all, axis=0)
    s_std = np.std(s_learn_all, axis=0)
    
    ax2.plot(s_mean, label="Ensemble Mean SVD", color="crimson", linewidth=3)
    ax2.fill_between(range(len(s_mean)), s_mean - s_std, s_mean + s_std, color="crimson", alpha=0.2, label="Model Variance")
    ax2.set_title(f"Effective Dimensionality Stability", fontweight="bold")
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Normalized Singular Value")
    ax2.legend()

    # Panel C: Margin Stability Boxplot
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(y=margins, color="teal", width=0.3, ax=ax3, boxprops=dict(alpha=0.6))
    sns.swarmplot(y=margins, color="black", size=8, ax=ax3)
    ax3.axhline(0.4, color="red", linestyle="--", label="Theoretical Triplet Margin (0.4)")
    ax3.set_title(f"Empirical Margin Consistency", fontweight="bold")
    ax3.set_ylabel("Achieved Distance Gap (sim_AP - sim_AN)")
    ax3.legend()

    # Panel D: Information Horizon Consistency
    ax4 = fig.add_subplot(gs[1, 1])
    h_mean = np.mean(ensemble_horizons, axis=0)
    h_std = np.std(ensemble_horizons, axis=0)
    
    # Plot individual models faintly
    for curve in ensemble_horizons:
        ax4.plot(L_vals, curve, color="darkorange", alpha=0.3, linewidth=1)
        
    ax4.plot(L_vals, h_mean, color="darkorange", marker="o", linewidth=3, label="Ensemble Mean SNR")
    ax4.fill_between(L_vals, h_mean - h_std, h_mean + h_std, color="darkorange", alpha=0.2, label="Model Variance")
    ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax4.set_title("Information Horizon Stability", fontweight="bold")
    ax4.set_xlabel("Window Size (L)")
    ax4.set_ylabel("Signal-to-Noise Ratio")
    ax4.legend()

    plt.suptitle(f"Neural-ECF Meta-Analysis: Architectural Stability (d={dim}, K={num_models})", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("ecf_ensemble_analysis.png", dpi=300, bbox_inches="tight")
    print("\nEnsemble Analysis complete. Plot saved to ecf_ensemble_analysis.png.")

if __name__ == "__main__":
    run_ensemble_analysis(dim=10, num_models=100)