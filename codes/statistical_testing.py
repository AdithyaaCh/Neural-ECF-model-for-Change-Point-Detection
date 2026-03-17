import torch
import numpy as np
from scipy.signal import find_peaks

def scan_series(model, X_series, L, gap, scan_step, device):
    """Step 1: Fast heuristic sliding window scan."""
    model.eval()
    n_samples = len(X_series)
    indices = np.arange(L, n_samples - L - gap, scan_step)
    scores = []
    with torch.no_grad():
        for t in indices:
            past = torch.tensor(X_series[t-L : t ], dtype=torch.float32).unsqueeze(0).to(device)
            future = torch.tensor(X_series[t+gap : t+L+gap], dtype=torch.float32).unsqueeze(0).to(device)
            
            z_p = model.get_fingerprint(past)
            z_f = model.get_fingerprint(future)
            
            dissimilarity = 1.0 - torch.sum(z_p * z_f, dim=-1).item()
            scores.append(dissimilarity)
            
    return indices, np.array(scores)

def compute_batched_p_value(model, past_tensor, future_tensor, n_permutations=100):
    """Step 3: Rigorous verification using Batched PyTorch Permutations."""
    device = next(model.parameters()).device
    L = past_tensor.shape[1]
    
    with torch.no_grad():
        # Actual Score
        z_p = model.get_fingerprint(past_tensor)
        z_f = model.get_fingerprint(future_tensor)
        actual_score = 1.0 - torch.sum(z_p * z_f, dim=-1).item()
        
        # Batch and Shuffle Data
        pooled = torch.cat([past_tensor, future_tensor], dim=1) 
        pooled_expanded = pooled.expand(n_permutations, -1, -1) 
        
        noise = torch.rand(n_permutations, 2 * L, device=device)
        perm_indices = noise.argsort(dim=1).unsqueeze(-1).expand(-1, -1, pooled.shape[-1])
        
        shuffled = torch.gather(pooled_expanded, 1, perm_indices)
        fake_past = shuffled[:, :L, :]
        fake_future = shuffled[:, L:, :]
        
        # Fake Scores
        z_fake_p = model.get_fingerprint(fake_past)
        z_fake_f = model.get_fingerprint(fake_future)
        fake_scores = 1.0 - torch.sum(z_fake_p * z_fake_f, dim=-1).cpu().numpy()
        
    p_value = (np.sum(fake_scores >= actual_score) + 1) / (n_permutations + 1)
    return p_value


def run_detection_pipeline(model, X_series, L, gap, scan_step, alpha=0.05, device='cpu', expected_cps=None):
    """
    Final Pipeline with Top-K Support.
    """
    indices, scores = scan_series(model, X_series, L, gap, scan_step, device)
    
    dyn_prominence= 0
    # Find all topographical hills separated by the exclusion zone
    
    peak_idxs, _ = find_peaks(scores, prominence=dyn_prominence, distance=L//scan_step)
    
    # Pair indices with their raw scores and sort highest to lowest
    candidates = [(idx, scores[idx]) for idx in peak_idxs]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Force the algorithm to only evaluate the Top K hills
    if expected_cps is not None:
        candidates = candidates[:expected_cps]
        
    verified_points = []
    for idx, score in candidates:
        t = indices[idx]
        
        past = torch.tensor(X_series[t-L : t], dtype=torch.float32).unsqueeze(0).to(device)
        future = torch.tensor(X_series[t+gap : t+L+gap], dtype=torch.float32).unsqueeze(0).to(device)
        
        p_val = compute_batched_p_value(model, past, future)
        if p_val <= alpha:
            verified_points.append((t, score, p_val))
            
    # Sort chronologically for final output
    verified_points.sort(key=lambda x: x[0])
    return verified_points