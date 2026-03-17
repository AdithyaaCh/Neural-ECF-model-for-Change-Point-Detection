import torch
import torch.nn as nn
import torch.optim as optim
from models import SpectralECFDetector
from data_generator import TripletDataGenerator

def train_step(model, optimizer, batch, device='cpu', margin=0.4):
    """
    Executes a single contrastive training step using Hard-Negative Triplets.
    """
    model.train()
    anchor, positive, negative = [b.to(device) for b in batch]
    
    optimizer.zero_grad()
    
    # Forward pass through the CNN-ECF architecture
    z_a, z_p, z_n = model(anchor, positive, negative)
    
    # 1. Triplet Margin Loss
    # Forces Euclidean distance: d(a, p) + margin < d(a, n)
    # Since embeddings are L2 normalized, Euclidean distance is tied to Cosine Similarity.
    triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2)
    loss_triplet = triplet_criterion(z_a, z_p, z_n)
    
    # 2. Sharpness Penalty (InfoNCE variant for the positive pair)
    # We want the Anchor and Positive to be almost identical (Cosine Sim is 1.0)
    # This ensures a low "noise floor" (H0) during our statistical calibration.
    cos_sim_ap = torch.sum(z_a * z_p, dim=-1)
    loss_sim = 1.0 - cos_sim_ap.mean()
    
    # 3. Total Loss
    total_loss = loss_triplet + (0.5 * loss_sim)
    
    total_loss.backward()
    optimizer.step()
    
    # Calculate Anchor-Negative similarity for monitoring
    with torch.no_grad():
        cos_sim_an = torch.sum(z_a * z_n, dim=-1).mean()
        
    return {
        'loss': total_loss.item(),
        'sim_ap': cos_sim_ap.mean().item(), # Should approach 1.0
        'sim_an': cos_sim_an.item()         # Should be lower than sim_ap
    }

def train_model(p_dim=2, L=200, epochs=5, steps_per_epoch=100, batch_size=32, device='cpu'):
    """
    The main training loop. Initializes the generator and trains the Spectral CNN.
    """
    print(f"Initializing Spectral CNN Training (Dimension: {p_dim}, Window: {L})...")
    
    #model = SpectralCNNDetector(in_channels=p_dim, hidden_dim=16, M=128).to(device)
    model = SpectralECFDetector(in_channels=p_dim, M=128).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    generator = TripletDataGenerator(p=p_dim, L=L)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_sim_ap = 0.0
        epoch_sim_an = 0.0
        
        for step in range(steps_per_epoch):
            # Generate a fresh batch of triplets featuring Student-t and Sub-Gaussian
            batch = generator.generate_triplet_batch(batch_size=batch_size)
            
            metrics = train_step(model, optimizer, batch, device=device, margin=0.4)
            
            epoch_loss += metrics['loss']
            epoch_sim_ap += metrics['sim_ap']
            epoch_sim_an += metrics['sim_an']
            
            if (step + 1) % 25 == 0:
                print(f"  Step [{step+1}/{steps_per_epoch}] | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Sim(A,P): {metrics['sim_ap']:.4f} | "
                      f"Sim(A,N): {metrics['sim_an']:.4f}")
                
        avg_loss = epoch_loss / steps_per_epoch
        avg_sim_ap = epoch_sim_ap / steps_per_epoch
        avg_sim_an = epoch_sim_an / steps_per_epoch
        
        print(f"=== EPOCH {epoch+1} COMPLETE ===")
        print(f"Avg Loss: {avg_loss:.4f} | Avg Sim(A,P): {avg_sim_ap:.4f} | Avg Sim(A,N): {avg_sim_an:.4f}\n")
        
    torch.save(model.state_dict(), f"spectral_cnn_d{p_dim}.pt")
    print(f"Model saved to spectral_cnn_d{p_dim}.pt")
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(p_dim=2, L=100, epochs=3, steps_per_epoch=100, batch_size=32, device=device)