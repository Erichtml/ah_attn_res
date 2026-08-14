import os
import sys
sys.path.append(os.getcwd())
import json
import torch
import torch.nn.functional as F
import numpy as np

from src.dataset import get_dataloaders
from src.modules.transformer import ModuleAwareTransformer

def compute_metrics(weights_list, block_size=4):
    """
    Computes for a list of layer attention weights:
    1. Shannon entropy H(l) = - sum_i alpha_i log2(alpha_i)
    2. Effective number of sources N_eff = 2^H
    3. Expected depth distance D = sum_i alpha_i * (current_block_idx - i)
    """
    entropies = []
    n_effs = []
    distances = []
    
    for layer_idx, w_vec in enumerate(weights_list):
        w = np.array(w_vec, dtype=np.float64)
        w = np.clip(w, 1e-12, 1.0)
        w = w / np.sum(w)
        
        # 1. Shannon Entropy
        entropy = -np.sum(w * np.log2(w))
        entropies.append(float(entropy))
        
        # 2. Effective Sources
        n_eff = float(2 ** entropy)
        n_effs.append(n_eff)
        
        # 3. Expected Depth Distance (in block units)
        # The sources are [b_0 (embedding), b_1, ..., b_{current_block-1}, (optional partial)]
        num_sources = len(w)
        current_block = (layer_idx * 2) // block_size # approximate block index
        source_indices = np.arange(num_sources)
        dist_vector = np.abs(num_sources - 1 - source_indices) # distance from current position
        exp_dist = float(np.sum(w * dist_vector))
        distances.append(exp_dist)
        
    return {
        "entropy": entropies,
        "n_eff": n_effs,
        "depth_distance": distances,
        "mean_entropy": float(np.mean(entropies)),
        "mean_n_eff": float(np.mean(n_effs)),
        "mean_depth_distance": float(np.mean(distances)),
    }

def run_extraction():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Extracting Full AttnRes mechanics on {device}...")
    
    vocab_size = 50257
    dim = 512
    n_layers = 12
    block_size = 4
    
    _, val_loader = get_dataloaders(vocab_size=vocab_size, seq_len=256, batch_size=4)
    
    # Load Full AttnRes model
    model = ModuleAwareTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        block_size=block_size,
        variant='full',
        max_seq_len=256
    ).to(device)
    
    ckpt_path = "results/model_full.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading trained weights from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"Warning: {ckpt_path} not found!")
        return
        
    model.eval()
    all_attn_weights = []
    all_mlp_weights = []
    
    with torch.no_grad():
        for i, (vx, vy) in enumerate(val_loader):
            vx, vy = vx.to(device), vy.to(device)
            _, stats = model(vx, track_stats=True)
            
            # Collect layer weights
            if not all_attn_weights:
                all_attn_weights = [w.cpu().numpy() for w in stats['attn_weights'] if w is not None]
                all_mlp_weights = [w.cpu().numpy() for w in stats['mlp_weights'] if w is not None]
            else:
                for idx, w in enumerate(stats['attn_weights']):
                    if w is not None:
                        all_attn_weights[idx] += w.cpu().numpy()
                for idx, w in enumerate(stats['mlp_weights']):
                    if w is not None:
                        all_mlp_weights[idx] += w.cpu().numpy()
            if i >= 10: # Average over 10 validation batches
                break
                
    all_attn_weights = [ (w / 11.0).tolist() for w in all_attn_weights ]
    all_mlp_weights = [ (w / 11.0).tolist() for w in all_mlp_weights ]
    
    mhsa_metrics = compute_metrics(all_attn_weights, block_size)
    mlp_metrics = compute_metrics(all_mlp_weights, block_size)
    
    print("\n=== Full AttnRes Internal Mechanisms Comparison ===")
    print(f"MHSA Mean Shannon Entropy:       {mhsa_metrics['mean_entropy']:.3f} bits")
    print(f"MLP  Mean Shannon Entropy:       {mlp_metrics['mean_entropy']:.3f} bits")
    print(f"MHSA Mean Effective Sources:     {mhsa_metrics['mean_n_eff']:.2f}")
    print(f"MLP  Mean Effective Sources:     {mlp_metrics['mean_n_eff']:.2f}")
    print(f"MHSA Mean Expected Depth Dist:   {mhsa_metrics['mean_depth_distance']:.2f} blocks")
    print(f"MLP  Mean Expected Depth Dist:   {mlp_metrics['mean_depth_distance']:.2f} blocks")
    
    results = {
        "mhsa": mhsa_metrics,
        "mlp": mlp_metrics,
        "raw_attn_weights": all_attn_weights,
        "raw_mlp_weights": all_mlp_weights,
    }
    
    out_path = "results/mechanistic_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed comparison to {out_path}")

if __name__ == "__main__":
    run_extraction()
