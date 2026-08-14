import json
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_entropy(weights_list):
    """
    Computes Shannon Information Entropy H(l) = - sum_i alpha_{i->l} log2(alpha_{i->l})
    for each layer l.
    """
    entropies = []
    for layer_w in weights_list:
        w = np.array(layer_w, dtype=np.float64)
        # Avoid log(0)
        w = np.clip(w, 1e-12, 1.0)
        w = w / np.sum(w)
        entropy = -np.sum(w * np.log2(w))
        entropies.append(entropy)
    return entropies

def run_supplementary_analysis(results_dir="results", output_dir="figures"):
    print("=== Running Supplementary Experiments for Nature-Style Evidence ===")
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, "results_full.json")
    ah_path = os.path.join(results_dir, "results_ah.json")
    
    if not os.path.exists(full_path) or not os.path.exists(ah_path):
        print("Error: Missing result json files.")
        return

    with open(full_path, "r") as f:
        full_data = json.load(f)
    with open(ah_path, "r") as f:
        ah_data = json.load(f)

    # 1. Calculate Information Entropy H(l)
    full_attn_w = full_data.get("attn_weights_final", [])
    ah_attn_w = ah_data.get("attn_weights_final", [])
    
    full_entropies = compute_entropy(full_attn_w)
    ah_entropies = compute_entropy(ah_attn_w)
    
    # Save entropy data to JSON
    entropy_results = {
        "full_attn_entropies": full_entropies,
        "ah_attn_entropies": ah_entropies,
        "mean_full_entropy": float(np.mean(full_entropies)),
        "mean_ah_entropy": float(np.mean(ah_entropies)),
    }
    
    entropy_json_path = os.path.join(results_dir, "results_entropy.json")
    with open(entropy_json_path, "w") as f:
        json.dump(entropy_results, f, indent=2)
    print(f"Saved Quantitative Entropy Evidence to {entropy_json_path}")

    # 2. Plot Extended Data Figure 1: Layer-wise Information Entropy H(l)
    plt.figure(figsize=(8, 5))
    layers = list(range(1, len(full_entropies) + 1))
    
    plt.plot(layers, full_entropies, 's-', label="Full AttnRes (Pre-Attn)", color="#1f77b4", linewidth=2.5)
    plt.plot(layers, ah_entropies, '^-', label="AH-AttnRes (Pre-Attn, Ours)", color="#2ca02c", linewidth=2.5)
    
    # Theoretical Low Entropy Line for Standard Residual (Pre-MLP) -> Entropy = 0
    plt.axhline(y=0.0, color='gray', linestyle='--', label="Pre-MLP Standard Residual (Entropy = 0)")

    plt.title("Extended Data Fig. 1: Layer-wise Information Entropy H(l)", fontweight='bold', fontsize=14)
    plt.xlabel("Layer Depth (l)", fontsize=12)
    plt.ylabel("Entropy H(l) (bits)", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    ext_fig_path = os.path.join(output_dir, "extended_data_fig1_entropy.png")
    plt.savefig(ext_fig_path, dpi=300)
    plt.close()
    print(f"Saved Extended Data Figure 1 to {ext_fig_path}")

if __name__ == "__main__":
    run_supplementary_analysis()
