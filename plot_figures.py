import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def load_results(results_dir="results"):
    variants = ['baseline', 'full', 'ah', 'reverse']
    data = {}
    for var in variants:
        path = os.path.join(results_dir, f"results_{var}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data[var] = json.load(f)
    return data

def plot_all_figures(results_dir="results", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    data = load_results(results_dir)
    if not data:
        print("No result data available to plot.")
        return

    colors = {
        'baseline': '#7f7f7f', # Grey
        'full': '#1f77b4',     # Blue
        'ah': '#2ca02c',       # Green (Ours)
        'reverse': '#d62728'   # Red (Negative Control)
    }
    labels = {
        'baseline': 'Baseline (PreNorm)',
        'full': 'Full AttnRes (Kimi)',
        'ah': 'AH-AttnRes (Ours)',
        'reverse': 'Reverse Control (MLP-Only)'
    }
    markers = {'baseline': 'o', 'full': 's', 'ah': '^', 'reverse': 'd'}

    # -----------------------------------------------------------------
    # Figure 1: Validation Loss & Pareto Frontier (Loss vs Speed)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1a. Validation Loss Curves
    for var, res in data.items():
        if 'val_losses' in res and len(res['val_losses']) > 1:
            eval_steps = np.linspace(200, 2000, len(res['val_losses']))
            ax1.plot(eval_steps, res['val_losses'], label=labels.get(var, var), color=colors.get(var, 'black'),
                     linewidth=2.2, marker=markers.get(var, 'o'), markersize=5)
    ax1.set_title("(a) Validation Loss Trajectories (2,000 Steps)", fontweight='bold')
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Validation Loss")
    ax1.legend(frameon=True, fontsize=10)

    # 1b. Pareto Frontier: Loss vs Tokens/sec
    for var, res in data.items():
        if 'val_losses' in res and 'avg_tokens_per_sec' in res:
            tok_s = res['avg_tokens_per_sec']
            loss = res['val_losses'][-1]
            ax2.scatter(tok_s, loss, color=colors.get(var, 'black'), s=140, zorder=5, label=labels.get(var, var))
            offset_y = 0.008 if var == 'ah' else -0.012
            ax2.annotate(labels.get(var, var), (tok_s, loss + offset_y),
                         ha='center', fontweight='bold', fontsize=9.5)

    ax2.set_title("(b) Empirical Quality-Throughput Trade-off", fontweight='bold')
    ax2.set_xlabel("Training Throughput (Tokens / sec)")
    ax2.set_ylabel("Final Validation Loss")
    ax2.set_xlim(0, 42000)
    
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, "fig1_pareto_loss.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {fig_path1}")

    # -----------------------------------------------------------------
    # Figure 3: Full AttnRes Pre-MHSA vs Pre-MLP 2D Attention Heatmaps
    # -----------------------------------------------------------------
    mech_path = os.path.join(results_dir, "mechanistic_comparison.json")
    if os.path.exists(mech_path):
        with open(mech_path, "r") as f:
            mech_data = json.load(f)
            
        attn_w = mech_data.get("raw_attn_weights", [])
        mlp_w = mech_data.get("raw_mlp_weights", [])
        
        if attn_w and mlp_w:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            
            max_len_a = max(len(row) for row in attn_w)
            mat_a = np.zeros((len(attn_w), max_len_a))
            for i, row in enumerate(attn_w):
                mat_a[i, :len(row)] = row
                
            max_len_m = max(len(row) for row in mlp_w)
            mat_m = np.zeros((len(mlp_w), max_len_m))
            for i, row in enumerate(mlp_w):
                mat_m[i, :len(row)] = row
                
            sns.heatmap(mat_a, ax=axes[0], cmap="Blues", cbar=True, vmin=0, vmax=0.8)
            axes[0].set_title("(a) Full AttnRes: Pre-MHSA Depth Routing", fontweight='bold')
            axes[0].set_xlabel("Source Representation Index")
            axes[0].set_ylabel("Layer Depth Index")
            
            sns.heatmap(mat_m, ax=axes[1], cmap="Oranges", cbar=True, vmin=0, vmax=0.8)
            axes[1].set_title("(b) Full AttnRes: Pre-MLP Depth Routing", fontweight='bold')
            axes[1].set_xlabel("Source Representation Index")
            axes[1].set_ylabel("Layer Depth Index")
            
            plt.tight_layout()
            fig_path3 = os.path.join(output_dir, "fig3_dependency_heatmaps.png")
            plt.savefig(fig_path3, dpi=300)
            plt.close()
            print(f"Saved Figure 3 to {fig_path3}")

    # -----------------------------------------------------------------
    # Figure 4: Hidden State Norms Stability
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 4.8))
    for var, res in data.items():
        if 'hidden_norms_final' in res and res['hidden_norms_final']:
            norms = res['hidden_norms_final']
            plt.plot(range(1, len(norms) + 1), norms, label=labels.get(var, var),
                     color=colors.get(var, 'black'), linewidth=2.2, marker=markers.get(var, 'o'), markersize=4)
    
    plt.title(r"Figure 4: Hidden State Norm $\|h_l\|$ Across 24 Sub-layers", fontweight='bold')
    plt.ylabel(r"Hidden State Norm $\|h_l\|$")
    plt.xlabel("Sub-layer Index (1 to 24)")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig_path4 = os.path.join(output_dir, "fig4_stability_norms.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {fig_path4}")

    # -----------------------------------------------------------------
    # Extended Data Figure 1: Quantitative Mechanistic Breakdown
    # -----------------------------------------------------------------
    if os.path.exists(mech_path):
        with open(mech_path, "r") as f:
            mech = json.load(f)
            
        mhsa_ent = mech['mhsa']['entropy']
        mlp_ent = mech['mlp']['entropy']
        mhsa_dist = mech['mhsa']['depth_distance']
        mlp_dist = mech['mlp']['depth_distance']
        layers = np.arange(1, len(mhsa_ent) + 1)
        
        fig, (ax_e, ax_d) = plt.subplots(1, 2, figsize=(13, 4.8))
        
        ax_e.plot(layers, mhsa_ent, 's-', color='#1f77b4', linewidth=2.2, label=f"Pre-MHSA (Mean: {np.mean(mhsa_ent):.2f} bits)")
        ax_e.plot(layers, mlp_ent, 'o-', color='#ff7f0e', linewidth=2.2, label=f"Pre-MLP in Full AttnRes (Mean: {np.mean(mlp_ent):.2f} bits)")
        ax_e.axhline(0, color='gray', linestyle='--', label="Pre-MLP in AH-AttnRes (0 bits)")
        ax_e.set_title("(a) Layer-wise Shannon Entropy $H(l)$", fontweight='bold')
        ax_e.set_xlabel("Layer Index")
        ax_e.set_ylabel("Entropy $H(l)$ (bits)")
        ax_e.legend(frameon=True, fontsize=9.5)
        
        ax_d.plot(layers, mhsa_dist, 's-', color='#1f77b4', linewidth=2.2, label=f"Pre-MHSA (Mean: {np.mean(mhsa_dist):.2f} blocks)")
        ax_d.plot(layers, mlp_dist, 'o-', color='#ff7f0e', linewidth=2.2, label=f"Pre-MLP in Full AttnRes (Mean: {np.mean(mlp_dist):.2f} blocks)")
        ax_d.set_title("(b) Expected Depth Distance $D(l)$", fontweight='bold')
        ax_d.set_xlabel("Layer Index")
        ax_d.set_ylabel("Expected Distance (Blocks)")
        ax_d.legend(frameon=True, fontsize=9.5)
        
        plt.tight_layout()
        ext_fig1 = os.path.join(output_dir, "extended_data_fig1_entropy.png")
        plt.savefig(ext_fig1, dpi=300)
        plt.close()
        print(f"Saved Extended Data Figure 1 to {ext_fig1}")

if __name__ == "__main__":
    plot_all_figures()
