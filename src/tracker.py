import json
import os
import torch
import numpy as np

class ExperimentTracker:
    def __init__(self, variant: str, output_dir: str = "results"):
        self.variant = variant
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.val_ppls = []
        self.step_times = []
        self.tokens_per_sec = []
        
        self.hidden_norms_history = []
        self.delta_h_history = []
        self.attn_weights_history = []
        self.mlp_weights_history = []

    def record_step(self, train_loss: float, tokens_sec: float, step_time: float):
        self.train_losses.append(train_loss)
        self.tokens_per_sec.append(tokens_sec)
        self.step_times.append(step_time)

    def record_eval(self, val_loss: float, stats: dict | None = None):
        val_ppl = float(np.exp(val_loss)) if val_loss < 20 else 9999.0
        self.val_losses.append(val_loss)
        self.val_ppls.append(val_ppl)
        if stats is not None:
            if 'hidden_norms' in stats:
                self.hidden_norms_history.append(stats['hidden_norms'])
            if 'delta_h' in stats:
                self.delta_h_history.append(stats['delta_h'])
            if 'attn_weights' in stats and stats['attn_weights']:
                attn_weights_clean = []
                for w in stats['attn_weights']:
                    if w is not None:
                        attn_weights_clean.append(w.cpu().numpy().tolist())
                if attn_weights_clean:
                    self.attn_weights_history.append(attn_weights_clean)
            if 'mlp_weights' in stats and stats['mlp_weights']:
                mlp_weights_clean = []
                for w in stats['mlp_weights']:
                    if w is not None:
                        mlp_weights_clean.append(w.cpu().numpy().tolist())
                if mlp_weights_clean:
                    self.mlp_weights_history.append(mlp_weights_clean)

    def save(self):
        filepath = os.path.join(self.output_dir, f"results_{self.variant}.json")
        data = {
            "variant": self.variant,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ppls": self.val_ppls,
            "final_val_ppl": self.val_ppls[-1] if self.val_ppls else 0.0,
            "avg_tokens_per_sec": float(np.mean(self.tokens_per_sec)) if self.tokens_per_sec else 0.0,
            "hidden_norms_final": self.hidden_norms_history[-1] if self.hidden_norms_history else [],
            "delta_h_final": self.delta_h_history[-1] if self.delta_h_history else [],
            "attn_weights_final": self.attn_weights_history[-1] if self.attn_weights_history else [],
            "mlp_weights_final": self.mlp_weights_history[-1] if self.mlp_weights_history else [],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Recorded results for '{self.variant}' to {filepath}")
