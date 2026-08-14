import os
import sys
sys.path.append(os.getcwd())
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import get_dataloaders
from src.modules.transformer import ModuleAwareTransformer

def benchmark_depth(n_layers_list=[6, 12, 18], steps=150):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Benchmarking Depth Scaling on {device} across layer counts: {n_layers_list}...")
    
    vocab_size = 50257
    seq_len = 256
    batch_size = 4
    dim = 512
    block_size = 4
    
    train_loader, val_loader = get_dataloaders(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
    
    scaling_data = {"depths": [l*2 for l in n_layers_list], "full": {"tok_s": [], "losses": []}, "ah": {"tok_s": [], "losses": []}}
    
    for n_l in n_layers_list:
        sublayers = n_l * 2
        print(f"\n--- Depth = {sublayers} Sub-layers ({n_l} Blocks) ---")
        
        for var in ['full', 'ah']:
            model = ModuleAwareTransformer(
                vocab_size=vocab_size,
                dim=dim,
                n_layers=n_l,
                block_size=block_size,
                variant=var,
                max_seq_len=seq_len
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_iter = iter(train_loader)
            step_times = []
            
            model.train()
            for step in range(1, steps + 1):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                
                t0 = time.time()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t1 = time.time()
                if step > 20: # warmup
                    step_times.append(t1 - t0)
                    
            # Quick eval
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    vx, vy = vx.to(device), vy.to(device)
                    v_logits = model(vx)
                    val_loss += F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1)).item()
                    if i >= 10:
                        break
            val_loss /= 11.0
            tok_s = (batch_size * seq_len) / np.mean(step_times)
            
            scaling_data[var]["tok_s"].append(float(tok_s))
            scaling_data[var]["losses"].append(float(val_loss))
            print(f"  {var.upper():4s}: Val Loss = {val_loss:.4f}, Speed = {tok_s:.0f} tok/s")
            
    out_file = "results/depth_scaling_results.json"
    with open(out_file, "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\nSaved depth scaling results to {out_file}")

if __name__ == "__main__":
    benchmark_depth()
