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

def train_seed(variant, seed, steps=500, dim=512, n_layers=12, block_size=4, lr=1e-3, device="mps"):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    vocab_size = 50257
    seq_len = 256
    batch_size = 4
    
    train_loader, val_loader = get_dataloaders(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
    
    model = ModuleAwareTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        block_size=block_size,
        variant=variant,
        max_seq_len=seq_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    train_iter = iter(train_loader)
    model.train()
    
    step_times = []
    
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        t1 = time.time()
        step_times.append(t1 - t0)
        
    # Evaluate
    model.eval()
    val_loss = 0.0
    eval_steps = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_logits = model(vx)
            v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1))
            val_loss += v_loss.item()
            eval_steps += 1
            if eval_steps >= 20: # eval on 20 batches
                break
    val_loss /= eval_steps
    val_ppl = math.exp(min(val_loss, 20))
    avg_tok_sec = (4 * 256) / np.mean(step_times) if step_times else 0.0
    
    return val_loss, val_ppl, avg_tok_sec

def run_all_seeds():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running 3-seed statistical evaluation on {device}...")
    
    seeds = [42, 43, 44]
    variants = ['baseline', 'full', 'ah', 'reverse']
    results = {}
    
    for var in variants:
        results[var] = {"losses": [], "ppls": [], "tok_sec": []}
        print(f"\n--- Evaluating {var} across seeds {seeds} ---")
        for s in seeds:
            print(f"  Running seed {s}...", end="", flush=True)
            loss, ppl, tok_s = train_seed(var, s, steps=300, device=device)
            results[var]["losses"].append(loss)
            results[var]["ppls"].append(ppl)
            results[var]["tok_sec"].append(tok_s)
            print(f" -> Val Loss: {loss:.4f}, Val PPL: {ppl:.2f}, Speed: {tok_s:.0f} tok/s")
            
        results[var]["loss_mean"] = float(np.mean(results[var]["losses"]))
        results[var]["loss_std"] = float(np.std(results[var]["losses"]))
        results[var]["ppl_mean"] = float(np.mean(results[var]["ppls"]))
        results[var]["ppl_std"] = float(np.std(results[var]["ppls"]))
        results[var]["tok_sec_mean"] = float(np.mean(results[var]["tok_sec"]))
        results[var]["tok_sec_std"] = float(np.std(results[var]["tok_sec"]))
        
    out_file = "results/multiseed_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved multi-seed results to {out_file}")
    
    print("\n=== Multi-Seed Summary (Mean ± Std) ===")
    for var in variants:
        d = results[var]
        print(f"{var:10s} | Loss: {d['loss_mean']:.4f} ± {d['loss_std']:.4f} | PPL: {d['ppl_mean']:.2f} ± {d['ppl_std']:.2f} | Speed: {d['tok_sec_mean']:.0f} ± {d['tok_sec_std']:.0f} tok/s")

if __name__ == "__main__":
    run_all_seeds()
