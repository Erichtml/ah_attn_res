import argparse
import math
import os
import time
import torch
import torch.nn.functional as F

from src.dataset import get_dataloaders
from src.modules.transformer import ModuleAwareTransformer
from src.tracker import ExperimentTracker

def train_variant(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting Training Experiment: Variant = '{args.variant}'")
    print(f" Device: {device} | Dimension: {args.dim} | Layers: {args.n_layers} (24 Sub-layers)")
    print(f"=======================================================")
    
    tracker = ExperimentTracker(variant=args.variant, output_dir=args.output_dir)
    
    train_loader, val_loader = get_dataloaders(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    
    model = ModuleAwareTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        block_size=args.block_size,
        variant=args.variant,
        max_seq_len=args.seq_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    
    train_iter = iter(train_loader)
    
    model.train()
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        t1 = time.time()
        step_time = t1 - t0
        tokens_processed = x.numel()
        tokens_sec = tokens_processed / step_time if step_time > 0 else 0.0
        
        tracker.record_step(train_loss=loss.item(), tokens_sec=tokens_sec, step_time=step_time)
        
        if step % args.log_every == 0 or step == args.steps:
            print(f"  Step [{step:4d}/{args.steps}] | Loss: {loss.item():.4f} | PPL: {math.exp(min(loss.item(), 20)):.2f} | LR: {scheduler.get_last_lr()[0]:.6f} | Speed: {tokens_sec:.1f} tok/s", flush=True)
            
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val_loss = 0.0
            eval_steps = 0
            stats = None
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_logits, v_stats = model(vx, track_stats=True)
                    v_loss = F.cross_entropy(v_logits.view(-1, args.vocab_size), vy.view(-1))
                    val_loss += v_loss.item()
                    eval_steps += 1
                    if stats is None:
                        stats = v_stats
            val_loss /= eval_steps
            tracker.record_eval(val_loss=val_loss, stats=stats)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  >>> Eval @ Step {step:4d} | Validation Loss: {val_loss:.4f} | Validation PPL: {val_ppl:.2f} <<<", flush=True)
            model.train()
            
    total_elapsed = time.time() - start_time
    model_save_path = os.path.join(args.output_dir, f"model_{args.variant}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f" Saved model checkpoint to {model_save_path}")
    print(f" Finished Training '{args.variant}' in {total_elapsed:.2f}s | Final Val Loss: {tracker.val_losses[-1]:.4f}")
    tracker.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AH-AttnRes Ablation Experiment Runner")
    parser.add_argument("--variant", type=str, required=True, choices=['baseline', 'full', 'ah', 'reverse'])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=12) # 24 sub-layers
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    train_variant(args)
