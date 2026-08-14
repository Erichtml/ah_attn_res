import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn_res import RMSNorm, BlockAttnRes

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv_proj(x) # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, T, head_dim]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # PyTorch scaled dot product attention with causal mask
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # [B, n_heads, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class ModuleAwareTransformer(nn.Module):
    """
    Modular Transformer supporting 4 Depth-Routing Ablation Variants:
    1. 'baseline': Standard PreNorm Residuals
    2. 'full': Full AttnRes (Pre-Attn AttnRes + Pre-MLP AttnRes)
    3. 'ah': AH-AttnRes (Pre-Attn AttnRes + Pre-MLP Standard Residual) - Ours
    4. 'reverse': Reverse/MLP-only (Pre-Attn Standard + Pre-MLP AttnRes)
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 12, # 12 Transformer Blocks = 24 Sub-layers
        n_heads: int = 8,
        block_size: int = 4, # 4 sub-layers per AttnRes block
        variant: str = 'ah',
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_sublayers = n_layers * 2
        self.block_size = block_size
        self.variant = variant
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        
        # Build Layers
        self.attn_layers = nn.ModuleList([CausalSelfAttention(dim, n_heads) for _ in range(n_layers)])
        self.mlp_layers = nn.ModuleList([MLP(dim) for _ in range(n_layers)])
        
        # Norms
        self.attn_norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.mlp_norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # AttnRes operators for each sub-layer
        self.attn_res_attn = nn.ModuleList([BlockAttnRes(dim) for _ in range(n_layers)])
        self.attn_res_mlp = nn.ModuleList([BlockAttnRes(dim) for _ in range(n_layers)])

    def forward(self, input_ids: torch.Tensor, track_stats: bool = False):
        B, T = input_ids.shape
        h1 = self.token_emb(input_ids) + self.pos_emb[:, :T, :]
        
        # AttnRes Block management
        # blocks list initialized with token embedding b_0 = h_1
        blocks = [h1]
        partial_block = None
        current_h = h1
        
        # Tracking dictionaries
        stats = {
            'hidden_norms': [],  # ||h_l|| per sub-layer
            'delta_h': [],         # ||f_l(h_l)|| / ||h_l||
            'attn_weights': [],   # alpha_{i->l} for Pre-Attn
            'mlp_weights': [],    # alpha_{i->l} for Pre-MLP
        }
        
        sublayer_idx = 0
        
        for i in range(self.n_layers):
            # -------------------------------------------------------------
            # Sub-layer 1: Self-Attention
            # -------------------------------------------------------------
            # Determine input to Self-Attention based on variant
            use_attnres_attn = self.variant in ['full', 'ah']
            
            if use_attnres_attn:
                h_attn, alpha_attn = self.attn_res_attn[i](blocks, partial_block)
                if track_stats:
                    stats['attn_weights'].append(alpha_attn)
            else:
                h_attn = current_h
                if track_stats:
                    stats['attn_weights'].append(None)
            
            if track_stats:
                norm_h = h_attn.norm(dim=-1).mean().item()
                stats['hidden_norms'].append(norm_h)
            
            # Forward Attention Layer
            attn_out = self.attn_layers[i](self.attn_norms[i](h_attn))
            
            if track_stats:
                delta = (attn_out.norm(dim=-1) / (h_attn.norm(dim=-1) + 1e-6)).mean().item()
                stats['delta_h'].append(delta)
            
            # Update intra-block residual
            if partial_block is None:
                partial_block = attn_out
            else:
                partial_block = partial_block + attn_out
            current_h = h_attn + attn_out
            sublayer_idx += 1
            
            # Check block boundary
            if sublayer_idx % self.block_size == 0:
                blocks.append(partial_block)
                partial_block = None
                
            # -------------------------------------------------------------
            # Sub-layer 2: MLP
            # -------------------------------------------------------------
            use_attnres_mlp = self.variant in ['full', 'reverse']
            
            if use_attnres_mlp:
                h_mlp, alpha_mlp = self.attn_res_mlp[i](blocks, partial_block)
                if track_stats:
                    stats['mlp_weights'].append(alpha_mlp)
            else:
                h_mlp = current_h
                if track_stats:
                    stats['mlp_weights'].append(None)
            
            if track_stats:
                norm_h = h_mlp.norm(dim=-1).mean().item()
                stats['hidden_norms'].append(norm_h)
            
            # Forward MLP Layer
            mlp_out = self.mlp_layers[i](self.mlp_norms[i](h_mlp))
            
            if track_stats:
                delta = (mlp_out.norm(dim=-1) / (h_mlp.norm(dim=-1) + 1e-6)).mean().item()
                stats['delta_h'].append(delta)
            
            # Update intra-block residual
            if partial_block is None:
                partial_block = mlp_out
            else:
                partial_block = partial_block + mlp_out
            current_h = h_mlp + mlp_out
            sublayer_idx += 1
            
            # Check block boundary
            if sublayer_idx % self.block_size == 0:
                blocks.append(partial_block)
                partial_block = None

        # Final Logits
        final_h = self.final_norm(current_h)
        logits = self.head(final_h)
        
        if track_stats:
            return logits, stats
        return logits
