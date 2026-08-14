import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals Operator as proposed by Kimi Team (2025/2026),
    with tracking capabilities for 2D depth-wise attention weights alpha_{i->l}.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        # Pseudo-query vector w_l per sub-layer, initialized to zeros to guarantee 
        # uniform weighting at the start of training and prevent volatility.
        self.w = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim, eps=eps)

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor | None):
        """
        Args:
            blocks: List of finished block tensors, each of shape [B, T, D].
                    blocks[0] is token embedding h_1 (b_0).
            partial_block: Intra-block partial sum tensor of shape [B, T, D] or None if block boundary.
        Returns:
            h_l: Aggregated hidden state input for current sub-layer, shape [B, T, D].
            attn_weights: Captured attention weights over sources, shape [N_sources].
        """
        # Form values V: finished blocks + (optional) current intra-block partial sum
        if partial_block is not None:
            V_list = blocks + [partial_block]
        else:
            V_list = blocks
        
        # V: shape [N_sources, B, T, D]
        V = torch.stack(V_list, dim=0)
        
        # Apply RMSNorm on keys (K = norm(V))
        K = self.norm(V)
        
        # Compute unnormalized logits: w_l^\top K
        # K shape: [N_sources, B, T, D], self.w shape: [D]
        # einsum ('d, n b t d -> n b t')
        logits = torch.einsum('d, n b t d -> n b t', self.w, K) # [N_sources, B, T]
        
        # Compute softmax over source dimension (dim=0)
        attn_weights = F.softmax(logits, dim=0) # [N_sources, B, T]
        
        # Aggregate hidden state: sum_i (alpha_{i->l} * v_i)
        # einsum ('n b t, n b t d -> b t d')
        h_l = torch.einsum('n b t, n b t d -> b t d', attn_weights, V)
        
        # Return average attention weights per token across batch for tracking
        avg_weights = attn_weights.mean(dim=(1, 2)).detach() # [N_sources]
        
        return h_l, avg_weights
