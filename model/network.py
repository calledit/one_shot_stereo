"""
One Shot Stereo — network architecture.

Forward pass:
  latents    (B, T, C, H, W)  encoded input frames with green holes
  token_mask (B, T, TH, TW)   True where a 4×4 latent patch contains a hole

  → patchify 4×4 → (B, 2730, 256) + append flag → (B, 2730, 257)
  → linear projection + 3D positional encoding → (B, 2730, 1024)
  → 10× SparseTransformerLayer
      local  attention (3×3×3 window, all tokens)
      global attention (full sequence, masked tokens only)
      FFN
  → linear projection → (B, 2730, 256)
  → unpatchify → (B, T, C, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import (
    TOKEN_T, TOKEN_H, TOKEN_W, TOKEN_DIM, PATCH_SIZE,
    LATENT_T, LATENT_C, LATENT_H, LATENT_W,
)

N_TOKENS  = TOKEN_T * TOKEN_H * TOKEN_W   # 2730
TOKEN_IN  = TOKEN_DIM + 1                  # 257  (256 patch values + 1 mask flag)
HIDDEN    = 1024
N_HEADS   = 16
HEAD_DIM  = HIDDEN // N_HEADS              # 64
N_LAYERS  = 10
LOCAL_K   = 3                              # 3×3×3 local window → 27 neighbors


# ---------------------------------------------------------------------------
# Patchify / Unpatchify
# ---------------------------------------------------------------------------

def patchify(latents):
    """(B, T, C, H, W) → (B, T·TH·TW, C·P·P)"""
    B, T, C, H, W = latents.shape
    ph, pw = H // PATCH_SIZE, W // PATCH_SIZE
    x = latents.reshape(B, T, C, ph, PATCH_SIZE, pw, PATCH_SIZE)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return x.reshape(B, T * ph * pw, C * PATCH_SIZE * PATCH_SIZE)


def unpatchify(tokens):
    """(B, T·TH·TW, C·P·P) → (B, T, C, H, W)"""
    B = tokens.shape[0]
    C = TOKEN_DIM // (PATCH_SIZE * PATCH_SIZE)
    x = tokens.reshape(B, TOKEN_T, TOKEN_H, TOKEN_W, C, PATCH_SIZE, PATCH_SIZE)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(B, TOKEN_T, C, TOKEN_H * PATCH_SIZE, TOKEN_W * PATCH_SIZE)


# ---------------------------------------------------------------------------
# 3D learned positional encoding
# ---------------------------------------------------------------------------

class Pos3D(nn.Module):
    """Factored learned embeddings for (T, H, W) summed per token."""

    def __init__(self, hidden):
        super().__init__()
        self.emb_t = nn.Embedding(TOKEN_T, hidden)
        self.emb_h = nn.Embedding(TOKEN_H, hidden)
        self.emb_w = nn.Embedding(TOKEN_W, hidden)
        t = torch.arange(TOKEN_T)
        h = torch.arange(TOKEN_H)
        w = torch.arange(TOKEN_W)
        tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
        self.register_buffer("idx_t", tt.reshape(-1))
        self.register_buffer("idx_h", hh.reshape(-1))
        self.register_buffer("idx_w", ww.reshape(-1))

    def forward(self):
        """Returns (1, N_TOKENS, hidden)."""
        return (self.emb_t(self.idx_t) +
                self.emb_h(self.idx_h) +
                self.emb_w(self.idx_w)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Local 3×3×3 attention
# ---------------------------------------------------------------------------

def _gather_neighbors(kv):
    """
    For each token, collect its 3×3×3 spatiotemporal neighborhood.

    kv      : (B, N, 2D)  K and V concatenated, tokens in TOKEN_T×TOKEN_H×TOKEN_W order
    Returns : (B, N, 27, 2D)
    """
    B, N, D2 = kv.shape
    pad = LOCAL_K // 2
    x3d = kv.reshape(B, TOKEN_T, TOKEN_H, TOKEN_W, D2).permute(0, 4, 1, 2, 3)
    x3d = F.pad(x3d, (pad, pad, pad, pad, pad, pad))   # pad W, H, T

    slices = []
    for dt in range(LOCAL_K):
        for dh in range(LOCAL_K):
            for dw in range(LOCAL_K):
                slices.append(x3d[:, :, dt:dt+TOKEN_T, dh:dh+TOKEN_H, dw:dw+TOKEN_W])
    # stack → (B, 2D, 27, T, H, W) → (B, N, 27, 2D)
    return (torch.stack(slices, dim=2)
            .permute(0, 3, 4, 5, 2, 1)
            .reshape(B, N, LOCAL_K**3, D2))


class LocalAttention(nn.Module):
    """Every token attends to its 3×3×3 neighborhood (27 tokens)."""

    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv      = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each (B, N, D)

        # Gather K and V neighbors in a single pass
        kv_nb = _gather_neighbors(torch.cat([k, v], dim=-1))   # (B, N, 27, 2D)
        k_nb, v_nb = kv_nb.chunk(2, dim=-1)                    # each (B, N, 27, D)

        # Reshape for sdpa: (B*N, H, seq, d)
        q    = q.reshape(B * N, 1, H, d).transpose(1, 2)       # (B*N, H, 1, d)
        k_nb = k_nb.reshape(B * N, 27, H, d).transpose(1, 2)   # (B*N, H, 27, d)
        v_nb = v_nb.reshape(B * N, 27, H, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k_nb, v_nb)    # (B*N, H, 1, d)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# Global attention — masked tokens only
# ---------------------------------------------------------------------------

class GlobalAttention(nn.Module):
    """
    Sparse global attention — only masked tokens generate queries.
    K and V come from all N tokens; Q comes only from the M masked tokens per sample.
    Cost: O(M×N) instead of O(N²). M is typically 5-10% of N.

    Q is padded to M_max across the batch for a single batched sdpa call.
    A small Python loop (size B) handles the gather/scatter around it.
    """

    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv      = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, mask_flat):
        """
        x         : (B, N, D)
        mask_flat : (B, N) bool
        Returns   : (B, N, D)  — non-zero only at masked positions
        """
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim

        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each (B, N, D)

        m_counts = mask_flat.sum(dim=1)           # (B,) masked tokens per sample
        M = int(m_counts.max().item())

        if M == 0:
            return torch.zeros_like(x)

        # Gather masked Q into (B, M, D), zero-padded where M_b < M
        q_sparse = q.new_zeros(B, M, D)
        for b in range(B):
            mb = int(m_counts[b].item())
            q_sparse[b, :mb] = q[b, mask_flat[b]]

        # Batched attention: Q(B,H,M,d) × K(B,H,N,d)ᵀ → out(B,H,M,d)
        q_sparse = q_sparse.reshape(B, M, H, d).permute(0, 2, 1, 3)
        k = k.reshape(B, N, H, d).permute(0, 2, 1, 3)
        v = v.reshape(B, N, H, d).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q_sparse, k, v)   # (B, H, M, d)
        out = self.out_proj(out.permute(0, 2, 1, 3).reshape(B, M, D))

        # Scatter results back to masked positions in the full token sequence
        result = torch.zeros_like(x)
        for b in range(B):
            mb = int(m_counts[b].item())
            result[b, mask_flat[b]] = out[b, :mb]

        return result


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, hidden, expand=4):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * expand)
        self.fc2 = nn.Linear(hidden * expand, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer layer
# ---------------------------------------------------------------------------

class SparseTransformerLayer(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm_local  = nn.LayerNorm(hidden)
        self.norm_global = nn.LayerNorm(hidden)
        self.norm_ffn    = nn.LayerNorm(hidden)
        self.local_attn  = LocalAttention(hidden, n_heads)
        self.global_attn = GlobalAttention(hidden, n_heads)
        self.ffn         = FFN(hidden)

    def forward(self, x, mask_flat):
        x = x + self.local_attn(self.norm_local(x))
        x = x + self.global_attn(self.norm_global(x), mask_flat)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class OneShotStereoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(TOKEN_IN, HIDDEN)
        self.pos_enc     = Pos3D(HIDDEN)
        self.layers      = nn.ModuleList([
            SparseTransformerLayer(HIDDEN, N_HEADS) for _ in range(N_LAYERS)
        ])
        self.norm        = nn.LayerNorm(HIDDEN)
        self.output_proj = nn.Linear(HIDDEN, TOKEN_DIM)
        self._init_weights()

    def _init_weights(self):
        # Standard transformer init; output_proj zeroed so the network starts
        # as an identity map (input latents pass through unchanged).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, latents, token_mask):
        """
        latents    : (B, LATENT_T, LATENT_C, LATENT_H, LATENT_W)
        token_mask : (B, TOKEN_T, TOKEN_H, TOKEN_W) bool
        Returns    : (B, LATENT_T, LATENT_C, LATENT_H, LATENT_W)
        """
        B = latents.shape[0]
        mask_flat = token_mask.reshape(B, -1)   # (B, 2730)

        tokens = patchify(latents)                                          # (B, 2730, 256)
        flag   = mask_flat.unsqueeze(-1).to(tokens.dtype)
        x      = self.input_proj(torch.cat([tokens, flag], dim=-1))        # (B, 2730, 1024)
        x      = x + self.pos_enc().to(dtype=x.dtype, device=x.device)

        for layer in self.layers:
            x = layer(x, mask_flat)

        delta = self.output_proj(self.norm(x))   # (B, 2730, 256)
        return unpatchify(tokens + delta)         # residual on input latents
