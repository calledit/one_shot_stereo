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

def _gather_neighbors(x):
    """
    For each token, collect its 3×3×3 spatiotemporal neighborhood.

    x       : (B, N, D)  tokens arranged in TOKEN_T × TOKEN_H × TOKEN_W order
    Returns : (B, N, 27, D)
    """
    B, N, D = x.shape
    pad = LOCAL_K // 2
    x3d = x.reshape(B, TOKEN_T, TOKEN_H, TOKEN_W, D).permute(0, 4, 1, 2, 3)
    x3d = F.pad(x3d, (pad, pad, pad, pad, pad, pad))   # pad W, H, T

    slices = []
    for dt in range(LOCAL_K):
        for dh in range(LOCAL_K):
            for dw in range(LOCAL_K):
                slices.append(x3d[:, :, dt:dt+TOKEN_T, dh:dh+TOKEN_H, dw:dw+TOKEN_W])
    # stack → (B, D, 27, T, H, W) → (B, N, 27, D)
    return (torch.stack(slices, dim=2)
            .permute(0, 3, 4, 5, 2, 1)
            .reshape(B, N, LOCAL_K**3, D))


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
        q, k, v = self.qkv(x).chunk(3, dim=-1)   # each (B, N, D)

        k_nb = _gather_neighbors(k)   # (B, N, 27, D)
        v_nb = _gather_neighbors(v)

        H, d = self.n_heads, self.head_dim
        q    = q.reshape(B, N, H, d)
        k_nb = k_nb.reshape(B, N, 27, H, d)
        v_nb = v_nb.reshape(B, N, 27, H, d)

        scale = d ** -0.5
        attn = torch.einsum("bnhd,bnkhd->bnhk", q, k_nb).mul_(scale).softmax(-1)
        out  = torch.einsum("bnhk,bnkhd->bnhd", attn, v_nb).reshape(B, N, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Global attention — masked tokens only
# ---------------------------------------------------------------------------

class GlobalAttention(nn.Module):
    """
    Masked tokens attend to every token in the sequence.
    Non-masked tokens are skipped entirely — output is zero for them.
    The residual connection in SparseTransformerLayer keeps them unchanged.
    """

    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj   = nn.Linear(hidden, hidden, bias=False)
        self.kv_proj  = nn.Linear(hidden, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, mask_flat):
        """
        x         : (B, N, D)
        mask_flat : (B, N) bool
        Returns   : (B, N, D)  — non-zero only at masked positions
        """
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim

        k, v = self.kv_proj(x).chunk(2, dim=-1)
        k = k.reshape(B, N, H, d).permute(0, 2, 1, 3)   # (B, H, N, d)
        v = v.reshape(B, N, H, d).permute(0, 2, 1, 3)

        out   = torch.zeros_like(x)
        scale = d ** -0.5

        for b in range(B):
            mask = mask_flat[b]
            if not mask.any():
                continue
            q = self.q_proj(x[b, mask])                           # (M, D)
            M = q.shape[0]
            q = q.reshape(M, H, d).permute(1, 0, 2)              # (H, M, d)
            attn = torch.einsum("hmd,hnd->hmn", q, k[b]).mul_(scale).softmax(-1)
            res  = torch.einsum("hmn,hnd->hmd", attn, v[b])      # (H, M, d)
            out[b, mask] = self.out_proj(res.permute(1, 0, 2).reshape(M, D))

        return out


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

        x = self.output_proj(self.norm(x))   # (B, 2730, 256)
        return unpatchify(x)                  # (B, 7, 16, 60, 104)
