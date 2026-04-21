"""
Latent-space transformer discriminator for the GAN loss — variant 2.

Changes vs discriminator.py:
  - TokenConvExtractor: 10 conv layers (7 extra at DISC_HIDDEN width)
  - FFN: 4 linear layers (2 extra), 512→2048→2048→2048→512
  - ScoreFunnel: per-token MLP (512→512→256→1) then final MLP (N_TOKENS→256→1)
    — no mean pooling, learned per-position weighting instead

Architecture:
  6 layers, 512 hidden, 8 heads, sparse attention:
    - Local 3×3×3 attention on all tokens
    - Global attention on masked tokens only
  Per-token scores aggregated by a learned final MLP → single real/fake value.

Hinge loss:
  D real : mean(relu(1 - score))
  D fake : mean(relu(1 + score))
  G      : mean(-score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

from data.dataset import TOKEN_DIM, TOKEN_T, TOKEN_H, TOKEN_W, LATENT_C, PATCH_SIZE
from model.network import Pos3D

DISC_HIDDEN   = 512
DISC_HEADS    = 8
DISC_LAYERS   = 6
LOCAL_K       = 3

TOKEN_FLAT = LATENT_C * PATCH_SIZE * PATCH_SIZE   # flattened token size = TOKEN_DIM (without mask)


# ---------------------------------------------------------------------------
# Local 3×3×3 attention
# ---------------------------------------------------------------------------

def _gather_neighbors(kv):
    B, N, D2 = kv.shape
    pad = LOCAL_K // 2
    x3d = kv.reshape(B, TOKEN_T, TOKEN_H, TOKEN_W, D2).permute(0, 4, 1, 2, 3)
    x3d = F.pad(x3d, (pad, pad, pad, pad, pad, pad))
    slices = []
    for dt in range(LOCAL_K):
        for dh in range(LOCAL_K):
            for dw in range(LOCAL_K):
                slices.append(x3d[:, :, dt:dt+TOKEN_T, dh:dh+TOKEN_H, dw:dw+TOKEN_W])
    return (torch.stack(slices, dim=2)
            .permute(0, 3, 4, 5, 2, 1)
            .reshape(B, N, LOCAL_K**3, D2))


class LocalAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv      = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        kv_nb   = _gather_neighbors(torch.cat([k, v], dim=-1))
        k_nb, v_nb = kv_nb.chunk(2, dim=-1)
        q    = q.reshape(B * N, 1, H, d).transpose(1, 2)
        k_nb = k_nb.reshape(B * N, 27, H, d).transpose(1, 2)
        v_nb = v_nb.reshape(B * N, 27, H, d).transpose(1, 2)
        out  = F.scaled_dot_product_attention(q, k_nb, v_nb)
        return self.out_proj(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# Global attention — masked tokens only
# ---------------------------------------------------------------------------

class GlobalAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden // n_heads
        self.qkv      = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, mask_flat):
        B, N, D = x.shape
        H, d = self.n_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        m_counts = mask_flat.sum(dim=1)
        M = int(m_counts.max().item())
        if M == 0:
            return torch.zeros_like(x)
        q_sparse = q.new_zeros(B, M, D)
        for b in range(B):
            mb = int(m_counts[b].item())
            q_sparse[b, :mb] = q[b, mask_flat[b]]
        q_sparse = q_sparse.reshape(B, M, H, d).permute(0, 2, 1, 3)
        k = k.reshape(B, N, H, d).permute(0, 2, 1, 3)
        v = v.reshape(B, N, H, d).permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_sparse, k, v)
        out = self.out_proj(out.permute(0, 2, 1, 3).reshape(B, M, D))
        result = torch.zeros_like(x)
        for b in range(B):
            mb = int(m_counts[b].item())
            result[b, mask_flat[b]] = out[b, :mb]
        return result


# ---------------------------------------------------------------------------
# FFN — 4 linear layers, 3 GELUs
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

class DiscTransformerLayer(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm_local  = nn.LayerNorm(hidden)
        self.norm_global = nn.LayerNorm(hidden)
        self.norm_ffn    = nn.LayerNorm(hidden)
        self.local_attn  = LocalAttention(hidden, n_heads)
        self.global_attn = GlobalAttention(hidden, n_heads)
        self.ffn         = FFN(hidden)

    def forward(self, x, mask_flat):
        x = x + self.local_attn(self.norm_local(x.float()).to(x.dtype))
        x = x + self.global_attn(self.norm_global(x.float()).to(x.dtype), mask_flat)
        x = x + self.ffn(self.norm_ffn(x.float()).to(x.dtype))
        return x


# ---------------------------------------------------------------------------
# TokenConvExtractor — 10 conv layers
# ---------------------------------------------------------------------------

class TokenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOKEN_FLAT, DISC_HIDDEN),
            nn.GELU(),
            nn.Linear(DISC_HIDDEN, DISC_HIDDEN),
            nn.GELU(),
        )

    def forward(self, tokens):
        # tokens: (B, N, TOKEN_DIM) — use only the latent dims, not the mask flag
        return self.net(tokens[..., :TOKEN_FLAT].float())


# ---------------------------------------------------------------------------
# ScoreFunnel — per-token MLP then final MLP over all token scores
# ---------------------------------------------------------------------------

class ScoreFunnel(nn.Module):
    """
    Per-token MLP scores each token independently → (B, 1, T, H, W) score volume.
    Strided Conv3d layers compress T×H×W down to 1×1×1 → scalar per sample.
    """

    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm      = nn.LayerNorm(hidden)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 16),
        )
        def _conv_out(s, stride, kernel=3, padding=1):
            return (s + 2 * padding - kernel) // stride + 1

        t = _conv_out(_conv_out(TOKEN_T, stride=1), stride=2)
        t = _conv_out(t, stride=2)
        h = _conv_out(_conv_out(TOKEN_H, stride=2), stride=2)
        h = _conv_out(h, stride=2)
        w = _conv_out(_conv_out(TOKEN_W, stride=2), stride=2)
        w = _conv_out(w, stride=2)

        self.aggregator = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GELU(),
            nn.Conv3d(64, 1, kernel_size=(t, h, w), stride=1, padding=0),
        )

    def forward(self, x, mask_flat):
        B = x.shape[0]
        per_token = self.token_mlp(self.norm(x.float()).to(x.dtype))                       # (B, N, 16)
        vol = per_token.permute(0, 2, 1).reshape(B, 16, TOKEN_T, TOKEN_H, TOKEN_W)        # (B, 16, T, H, W)
        return self.aggregator(vol).reshape(B)                                             # (B,)


# ---------------------------------------------------------------------------
# Full discriminator
# ---------------------------------------------------------------------------

class LatentDiscriminator(nn.Module):
    """
    Sparse transformer discriminator — variant 2.
    Input: patchified latents (B, N, TOKEN_DIM).
    Output: (B,) scores — one per sample.
    """

    def __init__(self):
        super().__init__()
        self.input_proj = TokenMLP()
        self.proj_down  = nn.Linear(DISC_HIDDEN + 1, DISC_HIDDEN, bias=False)
        self.proj_mlp   = nn.Sequential(
            nn.Linear(DISC_HIDDEN, DISC_HIDDEN),
            nn.GELU(),
            nn.Linear(DISC_HIDDEN, DISC_HIDDEN),
        )
        self.pos_enc    = Pos3D(DISC_HIDDEN)
        self.layers     = nn.ModuleList([
            DiscTransformerLayer(DISC_HIDDEN, DISC_HEADS) for _ in range(DISC_LAYERS)
        ])
        self.funnel     = ScoreFunnel(DISC_HIDDEN, DISC_HEADS)
        self._init_weights()
        self._apply_spectral_norm()
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.to(torch.float32)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_spectral_norm(self):
        for layer in self.layers:
            layer.local_attn.qkv      = SN(layer.local_attn.qkv)
            layer.local_attn.out_proj = SN(layer.local_attn.out_proj)
            layer.global_attn.qkv      = SN(layer.global_attn.qkv)
            layer.global_attn.out_proj = SN(layer.global_attn.out_proj)
            layer.ffn.fc1 = SN(layer.ffn.fc1)
            layer.ffn.fc2 = SN(layer.ffn.fc2)
        self.funnel.token_mlp = nn.Sequential(*[
            SN(m) if isinstance(m, nn.Linear) else m
            for m in self.funnel.token_mlp
        ])
        self.funnel.aggregator = nn.Sequential(*[
            SN(m) if isinstance(m, nn.Conv3d) else m
            for m in self.funnel.aggregator
        ])

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.to(torch.float32)
        return self

    def forward(self, tokens, mask_flat):
        """
        tokens    : (B, N, TOKEN_DIM)
        mask_flat : (B, N) bool
        Returns   : (B,) scores — one per sample
        """
        mask_feat = mask_flat.unsqueeze(-1).to(tokens.dtype)                        # (B, N, 1)
        x = self.proj_mlp(self.proj_down(torch.cat([self.input_proj(tokens), mask_feat], dim=-1)))
        x = x + self.pos_enc().to(dtype=tokens.dtype, device=tokens.device)
        for layer in self.layers:
            x = layer(x, mask_flat)
        return self.funnel(x, mask_flat)   # (B,)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def d_hinge(real_scores, fake_scores):
    """Discriminator hinge loss."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def g_hinge(fake_scores):
    """Generator hinge loss — maximise discriminator score on fakes."""
    return -fake_scores.mean()
