"""
Latent-space transformer discriminator for the GAN loss.

Architecture:
  4 layers, 512 hidden, 8 heads, sparse attention:
    - Local 3×3×3 attention on all tokens (non-masked tokens summarise neighbourhood)
    - Global attention on masked tokens only (pulls context from full sequence)
  Average score over masked tokens → single real/fake value per sample.

Hinge loss:
  D real : mean(relu(1 - score))
  D fake : mean(relu(1 + score))
  G      : mean(-score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import TOKEN_DIM, TOKEN_T, TOKEN_H, TOKEN_W, LATENT_C, PATCH_SIZE
from model.network import Pos3D

DISC_HIDDEN  = 512
DISC_HEADS   = 8
DISC_LAYERS  = 4
LOCAL_K      = 3
CONV_CHANNELS = [64, 128, DISC_HIDDEN]   # per-token conv feature extractor


# ---------------------------------------------------------------------------
# Local 3×3×3 attention  (reuses same gather logic as generator)
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
# FFN
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
# Full discriminator
# ---------------------------------------------------------------------------

class TokenConvExtractor(nn.Module):
    """
    Treats each token as a (LATENT_C, PATCH_SIZE, PATCH_SIZE) feature map and
    runs 3 conv layers over its spatial content before projecting to DISC_HIDDEN.
    Replaces the flat Linear(TOKEN_DIM, DISC_HIDDEN) input projection so the
    discriminator can detect spatial artifacts within a single patch.
    """

    def __init__(self):
        super().__init__()
        ch = [LATENT_C] + CONV_CHANNELS
        layers = []
        for in_c, out_c in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.GELU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, tokens):
        # tokens: (B, N, TOKEN_DIM)
        B, N, _ = tokens.shape
        x = tokens.reshape(B * N, LATENT_C, PATCH_SIZE, PATCH_SIZE)
        x = self.conv(x)                                    # (B*N, DISC_HIDDEN, H, W)
        mean = x.mean(dim=(-2, -1))                         # (B*N, DISC_HIDDEN)
        std  = x.std(dim=(-2, -1), unbiased=False)          # (B*N, DISC_HIDDEN)
        return torch.cat([mean, std], dim=-1).reshape(B, N, DISC_HIDDEN * 2)


# ---------------------------------------------------------------------------
# Score funnel: cross-attention from one learnable query → single score/sample
# ---------------------------------------------------------------------------

class ScoreFunnel(nn.Module):
    """Mean-pool masked tokens → 2-layer MLP score."""

    def __init__(self, hidden, n_heads):
        super().__init__()
        self.norm  = nn.LayerNorm(hidden)
        self.mlp   = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask_flat):
        counts = mask_flat.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B,1,1)
        pooled = (x * mask_flat.unsqueeze(-1)).sum(dim=1) / counts.squeeze(1)   # (B, D)
        return self.mlp(self.norm(pooled.float()).to(x.dtype)).squeeze(-1)       # (B,)


# ---------------------------------------------------------------------------
# Full discriminator
# ---------------------------------------------------------------------------

class LatentDiscriminator(nn.Module):
    """
    Sparse transformer discriminator.
    Input: patchified latents (B, N, TOKEN_DIM).
    Output: (B,) scores — one per sample.
    """

    def __init__(self):
        super().__init__()
        self.input_proj = TokenConvExtractor()
        self.proj_down  = nn.Linear(DISC_HIDDEN * 2, DISC_HIDDEN, bias=False)
        self.pos_enc    = Pos3D(DISC_HIDDEN)
        self.layers     = nn.ModuleList([
            DiscTransformerLayer(DISC_HIDDEN, DISC_HEADS) for _ in range(DISC_LAYERS)
        ])
        self.funnel     = ScoreFunnel(DISC_HIDDEN, DISC_HEADS)
        self._init_weights()
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.to(torch.float32)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        x = self.proj_down(self.input_proj(tokens)) + self.pos_enc().to(dtype=tokens.dtype, device=tokens.device)
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
