"""
Latent-space discriminator for the GAN loss.

Operates per-token on patchified latents (TOKEN_DIM = 256 values each),
applied only to masked (hole) tokens. Never needs a VAE decode.

Hinge loss:
  D real : mean(relu(1 - score))
  D fake : mean(relu(1 + score))
  G      : mean(-score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import TOKEN_DIM


class LatentDiscriminator(nn.Module):
    """
    Per-token MLP discriminator.
    Each masked token is scored independently; scores are averaged for the loss.
    Position-agnostic by design — forces the generator to produce locally
    realistic content regardless of where in the frame the hole appears.
    """

    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOKEN_DIM, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens, mask_flat):
        """
        tokens    : (B, N, TOKEN_DIM)  patchified latents
        mask_flat : (B, N) bool
        Returns   : (M,) scores — one per masked token across the batch
        """
        return self.net(tokens[mask_flat]).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def d_hinge(real_scores, fake_scores):
    """Discriminator hinge loss."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def g_hinge(fake_scores):
    """Generator hinge loss — maximise discriminator score on fakes."""
    return -fake_scores.mean()
