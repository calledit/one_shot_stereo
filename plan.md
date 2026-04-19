# One Shot Stereo — Plan

## Status

Data loading, augmentation, VAE integration, network skeleton, discriminator, loss functions,
checkpointing, and training loop scaffolding are all in place.

---

## Network review vs design_document.md

### Matches the design
- Token geometry 7×15×26=2730, 256+1 flag, 1024 hidden, 16 heads
- Pre-norm transformer, local 3×3×3, sparse global (masked-Q → all-KV)
- BF16 net, FP16 VAE, SDPA (auto flash), grad clip 1.0
- Hinge GAN on masked latent tokens, pixel L1 subset, loss-weight ramp

### Gaps to close before a real training run

**8. Loss-weight sanity check (MEDIUM)**
`λ_pixel_out=1.0` vs `λ_latent_out=0.1` — pixel L1 is in [0,1] space, latents are unbounded.
After VAE decode, first few runs will likely show pixel loss dominating. Log ranges for the
first 100 steps and re-tune.

**9. LayerNorm in fp32 under bf16 net**
Current `.to(torch.bfloat16)` casts LN params to bf16 too. Standard practice: keep LN in fp32
for stability. Either use `autocast` instead of blanket cast, or manually keep `LayerNorm`
parameters in fp32.

**10. Minor**
- Consider attention dropout / FFN dropout (optional, small task — probably skip).

**11. Check green flicker at step 10k**
Run inference at step 10k and inspect infilled regions for residual green flickering. If still
visible, raise `GAN_START_STEP` (e.g. to 20k–30k) so L1 has fully suppressed the green before
the GAN starts pushing for sharpness. GAN fighting a half-converged green baseline will hurt
more than help.
