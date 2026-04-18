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

**1. Weight init (HIGH — main convergence risk)**
10-layer transformer with PyTorch defaults is unstable. Add:
- Xavier/truncated-normal on all `nn.Linear`
- Near-zero init on `output_proj` so initial prediction ≈ 0 and residual-style loss behaves
- Small std (≤0.02) on `Pos3D` embeddings

**2. Predict a residual, not absolute latents (HIGH — design allows it implicitly)**
Currently the net must reconstruct non-masked latents exactly to satisfy `latent_l1_outside` +
`pixel_l1_outside`. Change to `out = patchify(input) + delta` (or `output_proj` zero-init), so
early training is near-identity. Big convergence win, no spec conflict.

**3. `train.py` entry point**
`__main__` block has `max_steps=1, batch_size=2` — smoke-test config. Swap to real defaults
(e.g. `max_steps=None`, `batch_size=4`), or expose CLI flags.

**4. AdamW hyperparams**
Set `betas=(0.9, 0.95)` and explicit `weight_decay=0.01` (transformer norm). Currently defaults.

**5. Memory headroom for batch ≥ 4**
With bf16 + 10 layers + FFN expand=4 + 2730 tokens, the local-attn `_gather_neighbors`
materialises `(B, 2730, 27, 2048)`. Add `torch.utils.checkpoint` on each
`SparseTransformerLayer` — cheap to add, unlocks the design's batch=4–8 target.

**6. Validation + sample previews**
`VAL_SUBFOLDERS = "f"` is defined but never used. Add a periodic eval (every 1–2k steps):
latent L1 in/out on val, and decode a couple of clips for visual inspection. Without it you
can't tell convergence from memorisation.

**7. GlobalAttention edge case**
`M = 0` returns zeros — correct but means a clip with no holes does nothing. Fine, but worth
confirming the dataset never produces such batches (it shouldn't given how clips are selected).

**8. Loss-weight sanity check (MEDIUM)**
`λ_pixel_out=1.0` vs `λ_latent_out=0.1` — pixel L1 is in [0,1] space, latents are unbounded.
After VAE decode, first few runs will likely show pixel loss dominating. Log ranges for the
first 100 steps and re-tune.

**9. LayerNorm in fp32 under bf16 net**
Current `.to(torch.bfloat16)` casts LN params to bf16 too. Standard practice: keep LN in fp32
for stability. Either use `autocast` instead of blanket cast, or manually keep `LayerNorm`
parameters in fp32.

**10. Minor**
- `num_workers=2` is low for a 3090 on pre-decoded mp4+npz; try 4–8.
- `input_proj` takes the concatenated (token, flag); `bias=True` already (default). OK.
- Consider attention dropout / FFN dropout (optional, small task — probably skip).

**11. Check green flicker at step 10k**
Run inference at step 10k and inspect infilled regions for residual green flickering. If still
visible, raise `GAN_START_STEP` (e.g. to 20k–30k) so L1 has fully suppressed the green before
the GAN starts pushing for sharpness. GAN fighting a half-converged green baseline will hurt
more than help.

### Suggested order
1, 2, 3, 4 first (~half a day, directly gate convergence). Then 5, 6 before kicking off the
long run. 8 tuned once you have real loss traces. 7, 9, 10 as you go.
