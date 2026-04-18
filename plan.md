# One Shot Stereo — Plan

## Status

Data loading, augmentation, VAE integration, network skeleton, discriminator, loss functions,
checkpointing, and training loop scaffolding are all in place.

---

## Remaining work

### 1. Fix training loop exhaustion (`train.py`)
The `for batch in loader:` only iterates one epoch then stops. Wrap in an infinite loop so
training continues until `max_steps` is reached or the user interrupts.

### 2. Call `disc.train()` (`train.py`)
`net.train()` is called but `disc.train()` is not. Add it alongside.

### 3. True sparse global attention (`model/network.py`)
`GlobalAttention` currently computes full N×N attention (N=2730) for every token then zeros
non-masked outputs. The design calls for only masked tokens to act as queries, attending to
all N tokens as keys/values. This reduces the attention cost from O(N²) to O(M×N) where
M ≈ 5–10% of N (~136–273 tokens). Implementation sketch:
- Extract Q only from masked positions: `(B, M, D)` — requires padding since M varies per sample
- Keep K, V from all N tokens: `(B, N, D)`
- Run sdpa: `(B, H, M, d)` × `(B, H, N, d)ᵀ`
- Scatter results back into the full `(B, N, D)` tensor at masked positions

### 4. Pre-encode training data to disk
The design calls for encoding all training videos to latents offline before training begins so
the VAE never runs during the training loop. Currently the VAE runs every step.
- Write an offline encoding script that walks `training_data/` and saves `.pt` latent files
  alongside the existing `.npz` and `.mp4` files
- Update the dataset to load pre-encoded latents directly instead of raw video frames
- This frees the full GPU budget for the network during training

### 5. Validation / sample logging
No validation loop exists yet. Add:
- Periodic decode of a fixed held-out clip to pixel space and save as a video
- Log PSNR inside holes vs ground truth
- Runs every N steps (e.g. every 2000 steps), separate from the training checkpoints
