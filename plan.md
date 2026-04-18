# One Shot Stereo — Plan

## 1. ~~Attention efficiency~~
Implemented
The local attention `_gather_neighbors` creates large intermediate tensors `(B, N, 27, D)`.
The global attention uses a Python loop over batch items.
Both should use `F.scaled_dot_product_attention` (PyTorch flash attention) for speed and memory.

## 2. ~~Pixel L1 loss (outside holes)~~
Implemented: latent L1 outside holes every step (0.1×) plus pixel L1 outside holes
every ~10th step via gradient-checkpointed VAE decode (1.0×).

## 3. ~~GAN loss (discriminator in latent space)~~
Implemented
A discriminator operating on the 2730-token latent representation. Pushes fill quality beyond
what L1 alone achieves. Needs adversarial training loop with a separate optimizer.

## 4. Loss scheduling
Dynamic λ weights that shift from latent L1 toward GAN loss as training progresses.

## 5. Training quality-of-life
Gradient clipping, LR warmup + decay, loss logging to file.

## 6. Pre-encode training data to disk
Walk `training_data/` and save `<stem>_latents.npz` files so the VAE never runs during
training iterations. Update `dataset.py` to load pre-encoded latents directly.
Not needed until training data generation is complete.
