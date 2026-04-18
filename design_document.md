# Stereo Disocclusion Infill Network — Design Document

## Problem Statement
When converting mono camera footage to stereo by shifting the camera in 3D space (using a depth map derived from the original image), the rendered stereo frames contain black disocclusion regions — areas that were hidden behind foreground objects in the original shot and have no pixel data. The goal is to inpaint these regions coherently using spatial and temporal context from surrounding frames.

## Overall Approach
One-shot latent space transformer. No diffusion, no iterative denoising. The network takes a fixed window of video latents with disocclusion regions encoded as green pixels, and directly predicts the filled latents in a single forward pass.

---

## Data Pipeline

### Training Data Generation
- Render synthetic stereo pairs from mono footage + depth maps
- Ground truth is available since we control the rendering
- Infinite data can be generated — generation speed is the practical bottleneck
- Render at **832×480** (not 848×480 — 832 is divisible by 16 and then by 2 for patching)

### Mask Encoding
- Paint disocclusion regions **green (RGB 0, 255, 0)** directly in pixel space before encoding
- Green is visually distinctive and encodes to a consistent, recognizable latent signature
- Sharp pixel-accurate boundaries are preserved through the VAE encoding naturally
- A separate **binary flag per token** (described below) disambiguates mask green from genuine green content in the scene

### Pre-encoding
- All training videos are encoded offline using **TAEW2_1** (Tiny AutoEncoder for Wan 2.1)
- Encoded latents are saved to disk
- The VAE never runs during training iterations — GPU is 100% focused on the network
- This dramatically speeds up training on the 3090

---

## VAE — TAEW2_1
- Tiny distilled version of the Wan 2.1 VAE
- 4-6× faster than the full Wan VAE
- Produces **identical 16-channel latents** to the full Wan 2.1 VAE
- Fully compatible — no architectural changes needed downstream
- Used for both training (encoding + occasional pixel loss decode) and inference
- Use 16 bit precision throughout
- Convention note: TAEW2_1 uses pixel values in [0,1] not [-1,1], and NTCHW dimension order not NCTHW. No latent scale/shift needed unlike Diffusers convention.

### Compression ratios
- Spatial: **8×** (832→104, 480→60)
- Temporal: **4×**
- Latent volume for 25 frames: **7 × 16 × 60 × 104**

---

## Input/Output Format

### Frames
- **25 input frames** at 832×480 with green painted disocclusion regions
- Encoded to **7 × 16 × 60 × 104** latents via TAEW2_1
- 25 frames chosen because temporal compression gives 1 + (24/4) = 7 latents — clean alignment with the 4× temporal compression (valid frame counts follow 1 + 4n pattern: 1, 5, 9, 17, 25...)

### Network Input
- 7 latent volume: **7 × 16 × 60 × 104**
- Plus binary mask flags (described in tokenization)

### Network Output
- 7 latent volume: **7 × 16 × 60 × 104**
- Decoded back to 25 frames via TAEW2_1

---

## Tokenization

### Spatial Patching
- **4×4 spatial patches** on the latent volume
- Each patch covers 16 latent positions × 16 channels = **256 values**
- Total tokens: 7 × 26 × 15 = **2,730 tokens**
- 16× fewer tokens than 1×1 patching → ~256× faster attention

### Binary Mask Flag
- Each token gets **+1 binary value** indicating whether any pixel in its 4×4 patch area was painted green
- Computed trivially before encoding from the known mask
- Disambiguates mask green from genuine green scene content (forest, cars, traffic lights etc.)
- The flag handles "am I a masked token?" globally and reliably
- The latent values themselves handle "where exactly within my 4×4 patch is the green?" locally
- Final token size: **257 values** (256 latent + 1 flag)

---

## Network Architecture

### Type
**Transformer** — not a UNet. Chosen because:
- Global attention lets masked tokens pull fill content from anywhere in space and time
- Green/mask detection is local (within token), fill content requires global context
- Attention naturally learns to weight nearby non-masked tokens most strongly
- More expressive than convolutions for context propagation across thin disocclusion strips

### Input Projection
- Linear layer: **257 → 1024** dimensions
- Expands each token into a rich representational space for the transformer to work in
- The projection matrix learns arbitrary combinations of all 65 input values

### Positional Encoding
- 3D positional encoding covering temporal (7), height (26), width (15) dimensions
- Required so the transformer knows where each token sits in the spatiotemporal volume
- Sinusoidal or learned — either works, learned may be slightly better for this domain

### Transformer Layers
- **10 layers** of self-attention + FFN if more is required we add more layers
- Shallower than Wan (which needs depth for novel content generation) because:
  - Green detection is trivial and local
  - Fill content is strongly constrained by surrounding pixels
  - Task is much simpler than general video generation
- **Sparse attention** with two levels per layer:

**Local attention — all tokens**
- Every token attends to its immediate spatial and temporal neighbors
- Small fixed window — 3×3×3 in token space
- Handles texture continuity and ensures non-green tokens stay stable
- Cheap — fixed cost regardless of sequence length

**Global attention — green (masked) tokens only**
- Only tokens with mask flag = 1 perform full global attention
- Green tokens can attend to any of the 2,730 tokens in the sequence
- This is where fill content is pulled from distant but relevant regions in space and time
- Non-green tokens never perform global attention — they don't need it

**Why sparse attention works here**
- Disocclusion regions are typically 5-10% of the frame
- So only ~5-10% of tokens (~136-273) ever do global attention
- Global attention cost: ~273² instead of 2,730² = ~100× cheaper
- The mask flag already computed at tokenization time defines the sparsity pattern — no additional computation needed to determine which tokens go global
- Flash attention for both local and global attention passes

### Output Projection
- Linear layer: **1024 → 256** dimensions
- Projects back down from transformer hidden space to patch values

### Reshape
- Unpatch: **2,730 × 256 → 7 × 16 × 60 × 104**
- This is the predicted filled latent volume

---

## Loss Function

### 1. Latent L1 Loss (inside holes only)
- Compare predicted latents vs ground truth latents **only within masked regions**
- Runs **every training step** — cheap, no decode needed
- Drives broad correctness of fill content
- Weighted heavily early in training

### 2. Pixel L1 Loss (outside holes only)
- Decode predicted latents via TAEW2_1 to pixel space
- Compare against ground truth pixels **only outside masked regions**
- Purpose: ensure the network doesn't shift or distort real pixel content — output must align with input
- Runs on a **subset of batches** (e.g. every 10th, or ~10-20% probability per batch) to avoid paying the decode cost every step
- TAEW2_1's 4-6× speed advantage over full VAE makes this more affordable than it would otherwise be

### 3. Latent GAN Loss (inside holes only)
- A discriminator operating purely in **latent space** — never needs pixel decoding
- Pushes fill latents to be indistinguishable from real content latents
- Encourages sharpness and realistic texture in filled regions
- Justified here because perceptual loss (VGG features) requires pixel decoding and is therefore too expensive to run frequently
- Discriminator operates on the 2,730 token latent representation

### Combined Loss
```
loss = λ_latent_l1 * latent_l1_loss (holes)
     + λ_pixel_l1  * pixel_l1_loss  (outside holes, subset of batches)
     + λ_gan       * gan_loss        (holes)
```
Shift weight from latent L1 toward GAN loss as training progresses for sharper results.

---

## Hardware & Training Setup

- **GPU:** RTX 3090 (24GB VRAM)
- **Target:** train within one week (~168 hours)
- **Precision:** BF16 throughout, fp32 only where needed — network weights, activations, VAE
- **Pre-encoding:** All training data encoded to latents offline and saved to disk before training begins. Training loop never touches the VAE encoder.
- **Batch size:** Experiment to find what fits — likely 4–8 given the small network and pre-encoded latents
- **Data:** Infinite synthetic stereo pairs available, generation speed is the bottleneck not data quantity

---

## Inference Pipeline
1. Take input stereo frame sequence (25 frames)
2. Paint disocclusion regions green
3. Compute binary mask flags for each 2×2 latent patch
4. Encode frames through TAEW2_1 → **7 × 16 × 60 × 104**
5. Tokenize: reshape + 4×4 patch → **2,730 × 257**
6. Project up: **2,730 × 257 → 2,730 × 1024**
7. Run through transformer layers
8. Project down: **2,730 × 1024 → 2,730 × 256**
9. Unpatch + reshape → **7 × 16 × 60 × 104**
10. Decode through TAEW2_1 → **25 frames** with disocclusions filled
11. Composite output pixels back onto input using the original mask — real pixels from input, filled pixels from network output

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Approach | One-shot | Diffusion dissolves green mask signal across denoising steps |
| Architecture | Transformer | Global attention needed for fill context propagation |
| VAE | TAEW2_1 | 4-6× faster, identical latents to full Wan 2.1 VAE |
| Resolution | 832×480 | Divisible by 16 (VAE) and 2 (patching) |
| Temporal window | 25 frames / 7 latents | Enough temporal context for consistency, manageable VRAM |
| Mask encoding | Green pixels | Full resolution, sharp boundaries, consistent latent signature |
| Mask disambiguation | Binary flag per token | Separates mask green from genuine scene green cheaply |
| Patching | 4×4 spatial | 256× attention speedup vs 1×1 |
| Token size | 257 (256 latent + 1 flag) | Natural grouping of channels, flag added for mask disambiguation |
| Hidden dim | 1024 | Standard middle ground for model capacity |
| Depth | 10 layers | Task is simpler than general video generation |
| Attention | Sparse (local all tokens, global green only) | ~100× cheaper global attention, mask flag defines sparsity for free |
| Precision | BF16 | Speed and memory efficiency |
| Latent loss | L1 inside holes | Cheap, every step, broad correctness |
| Pixel loss | L1 outside holes | Ensures pixel alignment, run occasionally |
| Perceptual/sharpness | Latent GAN | Can't do VGG features without decoding, GAN stays in latent space |



## Training Data

### File layout

All clips live under training_data/ split into 16 subfolders (0–f) based on the first hex digit of the video
filename's CRC32. Each processed video produces three files:

training_data/<hex>/
  <name>_f<NNNNNN>_gt.mp4   — ground-truth video
  <name>_f<NNNNNN>.npz      — hole mask
  <name>_f<NNNNNN>.txt      — status line (OK or rejection reason)

<NNNNNN> is the zero-padded start frame index within the source video.

---
### Ground-truth video (_gt.mp4)

- Resolution: 832 × 480 RGB
- 25 frames (source video < 50 frames) or 50 frames (source video ≥ 50 frames)
- Content: the stereo-shifted frame with disocclusion holes filled in from the reference view. This is what the
network must learn to output.
The 50 frames clips can be used for data augmenation you can use it to generate three 25 frame samples. Start, middle, end.

---
### Hole mask (.npz)

Single array hole_mask, shape (N, 480, 832) bool, where N matches the frame count of the paired video (25 or 50).

True = this pixel is a disocclusion hole in this frame.

---
## Use of traning data

To use the traning data use the hole mask to paint the ground truth frames green where needed. Also create token masks.
In 10% of the clip the first frame will have the hole filed with blured ground truth in it instead of green.
So we can give the last frame as the first to achive temporal consistency between chunks when doing inference.

The data can also be augmented by reversing it in time or fliping the frames left-right.