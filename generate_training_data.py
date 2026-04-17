# generate_training_data.py
# Generates 25-frame training clips for the Stereo Disocclusion Infill Network.
#
# Each .npz sample contains:
#   green_frames : (25, 480, 832, 3) uint8 RGB  — stereo frames, disocclusions painted green
#   gt_frames    : (25, 480, 832, 3) uint8 RGB  — stereo frames, disocclusions filled from reference
#   token_mask   : (7, 15, 26) bool             — per-token binary mask for network input
#
# Token grid math (TAEW2_1 + 2×2 patching):
#   spatial  832 / (16 * 2) = 26 cols,  480 / (16 * 2) = 15 rows
#   temporal (25 - 1) / 4 + 1 = 7  (frames map: t0→f0, t1→f1-4, ..., t6→f21-24)

import os
import cv2
import zlib
import zipfile
import numpy as np
import sys

sys.path.append("C:\\Users\\calle\\projects\\metric_depth_video_toolbox")
import depth_frames_helper
import depth_map_tools
import video_da3
import unik3d_video
from contextlib import redirect_stdout
import torch
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt, binary_dilation
import random
import re

dataset_folder = "F:\\zip\\Moments_in_Time_Raw_v2.zip"
training_output_folder = "training_data" + os.sep
video_cache_folder = "video_cache" + os.sep

depth_resolution = 504       # DA3 native resolution; must be 252 or 504
MODEL_maxOUTPUT_depth = 100.0

OUTPUT_W = 832
OUTPUT_H = 480
FRAMES_PER_CLIP = 25

# Token-level mask dimensions
TOKEN_T = 7    # temporal latents: (25-1)/4 + 1
TOKEN_H = 15   # height tokens:    480 / (16*2)
TOKEN_W = 26   # width tokens:     832 / (16*2)
PIXELS_PER_TOKEN = 32  # 16 (VAE spatial) * 2 (patch size)

da3model = None
unk3dmodel = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rename(src, dst):
    if os.path.exists(dst) and os.path.exists(src):
        os.remove(dst)
    os.rename(src, dst)


def fill_depth_nearest(depth, invalid_mask):
    """Replace invalid depth pixels with nearest valid depth value."""
    _, (iy, ix) = distance_transform_edt(invalid_mask, return_indices=True)
    filled = depth.copy()
    filled[invalid_mask] = depth[iy[invalid_mask], ix[invalid_mask]]
    return filled


def compute_token_mask(frame_masks):
    """
    Convert per-frame pixel-space hole masks to per-token binary flags.

    frame_masks : (25, OUTPUT_H, OUTPUT_W) bool
    Returns     : (TOKEN_T, TOKEN_H, TOKEN_W) bool

    Temporal grouping matches TAEW2_1 4× compression:
        t=0 → frame 0 only
        t=k → frames 4k-3 … 4k  (k ≥ 1)
    Spatial: each token covers PIXELS_PER_TOKEN × PIXELS_PER_TOKEN pixels.
    A token is flagged if ANY covered pixel is a hole in ANY covered frame.
    """
    assert frame_masks.shape == (FRAMES_PER_CLIP, OUTPUT_H, OUTPUT_W), \
        f"Expected ({FRAMES_PER_CLIP}, {OUTPUT_H}, {OUTPUT_W}), got {frame_masks.shape}"

    temporal_groups = [
        frame_masks[0:1],    # t=0: frame 0
        frame_masks[1:5],    # t=1: frames 1-4
        frame_masks[5:9],    # t=2: frames 5-8
        frame_masks[9:13],   # t=3: frames 9-12
        frame_masks[13:17],  # t=4: frames 13-16
        frame_masks[17:21],  # t=5: frames 17-20
        frame_masks[21:25],  # t=6: frames 21-24
    ]

    token_mask = np.zeros((TOKEN_T, TOKEN_H, TOKEN_W), dtype=bool)
    for t, group in enumerate(temporal_groups):
        # OR across time → (OUTPUT_H, OUTPUT_W)
        spatial_or = group.any(axis=0)
        # Spatial maxpool: reshape to (TOKEN_H, PX, TOKEN_W, PX) and OR
        spatial_or = spatial_or.reshape(TOKEN_H, PIXELS_PER_TOKEN,
                                        TOKEN_W, PIXELS_PER_TOKEN)
        token_mask[t] = spatial_or.any(axis=(1, 3))

    return token_mask


def clean_filename(s, max_len=150):
    s = s.split('?', 1)[0]
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    if len(s) > max_len:
        root, ext = os.path.splitext(s)
        s = root[:max_len - len(ext)] + ext
    return s


def write_report(msg, path):
    print(msg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(msg)
    return None


# ---------------------------------------------------------------------------
# Stereo rendering — one frame at a time
# ---------------------------------------------------------------------------

def make_sample_for_clip(depth, img_rgb, cam_matrix, do_right,
                         simulate_convergense, ipd_baseline,
                         output_W, output_H, transform_to_ref,
                         use_ref_as_base, do_dilate):
    """
    Render one synthetic stereo frame and compute its disocclusion mask.
    
    In this function we start of with the ground truth img_rgb
    and we will pretend that it was generated from a diffrent image (The "virtual original")
    
    We start of by morphing our ground truth in to the "virtual original".
    We then take that "virtual original" and morph it back in to a syntetic stereo image which is how the ground truth would have looked if it was generated from the "virtual original".
    

    Returns:
        green_frame : (output_H, output_W, 3) uint8 RGB — disocclusions painted green (0,255,0)
        gt_frame    : (output_H, output_W, 3) uint8 RGB — disocclusions filled from reference view
        hole_mask   : (output_H, output_W) bool          — True where disocclusion hole
    """
    near = 0.02
    far = 210.0

    H, W = depth.shape
    down_scaled_org = cv2.resize(img_rgb, (W, H)).astype(np.uint8)
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)

    # Camera pose (same convention as original script)
    open_cv_w2c = np.linalg.inv(transform_to_ref)
    A = np.diag([1, -1, -1, 1]).astype(np.float32)
    V_gl_row = A @ open_cv_w2c @ A
    base_pos = np.linalg.inv(V_gl_row)

    cam_world = np.linalg.inv(base_pos)
    cam_pos = cam_world[:3, 3]
    forward = cam_world[:3, :3] @ np.array([0, 0, -1], dtype=np.float32)
    forward /= np.linalg.norm(forward)
    delta_d = float(np.dot(cam_pos, forward))

    depth = depth.clip(near, 100)
    convergence_distance = float(depth.mean())

    mesh, _ = depth_map_tools.mesh_from_depth_and_rgb(depth, down_scaled_org, cam_matrix)

    convergence_angle_rad = 0.0
    if simulate_convergense:
        convergence_angle_rad = depth_map_tools.convergence_angle(convergence_distance, 0.063)

    if do_right:
        center_movement = 0.0351
        center_angle = -convergence_angle_rad
    else:
        center_movement = -0.0351
        center_angle = convergence_angle_rad

    proj = depth_map_tools.open_gl_projection_from_camera_matrix(cam_matrix, near, far)
    model = np.eye(4, dtype=np.float32)

    # --- Reference view render ---
    mvp_ref = proj @ base_pos @ model
    ref_image, ref_depth, _, ref_ids = depth_map_tools.gl_render(
        mesh, mvp_ref, W, H, near, far, bg_color=[0.0, 0.0, 0.0])
    ref_depth = ref_depth.copy() - delta_d
    ref_depth[ref_ids == 0] = 0
    ref_depth = ref_depth.clip(0, 100)

    # --- First morph: shift to stereo offset (to the virtual original iamge) ---
    vc1 = depth_map_tools.get_cam_view(center_movement, center_angle, reverse=True)
    if ipd_baseline == 0.063:
        view1 = depth_map_tools.get_cam_view(-center_movement, -center_angle) @ vc1
    else:
        view1 = vc1

    gen_img, gen_depth, _, first_morph_ids = depth_map_tools.gl_render(
        mesh, proj @ view1 @ model, W, H, near, far, bg_color=[1.0, 0.0, 1.0])

    background1 = first_morph_ids == 0
    gen_depth = fill_depth_nearest(gen_depth, background1)
    gen_depth = gen_depth.clip(0, 165)
    
    edge_mask_l, edge_mask_r = depth_map_tools.steep_disparity_lr(
        gen_depth, cam_matrix, parallax_shift=ipd_baseline)
    if not do_right:
        edge_mask = edge_mask_r
    else:
        edge_mask = edge_mask_l

    mesh_morphed, _ = depth_map_tools.mesh_from_depth_and_rgb(gen_depth, gen_img, cam_matrix)

    # IDs that were background in first morph (edge artefacts, not true disocclusions)
    first_morph_ids = (np.arange(H * W, dtype=np.uint32) + 1).reshape(H, W)
    first_morph_edge_ids = first_morph_ids[edge_mask]
    first_morph_bg_ids = first_morph_ids[background1]

    # --- Second render: back to original view at 2× resolution ---
    if ipd_baseline == 0.063:
        vc2 = depth_map_tools.get_cam_view(-center_movement, -center_angle, reverse=True)
        view2 = depth_map_tools.get_cam_view(center_movement, center_angle) @ vc2
    else:
        view2 = depth_map_tools.get_cam_view(center_movement, center_angle)
    view2 = view2 @ base_pos

    cam_matrix_2x = depth_map_tools.compute_camera_matrix(fovx, fovy, W * 2, H * 2)
    proj_2x = depth_map_tools.open_gl_projection_from_camera_matrix(cam_matrix_2x, near, far)

    original_perspective_img, _, _, ids_2x = depth_map_tools.gl_render(
        mesh_morphed, proj_2x @ view2 @ model,
        W * 2, H * 2, near, far, bg_color=[0.0, 1.0, 0.0])

    # background2x = true disocclusion holes (no geometry rendered here)
    background2x = ids_2x == 0
    background_from_first_morph = np.isin(ids_2x, first_morph_bg_ids)
    edges_from_first_morph = np.isin(ids_2x, first_morph_edge_ids)

    # We could use the real original image as ground truth but then we would not capture any quality loss incured in rendering
    # So we use as many pixels as posible from the re render
    org_rgb2x = cv2.resize(ref_image, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
    # Ground truth: fill disocclusions AND first-morph artefacts with reference content
    degraded_ground_truth = original_perspective_img.copy()
    # Some pixels where lost in the first morph so we fill them in
    degraded_ground_truth[background_from_first_morph] = org_rgb2x[background_from_first_morph]
    degraded_ground_truth = cv2.resize(degraded_ground_truth, (output_W, output_H), interpolation=cv2.INTER_AREA).astype(np.uint8)
    
    
    if use_ref_as_base:
        ref_resized = cv2.resize(ref_image, (output_W, output_H), interpolation=cv2.INTER_AREA).astype(np.uint8)
        green_frame = ref_resized
        gt_frame = ref_resized.copy()
    else:
        green_frame = degraded_ground_truth
        gt_frame = degraded_ground_truth.copy()

    hole_mask2x = edges_from_first_morph
    hole_mask2x[background2x] = 1
    # Hole mask at output resolution (nearest-neighbour — holes are never single pixels)
    hole_mask_out = cv2.resize(
        hole_mask2x.astype(np.uint8), (output_W, output_H),
        interpolation=cv2.INTER_NEAREST).astype(bool)
    
    
    if do_dilate:
        hole_mask_out = binary_dilation(hole_mask_out, iterations=1)

    # Green frame: GT with holes repainted green
    green_frame[hole_mask_out] = [0, 255, 0]

    return green_frame, gt_frame, hole_mask_out


# ---------------------------------------------------------------------------
# Per-video sample generation
# ---------------------------------------------------------------------------

def convert_to_training_data(video_path, from_zip=False, zip_ref=None):
    base_name = os.path.basename(video_path)
    name_only = os.path.splitext(base_name)[0]

    crc_hex = f"{zlib.crc32(name_only.encode()):08x}"
    subfolder = crc_hex[0]

    img_output_folder = training_output_folder + subfolder + os.sep
    local_cache_folder = video_cache_folder + subfolder + os.sep
    org_video_path = local_cache_folder + clean_filename(base_name)
    name_only = clean_filename(name_only)

    meta_output = img_output_folder + name_only + '.txt'

    if os.path.exists(meta_output):
        print(f"already done: {meta_output}")
        return

    os.makedirs(img_output_folder, exist_ok=True)

    # Derive rendering settings from filename hash (same logic as original)
    first_nibble  = int(crc_hex[1], 16)
    second_nibble = int(crc_hex[2], 16)
    third_nibble  = int(crc_hex[3], 16)

    do_right = (first_nibble % 2 == 0)

    simulate_convergense = False
    if first_nibble < 8:
        if first_nibble < 4:
            simulate_convergense = True
        ipd_baseline = 0.0351   # center → left/right
    else:
        if first_nibble < 12:
            simulate_convergense = True
        ipd_baseline = 0.063    # left → right or right → left

    use_ref_as_base = (second_nibble >= 8)   # ~50 % of clips use ref render as base image
    do_dilate       = (third_nibble  >= 8)   # ~50 % of clips get a 1-px hole dilation

    # Extract video from zip to local cache if needed
    if from_zip:
        if not os.path.exists(org_video_path):
            data = zip_ref.read(video_path)
            os.makedirs(os.path.dirname(org_video_path), exist_ok=True)
            with open(org_video_path, 'wb') as f:
                f.write(data)
    else:
        org_video_path = video_path

    # Check video properties
    video = cv2.VideoCapture(org_video_path)
    vid_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  video: {vid_w}x{vid_h}  fps:{fps:.1f}  frames:{video_len}")

    def bail(msg):
        video.release()
        if from_zip and os.path.exists(org_video_path):
            os.remove(org_video_path)
        return write_report(msg, meta_output)

    if max(vid_w, vid_h) < 480:
        return bail(f"resolution too low: {vid_w}x{vid_h}")
    if video_len < FRAMES_PER_CLIP:
        return bail(f"video too short: {video_len} frames")

    # Read 25 consecutive frames from a random start position
    start_frame = random.randint(0, video_len - FRAMES_PER_CLIP)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_bgr = []
    for _ in range(FRAMES_PER_CLIP):
        ret, frame = video.read()
        if not ret:
            break
        frames_bgr.append(frame)
    video.release()

    if from_zip and os.path.exists(org_video_path):
        os.remove(org_video_path)

    if len(frames_bgr) != FRAMES_PER_CLIP:
        return write_report(f"could not read {FRAMES_PER_CLIP} frames", meta_output)

    # Resize to output resolution immediately — handles any source aspect ratio
    rgb_frames = [
        cv2.cvtColor(cv2.resize(f, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2RGB)
        for f in frames_bgr
    ]

    # Run depth estimation on all 25 frames
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        depth_out = da3model.inference(rgb_frames, process_res=depth_resolution)

    cam_matrix = depth_out.intrinsics[0]
    depths = depth_out.depth.clip(0, 100)
    H_d, W_d = depths[0].shape

    std_dev = float(depths[0].std())
    if std_dev < 0.12:
        return write_report(
            f"depth std too low ({std_dev:.3f}) — flat scene or letterboxed", meta_output)

    mid = FRAMES_PER_CLIP // 2   # frame 12

    # Quality filter: compare DA3 and UniK3D depth on middle frame
    mid_rgb = cv2.resize(rgb_frames[mid], (W_d, H_d)).astype(np.uint8)
    rgb_torch = torch.from_numpy(mid_rgb).permute(2, 0, 1)
    unik_pred = unk3dmodel.infer(rgb_torch)
    unik_depth = unik_pred["depth"].squeeze().cpu().numpy().clip(0, 100)

    norm_d = depth_frames_helper.normalize_depth(depths[mid])
    unik_norm = depth_frames_helper.normalize_depth(unik_depth)
    ssim_val = float(ssim(norm_d, unik_norm, data_range=1.0))

    if ssim_val < 0.72:
        return write_report(f"depth disagreement ssim={ssim_val:.3f}", meta_output)

    # Interpolate depth maps to output resolution
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    depths_proc = np.stack([
        cv2.resize(d, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)
        for d in depths
    ])
    proc_cam = depth_map_tools.compute_camera_matrix(fovx, fovy, OUTPUT_W, OUTPUT_H)

    # No camera stabilisation — each frame rendered from identity transform
    identity = np.eye(4, dtype=np.float32)

    # Render all 25 frames
    green_frames_list = []
    gt_frames_list = []
    hole_masks_list = []

    for i in range(FRAMES_PER_CLIP):
        green_frame, gt_frame, hole_mask = make_sample_for_clip(
            depths_proc[i], rgb_frames[i], proc_cam,
            do_right, simulate_convergense, ipd_baseline,
            OUTPUT_W, OUTPUT_H, identity,
            use_ref_as_base, do_dilate,
        )
        green_frames_list.append(green_frame)
        gt_frames_list.append(gt_frame)
        hole_masks_list.append(hole_mask)

    green_arr = np.stack(green_frames_list)   # (25, H, W, 3) uint8
    gt_arr    = np.stack(gt_frames_list)      # (25, H, W, 3) uint8
    masks_arr = np.stack(hole_masks_list)     # (25, H, W) bool

    # 20% of clips: replace green holes in frame 0 with blurred GT content.
    # Simulates the inference chunk-boundary condition where the previous chunk's
    # last output frame is fed as frame 0 of the next chunk (network output is
    # slightly soft, not perfectly sharp like the original).
    if random.random() < 0.20:
        blurred_gt0 = cv2.GaussianBlur(gt_arr[0], (21, 21), 0)
        green_arr[0][masks_arr[0]] = blurred_gt0[masks_arr[0]]

    token_mask = compute_token_mask(masks_arr)   # (7, 15, 26) bool

    # Skip clips with no useful holes or implausibly many (bad depth)
    hole_ratio = float(masks_arr.mean())
    if hole_ratio < 0.001:
        return write_report(f"too few holes ({hole_ratio:.4f})", meta_output)
    if hole_ratio > 0.40:
        return write_report(f"too many holes ({hole_ratio:.4f}) — likely bad depth", meta_output)

    # Write atomically via temp files
    out_stem    = img_output_folder + name_only + f"_f{start_frame:06d}"
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")

    def write_video(frames_rgb, tmp_path, final_path):
        writer = cv2.VideoWriter(tmp_path, fourcc, 25.0, (OUTPUT_W, OUTPUT_H))
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        rename(tmp_path, final_path)

    write_video(green_arr, out_stem + "_tmp_green.mp4", out_stem + "_green.mp4")
    write_video(gt_arr,    out_stem + "_tmp_gt.mp4",    out_stem + "_gt.mp4")

    # .npz — token mask only
    tmp_npz   = out_stem + "_tmp.npz"
    final_npz = out_stem + ".npz"
    np.savez_compressed(tmp_npz, token_mask=token_mask)
    rename(tmp_npz, final_npz)

    with open(meta_output, "w") as f:
        f.write(f"OK ssim={ssim_val:.3f} std={std_dev:.3f} "
                f"holes={hole_ratio:.4f} start={start_frame}")

    print(f"  saved: {out_stem}_green.mp4  _gt.mp4  .npz")
    return True


# ---------------------------------------------------------------------------
# Dataset walker
# ---------------------------------------------------------------------------

def process_dataset_folder(dataset_path):
    exts = (".mp4",)   # videos only — we need temporal frames

    if dataset_path.lower().endswith(".zip"):
        print(f"Opening ZIP: {dataset_path}")
        with zipfile.ZipFile(dataset_path, 'r') as z:
            names = [
                n for n in z.namelist()
                if (n.startswith("unlabeled2017/") or
                    n.startswith("Moments_in_Time_Raw/training/"))
                and n.lower().endswith(exts)
            ]
            random.seed(42)
            random.shuffle(names)
            print(f"Found {len(names)} videos")
            for idx, name in enumerate(names):
                print(f"[{idx+1}/{len(names)}] {name}")
                convert_to_training_data(name, from_zip=True, zip_ref=z)
        return

    if os.path.isdir(dataset_path):
        print(f"Walking folder: {dataset_path}")
        paths = []
        for root, _, files in os.walk(dataset_path):
            for fname in files:
                if fname.lower().endswith(exts):
                    paths.append(os.path.join(root, fname))
        random.seed(42)
        random.shuffle(paths)
        print(f"Found {len(paths)} videos")
        for idx, p in enumerate(paths):
            print(f"[{idx+1}/{len(paths)}] {p}")
            convert_to_training_data(p)
        return

    print("ERROR: dataset_path is neither a folder nor a .zip file.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(training_output_folder, exist_ok=True)
    os.makedirs(video_cache_folder, exist_ok=True)
    print("Loading depth models...")
    da3model = video_da3.load_model()
    unk3dmodel = unik3d_video.load_model()
    process_dataset_folder(dataset_folder)
