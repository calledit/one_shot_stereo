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
import random
import re
import json
dataset_folder =  "E:\\zip\\Moments_in_Time_Raw_v2.zip" #"unlabeled2017.zip" # 
#dataset_folder = "dataset_imgs"
morph_to_ref_view = True
if morph_to_ref_view:
    training_output_folder = "training_data_extrin"+os.sep
else:
    raise Exception("Place them in the same folder, But only do this when the model is good at fxing morphed images")
    training_output_folder = "training_data_extrin"+os.sep

video_cache_folder = "video_cache"+os.sep

depth_resolution = 504#252 #we need to pick a resolution that aligns with the depth model so we can skip rescaling after the depth model da3 supports resolutions like 252, 504 ...
MODEL_maxOUTPUT_depth = 100.0#max 100m cut the rest of it wont make much paralx so it does not matter
save_resolution = 256

da3model = None
unk3dmodel = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def rename(from_path, to_path):
    if os.path.exists(to_path) and os.path.exists(from_path):
        os.remove(to_path)
    os.rename(from_path, to_path)

# ---------------------------------------------------------
# convert_to_training_data
# ---------------------------------------------------------

def downsample_mask_maxpool(mask, factor=2):
    H, W = mask.shape
    
    # pad to even
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor

    mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="edge")

    # reshape into blocks and OR
    new_h = mask_padded.shape[0] // factor
    new_w = mask_padded.shape[1] // factor

    return mask_padded.reshape(new_h, factor, new_w, factor).any(axis=(1, 3))

def show_imgs(list_of_imgs, titles=None, cols=3, figsize=(12, 8)):
    import matplotlib.pyplot as plt
    import math
    n_images = len(list_of_imgs)
    cols = min(cols, n_images)
    rows = math.ceil(n_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).reshape(-1)

    for i, img in enumerate(list_of_imgs):
        # Convert PIL → NumPy
        if not hasattr(img, "ndim"):
            img = np.array(img)

        if img.ndim == 2:
            axs[i].imshow(img, cmap="gray")
        else:
            axs[i].imshow(img)

        if titles:
            axs[i].set_title(titles[i])

        axs[i].axis("off")

    # Hide empty axes
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

from scipy.ndimage import distance_transform_edt

def fill_depth_nearest(depth, invalid_mask):
    # invalid_mask = True where depth is invalid
    # valid_mask   = True where depth is valid
    valid_mask = ~invalid_mask

    # distance transform: gives for each pixel the index of nearest valid pixel
    dist, (indices_y, indices_x) = distance_transform_edt(
        invalid_mask,
        return_indices=True
    )
    
    # copy the nearest valid depth into invalid ones
    filled = depth.copy()
    filled[invalid_mask] = depth[indices_y[invalid_mask], indices_x[invalid_mask]]
    return filled


def make_sample(depth, img_rgb, cam_matrix, do_right, simulate_convergense, ipd_baseline, ground_truth_side, virtual_org_side, output_W, output_H, transform_to_ref):
    near = 0.02
    far = 210.0
    
    H, W = depth.shape
    down_scaled_org_image = cv2.resize(img_rgb, (W, H)).astype(np.uint8)
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    
    
    #calculate transform
    #transform_to_ref = np.eye(4)
    open_cv_w2c = np.linalg.inv(transform_to_ref)   # this is your original w2c again

    # 2) Axis conversion CV -> OpenGL
    A = np.diag([1, -1, -1, 1]).astype(np.float32)

    # world_gl -> camera_gl (still row-major numbers)
    V_gl_row = A @ open_cv_w2c @ A

    # 3) Convert to column-major for OpenGL
    base_pos = V_gl_row#.T
    
    # 4. OpenGL view matrix is inverse of camera pose
    base_pos = np.linalg.inv(base_pos)
    
    # acount for camrea movment in new ref depth
    # --- Camera world transform ---
    V = base_pos#.T                   # row-major for math
    cam_world = np.linalg.inv(V)

    cam_pos = cam_world[:3, 3]
    forward_cam_world = cam_world[:3,:3] @ np.array([0,0,-1], dtype=np.float32)
    forward_cam_world /= np.linalg.norm(forward_cam_world)

    # --- Depth shift ---
    delta_d = np.dot(cam_pos, forward_cam_world)
    
    depth = depth.clip(near, 100)
    
    #lets hope our shourcut to convergence_distance dont have any adverse effects on the model
    convergence_distance = depth.mean()
    
    edge_mask_l, edge_mask_r = depth_map_tools.steep_disparity_lr(depth, cam_matrix, parallax_shift=ipd_baseline)
    full_edge_mask = depth_map_tools.steep_mask_disparity(depth, cam_matrix, parallax_shift=ipd_baseline)
    
    # Create a mesh from the data
    mesh, org_normals = depth_map_tools.mesh_from_depth_and_rgb(depth, down_scaled_org_image, cam_matrix)
    convergence_angle_rad = 0
    if simulate_convergense:
        convergence_angle_rad = depth_map_tools.convergence_angle(convergence_distance, 0.063)
    # Camera setup (origin, looking forward)
    
    if do_right:
        center_movment = 0.0351
        center_angle_rad = -convergence_angle_rad
    else:
        center_movment = -0.0351
        center_angle_rad = convergence_angle_rad


    

    
    proj = depth_map_tools.open_gl_projection_from_camera_matrix(cam_matrix, near, far)
    model = np.eye(4, dtype=np.float32)
    mvp = proj @ base_pos @ model
    
    ref_image, ref_depth, ref_normals, ref_ids = depth_map_tools.gl_render(mesh, mvp, W, H, near, far, bg_color = [0.0, 0.0, 0.0])
    
    #show_imgs([ref_image.astype(np.uint8), ref_depth, ref_normals])
    
    ref_depth = ref_depth.copy() - delta_d
    ref_background = ref_ids == 0
    ref_depth[ref_background] = 0
    
    ref_depth = ref_depth.clip(0, 100)

    view_center = depth_map_tools.get_cam_view(center_movment, center_angle_rad, reverse = True)
    if ipd_baseline == 0.063:
        view = depth_map_tools.get_cam_view(-center_movment, -center_angle_rad) @ view_center
    else:
        view = view_center
        
    
    
    mvp = proj @ view @ model
    
    
    
    # Render to a bigger surface to make sure that all pixels from the original image get renderd (or the visible ones) if we only render at orgiginal resolution then some pixels gets overwriten by their neighbors do to how the GPU does stuff
    generated_right_image, generated_right_depth, normals, first_morph_ids = depth_map_tools.gl_render(mesh, mvp, W, H, near, far, bg_color = [1.0, 0.0, 1.0])
    
    #view = depth_map_tools.get_cam_view(use_ipd_baseline, 0.0)
    #mvp = proj @ view @ model
    #generated_right_image_zero_convergence, _, _, _ = depth_map_tools.gl_render(mesh, mvp, W*2, H*2, near, far, bg_color = [1.0, 1.0, 0.0])
    #print("center view")
    #show_imgs([down_scaled_org_image, generated_right_image])
    #exit()
    
    background1 = first_morph_ids == 0
    
    #This is kind of ugly but to not fuck up normals we infill the depth of the background pixels. These pixels are not used later anyway.
    #The "correct" way if fixing this would be to change mesh_from_depth_and_rgb to take a mask and ignore calulations for those pixels. But that would be slow and allot of work.
    generated_right_depth = fill_depth_nearest(generated_right_depth, background1)
    
    #show_imgs([generated_right_image, background1, first_morph_ids])
    #background1 = generated_right_depth > 101
    #background1 = downsample_mask_maxpool(background1, factor=2)
    #Make sure that background pixels are not cliped away
    generated_right_depth = generated_right_depth.clip(0, 165)
    
    # Downscale after render
    #generated_right_depth = cv2.resize(generated_right_depth, (W, H), interpolation=cv2.INTER_NEAREST)
    #generated_right_image = cv2.resize(generated_right_image, (W, H)) 
    
    # Create a new mesh from the output of the first render a morphed mesh morped to the direction of choise the
    mesh_morpehed, normals = depth_map_tools.mesh_from_depth_and_rgb(generated_right_depth, generated_right_image, cam_matrix)
    
    #The ids of the pixels that have been calculated to need infill after paralax are marked
    first_morph_bg_ids = (np.arange(H * W, dtype=np.uint32) + 1).reshape(H, W)
    #first_morph_bg_ids = np.repeat(np.repeat(first_morph_bg_ids, 2, axis=0), 2, axis=1)
    first_morph_bg_ids = first_morph_bg_ids[background1]#This works cause it is referncing the id's in mesh_morpehed
    
    
    
    #ids_with_paralax = id_paralax_map[paralax_mask2]
    #hole_mask1 = np.isin(first_morph_ids, ids_with_paralax)
    
        
    
    # Generate a new projection matric to render the morphed mesh from the original camera position where the pixels that we callculated to probably have paralaxy will be more visible

    
    
    if ipd_baseline == 0.063:
        view_center = depth_map_tools.get_cam_view(-center_movment, -center_angle_rad, reverse = True)
        view = depth_map_tools.get_cam_view(center_movment, center_angle_rad) @ view_center
    else:
        view = depth_map_tools.get_cam_view(center_movment, center_angle_rad)
    
    view = view @ base_pos
    
    #Render the second image at 2x so we need to adjust the camera matrix
    cam_matrix = depth_map_tools.compute_camera_matrix(fovx, fovy, W*2, H*2)
    proj = depth_map_tools.open_gl_projection_from_camera_matrix(cam_matrix, near, far)
    
    mvp_left = proj @ view @ model
    
    
    #render
    image_generated_left, depth_generated_left, normals_generated_left, ids_duble_morp = depth_map_tools.gl_render(mesh_morpehed, mvp_left, W*2, H*2, near, far, bg_color = [0.0, 1.0, 0.0])
    
    #If a pixel has no id it means it is a background pixel
    background2x  = ids_duble_morp == 0
    
    background_from_first_morph = np.isin(ids_duble_morp, first_morph_bg_ids)
    
    ids, counts = np.unique(ids_duble_morp, return_counts=True)
    #This is the reason we render at 2x each id should appear 4 times, throgh trail and error more than 7 times has proven a good cutof that only gets expanded edges
    ids_with_2_or_more = ids[counts > 7] 
    
    hole_mask2x = np.isin(ids_duble_morp, ids_with_2_or_more)
    
    #Create a special normal image that will be used to fill in the normals of pixels that was not renderd
    xnormal = depth_map_tools.generate_normal_bg_image(W*2, H*2)
    
    full_edge_mask2x = np.repeat(np.repeat(full_edge_mask, 2, axis=0), 2, axis=1)
    
    #here we filter away unwanted pixels and only keep "true" edges
    hole_mask2x = full_edge_mask2x & hole_mask2x
    
    if not do_right:
        edge_mask_r2x = np.repeat(np.repeat(edge_mask_r, 2, axis=0), 2, axis=1)
        hole_mask2x = hole_mask2x & edge_mask_r2x
        hole_mask2x = edge_mask_r2x #overide hole mask with new simpler hole mask
    else:
        edge_mask_l2x = np.repeat(np.repeat(edge_mask_l, 2, axis=0), 2, axis=1)
        hole_mask2x = hole_mask2x & edge_mask_l2x
        hole_mask2x = edge_mask_l2x #overide hole mask with new simpler hole mask 
    
    #ref_background2x = np.repeat(np.repeat(ref_background, 2, axis=0), 2, axis=1)
    #Fix the normal map so that it can be exported
    normals_left_masked = normals_generated_left.copy()
    normals_left_masked[~hole_mask2x] = 0
    normals_left_masked[background2x] = xnormal[background2x]
    normals_left_masked[background_from_first_morph] = 0
    normals_left_masked = (normals_left_masked*255).astype(np.uint8)
    normals_out = cv2.resize(normals_left_masked, (output_W, output_H), interpolation=cv2.INTER_AREA)
    
    
    
    
    
    
    
    
    #Any ids that was marked as having a need for paralax infill are used to make a mask 
    #hole_mask = np.isin(ids_duble_morp, ids_with_paralax)
    
    
    
    
    
    #background = depth_generated_left > 205
    
    #Find the pixels that was lost in the first morph
    #background_from_first_morph = (depth_generated_left > 105) & ~background # we cliped it to 165
    
    
    
    
    org_rgb2x = cv2.resize(ref_image, (W*2, H*2), interpolation=cv2.INTER_AREA)
    # Create the generated out image
    out_generated = image_generated_left.copy()
    #Set any background pixels to be black
    out_generated[background2x] = 0
    #set any pixels that was lost in the first morpg wo their orgiginal value
    out_generated[background_from_first_morph] = org_rgb2x[background_from_first_morph]
    out_generated = cv2.resize(out_generated, (output_W, output_H), interpolation=cv2.INTER_AREA)
    
    
    
    
    depth_left_masked = depth_generated_left.copy()
    
    #Set any background pixels to 0 depth
    depth_left_masked[background2x] = 0
    #set any pixels that was lost in the first morpg wo their orgiginal value
    org_depth2x = cv2.resize(ref_depth, (W*2, H*2), interpolation=cv2.INTER_NEAREST)
    depth_left_masked[background_from_first_morph] = org_depth2x[background_from_first_morph]
    depth_left_masked = cv2.resize(depth_left_masked, (output_W, output_H), interpolation=cv2.INTER_NEAREST)
    
    #print("output:", output_W, output_H, 'depth:', W, H)
    #org_normals = (org_normals + 1) /2
    #show_imgs(
    #    [generated_right_image, down_scaled_org_image, out_generated, depth_left_masked, normals_generated_left, normals_out, org_normals],
    #    titles = ['Virtual org '+virtual_org_side, 'groud truth '+ground_truth_side, 'generated '+ground_truth_side, 'gen depth '+ground_truth_side, 'gen normals '+ground_truth_side, 'gen normals mask'+ground_truth_side, 'org_normals']
    #)
    #exit()
    
    
    # Encode depth for saving to file
    encoded_depth = depth_frames_helper.encode_depth_as_uint32(depth_left_masked.clip(0, 100), MODEL_maxOUTPUT_depth)
    encoded_depth_bgr24bit = depth_frames_helper.encode_data_as_BGR(encoded_depth, depth_left_masked.shape[1], depth_left_masked.shape[0], bit16 = True)
    
    return out_generated, normals_out, encoded_depth_bgr24bit
    

def write_repport(str_report, path, video_release = None, file_delete = None):
    print(str_report)
    with open(path, "w") as file:
        file.write(str_report)
    
    if video_release is not None:
        video_release.release()
    if file_delete is not None:
        os.remove(file_delete)
    return None
    
    
def clean_filename(s, max_len=150):
    # remove query params
    s = s.split('?', 1)[0]
    # replace illegal chars
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    # limit length
    if len(s) > max_len:
        root, ext = os.path.splitext(s)
        s = root[: max_len - len(ext)] + ext
    return s
    
mesh = None
def convert_to_training_data(img_path, from_zip=False, zip_ref=None):
    global mesh
    """
    Loads an image from either:
      - a filesystem path (from_zip=False)
      - or from inside a zip file (from_zip=True, zip_ref must be a ZipFile)
    
    Variable naming convention asumes you staart with a left image and generate a right one  but that is just naming convention it can be flipped using do_right = False
    
    """
    base_name = os.path.basename(img_path)
    name_parts = os.path.splitext(base_name)
    name_only = name_parts[0]
    extention = name_parts[1]
    crc_hex = f"{zlib.crc32(name_only.encode()):08x}"  # 8 hex chars
    subfolder = crc_hex[0]
    img_output_folder = training_output_folder + subfolder + os.sep
    local_video_cache_folder = video_cache_folder + subfolder + os.sep
    org_video_path = local_video_cache_folder + clean_filename(base_name)
    name_only = clean_filename(name_only)
    
    meta_output = img_output_folder + name_only + '.txt'
    out_transformations_file = img_output_folder + name_only + '_extrin.json'
    out_fovs_file = img_output_folder + name_only + '_fovs.json'
    
    #just pick some frames about 10-20 frames betwen might be reasonable to allow stuff to move 
    frames_to_use = [1,12,26]
    
    tmp_file_str = "_tmp.png"
    
    
    if os.path.exists(meta_output):
        print(f"file already done: {meta_output}")
        return #has already been converted
    
    #SEtup out directory
    os.makedirs(os.path.dirname(meta_output), exist_ok=True)
    
    # SET conversion settings based on filename
    first_nibble = int(crc_hex[1], 16)

    # Left or Right based on hex
    do_right = (first_nibble % 2 == 0)  # true if EVEN, false if ODD
    
    if do_right:
        ground_truth_side = "left"
        virtual_org_side = "right"
    else:
        ground_truth_side = "right"
        virtual_org_side = "left"
    
    simulate_convergense = False
    # IPD based on low/high range of nibble
    if first_nibble < 8:     # 0–7
        if first_nibble < 4:
            simulate_convergense = True
        virtual_org_side = "center"
        ipd_baseline = 0.0351 #when moving from center to left or right
    else:                    # 8–15
        if first_nibble < 12:
            simulate_convergense = True
        ipd_baseline = 0.063 #when moving from left to right or right to left
    
    is_video = False
    if extention == '.mp4':
        is_video = True
    #Read Data
    if from_zip:
        if is_video:
            if not os.path.exists(org_video_path):
                data = zip_ref.read(img_path)
                os.makedirs(os.path.dirname(org_video_path), exist_ok=True)
                
                with open(org_video_path, 'wb') as f:
                    f.write(data)
        else:
            data = zip_ref.read(img_path)
            img_array = np.frombuffer(data, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        if is_video:
            org_video_path = img_path
        else:
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if is_video:
        frames = []
        video = cv2.VideoCapture(org_video_path)
        width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps    = video.get(cv2.CAP_PROP_FPS)
        video_len = video.get(cv2.CAP_PROP_FRAME_COUNT)
        codec  = int(video.get(cv2.CAP_PROP_FOURCC))
        print("video info:", width, 'x', height, 'fps:', fps, 'frame count:', video_len, 'codec:', codec)
        if width < save_resolution or height < save_resolution:
            return write_repport("video resolution to low", meta_output, video_release = video, file_delete = org_video_path)
        frame_n = 0
        while video.isOpened():
            ret, bgr_frame = video.read()
            if not ret:
                return write_repport("could not read video frames", meta_output, video_release = video, file_delete = org_video_path)
            if frame_n in frames_to_use:
                frames.append(bgr_frame)
            if len(frames) == len(frames_to_use):
                break
            frame_n += 1
            
        video.release()
        
        if len(frames) != len(frames_to_use):
            return write_repport("could not extract enogh frames from video", meta_output, video_release = video, file_delete = org_video_path)
        os.remove(org_video_path)
    else:
        #Abort on failure
        if img_bgr is None:
            return write_repport(f"[WARN] Could not load image: {img_path}", meta_output)
        frames = [img_bgr]
    
    
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
    
    
    
    #Get the depth
    
    
    #Dont need to see what the depth model is doing so pipe it to null
    with open(os.devnull, "w") as f, redirect_stdout(f):
        depth_out = da3model.inference(
            rgb_frames,
            process_res = depth_resolution
        )
        
    cam_matrix = depth_out.intrinsics[0]
    depth1 = depth_out.depth[0]
    
    standard_deviation = depth1.std()
    if standard_deviation < 0.12:
        return write_repport(f"standard_deviation in depth to low {standard_deviation} probably a carton or a frame with letterboxing or something", meta_output)
    
    #100 is the max depth
    depths = depth_out.depth.clip(0, 100)
    
    
    out_transformations = []
    for extrin in depth_out.extrinsics:
        fixed_extrin = np.vstack([extrin, np.array([0, 0, 0, 1], dtype=extrin.dtype)])
        fixed_extrin = np.linalg.inv(fixed_extrin)#da3 outputs inverted transformations to waht we want
        out_transformations.append(fixed_extrin)
    
    # set midle frame as reference
    recerence_matrix_inv_4x4 = np.linalg.inv(out_transformations[1])
    for i, extrin in enumerate(out_transformations):

        #inverted_prediction = np.linalg.inv(extrin)
        #diff = recerence_matrix_4x4 @ inverted_prediction
        #fixd = diff @ extrin
        out_transformations[i] = extrin @ recerence_matrix_inv_4x4
    
    
    next_intersect = depth_map_tools.frusta_intersect(cam_matrix, out_transformations[1], out_transformations[2])
    prev_intersect = depth_map_tools.frusta_intersect(cam_matrix, out_transformations[1], out_transformations[0])
    
    if not next_intersect:
        print("NEXT frame does not intersect with frame")
        print(cam_matrix, out_transformations[1], out_transformations[2])
    
    if not prev_intersect:
        print("PREV frame does not intersect with frame")
        print(cam_matrix, out_transformations[1], out_transformations[0])
    
    out_xfovs = []
    for intrin in depth_out.intrinsics:
        fovx, fovy = depth_map_tools.fov_from_camera_matrix(intrin)
        out_xfovs.append(float(fovx))
    
    H, W = depth1.shape
    uniK3d_down_scaled_org_images = [cv2.resize(f, (W, H)).astype(np.uint8) for f in rgb_frames]
    
    
    #Calculate final output size
    scale = save_resolution / min(H, W)
    output_W = int(W * scale)
    output_H = int(H * scale)
    
    fovx, fovy = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    
    #If the output is bigger than the inferd depth we need to upscale the depth
    if output_W > W or output_H > H:
        W = output_W
        H = output_H
        uniK3d_down_scaled_org_images = [cv2.resize(f, (W, H)).astype(np.uint8) for f in rgb_frames]
        depths = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in depths]
        cam_matrix = depth_map_tools.compute_camera_matrix(fovx, fovy, W, H)
    
    #The model eats the save_resolution so we save as that no we dont we cant shift the images correcly if we do
    if False:
        output_W = save_resolution
        output_H = save_resolution
    
    #compare with unik3d
    rgb_torch = torch.from_numpy(uniK3d_down_scaled_org_images[0]).permute(2, 0, 1)
    unik_ored = unk3dmodel.infer(rgb_torch)
    unik_depth = unik_ored["depth"].squeeze().cpu().numpy()
    
    unik_depth = unik_depth.clip(0, 100)
    
    fx, fy = unik3d_video.estimate_focal_lengths(unik_ored['points'], W, H)

    #Get generic matrix that we can fill in
    unik3d_cam = depth_map_tools.compute_camera_matrix(90, None, W, H)
    
    #Fill in real focal lengths
    unik3d_cam[0][0] = fx
    unik3d_cam[1][1] = fy
    
    
    
    norm_d = depth_frames_helper.normalize_depth(depths[0])
    unik_norm_d = depth_frames_helper.normalize_depth(unik_depth)
    
    ssim_val = ssim(norm_d, unik_norm_d, data_range=1.0)
    
    
    
    
    
    
    
    #For debuging some general info about the data
    fovx, _ = depth_map_tools.fov_from_camera_matrix(cam_matrix)
    unik_fovx, _ = depth_map_tools.fov_from_camera_matrix(unik3d_cam)
    
    
    
    
    data = [
        meta_output, "ssim_val:", ssim_val, "standard_deviation:", standard_deviation, "xfov:", fovx, "unik_fovx:", unik_fovx, "min_depth:", depths[0].min(), "mean_depth:", depths[0].mean(),
        "max_depth:", depths[0].max(), "min_unik_depth:" , unik_depth.min(), "mean_unik_depth:", unik_depth.mean(), "max_unik_depth:", unik_depth.max(),
        'ipd_baseline:', ipd_baseline, 'simulate_convergense:', simulate_convergense, 'ground_truth_side:', ground_truth_side
    ]
    def to_str(val):
        if isinstance(val, (int, float, np.floating)):  # round numbers
            return f"{val:.2f}"
        return str(val)

    str_desc = " ".join(to_str(v) for v in data)
    #print(str_desc)
    
    #tested and 0.72 semas like a decent cutoff, there is still some jank but gets rid of the worst cases
    if ssim_val < 0.72:
        return write_repport("ignoring image: depth models dont agree enogh about depth "+str_desc, meta_output)
    
    for frame_no, _ in enumerate(depths):
    
        out_name = img_output_folder + name_only
        if len(rgb_frames) == 1:
            out_name += "_vid_0"
        else:
            actual_frame_no = frames_to_use[frame_no]
            out_name += "_vid_" + str(actual_frame_no)
            
        cam_extrin = np.eye(4)
        if morph_to_ref_view:
            cam_extrin = out_transformations[frame_no]
        
        out_generated, normals_out, encoded_depth_bgr24bit =  make_sample(depths[frame_no], rgb_frames[frame_no], cam_matrix, do_right, simulate_convergense, ipd_baseline, ground_truth_side, virtual_org_side, output_W, output_H, cam_extrin)
        
        
        if False:
            alpha = 0.5
            #print(rgb_frames[frame_no])
            #print(out_generated)
            blended_org_and_warp = cv2.addWeighted(cv2.resize(rgb_frames[frame_no], (output_W, output_H)), alpha, cv2.resize(out_generated, (output_W, output_H)), 1 - alpha, 0)
            print(str_desc)
            show_imgs([
                blended_org_and_warp
            ])
        #exit()
        
        img_output = out_name + '.png'
        img_output_depth = img_output + "_depth.png"
        img_output_gen = img_output + "_gen.png"
        img_output_normal_gen = img_output + "_gen_normal.png"
        img_output_depth_gen = img_output + "_gen_depth.png"
        
        #resacle to specified save res
        out_org_rgb = cv2.resize(rgb_frames[frame_no], (output_W, output_H))
        save_depth = cv2.resize(depths[frame_no], (output_W, output_H), interpolation=cv2.INTER_NEAREST)
        
        encoded_depth = depth_frames_helper.encode_depth_as_uint32(save_depth, MODEL_maxOUTPUT_depth)
        org_bgr24bit = depth_frames_helper.encode_data_as_BGR(encoded_depth, save_depth.shape[1], save_depth.shape[0], bit16 = True)
        
        #First we write to tmp files
        cv2.imwrite(img_output+tmp_file_str, cv2.cvtColor(out_org_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_output_depth+tmp_file_str, org_bgr24bit)
        cv2.imwrite(img_output_depth_gen+tmp_file_str, encoded_depth_bgr24bit)
        cv2.imwrite(img_output_normal_gen+tmp_file_str, cv2.cvtColor(normals_out, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_output_gen+tmp_file_str, cv2.cvtColor(out_generated, cv2.COLOR_RGB2BGR))
        
        
        
        #rename the temp files when they are done
        rename(img_output+tmp_file_str, img_output)
        rename(img_output_depth+tmp_file_str, img_output_depth)
        rename(img_output_depth_gen+tmp_file_str, img_output_depth_gen)
        rename(img_output_normal_gen+tmp_file_str, img_output_normal_gen)
        rename(img_output_gen+tmp_file_str, img_output_gen)
    
    #Save predicted Field of views and extrinsiscs
    with open(out_fovs_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_xfovs))
    
    with open(out_transformations_file, "w") as json_file_handle:
        json_file_handle.write(json.dumps(out_transformations, cls=NumpyEncoder)) 
    
    with open(meta_output, "w") as file:
        file.write("OK: "+str_desc)


    return True


# ---------------------------------------------------------
# process_dataset_folder
# ---------------------------------------------------------
def process_dataset_folder(dataset_path):
    """
    Walks through the dataset folder and processes all images
    by calling convert_to_training_data(img_path).
    """
    # Allowed image extensions
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", '.mp4')

    if dataset_path.lower().endswith(".zip"):
        print(f"Opening ZIP file: {dataset_path}")

        with zipfile.ZipFile(dataset_path, 'r') as z:
            # Filter: only files under unlabeled2017/
            image_names = [
                n for n in z.namelist()
                if (n.startswith("unlabeled2017/") or n.startswith("Moments_in_Time_Raw/training/")) and n.lower().endswith(exts)
            ]
            random.seed(42)
            random.shuffle(image_names)
            print(f"Found {len(image_names)} images inside zip under unlabeled2017/")

            for idx, name in enumerate(image_names):
                print(f"[{idx+1}/{len(image_names)}] Processing: {name}")
                img = convert_to_training_data(name, from_zip=True, zip_ref=z)

        return

    # Case B: Folder
    if os.path.isdir(dataset_path):
        print(f"Walking folder: {dataset_path}")

        image_paths = []
        for root, _, files in os.walk(dataset_path):
            for fname in files:
                if fname.lower().endswith(exts):
                    image_paths.append(os.path.join(root, fname))
        random.seed(42)
        random.shuffle(image_paths)
        print(f"Found {len(image_paths)} images in folder.")

        for idx, img_path in enumerate(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] Processing: {img_path}")
            img = convert_to_training_data(img_path, from_zip=False)

        return

    print("ERROR: Dataset path is neither folder nor .zip file.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(training_output_folder, exist_ok=True)
    os.makedirs(video_cache_folder, exist_ok=True)
    print("loading depth model")
    da3model = video_da3.load_model()
    unk3dmodel = unik3d_video.load_model()
    process_dataset_folder(dataset_folder)
