#!/usr/bin/env python3

import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from fvcore.common.file_io import PathManager
from torch.utils.data.distributed import DistributedSampler
from scipy.spatial.transform import Rotation

# from . import transform as transform

logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with PathManager.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        #assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def spatial_sampling_and_gaze2D_mod(
    frames,
    gaze_tensor_2D,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        gaze_tensor_2D (tensor): (num frames, 2), 2D coordinate of th
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        original_size = frames.shape[1:3]  # (height, width)
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        new_size = frames.shape[1:3]  # (height, width)
        frames, _, [y_offset,x_offset] = transform.random_crop(frames, crop_size)
        if random_horizontal_flip: #this should be false
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        #assert len({min_scale, max_scale, crop_size}) == 1
        original_size = frames.shape[1:3]  # (height, width)
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        new_size = frames.shape[1:3]  # (height, width)
        frames, _, [y_offset,x_offset] = transform.uniform_crop(frames, crop_size, spatial_idx)

    # Convert normalized gaze to absolute pixel coordinates in the original frame
    gaze_tensor_2D[:, 0] *= original_size[1]  # width
    gaze_tensor_2D[:, 1] *= original_size[0]  # height

    # Scale to new frame size after resize
    gaze_tensor_2D[:, 0] *= new_size[1] / original_size[1]  # width scaling
    gaze_tensor_2D[:, 1] *= new_size[0] / original_size[0]  # height scaling

    # Apply crop offset
    gaze_tensor_2D[:, 0] -= x_offset
    gaze_tensor_2D[:, 1] -= y_offset

    # Normalize to final crop_size × crop_size frame
    gaze_tensor_2D[:, 0] /= crop_size
    gaze_tensor_2D[:, 1] /= crop_size

    # Optionally, clamp to [0, 1]
    gaze_tensor_2D = gaze_tensor_2D.clamp(0.0, 1.0)

    return frames, gaze_tensor_2D


def spatial_sampling_2crops(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        frames, _ = transform.random_short_side_scale_jitter(
            images=frames,
            min_size=min_scale,
            max_size=max_scale,
            inverse_uniform_sampling=inverse_uniform_sampling,
        )
        frames, _ = transform.random_crop(frames, crop_size)
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        #assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = transform.uniform_crop_2crops(frames, crop_size, spatial_idx)
    return frames


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    Join a list of label list.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): The joint list of all lists in input.
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def convert_to_video_level_labels(labels):
    """
    Aggregate annotations from all frames of a video to form video-level labels.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): Same as input, but with each label replaced by
        a video-level one.
    """
    for video_id in range(len(labels)):
        video_level_labels = aggregate_labels(labels[video_id])
        for i in range(len(labels[video_id])):
            labels[video_id][i] = video_level_labels
    return labels


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with PathManager.open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None


def normalize_gaze_2(gaze_tensor):
    # Replace NaNs with 0
    gaze_tensor = torch.nan_to_num(gaze_tensor, nan=0.0)
    output_gaze_tensor = torch.zeros(gaze_tensor.shape[0], 16) #3+3+3+4+2+1

    output_gaze_tensor[:, 0:3] = gaze_tensor[:, 0:3] / 10
    output_gaze_tensor[:, 0:3] = output_gaze_tensor[:, 0:3] - gaze_tensor[:, 6:9].mean(dim=0, keepdim=True)
    vec = gaze_tensor[:, 3:6]
    norm = torch.norm(vec, dim=1, keepdim=True)
    output_gaze_tensor[:, 3:6] = torch.where(norm > 0, vec / norm, vec)
    output_gaze_tensor[:, 3:6] -= output_gaze_tensor[0:1, 3:6]

    head_7d = torch.cat((gaze_tensor[:, 6:9], gaze_tensor[:, 17:]), dim=-1)
    head_7d = head_7d.numpy()
    head_7d = absolute_to_relative_from_t0(head_7d)
    head_7d = torch.tensor(head_7d)
    output_gaze_tensor[:, 6:13] = head_7d

    image_size = gaze_tensor[:,-1].unsqueeze(-1)
    gaze_tensor[:,12:14] /= image_size  # Normalize (x, y) by image size
    output_gaze_tensor[:,13:15] = torch.clamp(gaze_tensor[:,12:14], min=0.0, max=1.0)
    output_gaze_tensor[:, 15] = gaze_tensor[:, 14] / 10

    return output_gaze_tensor
    

def normalize_gaze(gaze_tensor):
    # Replace NaNs with 0
    gaze_tensor = torch.nan_to_num(gaze_tensor, nan=0.0)

    # Normalize 3D vectors in-place
    # gaze_tensor[:, 0:3] -= gaze_tensor[:, 6:9].mean(dim=0, keepdim=True)
    gaze_tensor[:, 0:3] /= 10
    gaze_tensor[:, 0:3] -= gaze_tensor[:, 6:9].mean(dim=0, keepdim=True)#gaze_tensor[:, 0:3].mean(dim=0, keepdim=True) # Normalize by mean

    #cpf frame
    vec = gaze_tensor[:, 3:6]
    norm = torch.norm(vec, dim=1, keepdim=True)
    gaze_tensor[:, 3:6] = torch.where(norm > 0, vec / norm, vec)
    gaze_tensor[:, 3:6] -= gaze_tensor[0:1, 3:6] #

    #trajectory
    gaze_tensor[:, 6:9] -= gaze_tensor[:, 6:9].mean(dim=0, keepdim=True) #gaze_tensor[:, 6:9].mean(dim=0, keepdim=True)

    #head pose
    # gaze_tensor[:, 9:12] /= 90
    # gaze_tensor[:, 9] -= gaze_tensor[0, 9]  # Normalize yaw
    gaze_tensor[:, 9:12] = relative_ypr(gaze_tensor[:, 9:12])
    gaze_tensor[:, 9:12] /= 90

    #2D coordinate
    image_size = gaze_tensor[:,-1].unsqueeze(-1)
    gaze_tensor[:,12:14] /= image_size  # Normalize (x, y) by image size
    gaze_tensor[:,12:14] = torch.clamp(gaze_tensor[:,-4:-2], min=0.0, max=1.0)

    #depth
    gaze_tensor[:,14] /= 10  #depth

    # Remove image size column 
    gaze_tensor = gaze_tensor[:, :15]

    return gaze_tensor

def crop_using_gaze(frames, gaze_2D, crop_size):
    """
    frames: (channel, num_frames, height, width)
    gaze_2D: (num_frames, 2) - normalized gaze coordinates in [0, 1]
    crop_size: int - size of the square crop

    goal: crop and resize to original size, centered around the gaze coordinates
    """
    channels, num_frames, height, width = frames.shape
    frames = frames.permute(1, 0, 2, 3)  # (num_frames, channels, height, width)
    cropped_frames = torch.zeros((num_frames, channels, height, width), dtype=frames.dtype)

    for i in range(num_frames):
        # Convert normalized gaze coordinates to pixel coordinates
        gaze_x = int(gaze_2D[i, 0] * width)
        gaze_y = int(gaze_2D[i, 1] * height)

        # Calculate crop boundaries
        x_start = max(0, gaze_x - crop_size // 2)
        x_end = min(width, gaze_x + crop_size // 2)
        y_start = max(0, gaze_y - crop_size // 2)
        y_end = min(height, gaze_y + crop_size // 2)

        # Adjust for cases where the crop exceeds the image boundaries
        if x_end - x_start < crop_size:
            if x_start == 0:
                x_end = min(x_start + crop_size, width)
            else:
                x_start = max(x_end - crop_size, 0)

        if y_end - y_start < crop_size:
            if y_start == 0:
                y_end = min(y_start + crop_size, height)
            else:
                y_start = max(y_end - crop_size, 0)

        # Crop and resize the frame
        cropped_frame = frames[i, :, y_start:y_end, x_start:x_end]
        cropped_frame = torch.nn.functional.interpolate(
            cropped_frame.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False
        ).squeeze(0)

        cropped_frames[i] = cropped_frame

    cropped_frames = cropped_frames.permute(1, 0, 2, 3)  # (channels, num_frames, crop_size, crop_size)

    return cropped_frames


def ypr_to_matrix(ypr):
    """yaw, pitch, roll in radians -> rotation matrix [B, 3, 3]"""
    yaw, pitch, roll = ypr[:, 0], ypr[:, 1], ypr[:, 2]
    
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    Rz = torch.stack([cy, -sy, torch.zeros_like(cy),
                      sy,  cy, torch.zeros_like(cy),
                      torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)], dim=-1).reshape(-1, 3, 3)

    Ry = torch.stack([cp, torch.zeros_like(cp), sp,
                      torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp),
                      -sp, torch.zeros_like(cp), cp], dim=-1).reshape(-1, 3, 3)

    Rx = torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr),
                      torch.zeros_like(cr), cr, -sr,
                      torch.zeros_like(cr), sr,  cr], dim=-1).reshape(-1, 3, 3)

    return Rz @ Ry @ Rx  # yaw → pitch → roll

def relative_ypr(ypr_deg):
    """
    Normalize yaw/pitch/roll sequence relative to first orientation.
    Input: [T, 3] in degrees.
    Output: [T, 3] in degrees.
    """
    ypr_rad = torch.deg2rad(ypr_deg)

    R_all = ypr_to_matrix(ypr_rad)
    R0_inv = torch.linalg.inv(R_all[0])  # reference inverse

    R_rel = R0_inv @ R_all  # [T, 3, 3]

    # Convert back to yaw, pitch, roll
    # pitch = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
    pitch = torch.atan2(-R_rel[:, 2, 0], torch.sqrt(R_rel[:, 0, 0]**2 + R_rel[:, 1, 0]**2))
    yaw   = torch.atan2(R_rel[:, 1, 0], R_rel[:, 0, 0])
    roll  = torch.atan2(R_rel[:, 2, 1], R_rel[:, 2, 2])

    return torch.rad2deg(torch.stack([yaw, pitch, roll], dim=-1))


def pose7d_to_matrix(pose):
    t = pose[:3]
    q = pose[3:]
    R = Rotation.from_quat(q).as_matrix()  # shape (3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def matrix_to_pose7d(T):
    """
    Convert (3, 4) or (N, 3, 4) pose matrix to 7D pose vector(s): tx, ty, tz, qx, qy, qz, qw
    Args:
        T (np.ndarray): Pose matrix of shape (3, 4) or (N, 3, 4)
    Returns:
        pose7d: (7,) if input is (3, 4), or (N, 7) if input is (N, 3, 4)
    """
    is_batched = T.ndim == 3
    if not is_batched:
        T = T[None, ...]  # convert to shape (1, 3, 4)
    if T.shape[-2:] == (4, 4):
        T = T[:, :3, :]
    Rs = T[:, :3, :3]  # (N, 3, 3)
    ts = T[:, :3, 3]   # (N, 3)
    qs = Rotation.from_matrix(Rs).as_quat()  # (N, 4)
    poses7d = np.concatenate([ts, qs], axis=1)  # (N, 7)
    return poses7d[0] if not is_batched else poses7d
    
def absolute_to_relative_from_t0(absolute_pose):  # shape: (N, 7)
    T0 = pose7d_to_matrix(absolute_pose[0])
    rel_poses = []
    for i in range(len(absolute_pose)):
        Ti = pose7d_to_matrix(absolute_pose[i])
        T_rel = np.linalg.inv(T0) @ Ti
        rel_pose = matrix_to_pose7d(T_rel)
        rel_poses.append(rel_pose)
    return np.stack(rel_poses, axis=0)  # shape: (N, 7)