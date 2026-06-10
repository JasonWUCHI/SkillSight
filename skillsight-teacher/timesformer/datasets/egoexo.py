# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)

from PIL import Image
import pandas as pd
import json
import numpy as np


@DATASET_REGISTRY.register()
class Egoexo(torch.utils.data.Dataset):
    """
    EgoExo video loader. Construct the EgoExo video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the EgoExo video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for EgoExo".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = cfg.TRAIN.NUM_CLIPS
        elif self.mode in ["test", "val"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EgoExo {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.cfg.DATA.EGO_EXO_VERSION != "v1":
            path_to_file = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR + "_" + self.cfg.DATA.EGO_EXO_VERSION, self.cfg.DATA.CAMERA_VIEW, "{}.csv".format(self.mode)
            )
        else:
            path_to_file = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.CAMERA_VIEW, "{}.csv".format(self.mode)
            )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        with open("/vision/cwuau/egoexo4d_gaze/takes.json") as f:
            takes_info = json.load(f)
        case_id = ["Soccer", "Basketball", "Rock Climbing", "Dance", "Cooking", "Music"]
                
        takes_case_dict = {info["take_name"]:info["parent_task_name"] for info in takes_info}
        subtask_case_dict = {info["take_name"]:info["task_name"] for info in takes_info}

        subtask_list = {"Soccer": [], "Basketball":[], "Rock Climbing":[], "Dance":[], "Cooking":[], "Music":[]} 
        for take_name in subtask_case_dict:
            subtask = subtask_case_dict[take_name]
            if takes_case_dict[take_name] not in subtask_list:
                continue
            if subtask not in subtask_list[takes_case_dict[take_name]]:
                subtask_list[takes_case_dict[take_name]].append(subtask)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.task_names = []
        self.takes_scenario = []
        self.subtask_ids = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )

                task_name = path[path.index("v2")+3:path.index("frame")-1]
                self.task_names.append(task_name)
                scenario_id = case_id.index(takes_case_dict[task_name])
                self.takes_scenario.append(scenario_id)

                if takes_case_dict[task_name] != "Music":
                    self.subtask_ids.append(subtask_list[takes_case_dict[task_name]].index(subtask_case_dict[task_name]))
                else:
                    if "Piano" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(0)
                    elif "Violin" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(1)
                    elif "Guitar" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(2)
                    else:
                        print("ERRRRRRRRor")

                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load EgoExo split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing EgoExo dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def preprocess_images(self, images):
        """
        Preprocess a batch of images for CLIP.

        Args:
            images (np.ndarray): Batch of images with shape (C, T, H, W), pixel values in [0, 255].

        Returns:
            np.ndarray: Preprocessed batch of images with shape (C, T, H, W), normalized.
        """

        # resume the mean and std
        images = images.permute(1,2,3,0)  # Convert to (T, H, W, C)
        mean, std = self.cfg.DATA.MEAN, self.cfg.DATA.STD
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        images = images * std + mean
        images = (images * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

        #CLIP preprocessing
        if self.cfg.MODEL.IMG_MODEL == "clip":
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
        elif self.cfg.MODEL.IMG_MODEL == "dinov2":
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        B, H, W, C = images.shape

        # Resize images
        resized_images = np.array([np.array(Image.fromarray(img).resize((224, 224), Image.BICUBIC)) for img in images])

        # Convert to RGB and normalize
        rgb_images = resized_images.astype(np.float32) / 255.0
        normalized_images = (rgb_images - mean) / std

        # Convert to (C, T, H, W)
        preprocessed_images = np.transpose(normalized_images, (3, 0, 1, 2))
        preprocessed_images = torch.from_numpy(preprocessed_images).float()  # to torch.Tensor

        return preprocessed_images

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0] #448
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1] #448
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE #448
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test","val"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = ( #448,448,448
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        # sampling_rate = utils.get_random_sampling_rate(
        #     self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        #     self.cfg.DATA.SAMPLING_RATE,
        # )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                index = random.randint(0, len(self._path_to_videos) - 1)
                if self.mode not in ["test", "val"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            # JW: Decode the video and perform temporal sampling.
            ### JW: THIS IS THE PART TO CHANGE
            frames, frame_start_idx, frame_end_idx = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES, #16
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS, #10
                video_meta=self._video_meta[index], # None
                target_fps=self.cfg.DATA.TARGET_FPS, # 30
                backend=self.cfg.DATA.DECODING_BACKEND, #"pyav"
                max_spatial_scale=min_scale, #448, not used for pyav
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test", "val"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            ### 

            #JW: Load the gaze data
            gaze_tensor = None
            task_name = self._path_to_videos[index]
            task_name = task_name[task_name.index("v2")+3:task_name.index("frame")-1]

            start_time, end_time = frame_start_idx/self.cfg.DATA.TARGET_FPS, frame_end_idx/self.cfg.DATA.TARGET_FPS
            gaze_start_idx, gaze_end_idx = int(frame_start_idx/self.cfg.DATA.TARGET_FPS * 10), int(frame_end_idx/self.cfg.DATA.TARGET_FPS * 10)
            try:
                gaze_data = pd.read_csv(f"/vision/vision_data_2/EgoExo4D_public_v1/takes/{task_name}/eye_gaze/general_world_eye_gaze.csv")
                # print(gaze_data.head())
                gaze_data = gaze_data.iloc[:, 1:]
                gaze_tensor = torch.tensor(gaze_data.values, dtype=torch.float32)
                # gaze_tensor = gaze_tensor[gaze_start_idx:gaze_end_idx]
                gaze_tensor = decoder.temporal_sampling(gaze_tensor, gaze_start_idx, gaze_end_idx, len(frames))
            except Exception as e:
                logger.info(
                    "Failed to load gaze from {} with error {}".format(
                        task_name, e
                    )
                )
            if gaze_tensor is None:
                if self.mode in ["val", "test"]:
                    gaze_tensor = torch.ones(len(frames),21)
                    gaze_tensor[:,12:14] = 704
                    gaze_tensor[:,15] = 1408
                else:
                    logger.warning(
                        "Failed to get gaze from video idx {} from {}; trial {}".format(
                            index, task_name, i_try
                        )
                    )
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
            #

            #JW: Normalize the gaze
            # gaze_tensor = utils.normalize_gaze(gaze_tensor)
            gaze_tensor = utils.normalize_gaze_2(gaze_tensor)
            gaze_tensor = torch.nan_to_num(gaze_tensor, nan=0.0)
            #

            #JW get scenario
            scenario_id = self.takes_scenario[index//self._num_clips]
            subtask_id = self.subtask_ids[index//self._num_clips]
            #


            label = self._labels[index]
    
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            
            # Perform data augmentation.
            # REMOVING AUG
            # frames, gaze_2D_mod = utils.spatial_sampling_and_gaze2D_mod(
            #     frames,
            #     gaze_tensor[:,-3:-1],
            #     spatial_idx=spatial_sample_index,
            #     min_scale=min_scale,
            #     max_scale=max_scale,
            #     crop_size=crop_size,
            #     random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            #     inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            # )
            # gaze_tensor[:,-3:-1] = gaze_2D_mod

            # JW: crop using gaze
            gaze_crop_size = self.cfg.DATA.CROP_WITH_GAZE_SIZE
            frames_cropped = utils.crop_using_gaze(
                frames, gaze_tensor[:,-3:-1], gaze_crop_size,
            )

            gaze_tensor[:,-3:-1] = gaze_tensor[:,-3:-1].clamp(0.0, 1.0)

            # preprocess frames_cropped for CLIP
            frames_cropped = self.preprocess_images(frames_cropped)


            if not self.cfg.MODEL.ARCH in ['vit']:
                frames = utils.pack_pathway_output(self.cfg, frames)
                frames_cropped = utils.pack_pathway_output(self.cfg, frames_cropped)
            else:
                # Perform temporal sampling from the fast pathway.
                frames = torch.index_select(
                     frames,
                     1,
                     torch.linspace(
                         0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )
                frames_cropped = torch.index_select(
                     frames_cropped,
                     1,
                     torch.linspace(
                         0, frames_cropped.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                     ).long(),
                )

            return frames, frames_cropped, gaze_tensor, label, index, start_time, end_time, scenario_id, subtask_id, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
