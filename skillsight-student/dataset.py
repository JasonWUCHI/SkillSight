import os
import json
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from fvcore.common.file_io import PathManager
import random

import utils
import decoder
import video_container as container


from PIL import Image



TOY_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_ROOT = os.environ.get("EGOEXO_SPLIT_ROOT", os.path.abspath(os.path.join(TOY_ROOT, "..", "data", "ego_exo_splits")))
DATA_ROOT = os.environ.get("EGOEXO_DATA_ROOT", "/vision/asomaya1/share/ego_exo_demonstrator_proficiency/data")
TAKES_JSON = os.environ.get("EGOEXO_TAKES_JSON", "/vision/cwuau/egoexo4d_gaze/takes.json")
GAZE_ROOT = os.environ.get("EGOEXO_GAZE_ROOT", "/vision/vision_data_2/EgoExo4D_public_v1/takes")


class Egoexo(data.Dataset):
    """
    EgoExo video loader. Construct the EgoExo video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, mode, category, num_retries=10, shifting = False):
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
        self.shifting = shifting

        self._video_meta = {}
        self._num_retries = num_retries
        self.category = category

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["test", "val"]:
            self._num_clips = 10

        # logger.info("Constructing EgoExo {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        path_to_file = os.path.join(SPLIT_ROOT, f"448pFull_{self.category}_v2", "ego", f"{self.mode}.csv")

        with open(TAKES_JSON) as f:
            takes_info = json.load(f)
        case_id = ["Soccer", "Basketball", "Rock Climbing", "Dance", "Cooking", "Music"]
                
        takes_case_dict = {info["take_name"]:info["parent_task_name"] for info in takes_info}
        subtask_case_dict = {info["take_name"]:info["task_name"] for info in takes_info}
        self.duration_dict = {info["take_name"]:info["duration_sec"] for info in takes_info}

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
        self.full_task_names = []
        self.takes_scenario = []
        self.full_takes_scenario = []
        self.subtask_ids = []
        self.full_subtask_ids = []
        
        count = 0
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path, label = path_label.split(" ")
                task_name = path[path.index("v2")+3:path.index("frame")-1]
                self.task_names.append(task_name)
                scenario_id = case_id.index(takes_case_dict[task_name])
                self.takes_scenario.append(scenario_id)

                if takes_case_dict[task_name] == "Music":
                    if "Piano" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(0)
                    elif "Violin" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(1)
                    elif "Guitar" in subtask_case_dict[task_name]:
                        self.subtask_ids.append(2)
                elif takes_case_dict[task_name] == "Rock Climbing" or takes_case_dict[task_name] == "Cooking" or takes_case_dict[task_name] == "Dance":
                    self.subtask_ids.append(0)
                else:
                    self.subtask_ids.append(subtask_list[takes_case_dict[task_name]].index(subtask_case_dict[task_name]))

                if self.shifting==False:

                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(DATA_ROOT, path)
                        )
                        self.full_task_names.append(task_name)
                        self.full_takes_scenario.append(scenario_id)
                        self.full_subtask_ids.append(self.subtask_ids[-1])
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}

                else:
                    num_clip = int(self.duration_dict[task_name] - 8)
                    for idx in range(num_clip):
                        self._path_to_videos.append(
                            os.path.join(DATA_ROOT, path)
                        )
                        self.full_task_names.append(task_name)
                        self.full_takes_scenario.append(scenario_id)
                        self.full_subtask_ids.append(self.subtask_ids[-1])
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[count] = {}
                        count +=1

                

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load EgoExo split {} from {}".format(
            self._split_idx, path_to_file
        )
        # logger.info(
        #     "Constructing EgoExo dataloader (size: {}) from {}".format(
        #         len(self._path_to_videos), path_to_file
        #     )
        # )

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
        mean, std = [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        images = images * std + mean
        images = (images * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

        #CLIP preprocessing
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
            min_scale = 448
            max_scale = 512
            crop_size = 448
        
        elif self.mode in ["test","val"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = 1

            min_scale, max_scale, crop_size = 448,448,448
                
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        
        sampling_rate = 15

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            # video_container = container.get_video_container(
            #     self._path_to_videos[index],
            #     False,
            #     "pyav",
            # )
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    False,
                    "pyav",
                )
            except Exception as e:
                print("first error")
                # logger.info(
                #     "Failed to load video from {} with error {}".format(
                #         self._path_to_videos[index], e
                #     )
                # )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            # JW: Decode the video and perform temporal sampling.
            ### JW: THIS IS THE PART TO CHANGE
            task_name = self._path_to_videos[index]
            task_name = task_name[task_name.index("v2")+3:task_name.index("frame")-1]
            frames, frame_start_idx, frame_end_idx = decoder.decode(
                video_container,
                sampling_rate,
                16,
                temporal_sample_index,
                10 if self.shifting==False else int(self.duration_dict[task_name]-8),
                video_meta=self._video_meta[index], # None
                target_fps=30,
                backend="pyav",
                max_spatial_scale=min_scale, #448, not used for pyav
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                
                if self.mode not in ["test", "val"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            ### 

            #JW: Load the gaze data
            gaze_tensor = None
            task_name = self._path_to_videos[index]
            task_name = task_name[task_name.index("v2")+3:task_name.index("frame")-1]

            start_time, end_time = frame_start_idx/30, frame_end_idx/30
            gaze_start_idx, gaze_end_idx = int(frame_start_idx/30 * 10), int(frame_end_idx/30 * 10)
            try:
                gaze_data = pd.read_csv(os.path.join(GAZE_ROOT, task_name, "eye_gaze", "general_world_eye_gaze.csv"))
                gaze_data = gaze_data.iloc[:, 1:]
                gaze_tensor = torch.tensor(gaze_data.values, dtype=torch.float32)
                gaze_tensor = decoder.temporal_sampling(gaze_tensor, gaze_start_idx, gaze_end_idx, len(frames)*5)
            except Exception as e:
                # print(e)
                pass
            if gaze_tensor is None:
                if self.mode in ["val", "test"]:
                    gaze_tensor = torch.ones(80,21)
                    gaze_tensor[:,12:14] = 704
                    gaze_tensor[:,15] = 1408
                else:
                    
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
            scenario_id = self.full_takes_scenario[index]
            subtask_id = self.full_subtask_ids[index]
            #


            label = self._labels[index]
    
            frames = utils.tensor_normalize(
                frames, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225]
            )

            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            
            # Perform data augmentation.
            # REMOVING AUG
           

            # JW: crop using gaze
            gaze_crop_size = 224
            frames_cropped = utils.crop_using_gaze(
                frames, gaze_tensor[:,-3:-1], gaze_crop_size,
            )

            gaze_tensor[:,-3:-1] = gaze_tensor[:,-3:-1].clamp(0.0, 1.0)

            # preprocess frames_cropped for CLIP
            frames_cropped = self.preprocess_images(frames_cropped)


            
            # Perform temporal sampling from the fast pathway.
            frames = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, 16

                    ).long(),
            )
            frames_cropped = torch.index_select(
                    frames_cropped,
                    1,
                    torch.linspace(
                        0, frames_cropped.shape[1] - 1, 16

                    ).long(),
            )
            # print(gaze_tensor.shape)

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



