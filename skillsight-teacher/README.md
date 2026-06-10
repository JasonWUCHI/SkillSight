# SkillSight Teacher

This folder contains the focused teacher model code for **SkillSight**.

The teacher jointly uses egocentric RGB video, gaze attention, gaze sequence features, and DINOv2 crop-gaze features to predict skill level.

## Model architecture

Main class:

```text
timesformer/models/custom_video_model_builder.py
class SkillSightTeacher
```

Supporting pieces in the same file include the TimeSformer-style video backbone, gaze encoder, crop-gaze image sequence encoder, and prediction head.

The model is registered through:

```text
timesformer/models/__init__.py
```

and selected in the config with:

```yaml
MODEL:
  MODEL_NAME: SkillSightTeacher
```

## Config

```text
configs/EgoExo/SkillSightTeacher.yaml
```

The included config uses the shared split folder under `../data/ego_exo_splits` when run from this directory:

```yaml
DATA:
  PATH_TO_DATA_DIR: ../data/ego_exo_splits/448pFull_downloaded
  EGO_EXO_VERSION: v2
MODEL:
  IMG_MODEL: dinov2
  USE_GAZE_ATTENTION: True
  USE_GAZE_SEQ: True
  CROP_GAZE: True
```

Because `EGO_EXO_VERSION` is `v2`, the dataset resolves the split folder to:

```text
../data/ego_exo_splits/448pFull_downloaded_v2/ego/{train,val,test}.csv
```

## Checkpoints

Checkpoints are not included in git. Download them from the release location:

```text
Teacher evaluation checkpoint: <CHECKPOINT_DOWNLOAD_URL>/checkpoint_epoch_00020.pyth
Teacher training init:         <CHECKPOINT_DOWNLOAD_URL>/egovlpv2.pth
```

Use these paths as follows:

- `TEST.CHECKPOINT_FILE_PATH` — path to the trained SkillSight teacher checkpoint for evaluation, for example `/path/to/checkpoint_epoch_00020.pyth`.
- `TRAIN.CHECKPOINT_FILE_PATH` — path to the initialization checkpoint for training/fine-tuning, for example `/path/to/egovlpv2.pth`.
- `OUTPUT_DIR` — where logs, checkpoints, and segment-result JSON files are written.

The expected result for the released teacher checkpoint is:

```text
Expected overall/video-level accuracy: <number>
```

DINOv2 is loaded with `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`, so DINOv2 must be available online or in the local torch hub cache.

## Required external data

The split CSVs are included in `../data/ego_exo_splits`, but the large data files are not. Download the external data from:

```text
<DATA_DOWNLOAD_URL>/
```

The dataset expects:

- raw/downscaled video files, rooted at `DATA.PATH_PREFIX`
- `takes.json` metadata
- per-take gaze CSVs such as `eye_gaze/general_world_eye_gaze.csv`

A recommended data layout is:

```text
/path/to/SkillSight_data/
  videos/                                      # use this for DATA.PATH_PREFIX
    takes_448pFull_v2/
      upenn_0706_Dance_4_5/
        frame_aligned_videos/
          downscaled/
            448/
              aria01_214-1.mp4                # example path from the split CSVs
    <other video paths referenced by ../data/ego_exo_splits/448pFull_downloaded_v2/ego/*.csv>
  EgoExo4D_public_v1/
    takes/
      upenn_0706_Dance_4_5/
        eye_gaze/
          general_world_eye_gaze.csv
      <other_take_name>/
        eye_gaze/
          general_world_eye_gaze.csv
  takes.json
```

Set the video root in the evaluation/training command:

```bash
DATA.PATH_PREFIX /path/to/SkillSight_data/videos
```

For example, if a split CSV contains `takes_448pFull_v2/upenn_0706_Dance_4_5/frame_aligned_videos/downscaled/448/aria01_214-1.mp4`, then the command above makes the loader read `/path/to/SkillSight_data/videos/takes_448pFull_v2/upenn_0706_Dance_4_5/frame_aligned_videos/downscaled/448/aria01_214-1.mp4`.

This release expects `general_world_eye_gaze.csv` files to already be available under each take's `eye_gaze/` folder. These files are generated from EgoExo4D-provided VRS files and auxiliary CSV files; see `../processing_gaze/README.md` if you need to regenerate them.

Current implementation note: the split CSV path is configurable through `DATA.PATH_TO_DATA_DIR`, and the video root is configurable through `DATA.PATH_PREFIX`. The teacher dataset code still contains original local defaults for `takes.json` and `general_world_eye_gaze.csv`; for a public release, either mirror those files into the expected layout or make those paths configurable.

## Evaluation and `test_epoch20_segment_result.json`

From this folder:

```bash
python tools/run_net.py --cfg configs/EgoExo/SkillSightTeacher.yaml \
  TRAIN.ENABLE False VAL.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint_epoch_00020.pyth \
  DATA.PATH_PREFIX /path/to/SkillSight_data/videos \
  OUTPUT_DIR ./outputs/skillsight-teacher \
  NUM_GPUS 1
```

This loads `/path/to/checkpoint_epoch_00020.pyth` and writes:

```text
./outputs/skillsight-teacher/test_epoch20_segment_result.json
```

The filename is generated from the explicit `TEST.CHECKPOINT_FILE_PATH`: `checkpoint_epoch_00020.pyth` produces `test_epoch20_segment_result.json`. The JSON contains segment-level predictions keyed by video/take name. Video-level accuracy is computed by averaging the segment `pred` values per video and then comparing the resulting video prediction with the video label.

## Training / fine-tuning

From this folder:

```bash
python tools/run_net.py --cfg configs/EgoExo/SkillSightTeacher.yaml \
  TRAIN.ENABLE True VAL.ENABLE False TEST.ENABLE True \
  TRAIN.CHECKPOINT_FILE_PATH /path/to/egovlpv2.pth \
  TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint_epoch_00020.pyth \
  DATA.PATH_PREFIX /path/to/SkillSight_data/videos \
  OUTPUT_DIR ./outputs/skillsight-teacher \
  NUM_GPUS 1
```

Notes:

- Checkpoints and original outputs are intentionally not included.
- Use `TEST.CHECKPOINT_FILE_PATH` for evaluation checkpoints.
- Use `TRAIN.CHECKPOINT_FILE_PATH` for the `egovlpv2.pth` training/fine-tuning initialization checkpoint.
