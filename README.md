# [CVPR 2026] SkillSight: Efficient First-Person Skill Assessment with Gaze

This repository is the official code release for the paper **SkillSight: Efficient First-Person Skill Assessment with Gaze**. We have not finished the release yet. We will include teacher checkpoint, gaze data, env, prediction logits for each test case in 2026 June.

SkillSight uses a two-stage design:

1. **SkillSight teacher** — a video+gaze model for skill assessment.
2. **SkillSight student** — a gaze-only student distilled from the teacher for efficient inference.

## Repository layout

```text
skillsight-teacher/     # Teacher model training/evaluation code
skillsight-student/     # Gaze-only student distillation code
processing_gaze/        # Utilities to create general_world_eye_gaze.csv
data/                   # Lightweight metadata/split CSVs copied for convenience
```

## Included lightweight data

```text
data/takes.json

data/ego_exo_splits/
  448pFull_downloaded_v2/ego/{train,val,test}.csv      # teacher split
  448pFull_Soccer_v2/ego/{train,val,test}.csv          # student category splits
  448pFull_Basketball_v2/ego/{train,val,test}.csv
  448pFull_RockClimbing_v2/ego/{train,val,test}.csv
  448pFull_Music_v2/ego/{train,val,test}.csv
  448pFull_Dance_v2/ego/{train,val,test}.csv
  448pFull_Cooking_v2/ego/{train,val,test}.csv
```

All split CSVs live in the shared `data/ego_exo_splits` folder. The teacher config points to this shared folder, and the student loader defaults to it as well.

## Downloads before running

Large data/model artifacts are intentionally not included in git. Download them separately:

```text
Teacher checkpoint:      <CHECKPOINT_DOWNLOAD_URL>/checkpoint_epoch_00020.pyth
Teacher init checkpoint: <CHECKPOINT_DOWNLOAD_URL>/egovlpv2.pth
EgoExo4D gaze data:  <DATA_DOWNLOAD_URL>
```

`checkpoint_epoch_00020.pyth` is used for teacher evaluation and as the teacher checkpoint for student distillation. `egovlpv2.pth` is used as the teacher training/fine-tuning initialization checkpoint.

## Gaze data

See `processing_gaze/README.md` for how we generate `general_world_eye_gaze.csv`. It is not necessary to run processing gaze. We already provide the general_world_eye_gaze.csv. 

## Expected external data layout

The teacher and student loaders expect split CSVs plus external video/gaze data. A typical local layout is:

```text
/path/to/SkillSight_data/
  videos/                                      # passed as DATA.PATH_PREFIX or EGOEXO_DATA_ROOT
    takes_448pFull_v2/
      upenn_0706_Dance_4_5/
        frame_aligned_videos/
          downscaled/
            448/
              aria01_214-1.mp4                # example path from the split CSVs
    <other video paths referenced by the split CSV files>
  EgoExo4D_public_v1/
    takes/                                    # passed as EGOEXO_GAZE_ROOT when using the student code
      upenn_0706_Dance_4_5/
        eye_gaze/
          general_world_eye_gaze.csv
      <other_take_name>/
        eye_gaze/
          general_world_eye_gaze.csv
  takes.json                                  # copy of data/takes.json or the EgoExo metadata file
```

For teacher evaluation, the video root is specified with `DATA.PATH_PREFIX`. For example, if a split CSV contains `takes_448pFull_v2/upenn_0706_Dance_4_5/frame_aligned_videos/downscaled/448/aria01_214-1.mp4`, then `DATA.PATH_PREFIX /path/to/SkillSight_data/videos` makes the loader read `/path/to/SkillSight_data/videos/takes_448pFull_v2/upenn_0706_Dance_4_5/frame_aligned_videos/downscaled/448/aria01_214-1.mp4`.

## Main entry points

- Teacher model: `skillsight-teacher/timesformer/models/custom_video_model_builder.py`, class `SkillSightTeacher`
- Teacher config: `skillsight-teacher/configs/EgoExo/SkillSightTeacher.yaml`
- Teacher runner: `skillsight-teacher/tools/run_net.py`
- Student model: `skillsight-student/model.py`, class `SkillSightStudent`
- Student training: `skillsight-student/train_distill.py`

## Quick start

Teacher evaluation with the released checkpoint:

```bash
cd skillsight-teacher
python tools/run_net.py --cfg configs/EgoExo/SkillSightTeacher.yaml \
  TRAIN.ENABLE False VAL.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint_epoch_00020.pyth \
  DATA.PATH_PREFIX /path/to/SkillSight_data/videos \
  OUTPUT_DIR ./outputs/skillsight-teacher \
  NUM_GPUS 1
```

This writes:

```text
./outputs/skillsight-teacher/test_epoch20_segment_result.json
```

The `test_epoch20` name comes from `checkpoint_epoch_00020.pyth`. If you evaluate a different explicit checkpoint, the suffix follows that checkpoint epoch.

Student training/distillation:

```bash
cd skillsight-student
python train_distill.py \
  --category RockClimbing \
  --teacher-checkpoint /path/to/checkpoint_epoch_00020.pyth \
  --device cuda:0
```

Each subfolder has its own README with more details.
