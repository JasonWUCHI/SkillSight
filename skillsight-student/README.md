# SkillSight Student

This folder contains the focused gaze-only student distillation code for **SkillSight**.

The student is trained with multitask supervision and optional teacher-feature distillation:

- skill/proficiency prediction
- subtask/action prediction
- latent teacher/student feature alignment

## Model names

Main classes in `model.py`:

```text
SkillSightTeacherBackbone          # embedded teacher architecture used for feature extraction
SkillSightTeacherFeatureExtractor  # loads a teacher checkpoint and returns teacher features
SkillSightStudent                  # gaze-only student model
SkillSightProjectionHead           # maps teacher/student features into the distillation latent
```

## Evidence in the original code

The original source path was `/vision/cwuau/ProficiencyEstimation/toy`.

- `toy/train2.py` loaded the teacher feature extractor, gaze student transformer, and `Egoexo` dataset.
- The teacher checkpoint was loaded through the teacher feature extractor, then set to eval mode.
- Each train step obtained `teacher_feat = teacher_model(frames, frames_crop, teacher_gaze_input, scenario_id)`.
- The student received gaze-only input plus teacher features:
  `dis_out, target_feat, gaze_pred, subtask_preds = model(student_gaze_input, teacher_feat, True)`.
- The multitask objectives were skill prediction (`gaze_pred`) and subtask/action prediction (`subtask_preds`).
- The latent mapper was the projection head: `student_proj` maps the student distillation token and `teacher_proj` maps concatenated teacher features.

In this cleaned copy, those original names are organized as `SkillSightTeacherFeatureExtractor`, `SkillSightStudent`, and `SkillSightProjectionHead`.

## Distillation loss behavior

In the original `toy/train2.py`, the L1 distillation loss line existed but was commented out.
This copy keeps that behavior by default:

```text
--distill-loss-weight 0.0
```

Set it above zero to enable L1 alignment between `dis_out` and `target_feat`.

## Required data

The category split CSVs are included in the shared data folder:

```text
../data/ego_exo_splits/448pFull_<Category>_v2/ego/{train,val,test}.csv
```

For actual training/evaluation, you also need:

- raw/downscaled videos
- per-take `eye_gaze/general_world_eye_gaze.csv`
- `takes.json`
- a SkillSight teacher checkpoint, for example `<CHECKPOINT_DOWNLOAD_URL>/checkpoint_epoch_00020.pyth`

The teacher checkpoint path is specified with `--teacher-checkpoint /path/to/checkpoint_epoch_00020.pyth`.

The student uses the same teacher gaze trajectory channel selection as the SkillSight teacher path: columns `[0:13]` plus `[15:]`, padded/trimmed to the teacher checkpoint gaze width of 15. There is no separate default student gaze-index subset.

Environment variables for local data roots:

```bash
export EGOEXO_SPLIT_ROOT=/path/to/data/ego_exo_splits
export EGOEXO_DATA_ROOT=/path/to/video/data/root
export EGOEXO_TAKES_JSON=/path/to/takes.json
export EGOEXO_GAZE_ROOT=/path/to/EgoExo4D_public_v1/takes
```

Defaults preserve the original local paths where possible.

## Train/evaluate

```bash
python train_distill.py \
  --category RockClimbing \
  --teacher-checkpoint /path/to/checkpoint_epoch_00020.pyth \
  --device cuda:0 \
  --output-json outputs/RockClimbing.json \
  --save-checkpoint outputs/RockClimbing_student.pth
```

No teacher/student checkpoints are included in this copy. Download the released teacher checkpoint from `<CHECKPOINT_DOWNLOAD_URL>/checkpoint_epoch_00020.pyth` and pass its local path with `--teacher-checkpoint`.
