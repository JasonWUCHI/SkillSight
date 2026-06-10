# Processing EgoExo gaze into `general_world_eye_gaze.csv`

This folder contains the focused scripts used to generate the enriched gaze file consumed by the proficiency-estimation datasets:

```text
eye_gaze/general_world_eye_gaze.csv
```

The scripts are copied/organized from `/vision/cwuau/ego_workplace`:

- `gaze_trace.py` — geometry helpers that read VRS calibration, MPS trajectory, and MPS eye gaze.
- `get_gaze_csv.py` — driver that writes `general_world_eye_gaze.csv`.
- `uni_utils.py` — tiny JSON/pickle utility dependency.

## What the script does

For each take, it combines Project Aria / MPS outputs:

```text
<takes_root>/<take_name>/
  aria01.vrs                                  # or another .vrs file; used for calibration
  trajectory/closed_loop_trajectory.csv       # MPS closed-loop trajectory
  eye_gaze/general_eye_gaze.csv               # MPS generalized 3D eye-gaze angles/depth
  eye_gaze/general_eye_gaze_2d.csv            # MPS 2D gaze/timestamps
```

and writes:

```text
<takes_root>/<take_name>/eye_gaze/general_world_eye_gaze.csv
```

The VRS is **not** used here to estimate gaze from raw eye-camera streams. The script assumes MPS already produced `general_eye_gaze.csv`, `general_eye_gaze_2d.csv`, and `closed_loop_trajectory.csv`. The VRS is used for device/camera calibration and reprojection.

## The four required per-take inputs

If you already have these four files under a single take folder, you can generate `general_world_eye_gaze.csv`:

1. `*.vrs`
2. `trajectory/closed_loop_trajectory.csv`
3. `eye_gaze/general_eye_gaze.csv`
4. `eye_gaze/general_eye_gaze_2d.csv`

Example layout:

```text
/path/to/EgoExo4D_public_v1/takes/upenn_0706_Dance_4_5/
  aria01.vrs
  trajectory/closed_loop_trajectory.csv
  eye_gaze/general_eye_gaze.csv
  eye_gaze/general_eye_gaze_2d.csv
```

Then run from this folder:

```bash
python get_gaze_csv.py \
  --data-path /path/to/EgoExo4D_public_v1 \
  --take-name upenn_0706_Dance_4_5
```

Output:

```text
/path/to/EgoExo4D_public_v1/takes/upenn_0706_Dance_4_5/eye_gaze/general_world_eye_gaze.csv
```

## Optional metadata for better RGB reprojection

`get_gaze_csv.py` can use `takes.json` to find the RGB stream id used by each take:

```bash
python get_gaze_csv.py \
  --data-path /path/to/EgoExo4D_public_v1 \
  --takes-json /path/to/takes.json \
  --take-name upenn_0706_Dance_4_5
```

If `takes.json` or stream-id lookup is unavailable, the script still writes `general_world_eye_gaze.csv`; it falls back to `x`/`y` from `general_eye_gaze_2d.csv` for the `ego_x` and `ego_y` columns.

For a single take, you can also provide the RGB stream id directly:

```bash
python get_gaze_csv.py \
  --data-path /path/to/EgoExo4D_public_v1 \
  --take-name upenn_0706_Dance_4_5 \
  --stream-id 214-1
```

## Batch processing from a split CSV

If you have a split CSV with rows like:

```text
takes_448pFull_v2/<take_name>/frame_aligned_videos/downscaled/448/aria01_214-1.mp4 <label>
```

run:

```bash
python get_gaze_csv.py \
  --data-path /path/to/EgoExo4D_public_v1 \
  --takes-json /path/to/takes.json \
  --split-csv /path/to/ego/test.csv \
  --num-workers 8
```

## Dependencies

The scripts require Python packages used by the original code, especially:

```text
projectaria-tools
numpy
pandas
scipy
tqdm
matplotlib
plotly
```

`projectaria-tools` provides `projectaria_tools.core.data_provider` and `projectaria_tools.core.mps`.

## Output columns

The generated CSV includes:

- `gaze_world_x/y/z` — gaze intersection point transformed to world coordinates
- `gaze_cpf_x/y/z` — gaze point in CPF/device-related coordinates
- `traj_world_x/y/z` — camera/device trajectory in world coordinates
- `head_z_(yaw)`, `head_y_(pitch)`, `head_x_(roll)`
- `ego_x`, `ego_y` — 2D gaze projection in the egocentric RGB image
- `depth`, `image_size`, `timestamp_us`
- `head_quat_x/y/z/w`

