"""Generate EgoExo `general_world_eye_gaze.csv` files.

This is a focused/organized copy of the logic from
`/vision/cwuau/ego_workplace/get_gaze_csv.py`.

It combines, for each take:
  1. an Aria VRS file in the take folder (for calibration),
  2. `trajectory/closed_loop_trajectory.csv`,
  3. `eye_gaze/general_eye_gaze.csv`, and
  4. `eye_gaze/general_eye_gaze_2d.csv`,
then writes `eye_gaze/general_world_eye_gaze.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from gaze_trace import get_gaze_in_world, get_gaze_projection


DEFAULT_DATA_PATH = "/vision/vision_data_2/EgoExo4D_public_v1/"
DEFAULT_TAKES_JSON = "/vision/cwuau/egoexo4d_gaze/takes.json"


def _take_name_from_split_path(path: str) -> str:
    """Extract take name from split paths like takes_448pFull_v2/<take>/frame..."""
    return path[path.index("v2") + 3 : path.index("/frame")]


def load_take_names_from_split(split_csv: str) -> list[str]:
    names: list[str] = []
    df = pd.read_csv(split_csv, header=None)
    for _, row in df.iterrows():
        path = str(row[0]).split()[0]
        names.append(_take_name_from_split_path(path))
    return names


def load_takes_metadata(takes_json: str) -> dict[str, dict]:
    with open(takes_json, "r") as f:
        records = json.load(f)
    return {record["take_name"]: record for record in records}


def stream_id_from_metadata(take_meta: dict | None) -> str | None:
    """Return the RGB stream_id for the all-streams VRS when metadata provides it."""
    if not take_meta:
        return None
    try:
        aria = take_meta["vrs_all_streams_relative_path"]
        aria = aria[: aria.index(".")]
        return take_meta["frame_aligned_videos"][aria]["rgb"]["stream_id"]
    except Exception:
        return None


def generate_general_world_eye_gaze(
    take_name: str,
    data_path: str = DEFAULT_DATA_PATH,
    take_meta: dict | None = None,
    stream_id: str | None = None,
    output_path: str | None = None,
) -> str | None:
    """Generate one `general_world_eye_gaze.csv` file.

    The take folder is expected at `<data_path>/takes/<take_name>/`.
    Returns the output CSV path on success; returns None if required inputs are
    absent or inconsistent.
    """
    try:
        gaze_traj, gaze_cpf_traj, depth, _, traj, head_pose, head_quat = get_gaze_in_world(
            take_name, data_path=data_path
        )
    except Exception as exc:
        print(f"[skip] {take_name}: failed world gaze computation: {exc}")
        return None

    mps_sample_path = os.path.join(data_path, "takes", take_name)
    eye_gaze_2d_path = os.path.join(mps_sample_path, "eye_gaze", "general_eye_gaze_2d.csv")
    try:
        eye_gaze_raw_data = pd.read_csv(eye_gaze_2d_path)
    except FileNotFoundError:
        print(f"[skip] {take_name}: missing {eye_gaze_2d_path}")
        return None

    if not (len(gaze_traj) == len(traj) == len(depth) == len(eye_gaze_raw_data) == len(gaze_cpf_traj)):
        print(
            f"[skip] {take_name}: length mismatch "
            f"world={len(gaze_traj)} traj={len(traj)} depth={len(depth)} "
            f"2d={len(eye_gaze_raw_data)} cpf={len(gaze_cpf_traj)}"
        )
        return None

    n = len(gaze_traj)
    timestamp_us = eye_gaze_raw_data["tracking_timestamp_us"].values

    stream_id = stream_id or stream_id_from_metadata(take_meta)
    if stream_id:
        try:
            projections, calibration = get_gaze_projection(take_name, stream_id, data_path=data_path)
            egox = [point[0] for point in projections]
            egoy = [point[1] for point in projections]
            image_size = calibration.get_image_size()[0]
        except Exception as exc:
            print(f"[warn] {take_name}: projection via VRS stream_id failed ({exc}); using general_eye_gaze_2d x/y")
            egox = eye_gaze_raw_data["x"].values
            egoy = eye_gaze_raw_data["y"].values
            image_size = 1408
    else:
        egox = eye_gaze_raw_data["x"].values
        egoy = eye_gaze_raw_data["y"].values
        image_size = 1408

    df = pd.DataFrame(
        {
            "frame_num": np.arange(n),
            "gaze_world_x": gaze_traj[:, 0],
            "gaze_world_y": gaze_traj[:, 1],
            "gaze_world_z": gaze_traj[:, 2],
            "gaze_cpf_x": gaze_cpf_traj[:, 0],
            "gaze_cpf_y": gaze_cpf_traj[:, 1],
            "gaze_cpf_z": gaze_cpf_traj[:, 2],
            "traj_world_x": traj[:, 0],
            "traj_world_y": traj[:, 1],
            "traj_world_z": traj[:, 2],
            "head_z_(yaw)": head_pose[:, 0],
            "head_y_(pitch)": head_pose[:, 1],
            "head_x_(roll)": head_pose[:, 2],
            "ego_x": egox,
            "ego_y": egoy,
            "depth": depth,
            "image_size": image_size,
            "timestamp_us": timestamp_us,
            "head_quat_x": head_quat[:, 0],
            "head_quat_y": head_quat[:, 1],
            "head_quat_z": head_quat[:, 2],
            "head_quat_w": head_quat[:, 3],
        }
    )

    if output_path is None:
        output_path = os.path.join(mps_sample_path, "eye_gaze", "general_world_eye_gaze.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[ok] {take_name}: {output_path}")
    return output_path


def _worker(args_tuple):
    take_name, data_path, take_meta = args_tuple
    return generate_general_world_eye_gaze(take_name, data_path=data_path, take_meta=take_meta)


def generate_many(
    take_names: Iterable[str],
    data_path: str = DEFAULT_DATA_PATH,
    takes_json: str | None = DEFAULT_TAKES_JSON,
    num_workers: int = 1,
) -> list[str | None]:
    metadata = load_takes_metadata(takes_json) if takes_json else {}
    jobs = [(name, data_path, metadata.get(name)) for name in take_names]
    if num_workers <= 1:
        return [_worker(job) for job in tqdm(jobs)]
    with Pool(num_workers) as pool:
        return list(tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EgoExo general_world_eye_gaze.csv")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Dataset root containing takes/<take_name>/")
    parser.add_argument("--take-name", action="append", default=[], help="Take name to process. Can be passed multiple times.")
    parser.add_argument("--split-csv", default="", help="Optional split CSV; take names are extracted from its video paths.")
    parser.add_argument("--takes-json", default=DEFAULT_TAKES_JSON, help="Optional takes.json for stream_id lookup. Use '' to disable.")
    parser.add_argument("--stream-id", default="", help="Optional RGB stream_id for a single --take-name run.")
    parser.add_argument("--output-path", default="", help="Optional output path for a single --take-name run.")
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    takes_json = args.takes_json or None

    take_names = list(args.take_name)
    if args.split_csv:
        take_names.extend(load_take_names_from_split(args.split_csv))
    # Preserve order while removing duplicates.
    take_names = list(dict.fromkeys(take_names))
    if not take_names:
        raise SystemExit("Provide --take-name and/or --split-csv")

    if len(take_names) == 1 and args.output_path:
        take_meta = None
        if takes_json:
            take_meta = load_takes_metadata(takes_json).get(take_names[0])
        generate_general_world_eye_gaze(
            take_names[0],
            data_path=args.data_path,
            take_meta=take_meta,
            stream_id=args.stream_id or None,
            output_path=args.output_path,
        )
    else:
        generate_many(take_names, data_path=args.data_path, takes_json=takes_json, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
