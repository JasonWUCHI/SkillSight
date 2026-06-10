from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import pandas as pd
import plotly.graph_objs as go
import json
import matplotlib
from matplotlib import cm
import argparse
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)

# === Helper to transform gaze point to world coordinate ===
def gaze_to_world_coord(vrs_data_provider, T_world_device, gaze):
    device_calibration = vrs_data_provider.get_device_calibration()
    T_device_cpf = device_calibration.get_transform_device_cpf()

    T_world_cpf = T_world_device @ T_device_cpf
    gaze_transformed = T_world_cpf @ gaze.T

    return gaze_transformed.flatten()

# === Paths ===
def get_gaze_in_world(take_name, task_start_sec=0, task_end_sec=1e9, start_time=None, end_time=None, max_gaze_length=None, data_path="/vision/vision_data_2/EgoExo4D_public_v1/"):

    mps_sample_path =  os.path.join(data_path, "takes", take_name)
    if start_time is None:
        start_time = float("-inf")
    if end_time is None:
        end_time = float("inf")
    if max_gaze_length is None:
        max_gaze_length = float("inf")

    vrs_files = [f for f in os.listdir(mps_sample_path) if f.endswith('.vrs')]
    if not vrs_files:
        raise FileNotFoundError("No .vrs file found in the given path.")
    vrsfile = os.path.join(mps_sample_path, vrs_files[0])
    
    trajectory_path = os.path.join(mps_sample_path, "trajectory", "closed_loop_trajectory.csv")
    eye_gaze_path = os.path.join(mps_sample_path, "eye_gaze", "general_eye_gaze.csv")

    # === Load data ===
    provider = data_provider.create_vrs_data_provider(vrsfile)
    T_device_RGB = provider.get_device_calibration().get_transform_device_sensor("camera-rgb")

    mps_trajectory = mps.read_closed_loop_trajectory(trajectory_path)
    generalized_eye_gazes = mps.read_eyegaze(eye_gaze_path)
    
    # === Crop data ===
    generalized_eye_gazes = generalized_eye_gazes[int(task_start_sec*10):int(task_end_sec*10)]

    # === Pair gaze timestamps with trajectory indices ===
    gaze_timestamps = [g.tracking_timestamp.total_seconds() for g in generalized_eye_gazes]
    traj_timestamps = [t.tracking_timestamp.total_seconds() for t in mps_trajectory]

    pairing_indices = []
    interval = [gaze_timestamps[0]+start_time, gaze_timestamps[0]+end_time]
    for gidx, gt in enumerate(gaze_timestamps):
        if gt < interval[0] or gt>interval[1]:
            continue
        midx = int((gt - traj_timestamps[0]) / 0.001)
        if midx < len(mps_trajectory):
            pairing_indices.append([gidx, midx])

    # === Compute normalized time and colormap ===
    gaze_times = [mps_trajectory[midx].tracking_timestamp.total_seconds() for _, midx in pairing_indices]

    # === Compute gaze points in world coordinates ===
    gaze_traj = np.empty((len(pairing_indices), 3))
    gaze_cpf_traj = np.empty((len(pairing_indices), 3))
    traj = np.empty([len(pairing_indices), 3])
    head_pose = np.empty((len(pairing_indices), 3))
    head_quat = np.empty((len(pairing_indices), 4))


    for i, (gidx, midx) in enumerate(pairing_indices):
        pose = mps_trajectory[midx]
        transform = pose.transform_world_device

        # Gaze intersection point (CPF space)
        gaze_x, gaze_y, gaze_z = mps.get_gaze_intersection_point(
            generalized_eye_gazes[gidx].vergence.left_yaw,
            generalized_eye_gazes[gidx].vergence.right_yaw,
            generalized_eye_gazes[gidx].pitch,
        )
        gaze_point_cpf = np.array([gaze_x, gaze_y, gaze_z])

        # Convert to scipy Rotation object
        # Rotation without consider head facing
        # r = R.from_matrix(transform.rotation().to_matrix())
        # yaw, pitch, roll = r.as_euler('zyx', degrees=True)
        # head_pose[i, :] = np.array([yaw, pitch, roll])

        # Rotationw with head facing
        r_world = R.from_matrix(transform.rotation().to_matrix())
        r_quat = r_world.as_quat()
        yaw_only = r_world.as_euler('zyx', degrees=False)[0]
        R_inv_yaw = R.from_euler('z', -yaw_only).as_matrix()
        R_pitch_roll_local = R_inv_yaw @ transform.rotation().to_matrix()
        r_local = R.from_matrix(R_pitch_roll_local)
        pitch, roll = r_local.as_euler('zyx', degrees=True)[1:]
        head_pose[i, :] = np.array([np.degrees(yaw_only), pitch, roll])

        gaze_cpf_traj[i,:] = gaze_point_cpf
        gaze_traj[i, :] = gaze_to_world_coord(provider, transform, gaze_point_cpf)
        traj[i, :] = transform.translation()
        head_quat[i, :] = r_quat

    # === Create a single 3D scatter trace ===
    norms = np.linalg.norm(gaze_traj, axis=1, keepdims=True)
    gaze_traj = np.where(
        norms > 10,
        gaze_traj * (10 / norms),
        gaze_traj
    )

    # === Load depth ===
    depths = np.array([data.depth for data in generalized_eye_gazes])

    return gaze_traj, gaze_cpf_traj, depths, np.array(gaze_times), traj, head_pose, head_quat


def get_gaze_projection(take_name, stream_id, task_start_sec=0, task_end_sec=1e9, start_time=None, end_time=None, max_gaze_length=None, data_path="/vision/vision_data_2/EgoExo4D_public_v1/"):
    mps_sample_path =  os.path.join(data_path, "takes", take_name)
    mps_data_paths_provider = mps.MpsDataPathsProvider(mps_sample_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)
    assert mps_data_provider.has_general_eyegaze(), "The sequence does not have Eye Gaze data"


    vrs_files = [f for f in os.listdir(mps_sample_path) if f.endswith('.vrs')]
    if not vrs_files:
        raise FileNotFoundError("No .vrs file found in the given path.")
    vrsfile = os.path.join(mps_sample_path, vrs_files[0])

    # === Load data ===
    vrs_data_provider = data_provider.create_vrs_data_provider(vrsfile)
    rgb_stream_id = StreamId(stream_id)
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()

    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    eye_gaze_path = os.path.join(mps_sample_path, "eye_gaze", "general_eye_gaze.csv")

    generalized_eye_gazes = mps.read_eyegaze(eye_gaze_path)

    camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    assert camera_calibration is not None, "no camera calibration"

    #depth
    generalized_gaze_center_in_pixels = []
    for generalized_eye_gaze in generalized_eye_gazes: 
        depth_m = generalized_eye_gaze.depth or 1.0
        gaze2d = get_gaze_vector_reprojection(generalized_eye_gaze, rgb_stream_label, device_calibration, camera_calibration, depth_m, make_upright=True)
        generalized_gaze_center_in_pixels.append(gaze2d)

    return generalized_gaze_center_in_pixels, camera_calibration

def get_gaze_in_cpf(take_name, task_start_sec=0, task_end_sec=float("-inf"), start_time=None, end_time=None, max_gaze_length=None, data_path="/vision/vision_data_2/EgoExo4D_public_v1/"):

    mps_sample_path =  os.path.join(data_path, "takes", take_name)
    if start_time is None:
        start_time = float("-inf")
    if end_time is None:
        end_time = float("inf")
    if max_gaze_length is None:
        max_gaze_length = float("inf")

    vrs_files = [f for f in os.listdir(mps_sample_path) if f.endswith('.vrs')]
    if not vrs_files:
        raise FileNotFoundError("No .vrs file found in the given path.")
    vrsfile = os.path.join(mps_sample_path, vrs_files[0])
    
    trajectory_path = os.path.join(mps_sample_path, "trajectory", "closed_loop_trajectory.csv")
    eye_gaze_path = os.path.join(mps_sample_path, "eye_gaze", "general_eye_gaze.csv")

    # === Load data ===
    provider = data_provider.create_vrs_data_provider(vrsfile)
    T_device_RGB = provider.get_device_calibration().get_transform_device_sensor("camera-rgb")

    mps_trajectory = mps.read_closed_loop_trajectory(trajectory_path)
    generalized_eye_gazes = mps.read_eyegaze(eye_gaze_path)
    
    # === Crop data ===
    generalized_eye_gazes = generalized_eye_gazes[int(task_start_sec*10):int(task_end_sec*10)]

    # === Pair gaze timestamps with trajectory indices ===
    gaze_timestamps = [g.tracking_timestamp.total_seconds() for g in generalized_eye_gazes]
    traj_timestamps = [t.tracking_timestamp.total_seconds() for t in mps_trajectory]

    pairing_indices = []
    interval = [gaze_timestamps[0]+start_time, gaze_timestamps[0]+end_time]
    for gidx, gt in enumerate(gaze_timestamps):
        if gt < interval[0] or gt>interval[1]:
            continue
        midx = int((gt - traj_timestamps[0]) / 0.001)
        if midx < len(mps_trajectory):
            pairing_indices.append([gidx, midx])

    # === Compute normalized time and colormap ===
    gaze_times = [mps_trajectory[midx].tracking_timestamp.total_seconds() for _, midx in pairing_indices]

    # === Compute gaze points in world coordinates ===
    gaze_traj = np.empty((len(pairing_indices), 3))
    traj = np.empty([len(pairing_indices), 3])


    for i, (gidx, midx) in enumerate(pairing_indices):
        pose = mps_trajectory[midx]
        transform = pose.transform_world_device

        # Gaze intersection point (CPF space)
        gaze_x, gaze_y, gaze_z = mps.get_gaze_intersection_point(
            generalized_eye_gazes[gidx].vergence.left_yaw,
            generalized_eye_gazes[gidx].vergence.right_yaw,
            generalized_eye_gazes[gidx].pitch,
        )
        gaze_point_cpf = np.array([gaze_x, gaze_y, gaze_z])
        gaze_traj[i, :] = gaze_point_cpf

    # === Create a single 3D scatter trace ===
    if max_gaze_length != None:
        norms = np.linalg.norm(gaze_traj, axis=1, keepdims=True)
        gaze_traj = np.where(
            norms > 10,
            gaze_traj * (10 / norms),
            gaze_traj
        )

    # === Load depth ===
    depths = [data.depth for data in generalized_eye_gazes]

    return gaze_traj, np.array(depths), np.array(gaze_times)

def get_plot(gaze_traj, gaze_times, save_path="gaze_points_colored_by_time.html"):

    gaze_times_norm = (np.array(gaze_times) - np.min(gaze_times)) / (np.max(gaze_times) - np.min(gaze_times))
    colormap = matplotlib.colormaps["viridis"]
    colors = [f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b, _ in colormap(gaze_times_norm)]

    # === Create a single 3D scatter trace ===
    gaze_trace = go.Scatter3d(
        x=gaze_traj[:, 0],
        y=gaze_traj[:, 1],
        z=gaze_traj[:, 2],
        mode="markers",
        marker=dict(size=3, color=colors, opacity=0.9),
        name="Gaze Points (colored by time)",
        hoverinfo="none",
    )

    # === Layout ===
    layout = go.Layout(
        scene=dict(
            bgcolor="white",
            dragmode="orbit",
            aspectmode="data",
            xaxis_visible=True,
            yaxis_visible=True,
            zaxis_visible=True,
            camera=dict(
                eye=dict(x=0.5, y=0.5, z=0.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        width=1100,
        height=1000,
        title="Gaze Points Colored by Time",
    )

    # === Render ===
    fig = go.Figure(data=[gaze_trace], layout=layout)
    fig.write_html("gaze_points_colored_by_time.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot gaze data.")

    parser.add_argument("--take_name", type=str, help="Name of the take (e.g., take_001)")
    parser.add_argument("--start_time", type=float, help="Start time in seconds")
    parser.add_argument("--end_time", type=float, help="End time in seconds")

    parser.add_argument("--max_gaze_length", type=int, default=10, help="Maximum length of gaze trajectory")
    parser.add_argument("--data_path", type=str, default="/vision/vision_data_2/EgoExo4D_public_v1/", help="Path to dataset")
    parser.add_argument("--save_path", type=str, default="gaze_points_colored_by_time.html", help="Path to save plot")

    args = parser.parse_args()

    gaze_traj, _, depths, gaze_times, traj, _, _ = get_gaze_in_world(
        args.take_name,
        args.start_time,
        args.end_time,
        max_gaze_length=args.max_gaze_length,
        data_path=args.data_path
    )

    get_plot(gaze_traj, gaze_times, save_path=args.save_path)





