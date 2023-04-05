from typing import List

import numpy as np
from utils.transformations import quaternion_matrix, translation_matrix


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def pose_to_matrix(pose: np.ndarray):
    r = quaternion_matrix(pose[3:])
    t = translation_matrix(pose[:3])

    return np.dot(t, r)


def transform_point_cloud(
        points: np.ndarray,
        src_pose: np.ndarray,
        dst_pose: np.ndarray
):
    src_m = pose_to_matrix(src_pose)
    dst_m = pose_to_matrix(dst_pose)

    points_homo = np.concatenate((points.T, np.ones((1, points.shape[0]))), 0)
    points_transformed = (np.linalg.inv(dst_m) @ src_m @ points_homo)[:3, :].T

    return points_transformed


def filter_points(
        points: np.ndarray,
        z_range: np.ndarray  # [z_min, z_max]
):
    points_in_range_mask = np.logical_and(
        points[:, -1] > z_range[0], points[:, -1] < z_range[1]
    )
    return points[points_in_range_mask]
