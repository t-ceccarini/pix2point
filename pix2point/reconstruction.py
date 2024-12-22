import cv2
import numpy as np

def triangulate_points_multiple(images, keypoints, matches, mtx):
    """
    Triangulates 3D points from multiple images.

    Args:
        images (list): List of input images.
        keypoints (list): Keypoints detected in each image.
        matches (list): Matches between consecutive images.
        mtx (numpy.ndarray): Intrinsic camera matrix.

    Returns:
        numpy.ndarray: Combined 3D point cloud.
    """
    points_3d = []
    R = np.eye(3)  # Initial rotation (identity for the first camera)
    T = np.zeros((3, 1))  # Initial translation (origin for the first camera)
    P_prev = np.hstack((R, T))  # Projection matrix for the first camera

    for i in range(len(images) - 1):
        # Get matched keypoints for the current pair of images
        pts1 = [keypoints[i][m.queryIdx].pt for m in matches[i]]
        pts2 = [keypoints[i + 1][m.trainIdx].pt for m in matches[i]]

        # Compute Essential Matrix and decompose to get R and T
        E, _ = cv2.findEssentialMat(np.array(pts1), np.array(pts2), mtx)
        _, R, T, _ = cv2.recoverPose(E, np.array(pts1), np.array(pts2), mtx)

        # Projection matrix for the second camera
        P_next = np.hstack((R, T))

        # Triangulate points between the two views
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        points_4d = cv2.triangulatePoints(P_prev, P_next, pts1.T, pts2.T)

        # Convert to 3D points
        points_3d_pair = points_4d[:3, :] / points_4d[3, :]
        points_3d.extend(points_3d_pair.T)

        # Update the projection matrix for the next iteration
        P_prev = np.dot(mtx, np.hstack((R, T)))

    return np.array(points_3d)
