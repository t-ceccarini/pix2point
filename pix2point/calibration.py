import cv2
import numpy as np

def calibrate_camera(images, pattern_size=(9, 6), square_size=1.0):
    """
    Calibrate the camera using a chessboard pattern.

    Args:
        images (list): List of input images containing the chessboard pattern.
        pattern_size (tuple): Number of corners per row and column of the chessboard.
        square_size (float): Physical size of each square in the chessboard.

    Returns:
        tuple: Calibration results including camera matrix, distortion coefficients, and extrinsics.
    """
    object_points = []  # 3D points in real-world space
    image_points = []  # 2D points in image plane

    # Generate the 3D coordinates of the chessboard corners
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            image_points.append(corners)
            object_points.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
