import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pyvista as pv

def visualize_point_cloud_matplotlib(points):
    """
    Visualizes a 3D point cloud using Matplotlib.

    Parameters:
    - points (numpy.ndarray): An array of shape (N, 3) containing the 3D points.
    """
    fig = plt.figure()  # Create a figure
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis

    # Extract x, y, z coordinates from the points array
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter plot of the 3D points
    ax.scatter(x, y, z, c='cyan', s=5)  # c=cyan, s=5 sets color and point size

    # Set axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Set the background color
    ax.set_facecolor('black')

    # Show the plot
    plt.show()

def visualize_point_cloud_pyvista(points):
    """
    Visualizes a 3D point cloud using PyVista.

    Parameters:
    - points (numpy.ndarray): An array of shape (N, 3) containing the 3D points.
    """
    # Create a PointCloud object in PyVista (PolyData holds points, cells, etc.)
    point_cloud = pv.PolyData(points)

    # Visualize the point cloud using PyVista's plot function
    point_cloud.plot(
        point_size=5,  # size of the points
        color="cyan",  # point color
        show_grid=False,  # hide the background grid
        window_size=[800, 600],  # window size
        background="black",  # background color
        interactive=False   # Disable interactive mode
    )


def visualize_point_cloud_open3d(points_3d):
    """
    Visualizes a 3D point cloud.

    Args:
        points_3d (numpy.ndarray): 3D points to visualize.
    """
    print(f"Number of 3D points: {len(points_3d)}")
    print(f"Shape of 3D points: {points_3d.shape}")
    np.savetxt("points_3d.txt", points_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    o3d.visualization.draw_geometries([pcd])

    # Ensure that OpenCV waits to display the window
    cv2.waitKey(0)
