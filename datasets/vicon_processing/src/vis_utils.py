import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_points_3d(ax, points, split_every=-1):
        if split_every > 0:
            for i in range(3):
                ps = points[13*i:13*(i+1)]
                ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=5)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def XYZ_frame(T):
    """
    Given a transform T, plot the corresponding frame
    """
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    points = (T[:3, :3] @ points.transpose()).transpose()
        
    p_origin = T @ np.array([0.0, 0.0, 0.0, 1.0])
    p_origin /= p_origin[-1]

    origins = np.tile(p_origin[:3], (3, 1))

    soa = np.hstack((origins, points[:, :]))
    X, Y, Z, U, V, W = zip(*soa)
    
    return X, Y, Z, U, V, W

def plot_frame(ax, T, arrow_length=200):
    """
    Given a transform T, plot the corresponding frame
    """
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    points = (T[:3, :3] @ points.transpose()).transpose()
        
    p_origin = T @ np.array([0.0, 0.0, 0.0, 1.0])
    p_origin /= p_origin[-1]

    origins = np.tile(p_origin[:3], (3, 1))

    soa = np.hstack((origins, points[:, :]))
    X, Y, Z, U, V, W = zip(*soa)
    
    colors = ['red', 'green', 'blue']

    ax.quiver(X, Y, Z, U, V, W, colors=colors, length=arrow_length, normalize=True)

def plot_2d_points(frame, points, color=(255, 0, 0)):
    
    for p in points:
        if np.isnan(p).any():
            continue
        cv2.circle(frame, p[:2].astype(int), 5, color, -1)

    return frame

def plot_2d_points_plt(ax, points, color=(0.0, 1.0, 0.0), alpha=1, graded=False):
    
    if graded:
        colors = np.zeros((points.shape[0], 3))
        w = np.arange(1, points.shape[0]+1) / points.shape[0]
        colors[:, 0] = color[0] * w
        colors[:, 1] = color[1] * w
        colors[:, 2] = color[2] * w
    else:
        colors = [color]

    ax.scatter(points[:, 0], points[:, 1], c=colors, alpha=alpha)

def plot_2d_difference(frame, image_points, projected_points):

    for ip, pp in zip(image_points, projected_points):
        # draw image point
        cv2.circle(frame, ip[:2].astype(int), 5, (0, 0, 255), -1)
        # draw projected point
        cv2.circle(frame, pp[:2].astype(int), 5, (255, 0, 0), -1)
        # draw line connection
        cv2.line(frame, ip[:2].astype(int), pp[:2].astype(int), (0, 255, 0), 2)

    return frame