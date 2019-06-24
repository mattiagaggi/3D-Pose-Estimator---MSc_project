

import numpy as np
from matplotlib import pyplot as plt


from utils.utils_H36M.common import H36M_CONF


def cam2pixel(cam_coord, f, c):

    """
    :param cam_coord: Nx3 camera coord
    :param f:
    :param c:
    :return: NX3 pixel coords
    """
    assert cam_coord.shape[-1] == 3
    assert f.shape == (1,2)
    assert c.shape == (1,2)

    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0,0] + c[0,0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[0,1] + c[0,1]
    z = cam_coord[..., 2]

    points = np.concatenate([x[..., np.newaxis],
                             y[..., np.newaxis],
                             z[..., np.newaxis]], axis=-1)

    return points





def pixel2cam(pixel_coord, f, c):
    """
    :param pixel_coord: Nx3 pixel coord
    :param f:
    :param c:
    :return: Nx3 camera coords
    """
    assert pixel_coord.shape[-1] == 3
    assert f.shape == (1,2)
    assert c.shape == (1,2)

    x = (pixel_coord[..., 0] - c[0,0]) / f[0,0] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[0,1]) / f[0,1] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]

    points = np.concatenate([x[..., np.newaxis],
                             y[..., np.newaxis],
                             z[..., np.newaxis]], axis=-1)

    return points


def world_to_camera(joints, n_joints, rot, t):
    """
    :param joints: 17x3 world joints
    :param n_joints: number of joints (17)
    :param rot: rotation matrix 3x3
    :param t: translation 1x3
    :return: 17x3 in camera coordinates
    """

    assert joints.shape == (n_joints, 3)
    assert t.shape == (1,3)
    assert rot.shape == (3, 3)
    return np.dot(joints - t, rot.T)


def camera_to_world(joints, n_joints, rot, t):
    """
    :param joints: 17x3 joints in camera coord
    :param n_joints: number of joints
    :param rot: rotation 3x3
    :param t: translation 1x3
    :return: joints in world coordinates
    """
    assert joints.shape == (n_joints, 3)
    assert t.shape == (1, 3)
    assert rot.shape == (3, 3)
    return np.dot(joints + t, rot)


def bounding_box_pixel(joints, root_idx,rot, t, f, c):
    """
    :param joints: joints in world coord
    :param root_idx: index of root joint which is centered in world coordinates
    :param rot: rotation
    :param t: translation
    :param f: intrinsic f
    :param c: intrinsic c
    :return: bounding box in pixel coors
    """

    assert f.shape == (1,2)
    assert c.shape == (1,2)
    assert rot.shape == (3, 3)

    bbox_3d=H36M_CONF.bbox_3d
    # build 3D bounding box centered on center_cam
    bbox_3d_center = np.array([bbox_3d[2] / 2,
                               bbox_3d[1] / 2,
                               0])
    #center cam in camera coord
    center_cam = np.dot(rot, joints[root_idx] - t.flatten())

    bbox3d_lt = center_cam - bbox_3d_center
    bbox3d_rb = center_cam + bbox_3d_center

    # back-project 3D BBox to 2D image
    temp = cam2pixel(bbox3d_lt, f, c)
    bbox2d_l, bbox2d_t = temp[..., 0], temp[..., 1]

    temp = cam2pixel(bbox3d_rb, f, c)
    bbox2d_r, bbox2d_b = temp[..., 0], temp[..., 1]

    bbox = np.array([bbox2d_l,  #x start
                     bbox2d_t, #y start
                     bbox2d_r - bbox2d_l,#width
                     bbox2d_b-bbox2d_t]) #height
    return bbox


def plot_bounding_box(fig,joints, root_idx,rot, t, f, c):
    bbox = bounding_box_pixel(joints, root_idx,rot, t, f, c)
    plt.scatter([bbox[0], bbox[2] + bbox[0]], [bbox[1], bbox[1]])
    plt.scatter([bbox[0], bbox[2] + bbox[0]], [bbox[3] + bbox[1], bbox[3] + bbox[1]])
    return fig




def world_to_pixel(joints, root_idx, n_joints, rot, t, f, c):
    """Project from world coordinates to the camera space

    Arguments:
        joints {numpy arrat} -- format (N_JOINTS x 3)
        root_idx {int} -- root joint index
        n_joints {int} -- N_JOINTS
        rot {numpy array} -- rotation matrix
        t {numpy array} -- translation matrix
        f {numpy array} -- focal length (format [1,2])
        c {numpy array} -- optical center (format [1,2])

    Returns:
        numpy array -- joints in pixel coordinates
        numpy array -- joints in camera coordinates
        numpy array -- camera center
    """

    assert joints.shape == (n_joints,3)
    assert t.shape == (1,3)
    assert f.shape == (1,2)
    assert c.shape == (1,2)
    assert rot.shape == (3, 3)

    # joints in camera reference system
    joint_cam = world_to_camera(joints, n_joints, rot, t)
    center_cam = joint_cam[root_idx]
    # joint in pixel coordinates
    joint_px = cam2pixel(joint_cam, f, c)

    return joint_px, center_cam

