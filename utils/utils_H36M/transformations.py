

import numpy as np
from matplotlib import pyplot as plt
import cv2
from utils.utils_H36M.common import H36M_CONF


def cam2pixel(cam_coord, f, c):

    """
    :param cam_coord: Nx3 camera coord
    :param f: focal length
    :param c: image center
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

    bbox_pixel = np.array([bbox2d_l,  #x start
                     bbox2d_t, #y start
                     bbox2d_r - bbox2d_l,#width
                     bbox2d_b-bbox2d_t]) #height

    return bbox_pixel



def plot_bounding_box(fig,joints, root_idx,rot, t, f, c):
    bbox = bounding_box_pixel(joints, root_idx,rot, t, f, c)
    plt.scatter([bbox[0], bbox[2] + bbox[0]], [bbox[1], bbox[1]])
    plt.scatter([bbox[0], bbox[2] + bbox[0]], [bbox[3] + bbox[1], bbox[3] + bbox[1]])
    return fig


def world_to_pixel(joints, root_idx, n_joints, rot, t, f, c):
    """
    Project from world coordinates to the camera space
    :param joints:  format (N_JOINTS x 3)
    :param root_idx: 0
    :param n_joints: 17
    :param rot: rotation 3x3
    :param t: translation
    :param f: f intrinsic
    :param c: camera center intrinsics
    :return: Nx3 camera coords

    """
    assert joints.shape == (n_joints,3)
    assert t.shape == (1,3)
    assert f.shape == (1,2)
    assert c.shape == (1,2)
    assert rot.shape == (3, 3)

    # joints in camera reference system
    joint_cam = world_to_camera(joints, n_joints, rot, t)
    #center_cam = joint_cam[root_idx]
    # joint in pixel coordinates
    joint_px = cam2pixel(joint_cam, f, c)
    return joint_px#, center_cam


def rotate_x(angle_rad):
    """
    rotation around x axis in 3D
    :param angle_rad: angle in radiants
    :return: rotation matrix
    """

    return np.array([[1,                 0,                 0],
                     [0, np.cos(angle_rad),-np.sin(angle_rad)],
                     [0, np.sin(angle_rad), np.cos(angle_rad)]])


def rotate_y(angle_rad):
    """
    rotation around y axis in 3D
    :param angle_rad: angle in radiants
    :return: rotation matrix
    """

    return np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                     [0,                 1,                 0],
                     [-np.sin(angle_rad),0, np.cos(angle_rad)]])

def rotate_z(angle_rad):
    """
    rotation around z axis in 3D
    :param angle_rad: angle in radiants
    :return: rotation matrix
    """

    return np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                     [np.sin(angle_rad), np.cos(angle_rad), 0],
                     [0,                 0,                 1]])

#def in_plane_rotation


def cam_pointing_root(world_joints, root_idx, n_joints, rot, t):
    """
    rotation matrix so that the camera coordinates have z axis pointing at root index
    :param world_joints: Nx3 joints
    :param root_idx: 0
    :param n_joints: 17 joints
    :param rot: rotation matrix
    :param t: tranlation
    :return:
    """

    assert world_joints.shape == (n_joints, 3)
    assert rot.shape == (3, 3)
    assert t.shape == (1, 3)
    #reshape for matrix multiplication
    root_joint_world = world_joints[root_idx,:].reshape(1,3)
    root_joint_cam = np.dot(root_joint_world - t, rot.T)
    #need a rotation around the y coordinate of angle -x/z because of array indexing
    angle_y = np.arctan(- root_joint_cam[0, 0]/root_joint_cam[0, 2])
    rot_y = rotate_y(angle_y)
    root_joint_cam_y = np.dot( root_joint_cam, rot_y.T)
    # need a rotation around the x coordinate of angle y/z because of array indexing
    angle_x = np.arctan( root_joint_cam_y[0,1] / root_joint_cam_y[0,2])
    rot_x = rotate_x(angle_x)
    #the order is important here we first rotate y then x
    return np.dot(rot_x, rot_y)

def rotation_xy(cx,cy,angle):
    """
    :param image:
    :param joints_px:
    :param angle:
    :return:
    """
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
    return rot_mat



def get_affine(c_x, c_y, src_width, src_height, dst_width, dst_height, inv=False, rotation_angle=None):
    """
    get affine tranformation
    :param c_x: center x
    :param c_y: center y
    :param src_width: source width
    :param src_height: source height
    :param dst_width: target width
    :param dst_height: target height
    :param inv: True if inverse
    :return: np array 2x3
    """


    # augment size with scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5],
                          dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5],
                           dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0],
                            dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + [0, src_height * 0.5]
    src[2, :] = src_center + [src_width * 0.5, 0]

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    if rotation_angle is not None:
        # negative rotation because z axis is pointing at image not out of the image
        rotation = rotation_xy(dst_center[0],dst_center[1], - rotation_angle*180/np.pi)
        ones=np.ones((dst.shape[0],1))
        dst=np.concatenate([dst,ones], axis=1)
        dst = np.dot(dst, rotation.T)
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_patch_image(img, bbox, target_shape, rotation_angle=45):
    """
    :param img: numpy array
    :param bbox: bounding box [x, y, width, height] array
    :param target_shape: target  [width, height]
    :return: transformed image, transformation 2x3 array
    """

    assert len(bbox) == 4
    assert len(target_shape) == 2

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = get_affine(bb_c_x, bb_c_y,
                               bb_width, bb_height,
                               target_shape[1],
                               target_shape[0],
                               False,rotation_angle)
    img_patch = cv2.warpAffine(img, trans,
                               (int(target_shape[1]), int(target_shape[0])),
                               flags=cv2.INTER_LINEAR)

    return img_patch, trans




def transform_2d_joints(joints_px, transformation):

    """
    :param joints_px: Nx3 joint in pixel coords
    :param transformation: 2x3 affine
    :return: transformerd Nx3 pixel,visibility
    """

    transformed_joints = np.copy(joints_px)
    vis = np.ones(len(joints_px), dtype=bool)
    concatenated_ones=np.concatenate( [transformed_joints[:,:2],np.ones(shape=(transformed_joints.shape[0],1))], axis=1)
    transformed_joints[:,:2]=np.dot(concatenated_ones,transformation.T)
    # rescale points to output size
    transformed_joints[:, 2] *= H36M_CONF.depth_dim

    return transformed_joints, vis



def warp_joints_bbox(joints, bbox, cam_center,
                     src_shape, src_depth, trg_depth):

    """
    NOT USED
    Warp joints normalized according to the 2D bounding
    box back to their original version (bbox independent)
    :param joints: 17x3 joints
    :param bbox: [x, y, width, height]
    :param cam_center: numpy array size 2
    :param src_shape:  output resolution of the heatmaps
    :param src_depth: source depth
    :param trg_depth: target depth
    :return:
    """

    if joints.ndim == 1:
        joints = joints.reshape([-1, 3])

    p3d = np.zeros_like(joints)
    p3d[:, 0] = joints[:, 0] / src_shape * bbox[2]
    p3d[:, 1] = joints[:, 1] / src_shape * bbox[3]
    p3d[:, 0] += bbox[0]
    p3d[:, 1] += bbox[1]
    p3d[:, 2] = joints[:, 2] / src_depth - 0.5
    p3d[:, 2] *= trg_depth / 2.0
    p3d[:, 2] += cam_center[2]

    return p3d





