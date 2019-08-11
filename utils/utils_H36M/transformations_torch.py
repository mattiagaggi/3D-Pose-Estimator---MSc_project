import torch
from utils.trans_numpy_torch import create_one_float_tensor


#R x4, t x4 , f x4 , trans x4 , joints, mean_pose


def check_f_t_input(t):
    t_size=list(t.size())
    assert t_size[1]==1
    assert t_size[2] == 3

def check_rot_input(rot):

    rot_size=list(rot.size())
    assert rot_size[1] == 3
    assert rot_size[2] == 3

def check_joints_input(joints,n_joints):
    joint_size = list(joints.size())
    assert joint_size[1] == n_joints
    assert joint_size[2] == 3

def check_transformation_input(trans):
    trans_size = list(trans.size())
    assert trans_size[1] == 3
    assert trans_size[2] == 2


def world_to_camera_batch(joints, n_joints, rot, t):
    check_joints_input(joints,n_joints)
    check_rot_input(rot)
    check_f_t_input(t)
    return torch.bmm(joints - t, rot.transpose(1,2))

def camera_to_world_batch(joints,n_joints,rot, t):
    check_joints_input(joints, n_joints)
    check_rot_input(rot)
    check_f_t_input(t)
    return torch.bmm(joints + t, rot)




def camera_to_pixels_batch(joints_cam, f, c):
    check_joints_input(joints_cam)
    check_f_t_input(c)
    check_f_t_input(f)
    x = joints_cam[:, :, 0]
    y = joints_cam[:, :, 1]
    z = joints_cam[:, :, 0]
    fx = f[:, 0, 0].view(-1,1)
    fy = f[:, 0, 1].view(-1,1)
    cx = c[:, 0, 0].view(-1,1)
    cy = c[:, 0, 1].view(-1,1)
    new_x = torch.div(x,z) * fx + cx
    new_y = torch.div(y,z) * fy + cy
    joints_px = torch.stack([new_x,new_y], dim=2)
    #we erase z axis
    return joints_px



def transform_2d_joints_batch(joints_px, transformation):
    check_joints_input(joints_px)
    check_transformation_input(transformation)
    n, j, _ = list(joints_px.size())
    ones = create_one_float_tensor([n, j, 1])
    joints_concat = torch.cat([joints_px, ones], dim=2)
    return torch.bmm(joints_concat, transformation.transpose(1,2))


