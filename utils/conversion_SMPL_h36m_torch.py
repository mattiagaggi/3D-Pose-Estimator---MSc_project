from utils.trans_numpy_torch import numpy_to_long
from utils.utils_H36M.transformations_torch import rotate_x_torch
from utils.utils_H36M.transformations_torch import world_to_camera_batch, camera_to_pixels_batch, transform_2d_joints_batch
import torch
import numpy as np
from sample.config.smpl_config import PARAMS as SMPL_PARAMS

n_vertices = SMPL_PARAMS.fixed.n_vertices




def from_smpl_to_h36m_world_torch(points_smpl, root_position, from_camera=False, R_world_cam=None):
    batch_size = points_smpl.size()[0]
    #rotate so it matches h36m convertion
    if not from_camera:
        angle = -90. / 180 * np.pi
    else:
        assert R_world_cam is not None
        angle = - np.pi
    R = rotate_x_torch(angle, batch_size)
    points_smpl = torch.bmm(points_smpl, R.transpose(1, 2))
    #rescale
    points_smpl = points_smpl * 1000
    if from_camera:
        #change to world coords
        points_smpl = torch.bmm(points_smpl, R_world_cam)
    points_smpl = points_smpl + root_position
    return points_smpl

def from_h36m_world_to_smpl_torch(points_h36m, root_position):
    batch_size = points_h36m.size()[0]
    points_h36m = points_h36m - root_position
    points_h36m = points_h36m / 1000
    angle = -90. / 180 * np.pi
    R = rotate_x_torch(angle, batch_size)
    points_h36m = torch.bmm(points_h36m, R.transpose(1, 2))
    return points_h36m


def project_vertices_onto_mask(smpl_converted, dic):

    verts_cam = world_to_camera_batch( smpl_converted , n_vertices, dic['mask_R'], dic['mask_T'])
    verts_pix = camera_to_pixels_batch(verts_cam, n_vertices, dic['mask_f'], dic['mask_c'], return_z=True)
    verts_fin = transform_2d_joints_batch(verts_pix, dic['mask_trans_crop'])
    return verts_fin




class Convert_joints():

    def __init__(self):
        # root index 0 h36m index 0 smpl
        # right leg index 2 h36m index 5 smpl
        # right foot index 3 h36m index 8 smpl
        # left leg index 5 h36m index 4 smpl
        # left foot index 6 h36m index 7 smpl
        # spine index 7 h36m index 6 smpl
        # neck index 8 h36m index 12 smpl
        # head index 9 h36m index 15 smpl  # lower weight
        # left shoulder index 11 h36m index 16 smpl
        # left forearm index 12 h36m index 18 smpl
        # left wrist index 13 h36m index 20 smpl
        # right shoulder index 14 h36m index 17 smpl
        # right forearm index 15 h36m index 19 smpl
        # right wrist index 16 h36m index 21 smpl
        self.index_smpl = [0, 5, 8, 4, 7, 6, 12, 15, 16, 18, 20, 17, 19, 21]
        self.index_smpl = numpy_to_long(self.index_smpl)
        self.index_h36m = [0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
        self.index_h36m = numpy_to_long(self.index_h36m)

    def indexing_SMPL(self, smpl_joints, batch=True):
        if batch:
            return torch.index_select(smpl_joints, dim=1, index=self.index_smpl)
        return torch.index_select(smpl_joints, dim=0, index=self.index_smpl)

    def indexing_h36m(self, h36m_joints, batch=True):
        if batch:
            return torch.index_select(h36m_joints, dim=1, index=self.index_h36m)
        return torch.index_select(h36m_joints, dim=0, index=self.index_h36m)

    def match_joints(self, smpl_joints, h36m_joints, batch):
        smpl_joints_new = self.indexing_SMPL(smpl_joints, batch)
        h36m_joints_new = self.indexing_h36m(h36m_joints, batch)
        return smpl_joints_new, h36m_joints_new





