from utils.trans_numpy_torch import numpy_to_long
import torch

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

    def index_SMPL(self, smpl_joints, batch=True):
        if batch:
            return torch.index_select(smpl_joints, dim=1, index=self.index_smpl)
        return torch.index_select(smpl_joints, dim=0, index=self.index_smpl)

    def index_h36m(self, h36m_joints, batch=True):
        if batch:
            return torch.index_select(h36m_joints, dim=1, index=self.index_h36m)
        return torch.index_select(h36m_joints, dim=0, index=self.index_h36m)

    def match_joints(self, smpl_joints, h36m_joints, batch):
        smpl_joints = self .index_SMPL(smpl_joints, batch)
        h36m_joints = self.index_h36m(h36m_joints, batch)
        return smpl_joints, h36m_joints




