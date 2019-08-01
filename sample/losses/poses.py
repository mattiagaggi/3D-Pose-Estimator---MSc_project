from sample.base.base_metric import BaseMetric
from dataset_def.trans_numpy_torch import tensor_to_numpy
import torch
import torch_batch_svd

from numpy.linalg import det
import numpy as np


class MPJ(BaseMetric):


    def __init__(self):
        super().__init__()


    def forward(self, pose_pred, pose_label):
        assert pose_pred.size()[1] == 17
        assert pose_label.size()[1] == 17
        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        squared = torch.pow((pose_pred-pose_label),exponent=2)
        summed = torch.sum(squared, dim=2)
        dist = torch.sqrt(summed)
        #mean_over_joints=torch.mean(dist,dim=1)
        return torch.mean(dist) # mean over batch and joints


class Normalised_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()


    def forward(self, pose_pred, pose_label):
        assert pose_pred.size()[1] == 17
        assert pose_label.size()[1] == 17
        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        dot_pose_pose = torch.mul(pose_pred,pose_pred).sum(dim=2).sum(dim=1)
        dot_pose_gt = torch.mul(pose_pred,pose_label).sum(dim=2).sum(dim=1)
        s_op = dot_pose_gt / dot_pose_pose
        norm = s_op.view(-1,1,1).expand_as(pose_label)
        norm_pred = torch.mul(norm, pose_pred)
        return MPJ.forward(norm_pred, pose_label)



class Aligned_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()


    def forward(self, pose_pred, pose_label):
        assert pose_pred.size()[1] == 17
        assert pose_label.size()[1] == 17
        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        #align by translation
        dot_pose_pose = torch.mul(pose_pred,pose_pred).sum(dim=2).sum(dim=1)
        dot_pose_gt = torch.mul(pose_pred,pose_label).sum(dim=2).sum(dim=1)
        s_op = dot_pose_gt / dot_pose_pose
        norm = s_op.view(-1,1,1).expand_as(pose_label)
        norm_pred = torch.mul(norm, pose_pred)

        #reshape and mul H =

        U,S,VT = torch_batch_svd(H)
        V= VT.transpose(1,2)
        M=torch.bmm(V, U.transpose(1,2) )

        D = det( tensor_to_numpy(M) )
        U= tensor_to_numpy(U)
        V = tensor_to_numpy(V)
        ident = np.eye(3)
        iden_batch = np.tile(ident, (len(D), 1, 1))
        iden_batch[:,2,2] = D
        UT = np.transpose(U,(0,2,1))
        R = np.matmul(V,np.matmul(iden_batch, UT))

        return MPJ.forward(norm_pred, pose_label)


