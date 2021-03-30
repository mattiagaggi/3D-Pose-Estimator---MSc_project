
import torch

from sample.base.base_metric import BaseMetric
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from utils.trans_numpy_torch import numpy_to_tensor_float, tensor_to_numpy
import numpy as np




class MPJ(BaseMetric):


    def __init__(self):
        super().__init__()


    def forward(self, pose_pred, pose_label):

        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        squared = torch.pow((pose_pred-pose_label),exponent=2)
        summed = torch.sum(squared, dim=2)
        dist = torch.sqrt(summed)
        # debug line
        #return torch.mean(dist),pose_pred,pose_label
        return torch.mean(dist)# mean over batch and joints


class Normalised_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()
        self.MPJ=MPJ()


    def forward(self, pose_pred, pose_label):

        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        dot_pose_pose = torch.mul(pose_pred,pose_pred).sum(dim=2).sum(dim=1)
        dot_pose_gt = torch.mul(pose_pred,pose_label).sum(dim=2).sum(dim=1)
        s_op = dot_pose_gt / dot_pose_pose
        norm = s_op.view(-1,1,1).expand_as(pose_label)
        norm_pred = torch.mul(norm, pose_pred)
        return self.MPJ(norm_pred, pose_label)



class Aligned_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()
        self.MPJ = MPJ()
        self._logger.info("Aligned MPJ does not support Backprop - we use numpy")


    def forward(self, pose_pred, pose_label):

        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        #stand=torch.mul(pose_pred, pose_pred).sum(dim=2).sum(dim=1)
        pose_pred = tensor_to_numpy(pose_pred)
        pose_label = tensor_to_numpy(pose_label)
        batch_size = pose_label.shape[0]
        #pose1_lst=[]
        #pose2_lst=[]
        R_lst=[]
        for i in range(batch_size):
            #pose1, pose2, disparity = procrustes(pose_label[i], pose_pred[i])
            #pose1_lst.append(numpy_to_tensor_float(pose1))
            #pose2_lst.append(numpy_to_tensor_float(pose2))
            R,_ = orthogonal_procrustes(pose_label[i],pose_pred[i])
            R_lst.append(numpy_to_tensor_float(R))
        #pose1_stacked=torch.stack(pose1_lst, dim=0)*stand.view(-1,1,1)
        #pose2_stacked= torch.stack(pose2_lst, dim=0)*stand.view(-1,1,1)
        R_lst=torch.stack(R_lst,dim=0)
        one=torch.bmm(numpy_to_tensor_float(pose_label),R_lst)

        return self.MPJ(one, numpy_to_tensor_float(pose_pred))


