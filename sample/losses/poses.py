from sample.base.base_metric import BaseMetric
import torch

import torch_batch_svd

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

