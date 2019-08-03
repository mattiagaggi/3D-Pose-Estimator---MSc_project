
import torch

from sample.base.base_metric import BaseMetric
from utils.utils_alignment import batch_svd,tiled_identity,determinant



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
        # debug line
        #return torch.mean(dist),pose_pred,pose_label
        return torch.mean(dist) # mean over batch and joints


class Normalised_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()
        self.MPJ=MPJ()


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
        return self.MPJ(norm_pred, pose_label)



class Aligned_MPJ(BaseMetric):


    def __init__(self):
        super().__init__()
        self.MPJ = MPJ()
        self._logger.info("Aligned MPJ does not support Backprop")


    def forward(self, pose_pred, pose_label):
        assert pose_pred.size()[1] == 17
        assert pose_label.size()[1] == 17
        assert pose_pred.size()[2] == 3
        assert pose_label.size()[2] == 3
        #Kabsch algorithm
        pose_pred = pose_pred- torch.sum(pose_pred, dim=1).reshape((-1,1,3))
        pose_label = pose_label - torch.sum(pose_label, dim=1).reshape((-1,1,3))
        H = torch.bmm(pose_pred.transpose(1, 2), pose_label)
        U,S,V = batch_svd(H)
        M=torch.bmm(V, U.transpose(1,2))
        d = determinant(M)
        b_size = d.size()[0]
        D = tiled_identity(b_size)
        D[:,2,2] = d  #determinant correction -1 or 1
        UT = U.transpose(1, 2)
        R = torch.bmm(V, torch.bmm(D, UT))
        rot_pred = torch.bmm(pose_pred, R.transpose(1,2))
        return self.MPJ(rot_pred, pose_label)


