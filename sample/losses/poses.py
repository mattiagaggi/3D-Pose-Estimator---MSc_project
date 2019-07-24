import torch
import torch.nn

from sample.models.resnet18 import resnet18_loss


class MPJ(torch.nn.Module):


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
        return torch.mean(dist) # mean over batch


