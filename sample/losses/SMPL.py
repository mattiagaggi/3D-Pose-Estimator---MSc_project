import torch
from sample.base.base_metric import BaseMetric
import torch.nn
import numpy as np
from utils.conversion_SMPL_h36m_torch import Convert_joints
from sample.losses.poses import MPJ
from sample.losses.images import Cross_Entropy_loss
from utils.trans_numpy_torch import numpy_to_tensor_float
from torch.nn import BCELoss






class Masks_Loss(BaseMetric):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cross_entropy = Cross_Entropy_loss(self.batch_size)

    def forward(self, pred, gt):

        total_loss = self.cross_entropy(pred, gt)
        return torch.mean(total_loss)




class Pose_Loss_SMPL(BaseMetric):
    def __init__(self, criterium = None):
        super().__init__()
        self.conversion = Convert_joints()
        if criterium is None:
            self.sub_metric = MPJ()
        else:
            self.sub_metric = criterium
    def forward(self, smpl_pose_pred, h36m_pose_ground ):

        smpl_joints, h36m_joints = self.conversion.match_joints(smpl_pose_pred, h36m_pose_ground, batch=True)
        #exclude root
        return self.sub_metric( h36m_joints[:,1:,:], smpl_joints[:,1:,:])



class Loss_Pose_Zero(BaseMetric):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        pose_params, shape_params = params
        squared = torch.pow((pose_params), exponent=2)
        squared2 = torch.pow((shape_params), exponent=2)

        return torch.mean(torch.sum(squared, dim=1)+torch.sum(squared2, dim=1))






class SMPL_Loss(BaseMetric):

    #cross entropy loss!!!!

    def __init__(self, batch_size):

        self.init__ = super().__init__()
        self.batch_size = batch_size
        #self.masks_criterium = Masks_Loss(batch_size)
        self.masks_criterium = BCELoss()
        self.pose_criterium = Pose_Loss_SMPL()
        self.SMPL_init = Loss_Pose_Zero()
        self.beta=1
        self.alpha=10
        self.optimise_vertices = False


    def forward(self, dic_in, dic_out, global_iter):
        loss_pose = self.pose_criterium(dic_out['joints_im'], dic_in['joints_im']) #symmetric
        if "discr_output" in dic_out.keys():
            GAN_OUTPUT= dic_out["discr_output"]
            ones_label = numpy_to_tensor_float(np.ones((GAN_OUTPUT.size()[0], 1)))
            gan_loss = self.masks_criterium(GAN_OUTPUT, ones_label)
            loss_pose_GAN = loss_pose + self.alpha * gan_loss
        else:
            loss_pose_GAN= loss_pose
        total_loss = loss_pose_GAN
        if self.optimise_vertices:
            loss_mask = self.masks_criterium(dic_out['mask_image'].view(-1,1), dic_in['mask_image'].view(-1,1))
            total_loss = loss_pose_GAN + self.beta* loss_mask
            return total_loss, loss_pose, loss_mask  # , loss_mask
        return total_loss, loss_pose


        # im_loss 0.03 #pose loss 80
