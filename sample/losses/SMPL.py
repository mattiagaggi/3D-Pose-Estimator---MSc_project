import torch
from sample.base.base_metric import BaseMetric
import torch.nn
import numpy as np
from utils.conversion_SMPL_h36m_torch import Convert_joints
from sample.losses.poses import MPJ
from sample.losses.images import Cross_Entropy_loss
from utils.trans_numpy_torch import numpy_to_tensor_float






class Masks_Loss(BaseMetric):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cross_entropy = Cross_Entropy_loss(self.batch_size)

    def forward(self, mask_dic_in, mask_dic_out):
        total_loss = numpy_to_tensor_float(np.zeros((1)))
        for ca in range(1,5):
            image_in = mask_dic_in[ca]['image']
            image_out = mask_dic_out[ca]['image']
            total_loss = total_loss + self.cross_entropy(image_out, image_in)
        return torch.mean(total_loss)


class Pose_Loss(BaseMetric):
    def __init__(self):
        super().__init__()
        self.conversion = Convert_joints()
        self.pose_loss = MPJ()

    def forward(self, h36m_pose_ground, smpl_pose_pred ):

        smpl_joints, h36m_joints = self.conversion.match_joints(smpl_pose_pred, h36m_pose_ground, batch=True)
        #exclude root
        return self.pose_loss( h36m_joints[:,1:,:], smpl_joints[:,1:,:])



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
        self.masks_criterium = Masks_Loss(batch_size)
        self.pose_criterium = Pose_Loss()
        self.SMPL_init = Loss_Pose_Zero()


        self.w_m = 10*(-4) # roughly 1
        self._logger.error("for 50 iter loss is 0")
        self.optimise_vertices = False


    def forward(self, dic_in, dic_out, global_iter):



        loss_pose = self.pose_criterium(dic_in['joints_im'], dic_out['joints_im'])
        if self.optimise_vertices:
            loss_mask = self.masks_criterium(dic_in['masks'], dic_out['masks'])
            total_loss = self.w_m*loss_mask + loss_pose
            return total_loss, loss_pose, loss_mask  # , loss_mask
        total_loss = loss_pose
        return total_loss, loss_pose


        # im_loss 0.03 #pose loss 80
