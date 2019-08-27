import torch

from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
from utils.smpl_torch.display_utils import Drawer


if __name__ == '__main__':
    cuda = False
    batch_size = 1
    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='data/models_smpl')
    d = Drawer(kintree_table=smpl_layer.kintree_table,smpl_layer.fa)

    # Generate random pose and shape parameters
    pose_params = torch.rand(batch_size, 72) * 0.2
    shape_params = torch.rand(batch_size, 10) * 0.3
    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()
    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)


    # Draw output vertices and joints
    import matplotlib.pyplot as plt

    plt.show()
