import torch

from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
from utils.smpl_torch.display_utils import Drawer
from utils.trans_numpy_torch import numpy_to_tensor_float

if __name__ == '__main__':
    cuda = False
    batch_size = 1
    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='data/models_smpl')
    d = Drawer(kintree_table=smpl_layer.kintree_table)

    # Generate random pose and shape parameters
    import numpy as np
    pose_params =numpy_to_tensor_float(np.array([ 1.3438e-01,  1.0050e-03,  1.2732e-01,  8.8333e-02, -1.3887e+00,
        -1.5473e-01,  5.2549e-01, -5.0818e-01,  3.5654e-01,  4.1361e-01,
        -3.5565e-03, -2.1754e-01, -1.2578e-01, -2.0454e-01,  3.0172e-01,
        -1.0112e+00,  4.8895e-01,  3.9189e-01,  3.7336e-01,  5.0514e-01,
        -4.0682e-01,  4.8758e-01, -2.8710e-02, -2.0994e-02,  2.2309e-01,
         1.5088e-01, -1.1272e+00, -1.0828e+00,  8.7898e-02, -2.2011e-02,
         2.2926e-02, -2.0094e-01,  1.3080e-01, -1.3582e-01, -7.7722e-02,
        -3.1845e-01,  1.8695e-01, -5.3144e-01,  2.7748e-01,  6.7985e-02,
        -4.3104e-01, -1.5006e-01,  3.8521e-02, -5.5779e-01,  1.4236e-01,
        -1.8640e-01, -8.4205e-02,  6.0946e-01,  1.5840e-01,  1.3520e+00,
        -1.7617e-01, -1.2128e-01, -4.4586e-01, -5.8661e-01,  4.0632e-01,
         8.6259e-01,  3.4038e-01,  1.8232e-01, -3.3726e-01, -5.7435e-01,
         6.5781e-02, -2.7820e-01, -2.8747e-01, -1.7012e-01, -2.2815e-01,
        -2.5051e-01,  5.8695e-01, -9.4217e-02,  6.6951e-02, -1.7207e-01,
         4.9331e-02,  9.0434e-02])/2) #(torch.rand(batch_size, 72)-1 ) * 0.1
    pose_params = torch.stack([pose_params] * batch_size, dim=0)*0
    shape_params = torch.rand(batch_size, 10) *0
    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()
    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)


    # Draw output vertices and joints
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig = d.display_model(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=None,#smpl_layer.th_faces,
        with_joints=True,
        batch_idx=0,
        plot=True,
        fig=fig,
        savepath=None)
    plt.show()
