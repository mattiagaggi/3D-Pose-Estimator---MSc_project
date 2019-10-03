
import torch.nn
import numpy as np
from sample.base.base_model import BaseModel
from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
from utils.trans_numpy_torch import numpy_to_tensor_float


class SMPL_from_Latent(BaseModel):
    def __init__(self, d_in_3d, d_in_app, use_zero_shape=False):
        super().__init__()
        self._logger.info("Make sure weights encoder decoder are set as not trainable")
        d_hidden = 1024
        n_hidden = 4
        dropout = 0.3
        self.d_in_app = d_in_app
        self.d_in_3d = d_in_3d
        self.SMPL_pose_params = 72
        self.SMPL_shape_params = 10
        self.n_regressions = 4


        self.SMPL_layer_neutral = SMPL_Layer(center_idx=0, gender='neutral', model_root='data/models_smpl')
        self.faces = self.SMPL_layer_neutral.th_faces

        self.kintree_table = self.SMPL_layer_neutral.kintree_table
        self.dropout = dropout



        module_list = [torch.nn.Linear(d_in_3d+d_in_app, d_hidden),
                        torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]

        for i in range(n_hidden - 1):
            module_list.extend([
                torch.nn.Linear(d_hidden, d_hidden),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(d_hidden, affine=True),
                ])
        
        self.fully_connected = torch.nn.Sequential(*module_list)
        
        """
        regression_lst = [
            torch.nn.Linear(d_in_3d + d_in_app + self.SMPL_pose_params + self.SMPL_shape_params, d_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d( d_hidden, affine=True),
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.Linear(d_hidden, self.SMPL_pose_params +self.SMPL_shape_params)

        ]
        #hyperbolic tangent
        """
        #self.regression_module = torch.nn.Sequential(*regression_lst)
        #self.initial_pose = numpy_to_param(np.zeros((batch_size, self.SMPL_pose_params)))
        #self.initial_shape = numpy_to_tensor_float(np.zeros((batch_size, self.SMPL_shape_params)))

        shape_list = [
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d( d_hidden, affine=True),
            torch.nn.Linear(d_hidden, self.SMPL_shape_params)

        ]
        self.to_shape = torch.nn.Sequential(*shape_list)


        pose_list = [
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d( d_hidden, affine=True),
            torch.nn.Linear(d_hidden, self.SMPL_pose_params)

        ]

        self.to_pose = torch.nn.Sequential(*pose_list)


    def forward(self, dic_in):
        #change to IEF

        L_3D = dic_in["L_3d"]
        use_zero_shape = dic_in['use_zero_shape']
        L_app = dic_in["L_app"]
        L = torch.cat([L_3D, L_app], dim=1)


        #pose = self.initial_pose
        #shape = self.initial_shape
        #for i in range(self.n_regressions):
        #inreg = torch.cat([L, pose, shape], dim=1)
        #deltas = self.regression_module(inreg)
        #pose_delta = deltas[:, : self.SMPL_pose_params]
        #shape_delta = deltas[:, self.SMPL_pose_params:]
        #pose = pose + pose_delta
        #shape = shape + shape_delta
        output = self.fully_connected(L)

        pose = self.to_pose(output)

        shape = self.to_shape(output)

        if use_zero_shape:
            shape = numpy_to_tensor_float(np.zeros((pose.size()[0], self.SMPL_shape_params)))
        verts, joints = self.SMPL_layer_neutral(pose, th_betas=shape)
        #print("joints")
        #joints.register_hook(lambda grad: print(grad))
        #verts.register_hook(lambda grad: print(grad))

        dic_out={
            'pose' : pose,
            'shape' : shape,
            'verts' : verts,
            'joints' : joints
            }
        #make sure it is translated

        return dic_out