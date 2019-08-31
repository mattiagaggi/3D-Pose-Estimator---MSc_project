
import torch.nn
import numpy as np
from sample.base.base_model import BaseModel
from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
from utils.trans_numpy_torch import numpy_to_param


class SMPL_from_Latent(BaseModel):
    def __init__(self,batch_size, d_in_3d, d_in_app):
        super().__init__()
        self._logger.info("Make sure weights encoder decoder are set as not trainable")
        self.batch_size = batch_size
        d_hidden = 2024
        n_hidden = 3
        dropout = 0.3
        SMPL_pose_params = 72
        SMPL_shape_params = 10
        #self.scale= numpy_to_param(np.array([[[1]]]))

        self.SMPL_layer_neutral = SMPL_Layer(center_idx=0, gender='neutral', model_root='data/models_smpl')
        self.faces = self.SMPL_layer_neutral.th_faces
        self.kintree_table = self.SMPL_layer_neutral.kintree_table
        self.dropout = dropout

        module_list = [torch.nn.Linear(d_in_app+d_in_3d, d_hidden),
                        torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]

        for i in range(n_hidden - 1):
            module_list.extend([
                torch.nn.Dropout(inplace=True, p=self.dropout),
                torch.nn.Linear(d_hidden, d_hidden),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(d_hidden, affine=True),
                torch.nn.Dropout(inplace=True, p=self.dropout)
                ])

        self.fully_connected = torch.nn.Sequential(*module_list)
        to_vertices = [

                    torch.nn.Linear(d_hidden, SMPL_shape_params),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(SMPL_shape_params, affine=True)
        ]

        to_pose = [

                    torch.nn.Linear(d_hidden, SMPL_pose_params),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(SMPL_pose_params, affine=True)
        ]

        self.to_shape = torch.nn.Sequential(*to_vertices)
        self.to_pose = torch.nn.Sequential(*to_pose)




    def forward(self, dic_in):

        L_3D = dic_in["L_3d"]
        L_app = dic_in["L_app"]
        L_3D = L_3D.view(self.batch_size, -1)
        L_app = L_app.view(self.batch_size, -1)
        L = torch.cat([L_3D, L_app], dim=1)
        output = self.fully_connected(L)

        pose_params = self.to_pose(output)
        shape_params = self.to_shape(output)

        verts, joints = self.SMPL_layer_neutral(pose_params, th_betas=shape_params)
        #rescale should multiply by 1000
        #verts = torch.mul(verts, self.scale)
        #joints = torch.mul(joints, self.scale)
        dic_out={
            'pose' : pose_params,
            'shape' : shape_params,
            'verts' : verts,
            'joints' : joints
            }
        #make sure it is translated

        return dic_out