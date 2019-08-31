

from sample.base.base_model import BaseModel
from sample.models.encoder_decoder import Encoder_Decoder
from sample.models.SMPL_from_latent import SMPL_from_Latent
from utils.conversion_SMPL_h36m_torch import from_smpl_to_h36m_world_torch, project_vertices_onto_mask
from utils.rendering.rasterizer_silhuette import Rasterizer
import torch

class SMPL_enc_dec(BaseModel):
    def __init__(self, batch_size):

        super().__init__()

        self.batch_size = batch_size
        self.encoder_decoder=Encoder_Decoder(batch_size)
        dimension_L_3D = self.encoder_decoder.dimension_L_3D
        dimension_L_app = self.encoder_decoder.dimension_L_app
        self.SMPL_from_latent = SMPL_from_Latent(batch_size, d_in_3d=dimension_L_3D, d_in_app = dimension_L_app)
        self.rasterizer = Rasterizer(batch_size, self.SMPL_from_latent.faces)

    def fix_encoder_decoder(self):
        for par in self.encoder_decoder.parameters():
            par.requires_grad = False

    def forward(self, dic):

        im= dic['image']
        out_enc = self.encoder_decoder.encoder(im)
        out_smpl = self.SMPL_from_latent(out_enc)
        joints_converted_world = from_smpl_to_h36m_world_torch(out_smpl['joints'], dic['root_pos'],
                                                             from_camera=True, R_world_cam=dic['R'])
        vertices_converted_world = from_smpl_to_h36m_world_torch(out_smpl['verts'], dic['root_pos'],
                                                               from_camera=True, R_world_cam=dic['R'])
        #convert to world coord

        dic_out = {}
        dic_out["SMPL_params"] = (out_smpl['pose'], out_smpl['shape'])
        dic_out["SMPL_output"] = (out_smpl['joints'], out_smpl['verts'])
        dic_out['joints_im'] = joints_converted_world
        dic_out['masks'] = {1: {},
                            2: {},
                            3: {},
                            4: {}
                            }
        for ca in range(1,5):

            pix_vertices_ca = project_vertices_onto_mask(vertices_converted_world, dic['masks'][ca])
            dic_out['masks'][ca]['verts'] = pix_vertices_ca
            image = self.rasterizer(pix_vertices_ca)
            dic_out['masks'][ca]['image'] = image
        return dic_out