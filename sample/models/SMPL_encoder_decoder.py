

from sample.base.base_model import BaseModel
from sample.models.encoder_decoder import Encoder_Decoder
from sample.models.SMPL_from_latent import SMPL_from_Latent
from utils.conversion_SMPL_h36m_torch import from_smpl_to_h36m_world_torch, project_vertices_onto_mask
from utils.rendering.rasterizer_silhuette import Rasterizer
from matplotlib import pyplot as plt
import torch
from sample.models.GAN import GAN_SMPL

class SMPL_enc_dec(BaseModel):
    def __init__(self):

        super().__init__()


        self.encoder_decoder=Encoder_Decoder()
        dimension_L_3D = self.encoder_decoder.dimension_L_3D
        dimension_L_app = self.encoder_decoder.dimension_L_app
        self.SMPL_from_latent = SMPL_from_Latent( d_in_3d=dimension_L_3D, d_in_app = dimension_L_app)
        self.rasterizer = Rasterizer( self.SMPL_from_latent.faces)
        self.optimise_vertices = False
        self.use_zero_shape = False
        #self.GAN = None
        self.GAN= GAN_SMPL()

    def fix_encoder_decoder(self):
        for par in self.encoder_decoder.parameters():
            par.requires_grad = False

    def forward(self, dic):

        im= dic['image']
        out_enc = self.encoder_decoder.encoder(im)
        out_enc['use_zero_shape'] = self.use_zero_shape
        out_smpl = self.SMPL_from_latent(out_enc)
        joints_converted_world = from_smpl_to_h36m_world_torch(out_smpl['joints'], dic['root_pos'],
                                                             from_camera=True, R_world_cam=dic['R'])


        dic_out = {}
        dic_out["SMPL_params"] = (out_smpl['pose'], out_smpl['shape'])
        if self.GAN is not None:
            discr_input = torch.cat([out_smpl['pose'],out_smpl['shape']], dim=1)
            discr_output = self.GAN.discriminator(discr_input)
            dic_out["discr_output"] = discr_output

        dic_out["SMPL_output"] = (out_smpl['joints'], out_smpl['verts'])
        dic_out['joints_im'] = joints_converted_world

        if self.optimise_vertices:

            vertices_converted_world = from_smpl_to_h36m_world_torch(out_smpl['verts'], dic['root_pos'],
                                                                     from_camera=True, R_world_cam=dic['R'])
            vertices_converted_world = torch.index_select(vertices_converted_world, index=dic['mask_idx_all'], dim=0)
            pix_vertices_ca = project_vertices_onto_mask(vertices_converted_world, dic)
            dic_out['mask_verts'] = pix_vertices_ca
            image = self.rasterizer(pix_vertices_ca)
            dic_out['mask_image'] = image
        return dic_out