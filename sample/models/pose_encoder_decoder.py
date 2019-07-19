
import torch.nn as nn
import torch


from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.base.base_model import BaseModel
from sample.models.encoder_decoder import Encoder_Decoder
from sample.models.MLP_from_latent import MLP_from_Latent

class Pose_3D(BaseModel):
    def __init__(self,
                 args):

        super().__init__()

        self.batch_size = args.batch_size
        self.encoder_decoder=Encoder_Decoder(args)
        dimension_L_3D=self.encoder_decoder.dimension_L_3D
        self.pose_from_latent = MLP_from_Latent(d_in=dimension_L_3D)

    def fix_encoder_decoder(self):
        for par in self.encoder_decoder.parameters():
            par.requires_grad = False

    def forward(self, x):
        dic_out = self.encoder_decoder.encoder(x)
        l3D = dic_out['L_3d']
        pose_out = self.pose_from_latent(l3D)
        return pose_out

