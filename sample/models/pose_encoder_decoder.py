

from sample.base.base_model import BaseModel
from sample.models.encoder_decoder import Encoder_Decoder
from sample.models.pose_from_latent import Pose_from_Latent

class Pose_3D(BaseModel):
    def __init__(self):

        super().__init__()


        self.encoder_decoder=Encoder_Decoder()
        dimension_L_3D=self.encoder_decoder.dimension_L_3D
        self.pose_from_latent = Pose_from_Latent(d_in=dimension_L_3D)

    def fix_encoder_decoder(self):
        for par in self.encoder_decoder.parameters():
            par.requires_grad = False

    def forward(self, dic):
        im= dic['im_in']
        dic_out = self.encoder_decoder.encoder(im)
        l3D = dic_out['L_3d']
        pose_out = self.pose_from_latent(l3D)
        return pose_out

