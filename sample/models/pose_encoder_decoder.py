

from sample.base.base_model import BaseModel
from sample.models.encoder_decoder import Encoder_Decoder
from sample.models.resnet_to_pose import ResNet
from sample.models.pose_from_latent import Pose_from_Latent
from sample.base.base_modules import BasicBlock
from utils.training_utils import transfer_partial_weights
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    #'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    #'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Pose_3D(BaseModel):
    def __init__(self):

        super().__init__()

        #model = ResNet(BasicBlock, [2, 2, 2, 2])
        #self._logger.info("Loading partial weights")
        #transfer_partial_weights(model_zoo.load_url(model_urls['resnet18']),model)
        #self._logger.info("Done loading image net weights...")
        #self.encoder_decoder=model

        self.encoder_decoder=Encoder_Decoder()
        dimension_L_3D=self.encoder_decoder.dimension_L_3D
        #self._logger.info(self.encoder_decoder.dimension_L_3D)

        self.pose_from_latent = Pose_from_Latent(d_in=dimension_L_3D)

    def fix_encoder_decoder(self):
        for par in self.encoder_decoder.parameters():
            par.requires_grad = False

    def forward(self, dic):
        im= dic['im_in']
        dic_out = self.encoder_decoder.encoder(im)
        #dic_out=self.encoder_decoder(im)
        l3D = dic_out['L_3d']
        pose_out = self.pose_from_latent(l3D)
        return pose_out

