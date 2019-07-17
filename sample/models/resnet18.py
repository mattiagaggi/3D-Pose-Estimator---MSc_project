import torch.utils.model_zoo as model_zoo
from sample.models.resnet import ResNet
from sample.base.base_modules import BasicBlock
from utils.training_utils import transfer_partial_weights

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    #'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    #'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    #'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18_loss(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],  **kwargs)
    if pretrained:
        print("resnet_low_level: Loading image net weights...")
        transfer_partial_weights(model_zoo.load_url(model_urls['resnet18']), model)
        print("resnet_low_level: Done loading image net weights...")
        #model.load_state_dict( model_zoo.load_url(model_urls['resnet18']))
    return model