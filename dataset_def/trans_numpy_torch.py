import torch
import numpy as np
from sample.base.base_logger import FrameworkClass
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS

device=ENCODER_DECODER_PARAMS.encoder_decoder.device

class ImageToTensor(FrameworkClass):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        image = np.transpose(image, [2, 0, 1])
        return torch.from_numpy(image).float().to(device)


class NumpyToTensor(FrameworkClass):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        return torch.from_numpy(data).float().to(device)


def encoder_dictionary_to_pytorch(dic):
    for key in dic.keys():
        if key == 'invert_segments':
            dic[key] = torch.LongTensor(dic[key]).to(device)
        else:
            dic[key] = NumpyToTensor()(dic[key])
    return dic


