import torch
from data.config import device
from easydict import EasyDict as edict



ENCODER_DECODER_PARAMS = edict({
    'background':{
        'sampling': 64
    },
    'encoder_decoder' : {
        'sampling_train': 5,
        'sampling_test': 20,
        'im_size' : 128,
        'device_type' : device
    }
})

if ENCODER_DECODER_PARAMS['encoder_decoder']['device_type'] == 'cpu':
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cpu')
else:
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cuda')
