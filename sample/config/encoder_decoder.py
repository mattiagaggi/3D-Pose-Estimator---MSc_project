

from easydict import EasyDict as edict
import torch







ENCODER_DECODER_PARAMS = edict({
    'background':{
        'sampling': 10
    },
    'encoder_decoder' : {
        'batch_size': 10,
        'sampling': 64,
        'im_size' : 128,
        'device_type' : 'cpu'
    }
})

if ENCODER_DECODER_PARAMS['encoder_decoder']['device_type'] == 'cpu':
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cpu')
else:
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cuda:0')
