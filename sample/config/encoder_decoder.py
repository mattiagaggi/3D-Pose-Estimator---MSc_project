

from easydict import EasyDict as edict
import torch







ENCODER_DECODER_PARAMS = edict({
    'background':{
        'sampling': 64
    },
    'encoder_decoder' : {
        'sampling_train': 2,
        'sampling_test': 2,
        'im_size' : 128,
        'device_type' : 'gpu'
    }
})

if ENCODER_DECODER_PARAMS['encoder_decoder']['device_type'] == 'cpu':
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cpu')
else:
    ENCODER_DECODER_PARAMS['encoder_decoder']['device'] = torch.device('cuda')
