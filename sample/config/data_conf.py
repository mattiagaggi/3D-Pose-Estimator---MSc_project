import torch
from data.config import device
from easydict import EasyDict as edict



PARAMS = edict({
    'background':{
        'sampling': 64
    },
    'data' : {
        'sampling_train': 5,
        'sampling_test': 20,
        'im_size' : 128,
        'device_type' : device
    }
})

if PARAMS['data']['device_type'] == 'cpu':
    PARAMS['data']['device'] = torch.device('cpu')
else:
    PARAMS['data']['device'] = torch.device('cuda')
