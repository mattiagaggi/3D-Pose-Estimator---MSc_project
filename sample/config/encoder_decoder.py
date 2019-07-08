

from easydict import EasyDict as edict
import torch







ENCODER_DECODER_PARAMS = edict({
    'background':{
        'sampling': 10
    },
    'encoder_decoder':{
        'sampling':64,
        'im_size' : 128,
        'device' : torch.device('cpu') #torch.device('cuda')
    }
})