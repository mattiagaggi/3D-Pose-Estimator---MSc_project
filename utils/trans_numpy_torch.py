import torch
from sample.config.encoder_decoder import PARAMS

device=PARAMS.data.device



def create_zero_float_tensor(shape):
    return torch.zeros(shape, dtype=torch.float32, device=device)

def create_one_float_tensor(shape):
    return create_zero_float_tensor(shape) + 1

def numpy_to_tensor(data):
    return torch.from_numpy(data).float().to(device)


def numpy_to_long(data):
    return torch.LongTensor(data).to(device)

def encoder_dictionary_to_pytorch(dic):
    for key in dic.keys():
        if key == 'invert_segments':
            dic[key] = numpy_to_long(dic[key])
        else:
            dic[key] = numpy_to_tensor(dic[key])
    return dic



def tensor_to_numpy(tensor,from_gpu=True):
    if from_gpu:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()


def image_pytorch_to_numpy(image, batch_idx=False):

    if batch_idx:
        return tensor_to_numpy(image).transpose(0,2, 3, 1)
    return tensor_to_numpy(image).transpose(1, 2, 0)

def image_numpy_to_pytorch(image, batch_idx=False):

    if batch_idx:
        return numpy_to_tensor(image.transpose(0,3, 1, 2))
    else:
        return numpy_to_tensor(image.transpose(2, 0, 1))






