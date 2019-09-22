import torch
import torch.nn
from sample.config.data_conf import PARAMS

from torch.autograd import Variable
device=PARAMS.data.device
device_type = PARAMS.data.device_type


def create_zero_float_tensor(shape):
    return torch.zeros(shape, dtype=torch.float32, device=device)

def create_one_float_tensor(shape):
    return create_zero_float_tensor(shape) + 1

def numpy_to_tensor_float(data):
    return torch.from_numpy(data).float().to(device)

def numpy_to_tensor_float_cpu(data):
    return torch.from_numpy(data).float().to(torch.device('cpu'))

def numpy_to_long_cpu(arr):
    return torch.LongTensor(arr).to(torch.device('cpu'))

def numpy_to_tensor(data):
    return torch.from_numpy(data).to(device)

def numpy_to_tensor_cpu(data):
    return torch.from_numpy(data).to(torch.device('cpu'))

def numpy_to_long(arr):
    return torch.LongTensor(arr).to(device)




def numpy_to_param(data):
    data = numpy_to_tensor_float(data)
    return torch.nn.Parameter(data, requires_grad=True)

def encoder_dictionary_to_pytorch(dic):
    for key in dic.keys():
        if key == 'invert_segments':
            dic[key] = numpy_to_long(dic[key])
        else:
            dic[key] = numpy_to_tensor_float(dic[key])
    return dic



def tensor_to_numpy(tensor):
    if device_type == 'gpu': #transfer to cpu
        return tensor.cpu().data.numpy()
    return tensor.data.numpy()


def image_pytorch_to_numpy(image, batch_idx=False):

    if batch_idx:
        return tensor_to_numpy(image).transpose(0,2, 3, 1)
    return tensor_to_numpy(image).transpose(1, 2, 0)

def image_numpy_to_pytorch(image, batch_idx=False):

    if batch_idx:
        return numpy_to_tensor_float(image.transpose(0, 3, 1, 2))
    else:
        return numpy_to_tensor_float(image.transpose(2, 0, 1))






