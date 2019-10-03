import torch.nn
import numpy as np
from sample.base.base_model import BaseModel
import os
import utils.io as io
from collections import OrderedDict

class Discriminator(BaseModel):
    def __init__(self):
        super().__init__()
        hidden=2
        d_hidden =  1024
        self.SMPL_pose=72
        self.SMPL_shape = 10
        module_list = [torch.nn.Linear(self.SMPL_pose+self.SMPL_shape, d_hidden),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]
        for i in range(hidden-1):
            module_list+=[torch.nn.Linear(d_hidden, d_hidden),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]
        module_list += [torch.nn.Linear(d_hidden, 2)]
        self.to_label = torch.nn.Sequential(*module_list)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        out = self.to_label(x)
        out = self.softmax(out)
        out=out[:,0].view(-1,1)

        return out


class Generator(BaseModel):
    def __init__(self):
        super().__init__()
        hidden=1
        d_hidden =  256
        self.start_dim =20
        self.SMPL_pose=72
        self.SMPL_shape = 10
        module_list = [torch.nn.Linear(self.start_dim, d_hidden),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]
        for i in range(hidden-1):
            module_list+=[torch.nn.Linear(d_hidden, d_hidden),
                       torch.nn.ReLU(),
                       torch.nn.BatchNorm1d(d_hidden, affine=True)]
        module_list += [torch.nn.Linear(d_hidden, self.SMPL_pose+self.SMPL_shape)]
        self.to_SMPL = torch.nn.Sequential(*module_list)

    def forward(self, x):

        out = self.to_SMPL(x)
        return out


class GAN_SMPL(BaseModel):
    def __init__(self):
        super().__init__()
        self.generator=Generator()
        self.discriminator = Discriminator()

    def fix_gan(self):
        for par in self.discriminator.parameters():
            par.requires_grad = False
        for par in self.generator.parameters():
            par.requires_grad = False


