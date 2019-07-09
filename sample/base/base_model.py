
import torch.nn as nn
import numpy as np
from logger.console_logger import ConsoleLogger


class BaseModel(nn.Module):
    """
    Base class for all model
    """

    def __init__(self):
        super().__init__()

        self._logger = ConsoleLogger(self.__class__.__name__)

        self.model = None
        self.name = self.__class__.__name__

    def forward(self, x):
        raise NotImplementedError

    def summary(self):
        """Summary of the model"""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        enc_grad = np.sum(np.array([np.sum(x.grad.data.numpy()) for x in self.parameters()]))
        enc_mean = np.mean(np.array([np.sum(x.grad.data.numpy()) for x in self.parameters()]))
        self._logger.info('Trainable parameters: %d', int(params))
        self._logger.info('Sum and layers mean of gradients: %f, %f', enc_grad,enc_mean)


