
import torch.nn as nn
import numpy as np
import torch
from logger.console_logger import ConsoleLogger
from matplotlib import pyplot as plt


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

    def return_n_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        return params

    def plot_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is not None:
                    ave_grads.append(p.grad.abs().mean())
                else:
                    layers[-1]+=" Non"
                    ave_grads.append(0)
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)

    def clip_gradients(self):

        torch.nn.utils.clip_grad_value_(self.parameters(), 0.25)

    def summary(self):
        """Summary of the model"""

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = np.sum([np.prod(p.size()) for p in model_parameters])
        self._logger.info('Trainable parameters: %d', int(params))

