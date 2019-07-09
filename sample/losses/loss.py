# -*- coding: utf-8 -*-
"""
Custom loss

@author: Denis Tome'

"""
import torch


def MSE_loss(predicted, target):
    """Custom loss used when both prediction
    and target comes from the model

    Arguments:
        predicted {tensor} -- pytorch tensor
        target {tensor} -- pytorch tensor

    Returns:
        tensor -- loss
    """

    diff = torch.pow(predicted - target, 2)
    loss = torch.sum(diff, dim=2)
    return torch.mean(loss)
