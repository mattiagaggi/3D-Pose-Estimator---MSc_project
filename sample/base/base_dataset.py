# -*- coding: utf-8 -*-
"""
Base dataset class to be extended in dataset_def dir
depending on the different datasets.

@author: Denis Tome'

"""
import enum
from torch.utils.data import Dataset
from logger.console_logger import ConsoleLogger


__all__ = [
    'BaseDataset',
    'SubSet'
]


class SubSet(enum.Enum):
    """Type of subsets"""

    train = 0
    test = 1
    val = 2


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, subset=SubSet.train):
        super().__init__()

        logger_name = '{}'.format(self.__class__.__name__)

        self.subset = subset
        self._logger = ConsoleLogger(logger_name)



    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

