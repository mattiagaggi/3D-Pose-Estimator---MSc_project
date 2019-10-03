# -*- coding: utf-8 -*-
"""
Base tester class to be extended

@author: Denis Tome'

"""
import os
import collections
import torch
from torch.autograd import Variable
import numpy as np
import utils.io as io
import torch.nn
from sample.base.base_logger import FrameworkClass
from logger.model_logger import ModelLogger
from logger.train_hist_logger import TrainingLogger
import utils.io as io
from collections import OrderedDict
from utils.io import is_model_parallel



class BaseTester(FrameworkClass):
    """
    Base class for all dataset testers
    """

    def __init__(self, model,output, name, no_cuda):

        super().__init__()

        self.model = model
        self.training_name = name
        self.save_dir = output
        self.output_name = name
        self.save_dir = io.abs_path(output)
        self.with_cuda = not no_cuda
        self.single_gpu = True
        path = os.path.join(self.save_dir, self.training_name, 'log_test')
        self.model_logger = ModelLogger(path, self.training_name)
        path = os.path.join(self.save_dir, self.training_name, 'log_results_test')
        self.train_logger = TrainingLogger(path)

        # check that we can run on GPU
        if not torch.cuda.is_available():
            self.with_cuda = False

        if self.with_cuda and (torch.cuda.device_count() > 1):
            self.model = torch.nn.DataParallel(self.model)

            self.single_gpu = False

        io.ensure_dir(os.path.join(self.save_dir,
                                   self.output_name))



    def test(self):
        """Run test on the test-set"""
        raise NotImplementedError()

    def _resume_checkpoint(self, path):
        """Resume model specified by the path

        Arguments:
            path {str} -- path to directory containing the model
                                 or the model itself
        """

        if path == 'init':
            return

        # load model
        if not os.path.isfile(path):
            path = io.get_checkpoint(path)

        self._logger.info("Loading checkpoint: %s ...", path)
        if self.with_cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        trained_dict = checkpoint['state_dict']
        if is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = collections.OrderedDict((k.replace('module.', ''), val)
                                                       for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = collections.OrderedDict(('module.{}'.format(k), val)
                                                       for k, val in checkpoint['state_dict'].items())

        self.model.load_state_dict(trained_dict)
        self._logger.info("Checkpoint '%s' loaded", path)
