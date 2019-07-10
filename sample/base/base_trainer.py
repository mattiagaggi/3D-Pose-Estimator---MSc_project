# -*- coding: utf-8 -*-
"""
Base trainer class

@author: Denis Tome'

"""
import os
import math
import shutil
from torch.autograd import Variable
import torch
import torch.nn
import numpy as np
from logger.model_logger import ModelLogger
import utils.io as io
from utils.io import is_model_parallel
from sample.base.base_logger import FrameworkClass
from collections import OrderedDict


class BaseTrainer(FrameworkClass):
    """
    Base class for all trainers
    """

    def __init__(self, model, loss, metrics, optimizer, epochs,
                 name, output, save_freq, no_cuda, verbosity,
                 train_log_step, verbosity_iter,
                  eval_epoch,reset=False, **kwargs):
        """Init class"""

        super().__init__()


        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.epochs = epochs
        self.training_name = name
        self.save_dir = output
        self.save_freq = save_freq
        self.with_cuda = not no_cuda
        self.verbosity = verbosity
        self.verbosity_iter = verbosity_iter
        self.train_log_step = train_log_step
        self.eval_epoch = eval_epoch
        self.min_loss = math.inf
        self.start_epoch = 0
        self.start_iteration = 0
        path=os.path.join(self.save_dir,self.training_name,'log')
        self.model_logger = ModelLogger(path,self.training_name)
        self.training_info = None

        self.reset = reset
        self.single_gpu = True
        self.global_step = 0

        # check that we can run on GPU
        if not torch.cuda.is_available():
            self.with_cuda = False

        if self.with_cuda and (torch.cuda.device_count() > 1):
            if self.verbosity:
                self._logger.info("Let's use %d GPUs!",
                                  torch.cuda.device_count())

            #parallelise
            #self.single_gpu = False
            #self.model = torch.nn.DataParallel(self.model)

        io.ensure_dir(os.path.join(self.save_dir,
                                   self.training_name))

    def _get_var(self, var):
        """Generate variable based on CUDA availability

        Arguments:
            var {undefined} -- variable to be converted

        Returns:
            tensor -- pytorch tensor
        """

        var = torch.FloatTensor(var)
        var = Variable(var)

        if self.with_cuda:
            var = var.cuda()

        return var

    def train(self):
        """Train model"""

        self._dump_summary_info()
        if self.with_cuda:
            self.model.cuda()
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.verbosity:
                self._logger.info('Training epoch %d of %d',
                                  epoch, self.epochs)
            epoch_loss = self._train_epoch(epoch)
            if self.eval_epoch:
                self._logger.info('Evaluating epoch %d of %d',
                                  epoch, self.epochs)
                epoch_val_loss, epoch_val_metrics = self._valid_epoch()
                self.model_logger.val.add_scalar('loss/iterations', epoch_val_loss,
                                                 self.global_step)
                for i, metric in enumerate(self.metrics):
                    metric.log_res(logger=self.model_logger.val,
                                   iter=self.global_step,
                                   error=epoch_val_metrics[i])
            self._save_checkpoint(epoch, self.global_step, epoch_loss)

    def _dump_summary_info(self):
        """Save training summary"""

        info_file_path = os.path.join(self.save_dir,
                                      self.training_name,
                                      'INFO.json')
        if not io.file_exists(info_file_path):
            info = self._summary_info()
            io.write_json(info_file_path,
                          info)
        else:
            info = io.read_from_json(info_file_path)
        self.training_info = info

    def _update_summary(self, global_step, loss, metrics):
        """Update training summary details

        Arguments:
            global_step {int} -- global step in the training process
            loss {float} -- loss value
            metrics {Metric} -- metrics used for evaluating the model
        """

        self.training_info['global_step'] = global_step
        self.training_info['val_loss'] = loss
        for idx, metric in enumerate(self.metrics):
            m = metrics[idx]
            if isinstance(m, np.ndarray):
                m = m.tolist()
            self.training_info['val_{}'.format(metric._desc)] = m

        info_file_path = os.path.join(self.save_dir,
                                      self.training_name,
                                      'INFO.json')
        io.write_json(info_file_path,
                      self.training_info)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self):
        raise NotImplementedError

    def _summary_info(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, iteration, loss):
        """Save model

        Arguments:
            epoch {int} -- epoch number
            iteration {int} -- iteration number
            loss {float} -- loss value
        """

        if loss < self.min_loss:
            self.min_loss = loss
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'iter': iteration,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'min_loss': self.min_loss,
        }
        filename = os.path.join(
            self.save_dir, self.training_name,
            'ckpt_eph{:02d}_iter{:06d}_loss_{:.5f}.pth.tar'.format(
                epoch, iteration, loss))
        self._logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)
        if loss == self.min_loss:
            shutil.copyfile(filename,
                            os.path.join(self.save_dir, self.training_name,
                                         'model_best.pth.tar'))

    def _resume_checkpoint(self, resume_path):
        """Resume model to be fine-tuned

        Arguments:
            resume_path {str} -- path to the directory or model to be resumed
        """

        if resume_path == 'init':
            return

        if not os.path.isfile(resume_path):
            resume_path = io.get_checkpoint(resume_path)

        self._logger.info("Loading checkpoint: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']

        if is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = OrderedDict((k.replace('module.', ''), val)
                                                       for k, val in checkpoint['state_dict'].items())
            #############
            self.model.cuda()
            ###########
        else:
            if not self.single_gpu:
                trained_dict = OrderedDict(('module.{}'.format(k), val)
                                                       for k, val in checkpoint['state_dict'].items())


        self.model.load_state_dict(trained_dict)

        #############
        self.model.cuda()
        ###########

        if not self.reset:
            self.start_iteration = checkpoint['iter'] + 1
            self.start_epoch = checkpoint['epoch']
            try:
                self.global_step = checkpoint['global_step'] + 1
            except KeyError:
                self.global_step = self.start_iteration
            self.min_loss = checkpoint['min_loss']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self._logger.info("Checkpoint '%s' (epoch %d) loaded",
                          resume_path, self.start_epoch)
