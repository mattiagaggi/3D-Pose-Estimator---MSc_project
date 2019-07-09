# -*- coding: utf-8 -*-
"""
Train AutoEncoder

@author: Denis Tome'

"""
import datetime
from tqdm import tqdm
from base import BaseTrainer
from model.modules import LRDecay
from utils.draw import Drawer, Style
import utils

__all__ = [
    'AETrainer'
]


class AETrainer(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self, model, loss, metrics,
                 optimizer, data_loader, args):

        super().__init__(model, loss, metrics,
                         optimizer, **vars(args))

        self.batch_size = args.batch_size
        self.data_loader = data_loader
        self.img_log_step = args.img_log_step
        self.val_log_step = args.val_log_step
        self.len_trainset = len(self.data_loader)

        # load model
        self._resume_checkpoint(args.resume)

        # setting lr_decay
        self.lr_decay = None
        if not args.no_lr_decay:
            self.lr_decay = LRDecay(self.optimizer,
                                    lr=args.learning_rate,
                                    decay_rate=args.lr_decay_rate,
                                    decay_steps=args.lr_decay_step)

        # setting drawer
        self.drawer = Drawer(Style.EQ_AXES)

    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['batch_size'] = self.batch_size
        info['creation'] = str(datetime.datetime.now())
        info['size_dataset'] = len(self.data_loader)
        info['start_lr'] = utils.get_optimizer_lr(self.optimizer)
        info['lr_decay'] = True if self.lr_decay else False
        if self.lr_decay:
            info['lr_decay_rate'] = self.lr_decay.decay_rate
            info['lr_decay_step'] = self.lr_decay.decay_step
            info['lr_decay_mode'] = self.lr_decay.mode
        return info

    def _train_epoch(self, epoch):
        """Train model for one epoch

        Arguments:
            epoch {int} -- epoch number

        Returns:
            float -- epoch error
        """

        self.model.train()
        if self.with_cuda:
            self.model.cuda()

        total_loss = 0

        pbar = tqdm(self.data_loader)
        for bid, (hm, p3d_gt) in enumerate(pbar):

            hm = self._get_var(hm)
            p3d_gt = self._get_var(p3d_gt)

            # set right learning rate depending on decay
            if self.lr_decay:
                self.lr_decay.update_lr(self.global_step)

            self.optimizer.zero_grad()
            p3d = self.model(hm).view_as(p3d_gt)
            loss = self.loss(p3d, p3d_gt)
            loss.backward()
            self.optimizer.step()

            if (bid % self.verbosity_iter == 0) & (self.verbosity == 2):
                pbar.set_description(' Epoch {} Loss {:.3f}'.format(
                    epoch, loss.item()
                ))

            if (bid % self.train_log_step) == 0:
                val = loss.item()
                self.model_logger.train.add_scalar('loss/iterations', val,
                                                   self.global_step)

                for metric in self.metrics:
                    y_output = p3d.data.cpu().numpy()
                    y_target = p3d_gt.data.cpu().numpy()
                    metric.log(pred=y_output,
                               gt=y_target,
                               logger=self.model_logger.train,
                               iteration=self.global_step)

            if (bid % self.img_log_step) == 0 and self.img_log_step > -1:
                y_output = p3d.data.cpu().numpy()
                y_target = p3d_gt.data.cpu().numpy()

                img_poses = self.drawer.poses_3d(y_output[0], y_target[0])
                img_poses = img_poses.transpose([2, 0, 1])
                self.model_logger.train.add_image('3d_poses',
                                                  img_poses,
                                                  self.global_step)

            if (bid % self.save_freq) == 0:
                if total_loss:
                    self._save_checkpoint(epoch, self.global_step,
                                          total_loss / bid)

            self.global_step += 1
            total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)

        return avg_loss

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: loss and metrics
        """
        pass
