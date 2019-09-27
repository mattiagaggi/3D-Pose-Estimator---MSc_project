import datetime
import numpy as np
from utils.utils_H36M.visualise import Drawer
from utils import io
from utils.smpl_torch.display_utils import Drawer as DrawerSMPL
from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
import os
import numpy.random as random
from collections import OrderedDict
import matplotlib.pyplot as plt
from sample.config.data_conf import PARAMS
from utils.trans_numpy_torch import numpy_to_long




if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']


class Trainer_GAN(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self,
                 model,
                 loss_generator,
                 loss_discriminator,
                 metrics,
                 optimizer_generator,
                 optimizer_discriminator,
                 data_train,
                 args,
                 data_test=None,
                 no_cuda = no_cuda,
                 eval_epoch = True
                 ):

        super().__init__(model, optimizer_discriminator, no_cuda,eval_epoch,
                         args.epochs, args.name, args.output, args.save_freq,
                         args.verbosity, args.train_log_step,
                         args.verbosity_iter)

        self.batch_size = args.batch_size
        self.model.fix_encoder_decoder()

        self.loss_generator = loss_generator
        self.loss_discriminator = loss_discriminator
        self.optimizer_generator = optimizer_generator
        self.n_train_gen=5
        self.generator_start_dim = self.model.generator.start_dim

        self.metrics = metrics
        self.data_train = data_train
        self.data_test = data_test

        #test while training
        self.test_log_step = None
        self.img_log_step = args.img_log_step
        if data_test is not None:
            self.test_log_step = args.test_log_step

        self.log_images_start_training = [10, 100, 500,1000]
        self.parameters_show = self.train_log_step * 300
        self.length_test_set = len(self.data_test)
        self.len_trainset = len(self.data_train)
        self.drawer = Drawer()
        self.drawerSMPL = DrawerSMPL(self.model.SMPL_from_latent.kintree_table)
        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='neutral',
            model_root='data/models_smpl')




    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['creation'] = str(datetime.datetime.now())
        info['size_batches'] = len(self.data_train)
        info['batch_size'] = self.batch_size
        string=""
        info['details'] = string
        info['optimiser_discr']=str(self.optimizer)
        info['optimiser_gen'] = str(self.optimizer_generator)

        info['loss_discr']=str(self.loss_discriminator.__class__.__name__)
        info['loss_gen']  = str(self.loss_generator.__class__.__name__)
        return info


    def log_smpl(self, string, i, idx, generator_out):


        smpl=generator_out.cpu().detach()
        pose = smpl[:,72]
        shape = smpl[72:]
        verts, joints = self.smpl_layer(pose, th_betas=shape)
        fig = plt.figure()
        fig = self.drawerSMPL.display_model(
            {'verts': verts.cpu().detach(),
             'joints': joints.cpu().detach()},
            model_faces=self.model.SMPL_from_latent.faces,
            with_joints=True,
            batch_idx=idx,
            plot=True,
            fig=fig,
            savepath=None)
        self.model_logger.train.add_figure(str(string) + str(i) + "SMPL_plot", fig, self.global_step)






    def train_step(self, bid, smpl_real, pbar, epoch, optimise_discriminator):

        self.optimizer.zero_grad()
        self.optimizer_generator.zero_grad()
        input_generator = torch.randn(self.batch_size, self.generator_start_dim)
        zeros_labels = numpy_to_long(np.zeros((self.batch_size,1)))
        ones_label = numpy_to_long(np.ones((smpl_real.size()[0],1)))
        labels= torch.cat([ones_label,zeros_labels])
        if not no_cuda:
            input_generator = input_generator.cuda()
            labels = labels.cuda()
        generator_zero = zeros_labels + 1
        discriminator_0 = self.model.generator(input_generator)
        discriminator_1 = self.model.discriminator(smpl_real)
        preds = torch.cat([discriminator_1, discriminator_0], dim=0)
        loss_discriminator = self.loss_discriminator( preds, labels)
        loss_generator = self.loss_generator(zeros_labels,generator_zero)
        if optimise_discriminator:
            for param in self.model.generator.parameters():
                param.requires_grad = False
            loss_discriminator.backward()
            self.optimizer.step()
            for param in self.model.generator.parameters():
                param.requires_grad = True
        else:
            for param in self.model.discriminator.parameters():
                param.requires_grad = False
            loss_generator.backward()
            self.optimizer_generator.step()
            for param in self.model.discriminator.parameters():
                param.requires_grad = True

        if (bid % self.verbosity_iter == 0) and (self.verbosity == 2):
            pbar.set_description(' Epoch {} Loss Discr {:.3f} Gen {:.3f}'.format(
                epoch, loss_discriminator.item(), loss_generator.item()
            ))
        if bid % self.train_log_step == 0:
            val = loss_discriminator.item()
            val_gen = loss_generator.item()
            self.model_logger.train.add_scalar('discriminator loss/iterations', val,
                                               self.global_step)
            self.model_logger.train.add_scalar('generator loss/iterations', val_gen,
                                               self.global_step)
            self.train_logger.record_scalar('train_loss_discriminator', val,
                                               self.global_step)
            self.train_logger.record_scalar('train_loss_generator', val_gen,
                                               self.global_step)
        if (bid % self.img_log_step == 0) or (self.global_step in self.log_images_start_training):
            for i in range(5):
                idx = np.random.randint(self.batch_size)
                self.log_smpl("test", i,idx, discriminator_0 )
                ###save training images
        return loss_discriminator.item(),loss_generator.item(), pbar


    def test_step_on_random(self,bid):
        pass


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
        pbar = tqdm(self.data_train)
        for bid, smpl_real in enumerate(pbar):
            if bid % self.n_train_gen==0:
                loss_discr, loss_generator, pbar = self.train_step(bid, smpl_real, pbar, epoch,optimise_discriminator=True)
            else:
                loss_discr, loss_generator, pbar = self.train_step(bid, smpl_real, pbar, epoch,
                                                                   optimise_discriminator=False)
            if self.test_log_step is not None and (bid % self.test_log_step == 0):
                self.test_step_on_random(bid)
            if bid % self.save_freq == 0:
                if total_loss:
                    self._save_checkpoint(epoch, total_loss / bid)
                    self._update_summary(self.global_step,total_loss/bid)
            self.global_step += 1
            total_loss += loss_discr
        avg_loss = total_loss / len(self.data_train)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)
        self.train_logger.record_scalar('loss/epochs', avg_loss, epoch)
        self.train_logger.save_logger()

        return avg_loss



