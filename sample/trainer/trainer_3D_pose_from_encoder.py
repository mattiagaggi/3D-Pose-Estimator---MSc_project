import datetime
import numpy as np
from utils.utils_H36M.visualise import Drawer
from utils import io

from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
import os
import numpy.random as random
from collections import OrderedDict
import matplotlib.pyplot as plt
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from utils.trans_numpy_torch import numpy_to_tensor




if ENCODER_DECODER_PARAMS.encoder_decoder.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = ENCODER_DECODER_PARAMS['encoder_decoder']['device']

class Trainer_Enc_Dec_Pose(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 data_train,
                 args,
                 data_test=None,
                 no_cuda = no_cuda,
                 eval_epoch = True
                 ):

        super().__init__(model, loss, metrics, optimizer, no_cuda,eval_epoch,
                         args.epochs, args.name, args.output, args.save_freq,
                         args.verbosity, args.train_log_step,
                         args.verbosity_iter)

        self.model.fix_encoder_decoder()


        self.data_train = data_train
        self.data_test = data_test

        #test while training
        self.test_log_step = None
        self.img_log_step = args.img_log_step
        if data_test is not None:
            self.test_log_step = self.train_log_step * 10
            if self.img_log_step % self.test_log_step != 0:
                self._logger.error("Test images never recorded! %s %s" %(self.test_log_step,self.img_log_step))
        self.log_images_start_training = [10, 100, 500,1000]
        self.parameters_show = self.train_log_step * 300
        self.length_test_set = len(self.data_test)
        self.len_trainset = len(self.data_train)
        self.drawer = Drawer()
        mean= self.data_train.get_mean_pose()
        self.mean_pose = numpy_to_tensor(mean.reshape(1,17,3))
        #std = self.data_train.get_std_pose(mean).reshape(1, 17, 3)
        #self.std_pose = numpy_to_tensor(std)
        # load model
        #self._resume_checkpoint(args.resume)



    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['creation'] = str(datetime.datetime.now())
        info['size_dataset'] = len(self.data_train)
        string=""
        for number,contents in enumerate(self.data_train.index_file_list):
            string += "\n content :" + self.data_train.index_file_content[number]
            for elements in contents:
                string += " "
                string += " %s," % elements
        info['details'] = string
        info['optimiser']=str(self.optimizer)
        info['loss']=str(self.loss.__class__.__name__)
        info['sampling'] = str(self.data_train.sampling)
        return info

    def resume_encoder(self,resume_path):
        if not os.path.isfile(resume_path):
            resume_path = io.get_checkpoint(resume_path)
        self._logger.info("Loading Encoder: %s ...", resume_path)
        checkpoint = torch.load(resume_path)
        trained_dict = checkpoint['state_dict']
        if io.is_model_parallel(checkpoint):
            if self.single_gpu:
                trained_dict = OrderedDict((k.replace('module.', ''), val)
                                                       for k, val in checkpoint['state_dict'].items())
        else:
            if not self.single_gpu:
                trained_dict = OrderedDict(('module.{}'.format(k), val)
                                                       for k, val in checkpoint['state_dict'].items())
        self.model.encoder_decoder.load_state_dict(trained_dict)
        self._logger.info("Encoder Loaded '%s' loaded",
                          resume_path)



    def log_images(self, string, image_in, pose_out, ground_truth,):

        for i in range(5):
            idx=np.random.randint(self.model.batch_size)
            pt=pose_out[idx]
            gt=ground_truth[idx]
            pt_cpu=pt.cpu().data.numpy()
            gt_cpu=gt.cpu().data.numpy()
            self.model_logger.train.add_image(str(string)+str(i) + "Image", image_in[idx], self.global_step)
            fig = plt.figure()
            fig = self.drawer.poses_3d(pt_cpu, gt_cpu, plot=True, fig=fig)
            self.model_logger.train.add_figure(str(string)+str(i) + "GT", fig, self.global_step)
            fig2=plt.figure()
            fig2 = self.drawer.poses_3d(pt_cpu, gt_cpu, plot=True, fig=fig2, azim=-90, elev=0)
            self.model_logger.train.add_figure(str(string) + str(i) + "GT_depth", fig2, self.global_step)

    def log_gradients(self):

        gradients= np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.parameters()]))
        self.model_logger.train.add_scalar('Gradients', gradients,
                                           self.global_step)


    def gt_cam_mean_cam(self, dic_in, dic_out):
        mean = torch.bmm( self.mean_pose.repeat(self.model.batch_size,1,1), dic_in['R_world_im'].transpose(1,2))
        gt = torch.bmm( dic_out['joints_im'], dic_in['R_world_im'].transpose(1,2))
        return gt, mean


    def train_step(self, bid, dic_in, dic_out, pbar, epoch):

        self.optimizer.zero_grad()
        out_pose_norm = self.model(dic_in)
        gt, mean = self.gt_cam_mean_cam(dic_in, dic_out)
        #normalise gt
        gt_norm = gt-mean

        loss = self.loss( out_pose_norm, gt_norm)
        loss.backward()
        self.optimizer.step()
        if (bid % self.verbosity_iter == 0) and (self.verbosity == 2):
            pbar.set_description(' Epoch {} Loss {:.3f}'.format(
                epoch, loss.item()
            ))
        if bid % self.train_log_step == 0:
            val = loss.item()
            self.model_logger.train.add_scalar('loss/iterations', val,
                                               self.global_step)
            self.train_logger.record_scalar('train_loss', val,
                                               self.global_step)
        if (bid % self.img_log_step == 0) or (self.global_step in self.log_images_start_training):

            out_pose = out_pose_norm + mean

            self.log_images('train',dic_in['im_in'], out_pose, gt)
            self.train_logger.save_batch_images('train', dic_in['im_in'], self.global_step,
                                                pose_pred=out_pose, pose_gt=gt)

        return loss, pbar


    def test_step_on_random(self,bid):
        self.model.eval()
        idx = random.randint(self.length_test_set)
        in_test_dic, out_test_dic = self.data_test[idx]
        gt, mean = self.gt_cam_mean_cam(in_test_dic, out_test_dic)
        out_pose_norm = self.model(in_test_dic)
        self.model.train()
        #normalise gt
        gt_norm = gt - mean

        loss_test = self.loss(out_pose_norm, gt_norm)
        self.model_logger.val.add_scalar('loss/iterations', loss_test.item(),
                                         self.global_step)
        self.train_logger.record_scalar("test_loss", loss_test.item(),self.global_step)
        out_pose = out_pose_norm + mean

        if bid % self.img_log_step == 0:
            self.log_images('test',in_test_dic['im_in'], out_pose, gt)
            self.train_logger.save_batch_images('test', in_test_dic['im_in'],self.global_step,
                                                pose_pred=out_pose, pose_gt=gt)


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
        for bid, (dic_in,dic_out) in enumerate(pbar):
            loss, pbar = self.train_step(bid, dic_in, dic_out, pbar, epoch)
            if self.test_log_step is not None and (bid % self.test_log_step == 0):
                self.test_step_on_random(bid)
            if bid % self.save_freq == 0:
                if total_loss:
                    self._save_checkpoint(epoch, total_loss / bid)
                    self._update_summary(self.global_step,total_loss/bid,metrics=self.metrics)
            self.global_step += 1
            total_loss += loss.item()
        avg_loss = total_loss / len(self.data_train)
        self.model_logger.train.add_scalar('loss/epochs', avg_loss, epoch)
        self.train_logger.record_scalar('loss/epochs', avg_loss, epoch)
        self.train_logger.save_logger()

        return avg_loss

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: loss and metrics
        """
        self.model.eval()
        idx = random.randint(self.length_test_set)
        in_test_dic, out_test_dic = self.data_test[idx]
        gt, mean = self.gt_cam_mean_cam(in_test_dic, out_test_dic)
        out_pose_norm = self.model(in_test_dic)
        out_pose = out_pose_norm + mean

        for m in self.metrics:
            value = m(out_pose, gt)
            m.log_model(self.model_logger.val, self.global_step, value.item())
            m.log_train(self.train_logger, self.global_step, value.item())
        self.model.train()

