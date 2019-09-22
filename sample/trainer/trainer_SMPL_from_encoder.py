import datetime
import numpy as np
from utils.utils_H36M.visualise import Drawer
from utils.smpl_torch.display_utils import Drawer as DrawerSMPL
from utils import io

from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
import os
import numpy.random as random
from collections import OrderedDict
import matplotlib.pyplot as plt
from sample.config.data_conf import PARAMS
from sample.losses.SMPL import Pose_Loss_SMPL





if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']

class Trainer_Enc_Dec_SMPL(BaseTrainer):
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

        super().__init__(model, optimizer, no_cuda,eval_epoch,
                         args.epochs, args.name, args.output, args.save_freq,
                         args.verbosity, args.train_log_step,
                         args.verbosity_iter)


        #self.model.clip_gradients()
        #self._logger.error("Gradients clipped here")

        self.batch_size = args.batch_size
        self.loss = loss
        self.metrics = metrics
        self.SMPL_metrics = []
        for m in self.metrics:
            self.SMPL_metrics.append(Pose_Loss_SMPL(m))
        self.model.fix_encoder_decoder()
        self.data_train = data_train
        self.data_test = data_test

        #test while training
        self.start_optimising_vertices = 10000000
        self.optimise_vertices = False
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

        # load model
        #self._resume_checkpoint(args.resume)



    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['creation'] = str(datetime.datetime.now())
        info['size_batches'] = len(self.data_train)
        info['batch_size'] = self.batch_size
        string=""
        for number,contents in enumerate(self.data_train.dataset.index_file_list):
            string += "\n content :" + self.data_train.dataset.index_file_content[number]
            for elements in contents:
                string += " "
                string += " %s," % elements
        info['details'] = string
        info['optimiser'] = str(self.optimizer)
        info['loss'] = str(self.loss.__class__.__name__)
        info['sampling'] = str(self.data_train.dataset.sampling)
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

    def log_image_and_pose(self, string, i, idx, dic_in, dic_out):
        image = dic_in["image"][idx]
        self.model_logger.train.add_image(str(string) + str(i) + "Image", image, self.global_step)

        gt_cpu = dic_in["joints_im"][idx].cpu().data.numpy()
        pp_cpu = dic_out["joints_im"][idx].cpu().data.numpy()
        fig = plt.figure()
        fig = self.drawer.poses_3d(pp_cpu, gt_cpu, plot=True, fig=fig, azim=-90, elev=0)
        self.model_logger.train.add_figure(str(string) + str(i) + "GT", fig, self.global_step)

        fig = plt.figure()
        fig = self.drawer.poses_3d(pp_cpu, gt_cpu, plot=True, fig=fig, azim=-90, elev=-90)
        self.model_logger.train.add_figure(str(string) + str(i) + "GT_depth", fig, self.global_step)

    def log_masks_vertices(self,string, i, idx, dic_in, dic_out):
        mask_list = []
        index_list =[]
        out_verts_list =[]
        out_masks_list =[]
        from sample.losses.images import Cross_Entropy_loss
        #c = Cross_Entropy_loss(64)
        mask_indices = dic_in['mask_idx_n'].cpu().data.numpy()
        for ca in range(1,5):
            mask_cpu = dic_in["mask_image"].cpu().data.numpy()
            index_cpu = dic_in['mask_idx_all'].cpu().data.numpy()
            verts_cpu = dic_out['mask_verts'].cpu().data.numpy()
            mask_out_cpu = dic_out['mask_image'].cpu().data.numpy()
            mask_list.append(mask_cpu[mask_indices == ca]) #subset mask image according to idx
            index_list.append(index_cpu[mask_indices==ca]) #subset mask image according to idx
            out_verts_list.append(verts_cpu[mask_indices == ca]) #subset mask image according to idx
            out_masks_list.append(mask_out_cpu[mask_indices == ca]) ##subset mask image according to idx

        fig = self.drawer.plot_image_on_axis( idx, mask_list,out_verts_list, index_list)
        self.model_logger.train.add_figure(str(string) + str(i) + "masks_vertices", fig, self.global_step)
        fig = self.drawer.plot_image_on_axis( idx, out_masks_list, None, index_list)
        self.model_logger.train.add_figure(str(string) + str(i) + "rasterized", fig, self.global_step)

    def log_smpl(self, string, i, idx, dic_out):

        joints, verts = dic_out["SMPL_output"]
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


    def log_images(self, string, dic_in, dic_out):

        for i in range(5):
            idx=np.random.randint(self.batch_size)
            self.log_image_and_pose(string, i, idx, dic_in, dic_out)
            if self.optimise_vertices:
                self.log_masks_vertices(string, i, idx, dic_in, dic_out)
            self.log_smpl(string, i, idx, dic_out)

    def log_gradients(self):

        gradients= np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.parameters()]))
        self.model_logger.train.add_scalar('Gradients', gradients,
                                           self.global_step)





    def train_step(self, bid, dic, pbar, epoch):



        if self.start_optimising_vertices == self.global_step:
            self._logger.info("Start optimising vertices")
            self.optimise_vertices = True
            self.model.optimise_vertices = True
            self.loss.optimise_vertices = True
        self.optimizer.zero_grad()
        if not no_cuda:
            for k in dic.keys():
                dic[k] = dic[k].cuda()
        dic_out = self.model(dic)
        if self.optimise_vertices:
            loss, loss_pose, loss_vert = self.loss(dic, dic_out, self.global_step )
        else:
            loss, loss_pose = self.loss(dic, dic_out, self.global_step)
        #print("loss")
        #loss.register_hook(lambda grad: print(grad))
        loss.backward()
        #self.model.plot_grad_flow()
        #plt.show()
        self.optimizer.step()
        if (bid % self.verbosity_iter == 0) and (self.verbosity == 2):
            pbar.set_description(' Epoch {} Loss {:.3f}'.format(
                epoch, loss.item()
            ))
        if bid % self.train_log_step == 0:

            self.model_logger.train.add_scalar('loss/iterations', loss.item(),
                                               self.global_step)
            self.model_logger.train.add_scalar('loss_pose', loss_pose.item(),
                                               self.global_step)
            if self.optimise_vertices:
                self.model_logger.train.add_scalar('loss_verts', loss_vert.item(),
                                                self.global_step)

            self.train_logger.record_scalar('train_loss', loss.item(),
                                               self.global_step)

            self.train_logger.record_scalar('train_loss_pose', loss_pose.item(),
                                               self.global_step)
            if self.optimise_vertices:
                self.train_logger.record_scalar('train_loss_vert', loss_vert.item(),
                                                self.global_step)

        if (bid % self.img_log_step == 0) or (self.global_step in self.log_images_start_training):

            self.log_images("train", dic, dic_out)
            self.train_logger.save_dics("train", dic, dic_out, self.global_step)

        return loss.item(), pbar


    def test_step_on_random(self,bid):
        self.model.eval()
        idx = random.randint(self.length_test_set)
        dic= self.data_test[idx]
        if not no_cuda:
            for k in dic.keys():
                dic[k] = dic[k].cuda()
        dic_out = self.model(dic)
        if self.optimise_vertices:
            loss, loss_pose, loss_vert = self.loss(dic, dic_out, self.global_step)
        else:
            loss, loss_pose = self.loss(dic, dic_out, self.global_step)
        self.model.train()
        self.model_logger.val.add_scalar('loss/iterations', loss.item(),
                                           self.global_step)
        self.model_logger.val.add_scalar('loss_pose', loss_pose.item(),
                                           self.global_step)
        if self.optimise_vertices:
            self.model_logger.val.add_scalar('loss_verts', loss_vert.item(),
                                            self.global_step)

        self.train_logger.record_scalar('test_loss', loss.item(),
                                        self.global_step)
        self.train_logger.record_scalar('test_loss_pose', loss_pose.item(),
                                        self.global_step)
        if self.optimise_vertices:
            self.train_logger.record_scalar('test_loss_vert', loss_vert.item(),
                                            self.global_step)
        if bid % self.img_log_step == 0:

            self.log_images("test", dic, dic_out)
            self.train_logger.save_dics("test", dic, dic_out, self.global_step)


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
        for bid, dic in enumerate(pbar):
            for i in dic.keys():
                print(i)
                print(dic[i].size())
            loss, pbar = self.train_step(bid, dic, pbar, epoch)

            if self.test_log_step is not None and (bid % self.test_log_step == 0):

                self.test_step_on_random(bid)
            if bid % self.save_freq == 0:
                if total_loss:
                    self._save_checkpoint(epoch, total_loss / bid)
                    self._update_summary(self.global_step,total_loss/bid)
            self.global_step += 1
            total_loss += loss
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
        dic = self.data_test[idx]
        if not no_cuda:
            for k in dic.keys():
                dic[k] = dic[k].cuda()
        dic_out = self.model(dic)
        gt = dic["joints_im"]
        out_pose = dic_out["joints_im"]
        for m in self.SMPL_metrics:
            value = m(out_pose, gt)
            m.log_model(self.model_logger.val, self.global_step, value.item())
            m.log_train(self.train_logger, self.global_step, value.item())
        self.model.train()

