import datetime
import numpy as np
import pickle


from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torchvision.utils as vutils
import numpy.random as random
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.losses.images import ImageNetCriterium



if ENCODER_DECODER_PARAMS.encoder_decoder.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = ENCODER_DECODER_PARAMS['encoder_decoder']['device']

class Trainer_Enc_Dec(BaseTrainer):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 data_loader,
                 args,
                 data_test=None,
                 no_cuda = no_cuda,
                 eval_epoch = False
                 ):

        name=args.name
        output = args.output
        epochs = args.epochs
        save_freq = args.save_freq
        verbosity = args.verbosity
        verbosity_iter = args.verbosity_iter
        train_log_step = args.train_log_step


        super().__init__(model, loss, metrics, optimizer, epochs,
                 name, output, save_freq, no_cuda, verbosity,
                 train_log_step, verbosity_iter,eval_epoch)


        self.data_loader = data_loader
        self.data_test = data_test

        #test while training
        self.test_log_step = None
        self.img_log_step = args.img_log_step
        if data_test is not None:
            self.test_log_step = self.train_log_step

        self.parameters_show = self.train_log_step * 300
        self.length_test_set = len(self.data_test)
        self.len_trainset = len(self.data_loader)


        # load model
        #self._resume_checkpoint(args.resume)




    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['creation'] = str(datetime.datetime.now())
        info['size_dataset'] = len(self.data_loader)
        string=""
        for number,contents in enumerate(self.data_loader.index_file_list):
            string += "\n content :" + self.data_loader.index_file_content[number]
            for elements in contents:
                string += "\n numbers: "
                string += " %s," % elements
        info['details'] = string
        return info


    def log_grid(self,image_in, pose_out, ground_truth, string):

        scale, norm, nrow = True, True, 3
        grid = vutils.make_grid(image_in, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("1 Image In " + str(string), grid, self.global_step)

        grid1 = vutils.make_grid(pose_out, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("2 Model Output "+str(string), grid1, self.global_step)
        grid2 = vutils.make_grid(ground_truth, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("3 Ground Truth "+str(string), grid2, self.global_step)





    def log_gradients(self):

        gradients= np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.parameters()]))
        self.model_logger.train.add_scalar('Gradients', gradients,
                                           self.global_step)


    def log_metrics(self):
        pass

        # for metric in self.metrics:
        #    y_output = p3d.data.cpu().numpy()
        #    y_target = p3d_gt.data.cpu().numpy()
        #   metric.log(pred=y_output,
        #              gt=y_target,
        #              logger=self.model_logger.train,
        #              iteration=self.global_step)

    def log_3D_pose(self):
        pass

        # if (bid % self.img_log_step) == 0 and self.img_log_step > -1:
        #    y_output = p3d.data.cpu().numpy()
        #    y_target = p3d_gt.data.cpu().numpy()

        #    img_poses = self.drawer.poses_3d(y_output[0], y_target[0])
        #    img_poses = img_poses.transpose([2, 0, 1])
        #    self.model_logger.train.add_image('3d_poses',
        #                                      img_poses,
        #                                      self.global_step)


    def train_step(self, bid, dic_in, dic_out, pbar, epoch):

        self.optimizer.zero_grad()
        out_pose = self.model(dic_in)
        loss = self.loss(out_pose, dic_out['joints_im'])
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
            if bid % self.img_log_step:
                self.log_grid(dic_in['im_in'], out_pose, dic_out['joints_im'], 'train')
        return loss, pbar


    def test_step_on_random(self,bid):
        self.model.eval()
        idx = random.randint(self.length_test_set)
        in_test_dic, out_test_dic = self.data_test[idx]
        out_pose = self.model(in_test_dic)
        self.model.train()
        loss_test = self.loss(out_pose, out_test_dic['joints_im'])
        self.model_logger.val.add_scalar('loss/iterations', loss_test.item(),
                                         self.global_step)
        self.log_grid(in_test_dic['im_in'], out_pose, out_test_dic['joints_im'], 'test')


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

        for bid, (dic_in,dic_out) in enumerate(pbar):
            loss, pbar = self.train_step(bid, dic_in, dic_out, pbar, epoch)
            if self.test_log_step is not None and (bid % self.test_log_step == 0):
                self.test_step_on_random(bid)
            if bid % self.parameters_show == 0:
                self.log_gradients()
            if bid % self.save_freq == 0:
                if total_loss:
                    self._save_checkpoint(epoch, self.global_step,
                                          total_loss / bid)
                    self._update_summary(self.global_step,total_loss/bid,metrics=self.metrics)
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




