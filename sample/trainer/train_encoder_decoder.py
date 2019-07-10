

import datetime
from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torchvision.utils as vutils
import numpy.random as random
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS


if ENCODER_DECODER_PARAMS.encoder_decoder.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False

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
                 data_test=None,
                 epochs = 2,
                 name="enc_dec",
                 output = "sample/checkpoints/",
                 save_freq = 2000,
                 no_cuda = no_cuda,
                 verbosity=2,
                 verbosity_iter=10,
                 train_log_step = 1,
                 eval_epoch = False
                 ):
        super().__init__(model, loss, metrics, optimizer, epochs,
                 name, output, save_freq, no_cuda, verbosity,
                 train_log_step, verbosity_iter,eval_epoch)

        #self.batch_size = args.batch_size
        self.data_loader = data_loader

        self.data_test = data_test
        #test while training
        self.test_log_step = None
        self.length_test_set = len(self.data_test)
        if data_test is not None:
            self.test_log_step = 10 * self.train_log_step
        self.img_log_step = self.train_log_step*100

        #self.val_log_step = args.val_log_step

        self.len_trainset = len(self.data_loader)

        self.verbosity_iter=verbosity_iter

        # load model
        #self._resume_checkpoint(args.resume)


        # setting drawer
        #self.drawer = Drawer(Style.EQ_AXES)

    def _summary_info(self):
        """Summary file to differentiate
         between all the different trainings
        """
        info = dict()
        info['creation'] = str(datetime.datetime.now())
        info['size_dataset'] = len(self.data_loader)
        info['index_file_list'] = self.data_loader.index_file_list
        info['index_file_content'] = self.data_loader.index_file_content
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

        for bid, (dic_in,dic_out) in enumerate(pbar):

            self.optimizer.zero_grad()
            out_im = self.model(dic_in)
            loss = self.loss(out_im, dic_out['im_target'])
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
                if bid % self.img_log_step:
                    scale, norm, nrow = True, True, 3
                    grid = vutils.make_grid(dic_in['im_in'],nrow=nrow, normalize=norm, scale_each=scale)
                    self.model_logger.train.add_image("1 Image in", grid, self.global_step)
                    grid1 = vutils.make_grid(out_im,nrow=nrow,normalize=norm,scale_each=scale)
                    self.model_logger.train.add_image("2 Model Output", grid1, self.global_step)
                    grid2 = vutils.make_grid(dic_out['im_target'], nrow=nrow,normalize=norm, scale_each=scale)
                    self.model_logger.train.add_image("3 Ground Truth", grid2, self.global_step)
                    #for metric in self.metrics:
                    #    y_output = p3d.data.cpu().numpy()
                    #    y_target = p3d_gt.data.cpu().numpy()
                     #   metric.log(pred=y_output,
                     #              gt=y_target,
                     #              logger=self.model_logger.train,
                     #              iteration=self.global_step)


            if self.test_log_step is not None and (bid % self.test_log_step == 0):
                idx=random.randint(self.length_test_set)
                in_test_dic, out_test_dic =self.data_test[idx]
                out_test =  self.model(in_test_dic)
                loss_test = self.loss(out_test, out_test_dic['im_target'])

                self.model_logger.val.add_scalar('loss/iterations', loss_test.item(),
                                                       self.global_step)

                if bid % self.img_log_step ==0:
                    scale, norm, nrow = True, True,3
                    grid3 = vutils.make_grid(in_test_dic['im_in'], nrow=nrow, normalize=norm, scale_each=scale)
                    self.model_logger.val.add_image("4-Image in - Test", grid3, self.global_step)
                    grid4 = vutils.make_grid(out_test, nrow=nrow,normalize=norm, scale_each=scale)
                    self.model_logger.val.add_image("5 Model Output - Test", grid4, self.global_step)
                    grid5 = vutils.make_grid(out_test_dic['im_target'], normalize=norm,nrow=nrow, scale_each=scale)
                    self.model_logger.val.add_image("6 Ground Truth - Test", grid5, self.global_step)

            #if (bid % self.img_log_step) == 0 and self.img_log_step > -1:
            #    y_output = p3d.data.cpu().numpy()
            #    y_target = p3d_gt.data.cpu().numpy()

            #    img_poses = self.drawer.poses_3d(y_output[0], y_target[0])
            #    img_poses = img_poses.transpose([2, 0, 1])
            #    self.model_logger.train.add_image('3d_poses',
            #                                      img_poses,
            #                                      self.global_step)

            if (bid % self.save_freq) == 0:
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




