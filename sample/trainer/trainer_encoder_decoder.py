

import datetime
import numpy as np


from sample.base.base_trainer import BaseTrainer
from tqdm import tqdm
import torchvision.utils as vutils
import numpy.random as random
from sample.config.data_conf import PARAMS



if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']

class Trainer_Enc_Dec(BaseTrainer):
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
                 eval_epoch = False
                 ):

        super().__init__(model, optimizer,
                         no_cuda,eval_epoch, args.epochs,
                 args.name, args.output, args.save_freq, args.verbosity,
                 args.train_log_step, args.verbosity_iter
                         )
        self.batch_size = args.batch_size
        self.loss = loss
        self.metrics = metrics
        self.data_train = data_train
        self.data_test = data_test
        #test while training
        self.test_log_step = None
        self.img_log_step = args.img_log_step
        if data_test is not None:
            self.test_log_step = args.test_log_step

        self.log_images_start_training =[10,100,500,1000]
        self.parameters_show = self.train_log_step * 300
        self.length_test_set = len(self.data_test)
        self.len_trainset = len(self.data_train)


        # load model
        #self._resume_checkpoint(args.resume)

        self.encoder_parameters = self.model.encoder.return_n_parameters()
        self.decoder_parameters = self.model.decoder.return_n_parameters()
        self.rotation_parameters = self.model.rotation.return_n_parameters()



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
        info['one epoch'] = self.len_trainset
        info['optimiser'] = str(self.optimizer)
        info['loss'] = str(self.loss.__class__.__name__)
        info['sampling'] = str(self.data_train.dataset.sampling)

        return info


    def log_grid(self, string, image_in, image_out, ground_truth):

        scale, norm, nrow = True, True, 3
        grid = vutils.make_grid(image_in, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("1 Image in "+str(string), grid, self.global_step)
        grid1 = vutils.make_grid(image_out, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("2 Model Output "+str(string), grid1, self.global_step)
        grid2 = vutils.make_grid(ground_truth, nrow=nrow, normalize=norm, scale_each=scale)
        self.model_logger.train.add_image("3 Ground Truth "+str(string), grid2, self.global_step)



    def get_gradients(self):
        for name, parameters in self.model.named_parameters():
            self.model_logger.train.add_histogram(name, parameters.clone().cpu().data.numpy(), self.global_step)
        encoder_gradients = np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.encoder.parameters()]))
        decoder_gradients = np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.decoder.parameters()]))
        rotation_gradients = np.sum(np.array([np.sum(np.absolute(x.grad.cpu().data.numpy())) for x in self.model.rotation.parameters()]))
        return encoder_gradients,rotation_gradients,decoder_gradients

    def log_gradients(self):

        encoder_gradients, rotation_gradients, decoder_gradients = self.get_gradients()
        self.model_logger.train.add_scalar('rotation gradients ' + str(self.rotation_parameters), rotation_gradients,
                                           self.global_step)
        self.model_logger.train.add_scalar('encoder gradients ' + str((self.encoder_parameters)), encoder_gradients,
                                           self.global_step)
        self.model_logger.train.add_scalar('decoder gradients ' + str(self.decoder_parameters), decoder_gradients,
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




    def train_step(self, bid, dic_in, dic_out, pbar, epoch):

        self.optimizer.zero_grad()
        out_im = self.model(dic_in)
        loss = self.loss(out_im, dic_out['im_target'])
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
            self.train_logger.record_scalar("train_loss", val, self.global_step)
        if (bid % self.img_log_step==0) or (self.global_step in self.log_images_start_training):
            self.log_grid( 'train',dic_in['im_in'], out_im, dic_out['im_target'])
            self.train_logger.save_batch_images('train_img', dic_in['im_in'], self.global_step,
                                                image_target=dic_out['im_target'],
                                                image_pred= out_im)

        return loss.item(), pbar




    def test_step_on_random(self,bid):
        self.model.eval()
        idx = random.randint(self.length_test_set)
        in_test_dic, out_test_dic = self.data_test[idx]
        if not no_cuda:
            for k in in_test_dic.keys():
                in_test_dic[k] = in_test_dic[k].cuda()
            for k in out_test_dic.keys():
                out_test_dic[k] = out_test_dic[k].cuda()
        out_test = self.model(in_test_dic)
        self.model.train()
        loss_test = self.loss(out_test, out_test_dic['im_target'])
        self.model_logger.val.add_scalar('loss/iterations', loss_test.item(),
                                         self.global_step)
        self.train_logger.record_scalar("test_loss", loss_test.item(), self.global_step)
        if bid % self.img_log_step == 0:
            self.log_grid('test',in_test_dic['im_in'], out_test, out_test_dic['im_target'])
            self.train_logger.save_batch_images('test_img', in_test_dic['im_in'],self.global_step,
                                                image_target =  out_test_dic['im_target'],
                                                image_pred= out_test)



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
            if not no_cuda:
                for k in dic_in.keys():
                    dic_in[k] = dic_in[k].cuda()
                for k in dic_out.keys():
                    dic_out[k] = dic_out[k].cuda()
            loss, pbar = self.train_step(bid, dic_in, dic_out, pbar, epoch)
            if self.test_log_step is not None and (bid % self.test_log_step == 0):
                self.test_step_on_random(bid)
            #if bid % self.parameters_show == 0:
                #self.log_gradients()
            if bid % self.save_freq == 0:
                if total_loss:
                    self._save_checkpoint(epoch, total_loss / bid)
                    self._update_summary(self.global_step, total_loss/bid)
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
        in_test_dic, out_test_dic = self.data_test[idx]
        if not no_cuda:
            for k in in_test_dic.keys():
                in_test_dic[k] = in_test_dic[k].cuda()
            for k in out_test_dic.keys():
                out_test_dic[k] = out_test_dic[k].cuda()
        out_test = self.model(in_test_dic)
        for m in self.metrics:
            value = m(out_test,out_test_dic['im_target'])
            m.log_model(self.model_logger.test, self.global_step, value.item())
            m.log_train(self, self.train_logger, self.global_step, value.item())
        self.model.train()





