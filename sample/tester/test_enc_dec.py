
from sample.config.data_conf import PARAMS
from utils.collating_functions import collate_h36m
from sample.base.base_tester import BaseTester
from dataset_def.h36m_encoder_data_to_load import Data_3dpose_to_load
from dataset_def.h36m_encoder_data import Data_3dpose
from utils.trans_numpy_torch import numpy_to_tensor_float
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import os
from sample.parsers.parser_enc_dec import EncParser
from utils.utils_H36M.transformations_torch import rotate_x_torch,rotate_y_torch
parser= EncParser("Encoder parser")
args_pose = parser.get_arguments()

device=PARAMS['data']['device']
if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']



class Encoder_Tester(BaseTester):
    def __init__(self, model, output, name):

        super().__init__(model, output, name, no_cuda)


        self._resume_checkpoint(os.path.join(output,name))

    def test_on(self,s_list,sampling,name):
        self.model.eval()
        for act in range(3,17):
            self._logger.info("act %s",act)
            data_test = Data_3dpose(args_pose,  # subsampling_fno = 2,
                                    index_file_content=['s','act'],
                                    index_file_list=[s_list,[2,act]],
                                    sampling=sampling,
                                    no_apperance=False,
                                    randomise=True
                                    )  # 8,9
            pbar =tqdm(data_test)


            for bid, (in_test_dic, out_test_dic) in enumerate(pbar):
                if not no_cuda:
                    for k in in_test_dic.keys():
                        in_test_dic[k] = in_test_dic[k].cuda()
                    for k in out_test_dic.keys():
                        out_test_dic[k] = out_test_dic[k].cuda()
                b = in_test_dic['R_world_im_target']
                for r in [0,30,45,60,90,180]:
                    in_test_dic['background_target']= in_test_dic['background_target']
                    if r !=0:
                        in_test_dic['R_world_im_target']= rotate_y_torch(r*np.pi/180.,args_pose.batch_size)
                    out_im = self.model(in_test_dic)
                    out_test_dic["image_final"+str(r)]=out_im

                if bid in [0]:
                    self.train_logger.save_dics(name,in_test_dic,out_test_dic,"act_"+str(act)+"n"+str(bid))
                else:
                    break



    def test_on_train(self):
        self.test_on(s_list=[1,5,6,7,8], sampling=5, name="train")
    def test_on_test(self):
        self.test_on(s_list=[11,9], sampling=5, name="test")




