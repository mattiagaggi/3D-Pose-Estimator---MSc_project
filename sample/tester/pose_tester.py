
from sample.config.data_conf import PARAMS
from utils.collating_functions import collate_h36m
from sample.base.base_tester import BaseTester
from dataset_def.h36m_encoder_data_to_load import Data_3dpose_to_load
from utils.trans_numpy_torch import numpy_to_tensor_float
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

device=PARAMS['data']['device']
if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']



class Pose_Tester(BaseTester):
    def __init__(self, model, output, name, sampling_train, sampling_test,metrics,no_cuda):

        super().__init__(self, model, output, name, no_cuda)


        self._resume_checkpoint(os.path.join(output,name))
        self.sampling_train=sampling_train
        self.sampling_test = sampling_test
        self.metrics = metrics
        data_train_load = Data_3dpose_to_load(  # subsampling_fno = 1
            index_file_content=['s'],
            # index_file_list=[[1, 5, 6, 7],[1,2]])
            index_file_list=[[1]],  # 15678
            sampling=sampling_train,
            no_apperance=True
        )  # 8,9
        mean= data_train_load.get_mean_pose()
        self.mean_pose = numpy_to_tensor_float(mean.reshape(1, 17, 3))

    def gt_cam_mean_cam(self, dic_in, dic_out):
        batch_size = dic_in['R_world_im'].size()[0]
        mean = torch.bmm( self.mean_pose.repeat(batch_size,1,1), dic_in['R_world_im'].transpose(1,2))
        gt = torch.bmm( dic_out['joints_im'], dic_in['R_world_im'].transpose(1,2))
        return gt, mean

    def test_on_train(self):
        self.model.eval()
        for act in range(2,17):
            data_train_load = Data_3dpose_to_load(  # subsampling_fno = 1
                index_file_content=['s','act'],
                # index_file_list=[[1, 5, 6, 7],[1,2]])
                index_file_list=[[1],[act]],  # 15678
                sampling=self.sampling_train,
                no_apperance=True
            )  # 8,9
            train_data_loader = DataLoader(data_train_load,
                                           batch_size=10,
                                           shuffle=True,
                                           num_workers=0,
                                           collate_fn=collate_h36m, pin_memory=True)
            pbar =tqdm(train_data_loader)
            data_n =0
            for bid, (in_test_dic, out_test_dic) in enumerate(pbar):
                if not no_cuda:
                    for k in in_test_dic.keys():
                        in_test_dic[k] = in_test_dic[k].cuda()
                    for k in out_test_dic.keys():
                        out_test_dic[k] = out_test_dic[k].cuda()
                gt, mean = self.gt_cam_mean_cam(in_test_dic, out_test_dic)
                out_pose_norm = self.model(in_test_dic)
                out_pose = out_pose_norm + mean
                for idx,m in enumerate(self.metrics):
                    value = m(out_pose, gt)
                    # value.item()
                #save stuff
                data_n +=1






