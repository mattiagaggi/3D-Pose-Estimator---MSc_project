
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
from sample.parsers.parser_enc_dec import Pose_Parser
parser= Pose_Parser("Pose Parser")
args_pose = parser.get_arguments()

device=PARAMS['data']['device']
if PARAMS.data.device_type == 'cpu':
    no_cuda=True
else:
    no_cuda=False
device = PARAMS['data']['device']



class Pose_Tester(BaseTester):
    def __init__(self, model, output, name,metrics):

        super().__init__(model, output, name, no_cuda)


        self._resume_checkpoint(os.path.join(output,name))
        self.metrics = metrics
        self.dic_results = {}

        for n,i in enumerate(self.metrics):
            self.dic_results[n] = np.array([])
        self.batch_numbers =  np.array([])


    def gt_cam_mean_cam(self, dic_in, dic_out):
        #batch_size = dic_in['R_world_im'].size()[0]
        #mean = torch.bmm( self.mean_pose.repeat(batch_size,1,1), dic_in['R_world_im'].transpose(1,2))
        gt = torch.bmm( dic_out['joints_im'], dic_in['R_world_im'].transpose(1,2))
        return gt, None

    def test_on(self,s_list,sampling,name):
        self.model.eval()
        for act in range(2,17):
            self._logger.info("act %s",act)
            for n in self.dic_results.keys():
                self.dic_results[n]=np.append(self.dic_results[n], 0)
            self.batch_numbers =np.append(self.batch_numbers,0)
            data_test = Data_3dpose(args_pose,  # subsampling_fno = 2,
                                    index_file_content=['s','act'],
                                    index_file_list=[s_list,[act]],
                                    sampling=sampling,
                                    no_apperance=True,
                                    randomise=True
                                    )  # 8,9
            self._logger.info("acr %s, n %s" %(act,len(data_test)))
            
            pbar =tqdm(data_test)


            for bid, (in_test_dic, out_test_dic) in enumerate(pbar):
                break
                if not no_cuda:
                    for k in in_test_dic.keys():
                        in_test_dic[k] = in_test_dic[k].cuda()
                    for k in out_test_dic.keys():
                        out_test_dic[k] = out_test_dic[k].cuda()
                gt, _ = self.gt_cam_mean_cam(in_test_dic, out_test_dic)
                out_pose = self.model(in_test_dic)
                #out_pose = out_pose_norm + mean
                out_test_dic["pose_final"] = out_pose
                if bid in [0,1,2,3,4,5]:
                    self.train_logger.save_dics(name,in_test_dic,out_test_dic,"act_"+str(act)+"n"+str(bid))
                else:
                    break
                lst=[]
                for idx,m in enumerate(self.metrics):
                    value = m(out_pose, gt)
                    lst.append(value.item())
                arr=np.array(lst)
                if len(arr[arr>500])!=0:
                    self._logger.info("abnormal found %s " % arr[arr>500][-1])
                    self.train_logger.save_dics(name, in_test_dic, out_test_dic,
                                                "abnormal " + str(act) + "n" + str(arr[arr>100][-1]))
                else:
                    for idx, m in enumerate(self.metrics):
                        self.dic_results[idx][-1] += lst[idx]
                    #save stuff
                    self.batch_numbers[-1] +=1
                self._logger.info("%s  ,%s" %(self.dic_results[0][-1]/self.batch_numbers[-1],self.dic_results[2][-1]/self.batch_numbers[-1]))
            self._logger.info("act %s %s %s" % (self.dic_results[0][-1]/self.batch_numbers[-1],
                              self.dic_results[1][-1] / self.batch_numbers[-1],
                              self.dic_results[2][-1] / self.batch_numbers[-1]
                              ))

        for key in self.dic_results.keys():
            self.dic_results[key] = self.dic_results[key]/self.batch_numbers
        self._logger.info(self.dic_results)
        self.train_logger.save_dic(name,self.dic_results,0, gpu=False)

    def test_on_train(self):
        self.test_on(s_list=[1,5,6,7,8], sampling=5, name="train")
    def test_on_test(self):
        self.test_on(s_list=[11,9], sampling=5, name="test")






# act 2 143.3337716582819 130.90568948299327 130.7445228387278
# act 3

