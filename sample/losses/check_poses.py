

import torch
from matplotlib import plt
from sample.parsers.parser_enc_dec import EncParser
from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.losses.poses import Aligned_MPJ, MPJ, Normalised_MPJ
from utils.trans_numpy_torch import tensor_to_numpy
from utils.utils_H36M.visualise import Drawer


parser = EncParser("Encoder parser")
args_enc =parser.get_arguments()
data_train = Data_Encoder_Decoder(args_enc,
                            index_file_content =['s','act'],
                            index_file_list=[[1],[2,3]],
                            sampling=5,
                                randomise=False) #8,9

loss=MPJ(debug=True)
_,it=data_train[10]
poses= it['joints_im'][:5]
R = int['R_world_im'][:5]

#now apply transormation
pose_trans = torch.bmm(poses,R.transpose(1,2))
loss_al,pred,gt= l(poses,pose_trans)

#draw
print(tensor_to_numpy(loss_al))
dr = Drawer()
fig=plt.figure()
fig=dr.poses_3d(tensor_to_numpy(gt[0]),tensor_to_numpy(pred[0]),fig=fig,plot=True)
plt.show()