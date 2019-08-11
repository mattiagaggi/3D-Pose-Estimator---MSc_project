

import torch
from matplotlib import pyplot as plt
from sample.parsers.parser_enc_dec import EncParser
from dataset_def.h36m_3dpose_data import Data_3dpose
from sample.losses.poses import Aligned_MPJ, MPJ, Normalised_MPJ
from utils.trans_numpy_torch import tensor_to_numpy
from utils.utils_H36M.visualise import Drawer


parser = EncParser("Encoder parser")
args_enc = parser.get_arguments()
data_train = Data_3dpose(args_enc,
                         index_file_content =['s','act'],
                         index_file_list=[[1],[2,3]],
                         sampling=5,
                         randomise=False) #8,9

loss=Aligned_MPJ()
it,out=data_train[10]
poses= out['joints_im'][:5]
R = it['R_world_im'][:5]

#now apply transormation
pose_trans = torch.bmm(poses,R.transpose(1,2))
#uncomment line in losses.poses MPJ to output the GT and prediction
loss_al,pred,gt= loss(poses,pose_trans)


#draw
print(tensor_to_numpy(loss_al))
dr = Drawer()
fig=plt.figure()
fig=dr.poses_3d(tensor_to_numpy(gt[0]),tensor_to_numpy(pred[0]),fig=fig,plot=True)
plt.show()