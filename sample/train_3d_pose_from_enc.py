
import torch.nn
from dataset_def.h36m_encoder_data import Data_3dpose
from sample.models.pose_encoder_decoder import Pose_3D
from dataset_def.h36m_encoder_data_to_load import Data_3dpose_to_load
from utils.collating_functions import collate_h36m
from torch.utils.data import DataLoader
from sample.config.data_conf import PARAMS
from sample.parsers.parser_enc_dec import Pose_Parser
parser= Pose_Parser("Pose Parser")
args_pose = parser.get_arguments()
from sample.losses.poses import MPJ, Aligned_MPJ, Normalised_MPJ
from sample.trainer.trainer_3D_pose_from_encoder import Trainer_Enc_Dec_Pose



device=PARAMS['data']['device']
sampling_train=PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test

data_train_load = Data_3dpose_to_load( #subsampling_fno = 1
                         index_file_content =['s'],
                         #index_file_list=[[1, 5, 6, 7],[1,2]])
                         index_file_list=[[1]], #15678
                         sampling=sampling_train,
                            no_apperance= True
                            ) #8,9

train_data_loader = DataLoader(data_train_load,
                                   batch_size=args_pose.batch_size,
                                   shuffle=True,
                                   num_workers = args_pose.num_threads,
                                collate_fn = collate_h36m, pin_memory=True )

data_test_load = Data_3dpose_to_load( #subsampling_fno = 1
                         index_file_content =['s'],
                         #index_file_list=[[1, 5, 6, 7],[1,2]])
                         index_file_list=[[9,11]], #15678
                         sampling=sampling_test,
                            no_apperance= True
                            ) #8,9

test_data_loader = DataLoader(data_test_load,
                                   batch_size=args_pose.batch_size,
                                   shuffle=True,
                                   num_workers = args_pose.num_threads,
                                collate_fn = collate_h36m, pin_memory=True )

model = Pose_3D()



metrics=[MPJ(), Aligned_MPJ(), Normalised_MPJ()]
optimizer_pose = torch.optim.Adam(model.parameters(), lr=args_pose.learning_rate)
loss_pose=MPJ()
trainer_pose =Trainer_Enc_Dec_Pose(
        model,
        loss_pose,
        args=args_pose,
        metrics=metrics,
        optimizer=optimizer_pose,
        data_train=data_train_load,
        data_test = test_data_loader,
)

#trainer_pose.resume_encoder("data/checkpoints/enc_dec_S15678_no_rot")
#trainer_pose._resume_checkpoint("data/checkpoints/enc_dec_S15678_no_rotfinal23D")
trainer_pose.train()
