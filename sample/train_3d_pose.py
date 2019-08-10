
import torch.nn
import os
from dataset_def.h36m_3dpose_data import Data_3dpose
from sample.models.pose_encoder_decoder import Pose_3D
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.parsers.parser_enc_dec import Pose_Parser
from sample.losses.poses import MPJ, Aligned_MPJ, Normalised_MPJ
from sample.trainer.trainer_3D_pose_from_encoder import Trainer_Enc_Dec_Pose


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device=ENCODER_DECODER_PARAMS['encoder_decoder']['device']
sampling_train=ENCODER_DECODER_PARAMS.encoder_decoder.sampling_train
sampling_test= ENCODER_DECODER_PARAMS.encoder_decoder.sampling_test
parser= Pose_Parser("Pose Parser")
args_pose = parser.get_arguments()



data_train = Data_3dpose(args_pose,
                         index_file_content =['s'],
                         index_file_list=[[1]],
                         sampling=sampling_train) #8,9


data_test = Data_3dpose(args_pose,  #subsampling_fno = 2,
                        index_file_content =['s'],
                        index_file_list=[[9,11]],
                        sampling=sampling_test
                        ) #8,9



model = Pose_3D(args_pose.batch_size)



metrics=[MPJ, Aligned_MPJ, Normalised_MPJ]
optimizer_pose = torch.optim.Adam(model.parameters(), lr=args_pose.learning_rate)
loss_pose=MPJ()
trainer_pose =Trainer_Enc_Dec_Pose(
        model,
        loss_pose,
        args=args_pose,
        metrics=metrics,
        optimizer=optimizer_pose,
        data_train=data_train,
        data_test = data_test,
)

trainer_pose.resume_encoder("data/checkpoints/enc_dec_S15678_no_rot")

trainer_pose.train()
