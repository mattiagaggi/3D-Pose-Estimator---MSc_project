import torch.nn
import os
from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.models.pose_encoder_decoder import Pose_3D
from torch.utils.data import DataLoader
from sample.parsers.parser_enc_dec import EncParser,Pose_Parser
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.losses.images import L2_Resnet_Loss
from sample.trainer.trainer_encoder_decoder import Trainer_Enc_Dec
from sample.trainer.trainer_3D_pose_from_encoder import Trainer_Enc_Dec_Pose
from sample.losses.poses import MPJ





os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device=ENCODER_DECODER_PARAMS['encoder_decoder']['device']
sampling_train=ENCODER_DECODER_PARAMS.encoder_decoder.sampling_train
sampling_test= ENCODER_DECODER_PARAMS.encoder_decoder.sampling_test
parser = EncParser("Encoder parser")
args_enc =parser.get_arguments()
parser= Pose_Parser("Pose Parser")
args_pose = parser.get_arguments()


data_train = Data_Encoder_Decoder(args_enc,#subsampling_fno = 1,
                            index_file_content =['s'],
                            #index_file_list=[[1, 5, 6, 7],[1,2]])
                            index_file_list=[[1,5,6,7,8]],
                            sampling=sampling_train) #8,9


data_test = Data_Encoder_Decoder(args_enc, #subsampling_fno = 2,
                            index_file_content =['s'],
                            #index_file_list=[[1, 5, 6, 7],[1,2]])
                            index_file_list=[[9,11]],
                            sampling=sampling_test
                                 ) #8,9




"""


data_test=Data_Encoder_Decoder(args,
                            index_file_content =['s','act'],
                           # index_file_list=[[8, 9],[1,2]])
                            #index_file_list=[[1],[10,11,12],[1,2]],
                            index_file_list=[[1],[10, 11, 12]],
                            get_intermediate_frames=False)
"""


model = Pose_3D(args_enc.batch_size)
optimizer = torch.optim.Adam(model.encoder_decoder.parameters(), lr=args_enc.learning_rate)
loss = L2_Resnet_Loss(device)

train_data_loader = DataLoader(data_train,shuffle=True, num_workers=args_enc.num_threads)
metr=[]
trainer = Trainer_Enc_Dec(
        model.encoder_decoder,
        loss,
        args=args_enc,
        metrics=metr,
        optimizer=optimizer,
        data_loader=data_train,
        data_test = data_test,
)

trainer.train()

# Start training!
#trainer._resume_checkpoint("sample/checkpoints/enc_dec_more_cameras_new_loss")
#model.encoder_decoder = trainer.model

optimizer_pose = torch.optim.Adam(model.parameters(), lr=args_pose.learning_rate)
loss_pose=MPJ()
trainer_pose =Trainer_Enc_Dec_Pose(
        model,
        loss_pose,
        args=args_pose,
        metrics=metr,
        optimizer=optimizer_pose,
        data_loader=data_train,
        data_test = data_test,
)



trainer_pose.train()




