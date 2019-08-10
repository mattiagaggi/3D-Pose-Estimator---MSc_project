import torch.nn
import os
from dataset_def.h36m_3dpose_data import Data_3dpose
from sample.models.pose_encoder_decoder import Pose_3D
from torch.utils.data import DataLoader
from sample.parsers.parser_enc_dec import EncParser
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.losses.images import L2_Resnet_Loss
from sample.trainer.trainer_encoder_decoder import Trainer_Enc_Dec

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device=ENCODER_DECODER_PARAMS['encoder_decoder']['device']
sampling_train=ENCODER_DECODER_PARAMS.encoder_decoder.sampling_train
sampling_test= ENCODER_DECODER_PARAMS.encoder_decoder.sampling_test
parser = EncParser("Encoder parser")
args_enc =parser.get_arguments()



data_train = Data_3dpose(args_enc,  #subsampling_fno = 1,
                         index_file_content =['s'],
                         #index_file_list=[[1, 5, 6, 7],[1,2]])
                         index_file_list=[[1,5,6,7,8]],
                         sampling=sampling_train) #8,9


data_test = Data_3dpose(args_enc,  #subsampling_fno = 2,
                        index_file_content =['s'],
                        #index_file_list=[[1, 5, 6, 7],[1,2]])
                        index_file_list=[[9,11]],
                        sampling=sampling_test
                        ) #8,9

train_data_loader = DataLoader(data_train,shuffle=True, num_workers=args_enc.num_threads)




model = Pose_3D(args_enc.batch_size)
optimizer = torch.optim.Adam(model.encoder_decoder.parameters(), lr=args_enc.learning_rate)
loss = L2_Resnet_Loss(device)


trainer = Trainer_Enc_Dec(
        model.encoder_decoder,
        loss,
        args=args_enc,
        metrics=[],
        optimizer=optimizer,
        data_train=data_train,
        data_test = data_test,
)



# Start training!
#trainer._resume_checkpoint("data/checkpoints/enc_dec_S15678_no_rot")
#model.encoder_decoder = trainer.model

#trainer.train()





