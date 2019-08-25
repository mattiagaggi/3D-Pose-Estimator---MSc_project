import torch.nn
import os
from dataset_def.h36m_encoder_data import Data_3dpose
from sample.models.pose_encoder_decoder import Pose_3D
from sample.parsers.parser_enc_dec import EncParser
from sample.config.data_conf import PARAMS
from sample.losses.images import L2_Resnet_Loss
from sample.trainer.trainer_encoder_decoder import Trainer_Enc_Dec

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device=PARAMS['data']['device']
sampling_train= PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test
parser = EncParser("Encoder parser")
args_enc = parser.get_arguments()



data_train = Data_3dpose(args_enc,  #subsampling_fno = 1,
                         index_file_content =['s'],
                         #index_file_list=[[1, 5, 6, 7],[1,2]])
                         index_file_list=[[1,5,6,7,8]], #15678
                         sampling=sampling_train) #8,9


data_test = Data_3dpose(args_enc,  #subsampling_fno = 2,
                        index_file_content =['s'],
                        #index_file_list=[[1, 5, 6, 7],[1,2]])
                        index_file_list=[[9,11]],
                        sampling=sampling_test
                        ) #8,9




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
#trainer._resume_checkpoint("data/checkpoints/enc_dec_S15678_rot")
#model.encoder_decoder = trainer.model

trainer.train()






