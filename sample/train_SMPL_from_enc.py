import torch.nn
import os
from dataset_def.h36m_SMPL_data import SMPL_Data
from sample.models.SMPL_encoder_decoder import SMPL_enc_dec
from sample.config.data_conf import PARAMS
from sample.parsers.parser_enc_dec import SMPL_Parser
from sample.losses.poses import MPJ, Normalised_MPJ
from sample.losses.SMPL import SMPL_Loss
from sample.trainer.trainer_SMPL_from_encoder import Trainer_Enc_Dec_SMPL


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device=PARAMS['data']['device']
sampling_train=PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test
parser= SMPL_Parser("SMPL Parser")
args_SMPL = parser.get_arguments()




data_train = SMPL_Data(args_SMPL,
                         index_file_content =['s'],
                         index_file_list=[[1]],
                         sampling=sampling_train) #8,9


data_test = SMPL_Data(args_SMPL,  #subsampling_fno = 2,
                        index_file_content =['s'],
                        index_file_list=[[9,11]],
                        sampling=sampling_test
                        ) #8,9


model = SMPL_enc_dec(args_SMPL.batch_size)



metrics=[MPJ(), Normalised_MPJ()]

optimizer_pose = torch.optim.Adam(model.parameters(), lr=args_SMPL.learning_rate)
loss_smpl = SMPL_Loss(args_SMPL.batch_size)

trainer_SMPL =Trainer_Enc_Dec_SMPL(
        model,
        loss_smpl,
        args=args_SMPL,
        metrics=metrics,
        optimizer=optimizer_pose,
        data_train=data_train,
        data_test = data_test,
)

trainer_SMPL.resume_encoder("data/checkpoints/enc_dec_S15678_no_rot")

trainer_SMPL.train()
