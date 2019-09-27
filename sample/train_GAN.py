import torch.nn
from torch.utils.data import DataLoader
from dataset_def.surreal_SMPL_data_to_load import Surreal_data_load
from sample.models.GAN import 
from sample.config.data_conf import PARAMS
from sample.parsers.parser_enc_dec import SMPL_Parser
from sample.losses.poses import MPJ, Normalised_MPJ
from sample.losses.SMPL import SMPL_Loss
from sample.trainer.trainer_SMPL_from_encoder import Trainer_Enc_Dec_SMPL
from utils.collating_functions import collate_smpl



device=PARAMS['data']['device']
sampling_train=PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test
parser= SMPL_Parser("SMPL Parser")
args_SMPL = parser.get_arguments()



data_test = None

data_train_load = SMPL_Data_Load(
                        sampling_train,
                         index_file_content =['s','act'],
                         index_file_list=[[1],[1,2,3,4,5,6,7,8,9,10,11,12,13,14]],
                         )

train_data_loader = DataLoader(data_train_load,
                                   batch_size=args_SMPL.batch_size,
                                   shuffle=True,
                                   num_workers = args_SMPL.num_threads,
                                collate_fn = collate_smpl, pin_memory=True )


model = SMPL_enc_dec()



metrics=[MPJ(), Normalised_MPJ()]

optimizer_pose = torch.optim.Adam(model.parameters(), lr=args_SMPL.learning_rate)
loss_smpl = SMPL_Loss(args_SMPL.batch_size)

trainer_SMPL =Trainer_Enc_Dec_SMPL(
        model,
        loss_smpl,
        args=args_SMPL,
        metrics=metrics,
        optimizer=optimizer_pose,
        data_train=train_data_loader,
        data_test = data_test,
)

#trainer_SMPL.resume_encoder("data/checkpoints/enc_dec_S15678_no_rot")
#trainer_SMPL._resume_checkpoint("data/checkpoints/enc_dec_S15678_rot_finalSMPL")

trainer_SMPL.train()
