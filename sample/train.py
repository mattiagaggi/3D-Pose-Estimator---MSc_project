import torch.nn

from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.models.encoder_decoder import Encoder_Decoder
from torch.utils.data import DataLoader
from sample.trainer.trainer_encoder_decoder import Trainer_Enc_Dec
from sample.parsers.parser_enc_dec import EncParser
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from sample.losses.images import L2_Resnet_Loss

device=ENCODER_DECODER_PARAMS['encoder_decoder']['device']
sampling_train=ENCODER_DECODER_PARAMS.encoder_decoder.sampling_train
sampling_test= ENCODER_DECODER_PARAMS.encoder_decoder.sampling_test
parser = EncParser("Encoder parser")
args =parser.get_arguments()



data_train = Data_Encoder_Decoder(args,subsampling_fno = 1,
                            index_file_content =['s','act'],
                            #index_file_list=[[1, 5, 6, 7],[1,2]])
                            index_file_list=[[1],[2, 3, 4, 5, 6, 7, 8, 9]],
                            sampling=sampling_train) #8,9


data_test = Data_Encoder_Decoder(args, subsampling_fno = 2,
                            index_file_content =['s','act'],
                            #index_file_list=[[1, 5, 6, 7],[1,2]])
                            index_file_list=[[1],[2, 3, 4, 5, 6, 7, 8, 9]],
                            randomise=False,
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


model = Encoder_Decoder(args)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss = L2_Resnet_Loss(device)


train_data_loader = DataLoader(data_train,shuffle=True, num_workers=args.num_threads)

metr=[]


trainer = Trainer_Enc_Dec(
        model,
        loss,
        args=args,
        metrics=metr,
        optimizer=optimizer,
        data_loader=data_train,
        data_test = data_test,
)

trainer.train()

    # Start training!
#trainer._resume_checkpoint("sample/checkpoints/enc_dec")





