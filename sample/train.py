from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.models.encoder_decoder import Encoder_Decoder
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
import torch.nn
from torch.utils.data import DataLoader
from sample.trainer.train_encoder_decoder import Trainer_Enc_Dec



data_train = Data_Encoder_Decoder(batch_size= ENCODER_DECODER_PARAMS.encoder_decoder.batch_size,
                            sampling = ENCODER_DECODER_PARAMS.encoder_decoder.sampling,
                            index_file_content =['s','act'],
                            #index_file_list=[[1, 5, 6, 7],[1,2]])
                            index_file_list=[[1],[2, 3, 4, 5, 6, 7, 8, 9]]) #8,9


data_test=Data_Encoder_Decoder(batch_size= ENCODER_DECODER_PARAMS.encoder_decoder.batch_size,
                           sampling = ENCODER_DECODER_PARAMS.encoder_decoder.sampling,
                            index_file_content =['s','act'],
                           # index_file_list=[[8, 9],[1,2]])
                            #index_file_list=[[1],[10,11,12],[1,2]],
                            index_file_list=[[1],[10, 11, 12]],
                            get_intermediate_frames=True)




#sguffling only shuffles subelements

model = Encoder_Decoder(batch_size= 64,
                 input_im_size= ENCODER_DECODER_PARAMS.encoder_decoder.im_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.MSELoss()



train_data_loader = DataLoader(data_train,shuffle=True, num_workers=2)

metr=[]

    # Trainer instance
trainer = Trainer_Enc_Dec(
        model,
        loss,
        metrics=metr,
        optimizer=optimizer,
        data_loader=data_train,
        data_test = data_test,
        name ="enc_dec_test3", epochs=10
)

trainer.train()

    # Start training!
#trainer._resume_checkpoint("sample/checkpoints/enc_dec")





