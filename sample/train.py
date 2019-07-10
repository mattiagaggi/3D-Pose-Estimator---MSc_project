from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.models.encoder_decoder import Encoder_Decoder
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
import torch.nn
from torch.utils.data import DataLoader
from sample.trainer.train_encoder_decoder import Trainer_Enc_Dec
import numpy as np
import matplotlib.pyplot as plt




data_train = Data_Encoder_Decoder(batch_size= ENCODER_DECODER_PARAMS.encoder_decoder.batch_size,
                            sampling = ENCODER_DECODER_PARAMS.encoder_decoder.sampling,
                            index_file_content =['s','act'],
                            index_file_list=[[1],[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])



#sguffling only shuffles subelements

model = Encoder_Decoder(batch_size= ENCODER_DECODER_PARAMS.encoder_decoder.batch_size,
                 input_im_size= ENCODER_DECODER_PARAMS.encoder_decoder.im_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.MSELoss()



train_data_loader = DataLoader(data_train,shuffle=True)

metr=[]


    # Trainer instance
trainer = Trainer_Enc_Dec(
        model,
        loss,
        metrics=metr,
        optimizer=optimizer,
        data_loader=data_train
)



    # Start training!
trainer._resume_checkpoint("sample/checkpoints/enc_dec")

model=trainer.model

i=0

inp,out=data_train[i]
out_im=model(inp)
outtot=out_im.cpu().data.numpy()[0]
outtot2=out['im_target'].cpu().data.numpy()[0]


outtot = np.transpose(outtot,(1,2,0))
outtot=np.reshape(outtot,(128,128,3))
outtot2 = np.transpose(outtot2,(1,2,0))
outtot2=np.reshape(outtot2,(128,128,3))
plt.imshow(outtot)
plt.figure()
plt.imshow(outtot2)
plt.show()
#trainer.train()


#things to do: CHANGE SAVE MODEL

#INSTALL TENSORBOARD

#RESNET FEATURES



