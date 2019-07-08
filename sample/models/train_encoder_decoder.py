from dataset_def.h36_process_encoder_data import Data_Encoder_Decoder
from sample.models.encoder_decoder import Encoder_Decoder
from dataset_def.trans_numpy_torch import encoder_dictionary_to_pytorch

data = Data_Encoder_Decoder()

net = Encoder_Decoder(batch_size=20)

batch = data.process_batches(10)


batch = encoder_dictionary_to_pytorch(batch)
o=net(batch)





