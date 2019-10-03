import torch.nn
from torch.utils.data import DataLoader
from dataset_def.surreal_data_to_load import Surreal_data_load
from sample.models.GAN import GAN_SMPL
from sample.config.data_conf import PARAMS
from sample.parsers.parser_gan_smpl import GAN_Parser
from torch.nn import BCELoss
from sample.trainer.trainer_GAN import Trainer_GAN




device=PARAMS['data']['device']
sampling_train=PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test
parser= GAN_Parser("GAN Parser")
args_GAN = parser.get_arguments()



data_test = None

data_train_load = Surreal_data_load(
                        sampling_train, 0 # not used
                         )

train_data_loader = DataLoader(data_train_load,
                                   batch_size=args_GAN.batch_size,
                                   shuffle=True,
                                   num_workers = args_GAN.num_threads )


model = GAN_SMPL()



metrics=[]

optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=0.001)
optimizer_generator = torch.optim.Adam(model.generator.parameters(), lr=0.001)
loss_generator = BCELoss()
loss_discriminator = BCELoss()

trainer_GAN =Trainer_GAN(
        model,
        loss_generator,
        loss_discriminator,
        metrics,
        optimizer_generator,
        optimizer_discriminator,
        args=args_GAN,
        data_train=train_data_loader,
        data_test = data_test,
)



trainer_GAN.train()
