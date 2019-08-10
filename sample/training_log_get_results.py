import numpy
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from logger.train_hist_logger import TrainingLogger

log=TrainingLogger("data/checkpoints/enc_dec_S15678_no_rot/log_results")
log.load_logger()
print(log.scalars.keys())


plt.figure()
plt.plot(log.scalars['train_loss_idx'],log.scalars['train_loss'])
plt.plot(log.scalars['test_loss_idx'],log.scalars['test_loss'])
plt.show()

print(log.scalars['train_img_images'])
dic=log.load_batch_images("train_img_images",0)

print(dic.keys())
