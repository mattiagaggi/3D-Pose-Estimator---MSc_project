import numpy
import matplotlib.pyplot as plt

from logger.train_hist_logger import TrainingLogger

log=TrainingLogger("sample/checkpoints/enc_dec_more_cameras_S13D/log_results")
log.load_logger()
print(log.scalars.keys())


plt.figure()
plt.plot(log.scalars['loss/iterations_idx'],log.scalars['loss/iterations'])
plt.plot(log.scalars['test_loss_idx'],log.scalars['test_loss'])
plt.show()

print(log.scalars['test_images'])

dic=log.load_batch_images("test_images",63)
print(dic.keys())
