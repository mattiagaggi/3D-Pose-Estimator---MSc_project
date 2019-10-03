import numpy
import matplotlib.pyplot as plt
import matplotlib
import os
#matplotlib.use('Agg')
from logger.train_hist_logger import TrainingLogger
import pickle as pkl



log=TrainingLogger("data/checkpoints/enc_dec_S15678_rot_final/log_results")

def list_scalars_to_dic(lst):

    dic={}
    for i in lst:
        for key in i.keys():
            if key in dic:
                dic[key]+=i[key]
            else:
                dic[key]=i[key]
    return dic

def get_scalar_data(log):
    path=log.path
    count=1
    scalar_lst=[]
    while os.path.exists(os.path.join(path,'scalars%s.pkl' % count)):
        scal=pkl.load(open(os.path.join(path,'scalars%s.pkl' % count), "rb"))
        print(scal[0])
        scalar_lst.append(scal[1])
        count += 1
    return list_scalars_to_dic(scalar_lst)

scal = get_scalar_data(log)



plt.plot(scal['train_loss_idx'], scal['train_loss'])
plt.show()

dic=log.get_dic("train", 0, extra_str="")
print(dic.keys())







#dic=log.load_batch_images("train_img_images",0)

