import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import pickle as pkl
import os

from data.directories_location import backgrounds_location,h36m_location
from utils.utils_H36M.preprocess import Data_Base_class
from utils.utils_H36M.visualise import Drawer
from utils.io import file_exists



class Backgrounds(Data_Base_class):

    def __init__(self,train_val_test=0, location=h36m_location):
        super().__init__(train_val_test,location)


    def get_index_backgrounds(self):
        subj_list=[1]
        sampling = 1024
        self.create_index_file_subject(subj_list,sampling)

    def get_backgrounds(self,camera_number):
        lst_indices=[]
        for n, i in enumerate(self.index_file):
            if self.get_content(i['sequence_name'], 'ca') == camera_number:
                lst_indices.append(n)
        print("length",len(lst_indices))
        backgrounds=np.zeros((len(lst_indices),1000,1000,3),dtype=np.float64)
        for i,n in enumerate(lst_indices):
            if i %50 == 0:
                print("iter %s" % i)
            backgrounds[i, :, :, :] = self.extract_image(self.index_file[n]['path'])
        return np.median(backgrounds, axis=0)

    def save_backgrounds(self):
        b.get_index_backgrounds()
        backgrounds=[]
        for i in range(1,5):
            m = b.get_backgrounds(i)
            backgrounds.append(m)
        self._logger.info('Saving background file...')
        file_path = os.path.join(backgrounds_location, "backgrounds.pkl")
        if file_exists(file_path):
            self._logger.error("Overwriting previous file")
        pkl.dump(backgrounds, open(file_path, "wb"))





if __name__=="__main__":

    b=Backgrounds()
    b.save_backgrounds()
    m=pkl.load(open(os.path.join(backgrounds_location, "backgrounds.pkl"), "rb" ))
    print(m[0])
    d=Drawer()
    for i in m:
        fig = plt.figure()
        d.plot_image(fig,i)
    o=plt.figure()
    x=np.arange(10)
    plt.plot(x,x)
    plt.show()




