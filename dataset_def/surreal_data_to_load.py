from dataset_def.h36m_preprocess import Data_Base_class
from utils.io import get_files
from data.config import cmu_data
import scipy.io as sio
import numpy as np
import os

class Surreal_data_load(Data_Base_class):

    def __init__(self,
                 sampling,
                 type):
        super().__init__(sampling=sampling)
        if type==0:
            self.path = os.path.join(cmu_data,'train')
        elif type ==1:
            self.path = os.path.join(cmu_data,'test')
        else:
            self.path = os.path.join(cmu_data, 'val')

        file_paths, file_names = get_files(self.path,"mat")
        self.index_file =[]
        self.index_number =[]
        self.total_length =0
        self._logger.info("start")
        data_used = 10 ** 7
        self._logger.info("Remember amount of data is: %s" % data_used)
        for i in file_paths:
            if "info" in i:
                data = sio.loadmat(i)
                assert 'pose' in data.keys() and 'shape' in data.keys()
                _, N =data['pose'].shape
                self.total_length += N
                self.index_file.append(i)
                self.index_number.append(N)
                if len(self.index_number)>1:
                    self.index_number[-1] = self.index_number[-1]+self.index_number[-2]
                if self.total_length >= data_used:
                    break
        self._logger.info("dataset size %s" % self.total_length)



    def binary_search(self, idx, lst):
        current = len(lst)//2
        if current==0:
            if lst[0]<=idx:
                return 1
            else:
                return 0
        if lst[current]<= idx:
            return self.binary_search(idx,lst[current:]) + current
        else:
            return self.binary_search(idx,lst[:current])

    def __len__(self):
        return self.total_length

    def __getitem__(self, item):
        idx = self.binary_search(item,self.index_number)
        path = self.index_file[idx]
        data = sio.loadmat(path)
        N_idx= item
        if idx> 0:
            N_idx  -= self.index_number[idx-1]
        out=np.concatenate([data['pose'][:,N_idx], data['shape'][:,N_idx]])
        return out


