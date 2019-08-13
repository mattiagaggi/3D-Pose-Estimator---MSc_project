
from dataset_def.h36m_preprocess import Data_Base_class
import numpy as np
from numpy.random import randint
from numpy.random import normal
from random import shuffle

from utils.utils_H36M.common import H36M_CONF
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from dataset_def.h36m_preprocess import Data_Base_class

class SMPL_Data(Data_Base_class):
    def __init__(self,
                 args,
                 sampling,
                 index_file_content=['s'],
                 index_file_list=[[1]],
                 randomise=True,
                 get_intermediate_frames = False,
                 subsampling_fno = 0):
        """
        :param args:
        :param sampling:
        :param index_file_content:
        :param index_file_list:
        :param randomise:
        :param get_intermediate_frames:
        :param subsampling_fno:
        """

        super().__init__(sampling, get_intermediate_frames=get_intermediate_frames)
        self.batch_size = args.batch_size
        assert self.batch_size % 2 == 0
        self.create_index_file(index_file_content, index_file_list)
        self.index_file_content = index_file_content
        self.index_file_list = index_file_list
        self.randomise= randomise
        self.index_file_cameras =[]
        if subsampling_fno==0:
            pass
        elif subsampling_fno == 1:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=True)
        elif subsampling_fno == 2:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=False)
        else:
            self._logger.error("Subsampling not understood")
        if self.randomise:
            shuffle(self.index_file_cameras)
        self.elements_taken=0
        self._current_epoch=0


    def get_mean_pose(self):
        summed = np.zeros((17,3))
        N = 0
        for s in self.all_metadata:
            for act in self.all_metadata[s]:
                for subact in self.all_metadata[s][act]:
                    for ca in self.all_metadata[s][act][subact]:
                        metadata=self.all_metadata[s][act][subact][ca]['joint_world']
                        N += metadata.shape[0]
                        summed += np.sum(metadata, axis=0)

        return summed/N

    def get_std_pose(self,mean):
        summed = np.zeros((17,3))
        mean=mean.reshape(1,17,3)
        N = 0
        for s in self.all_metadata:
            for act in self.all_metadata[s]:
                for subact in self.all_metadata[s][act]:
                    for ca in self.all_metadata[s][act][subact]:
                        metadata=self.all_metadata[s][act][subact][ca]['joint_world']
                        N+= metadata.shape[0]
                        summed += np.sum((metadata-mean)**2, axis=0)
        summed /= N-1
        return np.sqrt(summed)