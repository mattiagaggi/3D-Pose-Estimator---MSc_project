"""

This class create the background.
The backgrounds are saved for each subject as an array of shape [camera number,L,W,3]
in background_location folder with the name "background_subject%s.npy" % subject_number

"""



import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import os
import cv2
from tempfile import TemporaryFile
outfile = TemporaryFile()


from data.directories_location import backgrounds_location,h36m_location
from utils.utils_H36M.preprocess import Data_Base_class
from utils.utils_H36M.visualise import Drawer
from utils.utils_H36M.common import H36M_CONF, ENCODER_DECODER_PARAMS
from utils.io import file_exists




class Backgrounds(Data_Base_class):

    def __init__(self,train_val_test = 0,
                 sampling = ENCODER_DECODER_PARAMS.background.sampling,
                 max_epochs = 1):

        super().__init__(train_val_test,sampling, max_epochs)

    def get_index_backgrounds(self,subject):
        """
        :param subject: subject to find background
        :return: None
        """
        self.create_index_file( 's', [subject])

    def get_backgrounds(self,camera_number):
        """
        :param camera_number: ...
        :return: median for one camera
        """

        self.iteration_start()
        s, act, subact, ca, fno = self.return_current_file()
        backgrounds=[]
        count=0
        while self._current_epoch < 1:
            if camera_number == ca:
                print(s, act, subact, ca, fno)
                print("image appended %s" % count)
                path=self.index_file[s][act][subact][ca][fno]
                print(path)
                backgrounds.append(self.extract_image(path))
                count += 1
            s, act, subact, ca, fno = self.next_file_content()
        print("out while loop")
        backgrounds = np.stack(backgrounds, axis=0)
        return np.median(backgrounds, axis=0)


    def save_backgrounds(self, subject_list):

        """
        created index file for each subject and goes through all 4 cameras in the dataset
        :param subject_list: ....
        :return: None
        """
        for i in subject_list:
            self.get_index_backgrounds(i)
            backgrounds= np.zeros((4, H36M_CONF.max_size,H36M_CONF.max_size, 3))
            for ca in range(1,5):
                m = b.get_backgrounds(ca)
                print("out of the get background")
                backgrounds[ca-1,:,:,:] = m
            self._logger.info('Saving background file...')
            file_path = os.path.join(backgrounds_location, "background_subject%s.npy" % i)
            if file_exists(file_path):
                self._logger.error("Overwriting previous file")
            np.save(file_path,backgrounds)



import copy

if __name__=="__main__":

    b=Backgrounds()
    b.save_backgrounds([1])
    path=os.path.join(backgrounds_location,"background_subject1.npy")

    m= np.load(path)
    d=Drawer()
    for i in range(m.shape[0]):
        plt.figure()
        im=d.get_image(m[i,...])
        plt.imshow(im)
    plt.show()




