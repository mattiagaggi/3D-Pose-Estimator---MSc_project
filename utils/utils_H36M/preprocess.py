
import os
import pickle as pkl
import scipy.io as sio

import cv2
import numpy as np


from data.directories_location import index_location,h36m_location
from utils.utils_H36M.common import H36M_CONF
from utils.io import get_sub_dirs,get_files,file_exists
from logger.console_logger import ConsoleLogger



class Data_Base_class:

    def __init__(self, train_val_test=0, location=index_location):

        self.index_file_loc= location
        self.sampling=10

        logger_name = '{}'.format(self.__class__.__name__)
        self._logger=ConsoleLogger(logger_name)

        self.index_file = None

        if train_val_test == 0:
            self.index_location = "index_train.pkl"
        elif train_val_test == 1:
            self.index_location = "index_val.pkl"
        elif train_val_test == 2:
            self.index_location = "index_test.pkl"
        else:
            self._logger.error("Argument to Data class must be 0 or 1 or 2 (train,val,test)")


    ################################ INDEX/DATA LOADING FUNCTIONS #############################


    def get_all_content(self,name):
        res = name.split('_')
        return int(res[1]), int(res[3]), int(res[5]), int(res[7]), int(res[9])

    def get_content(self, name, content):

        res = name.split('_')
        if content == 's':
            return int(res[1])
        if content == 'act':
            return int(res[3])
        if content == 'subact':
            return int(res[5])
        if content == 'ca':
            return int(res[7])
        if content == 'fno':
            return int(res[9])

    def get_name(self,s,act,sub,ca,fno):

        '{:04d}'.format(act)
        subdir="s_%s_act_%s_subact_%s_ca_%s/" % ('{:02d}'.format(s), '{:02d}'.format(act),
                                                     '{:02d}'.format(sub),'{:02d}'.format(ca))
        name="s_%s_act_%s_subact_%s_ca_%s_%s.jpg" % ('{:02d}'.format(s), '{:02d}'.format(act),
                                                     '{:02d}'.format(sub),'{:02d}'.format(ca),
                                                     '{:06d}'.format(fno))
        path=os.path.join(h36m_location, subdir, name)
        if not file_exists(path):
            self._logger.error("file found by path %s does not exist" % path)

        return path, name

    def create_index_file_subject(self, subj_list, sampling):

        self._logger.info('Indexing dataset...')
        index = []
        # get list of sequences
        names, paths = get_sub_dirs(self.index_file_loc)
        for name, path in zip(names, paths):
            # check data to load
            sid = self.get_content(name, 's')
            if sid not in subj_list:
                continue
            f_data, _ = get_files(path, 'jpg')
            #add only sequences sampled
            for fid, f_path in enumerate(f_data):
                if fid % sampling != 0:
                    continue
                file_details = {
                    'path': f_path,
                    'fid_sequence': fid,
                    'sequence_name': name
                }
                index.append(file_details)
        self.index_file = index

    def load_index_file(self):

        self._logger.info('Extract index file ...')
        file_path = os.path.join(self.index_file_loc, self.index_location)
        if not file_exists(file_path):
            self._logger.warning("index file to load does not exist")
        self.index_file = pkl.load(open( file_path, "rb" ))

    def save_index_file(self):

        if self.index_file is None:
            self._logger.error("File to save is None")
        self._logger.info('Saving index file...')
        file_path = os.path.join(self.index_file_loc, self.index_location)
        if file_exists(file_path):
            self._logger.error("Overwriting previous file")
        pkl.dump(self.index_file, open(file_path, "wb"))


    def load_metadata(self, subdir_path):
        path = os.path.join(subdir_path,"h36m_meta.mat")
        if not os.path.exists(path):
            self._logger.error('File %s not loaded', path)
            exit()
        metadata = {}
        data = sio.loadmat(path)
        metadata['joint_world'] = data['pose3d_world']
        metadata['R'] = data['R']
        metadata['T'] = data['T']
        metadata['c'] = data['c']
        metadata['f'] = data['f']
        metadata['img_widths'] = data['img_width']
        metadata['img_heights'] = data['img_height']
        return metadata
    #############################################################
    ########IMAGE FUNCTIONS #####################################

    def extract_image(self, path):
        im = cv2.imread(path)
        im=im[:H36M_CONF.max_size, :H36M_CONF.max_size, :]
        im = im.astype(np.float32)
        im /= 256
        return im








