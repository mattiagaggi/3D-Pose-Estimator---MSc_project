
import os
import pickle as pkl
import scipy.io as sio

import cv2
import numpy as np


from data.directories_location import index_location, h36m_location, backgrounds_location
from utils.utils_H36M.common import H36M_CONF
from utils.io import get_sub_dirs,get_files,file_exists,get_parent
from logger.console_logger import ConsoleLogger



class Data_Base_class:

    def __init__(self, train_val_test=0,
                 sampling=64,
                 max_epochs=1,
                 index_location = index_location,
                 h36m_location = h36m_location,
                 background_location = backgrounds_location):

        self._max_epochs=max_epochs

        self.index_file_loc = index_location
        self.h_36m_loc = h36m_location
        self.background_location = background_location
        self.sampling = sampling

        logger_name = '{}'.format(self.__class__.__name__)
        self._logger = ConsoleLogger(logger_name)



        self._current_epoch = None
        self.current_metadata = None
        self.current_background= None

        #once we create the index file we keep track of the current image being looked at
        #using  lists self.s_tot.... and indices in list self.current_s
        self.index_file = None
        self.s_tot, self.act_tot, self.subact_tot, self.ca_tot, self.fno_tot = \
            None, None, None, None, None
        self.current_s, self.current_act, self.current_subact, self.current_ca, self.current_fno = \
            None, None, None, None, None

        if train_val_test == 0:
            self.index_name = "index_train.pkl"
        elif train_val_test == 1:
            self.index_name = "index_val.pkl"
        elif train_val_test == 2:
            self.index_name = "index_test.pkl"
        else:
            self._logger.error("Argument to Data class must be 0 or 1 or 2 (train,val,test)")



    ################################ INDEX/DATA LOADING FUNCTIONS #############################


    def get_all_content_file_name(self, name, file = True):
        """
        :param name: string name of subdirectory or file
        :param file: true if file otherwise subdirectory
        :return: int content
        """

        res = name.split('_')
        if file:
            return int(res[1]), int(res[3]), int(res[5]), int(res[7]), int(res[8])
        else:
            return int(res[1]), int(res[3]), int(res[5]), int(res[7])


    def get_content(self, name, content, file=True):
        """
        :param name: string name of subdirectory or file
        :param content: type of content
        :param file: true if file otherwise subdirectory
        :return: int content
        """

        res = name.split('_')
        if content == 's':
            return int(res[1])
        elif content == 'act':
            return int(res[3])
        elif content == 'subact':
            return int(res[5])
        elif content == 'ca':
            return int(res[7])
        elif file and content == 'fno':
            return int(res[8])
        else:
            self._logger.error("Error in parsing %s for content %s" % (name, content))


    def get_name(self,s,act,sub,ca,fno):
        """
        :param s: subject
        :param act: act
        :param sub: subact
        :param ca: camera
        :param fno: sequence number
        :return: path of the file and file name as strings
        """

        '{:04d}'.format(act)
        subdir="s_%s_act_%s_subact_%s_ca_%s" % ('{:02d}'.format(s), '{:02d}'.format(act),
                                                     '{:02d}'.format(sub),'{:02d}'.format(ca))
        name="s_%s_act_%s_subact_%s_ca_%s_%s.jpg" % ('{:02d}'.format(s), '{:02d}'.format(act),
                                                     '{:02d}'.format(sub),'{:02d}'.format(ca),
                                                     '{:06d}'.format(fno))

        parent_path=os.path.join(h36m_location, subdir)
        path=os.path.join(parent_path,name)

        return path, name, parent_path


    def append_index(self, dictionary, s, act, subact, ca, fno):
        """
        transform self.index_file in nested dictionary such that
        self.index_file[s][act][subact][ca][fno]=path
        :param s: subject
        :param act: act
        :param subact: ...
        :param ca: ...
        :param fno: sequence number
        :return:
        """
        path, _, _ = self.get_name(s, act, subact, ca, fno)
        if not file_exists(path):
            self._logger.error("file found by path %s does not exist" % path)
        if s not in self.index_file.keys():
            self.index_file[s] = {act: {
                                    subact: {
                                        ca: {
                                            fno: path}
                                    }}}
        else:
            if act not in self.index_file[s].keys():
                self.index_file[s][act] = {subact: {
                                                ca: {
                                                    fno: path}
                }}
            else:
                if subact not in self.index_file[s][act].keys():
                    self.index_file[s][act][subact] = {ca: {fno: path}}
                else:
                    if ca not in self.index_file[s][act][subact].keys():
                        self.index_file[s][act][subact][ca] = {fno: path}
                    else:
                        if fno not in self.index_file[s][act][subact][ca].keys():
                            self.index_file[s][act][subact][ca][fno] = path
                        else:
                            self._logger.error(" adding path %s twice " % path)
        return dictionary




    def create_index_file(self, content,content_list):
        """
        creates nested dictionary from function above self.index_file[s][act][subact][ca][fno]=path
        :param content: one of: 's', 'act', 'subact' ,'ca', 'fno'
        :param content_list: list of contents
        :param sampling: sampling of fno
        """

        self._logger.info('Indexing dataset...')
        self.index_file = {}
        # get list of sequences
        names, paths = get_sub_dirs(self.h_36m_loc)
        for name, path in zip(names, paths):
            # check data to load
            if self.get_content(name, content) not in content_list:
                continue
            _, file_names = get_files(path, 'jpg')
            for name in file_names:  # add only sequences sampled
                s, act, subact, ca, fno = self.get_all_content_file_name(name, file=True)
                if fno % self.sampling != 1: # starts from 1
                    continue
                self.index_file=self.append_index(self.index_file,s, act, subact, ca, fno)

    def load_index_file(self):

        self._logger.info('Extract index file ... Note sampling might not correspond')
        file_path = os.path.join(self.index_file_loc, self.index_name)
        if not file_exists(file_path):
            self._logger.warning("index file to load does not exist")
        self.index_file = pkl.load(open( file_path, "rb" ))

    def save_index_file(self):

        if self.index_file is None:
            self._logger.error("File to save is None")
        self._logger.info('Saving index file...')
        file_path = os.path.join(self.index_file_loc, self.index_name)
        if file_exists(file_path):
            self._logger.info("Overwriting previous file")
        pkl.dump(self.index_file, open(file_path, "wb"))



    def reset(self, type):
        """
        resetting the tracking variables in the nested dictionary
        :param type: s, act, subact.....
        """
        if type=='s':
            self._logger.info("New Epoch")
            self.s_tot=list(self.index_file.keys())
            self.current_s = 0
            self.reset('act')
            if self._current_epoch  is not None: #meaning this is not the first reset
                self._current_epoch +=1
        elif type=='act':
            self.act_tot = list(self.index_file[self.s_tot[self.current_s]].keys())
            self.current_act = 0
            self.reset('subact')
        elif type == 'subact':
            self.subact_tot = list(self.index_file[
                                       self.s_tot[self.current_s]][
                                       self.act_tot[self.current_act]].keys())
            self.current_subact = 0
            self.reset('ca')
        elif type == 'ca':
            self.ca_tot = list(self.index_file[
                                   self.s_tot[self.current_s]][
                                   self.act_tot[self.current_act]][
                                   self.subact_tot[self.current_subact]].keys())
            self.current_ca = 0
            self.reset('fno')
        elif type == 'fno':
            self.fno_tot = list(self.index_file[
                                self.s_tot[self.current_s]][
                                self.act_tot[self.current_act]][
                                self.subact_tot[self.current_subact]][
                                self.ca_tot[self.current_ca]].keys())
            self.current_fno = 0
        else:
            self._logger.error("Reset type not understood %s" % type)


    def iteration_start(self):
        if self.index_file is None:
            self._logger.error("Can't start iteration if index file is None ")
        self.reset('s')
        self._current_epoch=0

    def increase_s(self):
        self.current_s += 1
        if self.current_s >= len(self.s_tot):
            self.reset('s')
        else:
            self.reset('act')
        return self.s_tot[self.current_s], \
               self.act_tot[self.current_act], \
               self.subact_tot[self.current_subact], \
               self.ca_tot[self.current_ca], \
               self.fno_tot[self.current_fno]

    def increase_act(self):
        self.current_act += 1
        if self.current_act >= len(self.act_tot):
            return self.increase_s()
        else:
            self.reset('subact')
            return self.s_tot[self.current_s], \
                   self.act_tot[self.current_act], \
                   self.subact_tot[self.current_subact], \
                   self.ca_tot[self.current_ca], \
                   self.fno_tot[self.current_fno]

    def increase_subact(self):
        self.current_subact +=1
        if self.current_subact >= len(self.subact_tot):
            return self.increase_act()
        else:
            self.reset('ca')
            return self.s_tot[self.current_s], \
                   self.act_tot[self.current_act], \
                   self.subact_tot[self.current_subact], \
                   self.ca_tot[self.current_ca], \
                   self.fno_tot[self.current_fno]

    def increase_camera(self):
        self.current_ca +=1
        if self.current_ca >= len(self.ca_tot):
            return self.increase_subact()
        else:
            self.reset('fno')
            return self.s_tot[self.current_s],\
                   self.act_tot[self.current_act],\
                   self.subact_tot[self.current_subact],\
                   self.ca_tot[self.current_ca],\
                   self.fno_tot[self.current_fno]


    def increase_fno(self):
        self.current_fno += 1
        if self.current_fno >= len(self.fno_tot):
            return self.increase_camera()
        else:
            return self.s_tot[self.current_s],\
                   self.act_tot[self.current_act],\
                   self.subact_tot[self.current_subact],\
                   self.ca_tot[self.current_ca],\
                   self.fno_tot[self.current_fno]

    def return_current_file(self):
        if self._current_epoch is None:
            self._logger.error("Can't return status if iterations not initialised")
        else:
            return self.s_tot[self.current_s], \
                   self.act_tot[self.current_act], \
                   self.subact_tot[self.current_subact], \
                   self.ca_tot[self.current_ca], \
                   self.fno_tot[self.current_fno]

    def next_file_content(self):

        s,act,subact,ca,fno= self.increase_fno()
        return s,act,subact,ca,fno


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


    def load_background(self,s):
        """

        loads backgrounds obtained in get_background file
        :param s: subject
        :return: array shape [4,L,W,3]
        """
        path=os.path.join(self.background_location, "background_subject%s.npy" % s)
        if not os.path.exists(path):
            self._logger.error('File %s not loaded', path)
            exit()
        return np.load(path)


    def load_metadata_backgrounds_image(self,s, act, subact, ca, fno, same_metadata=False, same_backgrounds=False):
        path = self.index_file[s][act][subact][ca][fno]
        parent = get_parent(path)
        if same_metadata:
            metadata=self.current_metadata
        else:
            metadata=self.load_metadata(parent)
        if same_backgrounds:
            backgrounds = self.current_background
        else:
            backgrounds= self.load_backgrounds(s)
        return metadata, backgrounds


    #############################################################
    ########IMAGE FUNCTIONS #####################################

    def extract_image(self, path):
        im = cv2.imread(path)
        im=im[:H36M_CONF.max_size, :H36M_CONF.max_size, :]
        im = im.astype(np.float32)
        im /= 256
        return im






    def __next__(self):
        same_backgrounds, same_metadata = False, False
        if self._current_epoch is None:
            self.iteration_start()
            s, act, subact, ca, fno = self.return_current_file()
        elif self._current_epoch >= self._max_epochs:
            self._logger.info("max epochs reached")
            return None
        else:
            s_prev, act_prev, subact_prev, ca_prev,fno_prev = self.return_current_file()
            s, act, subact, ca, fno = self.next_file_content()
            if s == s_prev:
                same_backgrounds=True
                if act == act_prev and subact == subact_prev and ca == ca_prev:
                    same_metadata=True
        metadata, backgrounds = self.load_metadata_backgrounds(s, act, subact, ca, fno, same_metadata, same_backgrounds)
        #get image, rotate crop
        #get pther camera















    #cropping function(










