
import os
import pickle as pkl
import scipy.io as sio

import cv2
import numpy as np


from data.directories_location import index_location, h36m_location, backgrounds_location
from utils.utils_H36M.common import H36M_CONF
from utils.io import get_sub_dirs,get_files,file_exists
from sample.base.base_dataset import BaseDataset,SubSet




class Data_Base_class(BaseDataset):

    def __init__(self,
                 sampling,
                 subset = SubSet.train,
                 index_as_dict = False,
                 index_location = index_location,
                 h36m_location = h36m_location,
                 background_location = backgrounds_location,
                 get_intermediate_frames = False):

        super().__init__()



        self.index_file_loc = index_location
        self.h_36m_loc = h36m_location
        self.background_location = background_location
        self.sampling = sampling
        self.subset = subset
        self.get_intermediate_frames = get_intermediate_frames
        if self.sampling ==1 and get_intermediate_frames:
            self._logger.error("sampling can't be one if we want intermediate frames")

        self.index_as_dict = index_as_dict
        self.index_file = None
        self.previous_chache = None
        self.previous_background = None
        self.all_metadata = {}

        if self.index_as_dict:

            #once we create the index file we keep track of the current image being looked at
            #using  lists self.s_tot.... and indices in list self.current_s
            self.s_tot, self.act_tot, self.subact_tot, self.ca_tot, self.fno_tot = \
                None, None, None, None, None
            self.current_s, self.current_act, self.current_subact, self.current_ca, self.current_fno = \
                None, None, None, None, None



        if self.subset==SubSet.train:
            self.index_name = "index_train.pkl"
        elif self.subset==SubSet.val:
            self.index_name = "index_val.pkl"
        elif self.subset==SubSet.test:
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
            self._logger.t("Error in parsing %s for content %s" % (name, content))


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


    def append_index_to_dic(self,dic,s, act, subact, ca, fno):
        """
        transform self.index_file in nested dictionary such that
        self.index_file[s][act][subact][ca][fno]=path
        :param dic: dictionary
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
        if s not in dic:
            dic[s] = {act: {
                            subact: {
                                     ca: {
                                            fno: path}
                        }}}
        else:
            if act not in dic[s].keys():
                dic[s][act] = {subact: {
                                        ca: {
                                             fno: path}
                }}
            else:
                if subact not in dic[s][act].keys():
                    dic[s][act][subact] = {ca: {fno: path}}
                else:
                    if ca not in dic[s][act][subact].keys():
                        dic[s][act][subact][ca] = {fno: path}
                    else:
                        if fno not in dic[s][act][subact][ca].keys():
                            dic[s][act][subact][ca][fno] = path
                        else:
                            self._logger.error(" adding path %s twice " % path)
        return dic


    def append_index_to_list(self, dic,s, act, subact, ca, fno):
        path, _, _ = self.get_name(s, act, subact, ca, fno)
        if not file_exists(path):
            self._logger.error("file found by path %s does not exist" % path)
        dic.append([s,act,subact,ca,fno])
        return dic


    def subsample_fno(self, index_as_list, percent, lower):
        dic={}
        for i in index_as_list:
            s, act, subact, ca, fno=i
            dic = self.append_index_to_dic(dic,s, act, subact, ca, fno)
        new_index_file=[]
        for s in dic.keys():
            for act in dic[s].keys():
                for subact in dic[s][act].keys():
                    for ca in dic[s][act][subact].keys():
                        max_fno = np.max(list(dic[s][act][subact][ca].keys()))
                        for fno in dic[s][act][subact][ca].keys():
                            if lower:
                                if fno < int(max_fno*percent):
                                    new_index_file.append([s, act, subact, ca, fno])
                            else:
                                if fno >= int(max_fno*percent):
                                    new_index_file.append([s, act, subact, ca, fno])
        return new_index_file

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

    def append_metadata(self, s, act, subact, ca):
        """
        transform self.all_metadata in nested dictionary such that
        self.all_metadata[s][act][subact][ca] = metadata
        :param s: subject
        :param act: act
        :param subact: ...
        :param ca: ...

        :return:
        """
        fno = 0 # not used only needed for get_name()
        _, _, path = self.get_name(s, act, subact, ca, fno)
        metadata = self.load_metadata(path)
        if s not in self.all_metadata.keys():
            self.all_metadata[s] = {act: {
                                    subact: {
                                        ca: metadata
                                    }}}
        else:
            if act not in self.all_metadata[s].keys():
                self.all_metadata[s][act] = {subact: {
                                                ca: metadata
                }}
            else:
                if subact not in self.all_metadata[s][act].keys():
                    self.all_metadata[s][act][subact] = {ca: metadata}
                else:
                    if ca not in self.all_metadata[s][act][subact].keys():
                        self.all_metadata[s][act][subact][ca] = metadata



    def create_index_file(self, contents,content_lists):
        """
        creates nested dictionary from function above self.index_file[s][act][subact][ca][fno]=path
        :param content: one of: 's', 'act', 'subact' ,'ca', 'fno'
        :param content_list: list of contents
        :param sampling: sampling of fno
        """

        self._logger.info('Indexing dataset...')
        self.all_metadata = {}
        if self.index_as_dict:
            self.index_file = {}
        else:
            self.index_file = []
        # get list of sequences
        names, paths = get_sub_dirs(self.h_36m_loc)
        for name, path in zip(names, paths):
            # check data to load
            breaking = False
            for i,content in enumerate(contents):
                if self.get_content(name, content) not in content_lists[i]:
                    breaking = True
                    continue
            if breaking:
                continue
            s,act,subact,ca = self.get_all_content_file_name(name, file = False)
            self.append_metadata(s,act,subact,ca)
            _, file_names = get_files(path, 'jpg')
            for name in file_names:  # add only sequences sampled
                s, act, subact, ca, fno = self.get_all_content_file_name(name, file=True)
                if not self.get_intermediate_frames:

                    if (fno-1) % self.sampling != 0 and self.sampling!=1: # starts from 1
                        continue
                else:
                    if (fno+ self.sampling//2) % self.sampling != 1: # starts from 1
                        continue
                if self.index_as_dict:
                    self.index_file=\
                        self.append_index_to_dic(self.index_file, s, act, subact, ca, fno)
                else:
                    self.index_file=\
                        self.append_index_to_list(self.index_file, s, act, subact, ca, fno)


    def load_index_file(self):

        self._logger.info('Extract index file ... Note sampling might not correspond')
        file_path = os.path.join(self.index_file_loc, self.index_name)
        if not file_exists(file_path):
            self._logger.warning("index file to load does not exist")
        file_indices=pkl.load(open(file_path, "rb"))
        self.index_file = file_indices[0]
        self.all_metadata = file_indices[1]

    def save_index_file(self):

        if self.index_file is None:
            self._logger.error("File to save is None")
        self._logger.info('Saving index file...')
        file_path = os.path.join(self.index_file_loc, self.index_name)
        if file_exists(file_path):
            self._logger.info("Overwriting previous file")
        file_indices =[self.index_file, self.all_metadata]
        pkl.dump(file_indices, open(file_path, "wb"))


    #############################index as dict functions##################
    def reset(self, type):
        """
        resetting the tracking variables in the nested dictionary
        :param type: s, act, subact.....
        """
        if type=='s':
            self.s_tot=list(self.index_file.keys())
            self.current_s = 0
            self.reset('act')
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

    ############################# end index as dict functions##################
    def iteration_start(self):
        if self.index_file is None:
            self._logger.error("Can't start iteration if index file is None ")
        self._logger.info("New Epoch")
        if self.index_as_dict:
            self.reset('s')




    def load_backgrounds(self, s):
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

    def load_backgrounds_image(self,s,same_backgrounds=False):

        if not same_backgrounds:
            back = self.load_backgrounds(s)
        else:
            back = self.previous_background
        return back


    def load_memory_backgrounds_image(self,s, same_backgrounds=False):
        if self.previous_background is None:
            same_background_= False
        else:
            same_background_=same_backgrounds
        self.previous_background = self.load_backgrounds_image(s, same_background_)



    #############################################################
    ########IMAGE FUNCTIONS #####################################

    def extract_image(self, path):
        im = cv2.imread(path)
        im=im[:H36M_CONF.max_size, :H36M_CONF.max_size, :]
        im = im.astype(np.float32)
        im /= 256
        return im

    def load_image(self,s,act, subact, ca, fno):

        if self.index_as_dict:
            path = self.index_file[s][act][subact][ca][fno]
        else:
            path, _, _= self.get_name(s,act,subact,ca,fno)
        if not file_exists(path):
            self._logger.error("path not loaded %s" % path)
        else:
            return self.extract_image(path)

    def extract_info(self,metadata, background,s, act, subact, ca, fno):

        background = background[ca-1,...]
        R = metadata['R']
        T = metadata['T']
        f = metadata['f']
        c = metadata['c']
        joints_world = metadata['joint_world'][fno-1]
        im = self.load_image(s, act, subact,ca, fno)
        return im, joints_world, R, T, f, c, background















