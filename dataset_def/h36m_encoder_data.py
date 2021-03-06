
import numpy as np
from numpy.random import randint
from random import shuffle

from sample.config.data_conf import PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from utils.trans_numpy_torch import encoder_dictionary_to_pytorch
from utils.utils_H36M.transformations import bounding_box_pixel, get_patch_image, cam_pointing_root, rotate_z, transform_2d_joints, world_to_pixel
from utils.utils_H36M.common import H36M_CONF
from utils.utils_H36M.visualise import Drawer
from matplotlib import pyplot as plt



class Data_3dpose(Data_Base_class):
    def __init__(self,
                 args,
                 sampling,
                 index_file_content=['s'],
                 index_file_list=[[1]],
                 randomise=True,
                 get_intermediate_frames = False,
                 subsampling_fno = 0,
                 no_apperance=False):
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
        self.no_apperance = no_apperance
        if subsampling_fno==0:
            pass
        elif subsampling_fno==1:
            self.index_file=self.subsample_fno(self.index_file, 0.75, lower=True)
        elif subsampling_fno==2:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=False)
        else:
            self._logger.error("Subsampling not understood")
        #self._logger.info("Only from 1 to 2")
        for i in self.index_file:
            s,act,subact,ca,fno = i
            for ca2 in range(1,5):
                if (ca2 != ca) and (ca2 in list(self.all_metadata[s][act][subact].keys())):###############
                #if ca==1 and ca2==2:
                    self.index_file_cameras.append([s,act,subact,ca,fno,ca2])
        if self.randomise:
            shuffle(self.index_file_cameras)

        self.invert_segments = list(range(self.batch_size//2, self.batch_size)) + list(range(self.batch_size//2))
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
                        metadata = metadata - np.reshape(metadata[:, H36M_CONF.joints.root_idx, :],
                                                         (metadata.shape[0], 1, metadata.shape[2]))
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

    def check_previous_background(self, s):
        same_backgrounds = False
        if self.previous_chache is not None:
            s_p, act_p, subact_p, ca_p, fno_p = self.previous_chache
            if s_p == s:
                same_backgrounds=True
        return same_backgrounds

    def testing(self,joints_world, imwarped, background_warped, R, T, f, c, trans):

        joint_px= world_to_pixel(
            joints_world,
            H36M_CONF.joints.number, R,T,f,c
        )
        plt.figure()
        b = Drawer()
        plt.figure()
        b_im=b.get_image(background_warped)
        plt.imshow(b_im)
        plt.figure()
        ax = plt.subplot()
        trsf_joints, vis = transform_2d_joints(joint_px, trans)
        ax = b.pose_2d(ax, imwarped, trsf_joints[:, :-1])
        plt.show()

    def crop_img(self, img, bbpx,rotation_angle):
        imwarped, trans = get_patch_image(img, bbpx,
                                          (PARAMS.data.im_size,
                                           PARAMS.data.im_size),
                                          rotation_angle)
        return imwarped, trans


    def extract_all_info(self, metadata, background,s, act, subact, ca, fno,):

        rotation_angle = None
        im, joints_world, R, T, f, c = self.extract_info(metadata, s, act, subact,ca, fno)
        bbpx = bounding_box_pixel(joints_world,H36M_CONF.joints.root_idx, R, T, f,c)
        im,  trans = self.crop_img(im, bbpx, rotation_angle)
        background, _ = self.crop_img(background, bbpx, rotation_angle)
        R_centre = cam_pointing_root(joints_world, H36M_CONF.joints.root_idx, H36M_CONF.joints.number, R, T)
        if rotation_angle is not None:
            R_centre = np.dot(rotate_z(rotation_angle), R_centre)
        R_pointing_centre = np.dot( R_centre, R)
        #self.testing(joints_world, im, background, R, T, f, c, trans)

        return im, R_pointing_centre,background, joints_world


    def extract_all_info_memory_background(self,s, act, subact, ca, fno):
        #print(s,act,subact,ca) #11 2 2 4
        metadata = self.all_metadata[s][act][subact][ca]
        background = self.previous_background[ca-1]
        im, R, background, joints = self.extract_all_info(metadata, background, s, act, subact, ca,fno)
        return im, R, background, joints

    def update_stored_info(self,s, act, subact, ca, fno):
        self.previous_chache = s, act, subact, ca, fno


    def return_random_from_list(self,lst, element_to_compare = None):

        new_index = randint(len(lst))
        if element_to_compare is not None:
            assert len(lst) > 1
            while lst[new_index] == element_to_compare:
                new_index = randint(len(lst))
        return lst[new_index]


    def return_apperance_contents(self,s,act):


        act_list = list(self.all_metadata[s].keys())
        if len(act_list) < 2:
            self._logger.error("Can't have appearance data if only one act selected")
        new_act = self.return_random_from_list(act_list, act)
        subact_list = list(self.all_metadata[s][new_act].keys())
        new_subact = self.return_random_from_list(subact_list)
        ca_list = list(self.all_metadata[s][new_act][new_subact].keys())
        new_ca =  self.return_random_from_list(ca_list)
        new_ca2 = self.return_random_from_list(ca_list, new_ca)
        fno_number = self.all_metadata[s][new_act][new_subact][new_ca]['joint_world'].shape[0]
        new_fno = np.random.randint(fno_number//self.sampling) * self.sampling + 1
        return new_act, new_subact, new_ca, new_ca2, new_fno

    def return_main_data(self,index):

        s, act, subact, ca, fno, ca2 = self.index_file_cameras[index]
        #print(s, act, subact, ca, fno, ca2)
        same_backgrounds = self.check_previous_background(s)
        self.load_memory_backgrounds_image(s,same_backgrounds)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, act, subact, ca, fno)
        imT, RT, backgroundT, jointsT = self.extract_all_info_memory_background(s, act, subact, ca2, fno)

        return im1, R1, background1, joints1, imT, RT, backgroundT, jointsT

    def return_apperance_data(self,index):
        s, act, _, _, _,_ = self.index_file_cameras[index]
        new_act, new_subact, new_ca, new_ca2, new_fno = self.return_apperance_contents(s,act)
        #print(s,new_act, new_subact, new_ca, new_ca2, new_fno)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca, new_fno)
        imT, RT, background1T, jointsT = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca2, new_fno)
        return im1, R1, background1, joints1, imT, RT, background1T, jointsT

    def testing_all(self,index):

        s, act, subact, ca, fno,_ = self.index_file_cameras[index]
        if s == 1 and act == 2 and subact == 1 and ca == 3 and fno == 65:
            self.return_main_data(index)
            self.return_apperance_data(index)
        self.update_stored_info(s, act, subact, ca, fno)


    def process_data(self,data):
        im, R, background, joints, imT, RT, backgroundT, jointsT = data
        #rot = np.dot(RT, R.T)
        return im, R, RT, backgroundT, imT, joints


    def return_batch(self, index):

        batch_1=self.return_main_data(index)
        if not self.no_apperance:
            batch_2 = self.return_apperance_data(index)
        s, act, subact, ca, fno, ca2 = self.index_file_cameras[index]
        self.update_stored_info(s, act, subact, ca, fno)
        if not self.no_apperance:
            return self.process_data(batch_1), self.process_data(batch_2), np.array([s, act, subact, ca, fno, ca2])
        else:
            return self.process_data(batch_1), None, np.array([s, act, subact, ca, fno, ca2])

    def create_dic_in(self):

        dic={ 'im_in' : [],
              'background_target': [],
              'R_world_im': [],
              'R_world_im_target': [],
              'invert_segments': list(range(self.batch_size//2, self.batch_size)) + list(range(self.batch_size//2))

        }
        return dic

    def create_dic_out(self):

        dic = { 'joints_im': [],
                'im_target': []
        }
        return dic

    def create_all_dic(self):
        dic_in = self.create_dic_in()
        dic_out = self.create_dic_out()
        dic_in_app = self.create_dic_in()
        dic_out_app = self.create_dic_out()
        return dic_in, dic_out, dic_in_app, dic_out_app

    def update_dic_in(self, dic, im, rot, rotT, backgroundT):
        dic['im_in'].append(im)
        dic['background_target'].append(backgroundT)
        dic['R_world_im'].append(rot)
        dic['R_world_im_target'].append(rotT)
        return dic

    def update_dic_out(self, dic, imT, joints):
        dic['joints_im'].append(joints)
        dic['im_target'].append(imT)
        return dic

    def joint_dic_in(self, dic1, dic2):
        assert self.batch_size//2 == len(dic1['im_in'])
        #assert self.batch_size//2 == len(dic2['im_in'])
        new_dic = {}
        if dic2 is not None:
            for key in dic1.keys():
                new_dic[key] = np.stack(dic1[key] + dic2[key], axis=0)
        else:
            for key in dic1.keys():
                new_dic[key] = np.stack(dic1[key], axis=0)
        new_dic['im_in'] = np.transpose(new_dic['im_in'], axes=[0,3,1,2])
        new_dic['background_target'] = np.transpose(new_dic['background_target'], axes=[0,3,1,2])
        new_dic['invert_segments'] = self.invert_segments
        return new_dic

    def join_dic_out(self, dic1, dic2):
        new_dic = {}
        if dic2 is not None:
            for key in dic1.keys():
                new_dic[key] = np.stack(dic1[key] + dic2[key], axis=0)
        else:
            for key in dic1.keys():
                new_dic[key] = np.stack(dic1[key], axis=0)
        new_dic['im_target'] = np.transpose(new_dic['im_target'], axes=[0, 3, 1, 2])
        N, J, T = new_dic['joints_im'].shape
        new_dic['joints_im'] -= np.reshape(new_dic['joints_im'][:, H36M_CONF.joints.root_idx, :], (N, 1, T))
        return new_dic



    def track_epochs(self):

        self.elements_taken += self.batch_size // 2
        if self.elements_taken //(self.batch_size // 2) == self.__len__():
            self._logger.info("New Epoch reset elements taken")
            self._current_epoch += 1
            self.elements_taken = 0
            if self.randomise:
                shuffle(self.index_file_cameras)

    def __len__(self):
        return len(self.index_file_cameras) // (self.batch_size//2)

    def __getitem__(self, item):
        """
        This looks horrible!
        :param item:
        :return:
        """
        index = item * (self.batch_size // 2)
        dic_in, dic_out, dic_in_app, dic_out_app = self.create_all_dic()
        for i in range(self.batch_size // 2):
            batches = self.return_batch(index + i)
            im, R, RT, backgroundT, imT,joints = batches[0]
            self.update_dic_in(dic_in,im, R, RT, backgroundT)
            self.update_dic_out(dic_out, imT, joints)
            if not self.no_apperance:
                im, R, RT, backgroundT, imT, joints = batches[1]
                self.update_dic_in(dic_in_app, im, R, RT, backgroundT)
                self.update_dic_out(dic_out_app, imT, joints)
        if not self.no_apperance:
            dic_in = self.joint_dic_in(dic_in, dic_in_app)
            dic_out = self.join_dic_out(dic_out, dic_out_app)
        else:
            dic_in = self.joint_dic_in(dic_in, None)
            dic_out = self.join_dic_out(dic_out, None)
        self.track_epochs()
        return encoder_dictionary_to_pytorch(dic_in), encoder_dictionary_to_pytorch(dic_out)

