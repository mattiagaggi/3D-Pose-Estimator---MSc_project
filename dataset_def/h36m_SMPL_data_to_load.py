import numpy as np
from random import shuffle
import torch
from utils.utils_H36M.common import H36M_CONF
from sample.config.data_conf import PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from utils.utils_H36M.transformations import get_patch_image, bounding_box_pixel, cam_pointing_root, rotate_z
from utils.trans_numpy_torch import numpy_to_tensor_float, image_numpy_to_pytorch, numpy_to_long, tensor_to_numpy, numpy_to_tensor

class SMPL_Data_Load(Data_Base_class):

    def __init__(self,
                 sampling,
                 index_file_content=['s'],
                 index_file_list=[[1]],
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
        self.create_index_file(index_file_content, index_file_list)
        self.index_file_content = index_file_content
        self.index_file_list = index_file_list
        if subsampling_fno == 0:
            pass
        elif subsampling_fno == 1:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=True)
        elif subsampling_fno == 2:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=False)
        else:
            self._logger.error("Subsampling not understood")




    def crop_img(self, img, bbpx,rotation_angle):
        imwarped, trans = get_patch_image(img, bbpx,
                                          (PARAMS.data.im_size,
                                           PARAMS.data.im_size),
                                          rotation_angle)
        return imwarped, trans


    def extract_image_info(self, s, act, subact, ca, fno, rotation_angle=None):

        metadata = self.all_metadata[s][act][subact][ca]
        im, joints_world, R, T, f, c= self.extract_info(metadata, s, act, subact, ca, fno)
        bbpx = bounding_box_pixel(joints_world, H36M_CONF.joints.root_idx, R, T, f,c)
        im, trans = self.crop_img(im, bbpx, rotation_angle)
        R_centre = cam_pointing_root(joints_world, H36M_CONF.joints.root_idx, H36M_CONF.joints.number, R, T)
        if rotation_angle is not None:
            R_centre = np.dot(rotate_z(rotation_angle), R_centre)
        R_pointing_centre = np.dot( R_centre, R)
        return im, joints_world, R_pointing_centre


    def extract_masks_info(self,s,act,subact,ca,fno, rotation_angle=None):
        metadata = self.all_metadata[s][act][subact][ca]
        im, joints_world, R, T, f, c = self.extract_mask_info(metadata, s, act, subact, ca, fno)
        bbpx = bounding_box_pixel(joints_world, H36M_CONF.joints.root_idx, R, T, f, c)
        im, trans = self.crop_img(im, bbpx, rotation_angle)
        return im, R, T, f, c, trans


    def create_dictionary_data(self):
        dic = {
              "masks": {1: {},
                        2: {},
                        3: {},
                        4: {}
                        }
              }
        return dic


    def update_dic_with_image(self, dic, s, act, subact, ca, fno, rotation_angle):
        im, joints_world, R = self.extract_image_info(s, act, subact, ca, fno, rotation_angle=rotation_angle)
        dic['image'] = im
        dic['joints_im'] = joints_world
        dic['R'] = R
        return dic


    def update_dic_with_mask(self,dic, i,  s, act, subact, mask_number, fno, rotation_angle):
        im, R, T, f, c, trans = self.extract_masks_info(s,act,subact,mask_number,fno,rotation_angle)
        dic['masks_'+str(mask_number)+'_idx'] = i
        dic['masks_' + str(mask_number) + '_image'] = im
        dic['masks_' + str(mask_number) + '_R'] = R
        dic['masks_' + str(mask_number) + '_T'] = T
        dic['masks_' + str(mask_number) + '_f'] = f
        dic['masks_' + str(mask_number) + '_c'] = c
        dic['masks_' + str(mask_number) + '_transcrop'] = trans
        return dic


    def __len__(self):
        return len(self.index_file)


    def __getitem__(self, item):
        dic= self.create_dictionary_data()
        rotation_angle = 0
        s, act, subact, ca, fno = self.index_file[item]
        dic = self.update_dic_with_image(dic,s, act, subact, ca, fno, rotation_angle)
        for mask_number in range(1,5):
            if mask_number in self.all_metadata[s][act][subact].keys():
                dic = self.update_dic_with_mask(dic, item, s, act, subact, mask_number, fno, rotation_angle)
        return dic




