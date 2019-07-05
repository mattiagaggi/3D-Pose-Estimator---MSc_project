
import numpy as np
from numpy.random import randint

from utils.utils_H36M.common import H36M_CONF, ENCODER_DECODER_PARAMS
from utils.utils_H36M.preprocess import Data_Base_class
from utils.utils_H36M.transformations import bounding_box_pixel, get_patch_image, cam_pointing_root, rotate_z, transform_2d_joints, world_to_pixel, world_to_camera
from utils.utils_H36M.visualise import Drawer
from matplotlib import pyplot as plt
import sys


class Data_Encoder_Decoder(Data_Base_class):

    def __init__(self,train_val_test = 0,
                 sampling = ENCODER_DECODER_PARAMS.encoder_decoder.sampling,
                 max_epochs = 1,
                 create_index_file = True,
                 index_file_content = 's',
                 index_file_list = [1]):

        super().__init__(train_val_test,sampling, max_epochs)

        if create_index_file:
            self.create_index_file(index_file_content, index_file_list)
        else:
            self.load_index_file()
        self.camera_list = [1,2,3,4]
        self.index_camera = 0

    def check_previous_image(self, s):
        same_backgrounds = False
        if self.previous_chache is not None:
            s_p, act_p, subact_p, ca_p, fno_p = self.previous_chache
            if s_p == s:
                same_backgrounds=True
        return same_backgrounds

    def testing(self,joints_world, imwarped, background_warped, R, T, f, c, trans):

        joint_px, center = world_to_pixel(
            joints_world,
            H36M_CONF.joints.root_idx,
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




    def extract_info(self,metadata, background,s, act, subact, ca, fno):

        background = background[ca-1,...]
        R = metadata['R']
        T = metadata['T']
        f = metadata['f']
        c = metadata['c']
        joints_world = metadata['joint_world'][fno-1]
        im = self.load_image(s, act, subact,ca, fno)
        return im, joints_world, R, T, f, c, background

    # extracxt apperance image

    def patch_images(self,im,background,bbpx, rotation_angle):
        imwarped, trans = get_patch_image(im, bbpx,
                                          (ENCODER_DECODER_PARAMS.encoder_decoder.im_size,
                                           ENCODER_DECODER_PARAMS.encoder_decoder.im_size),
                                          rotation_angle)  # in rotation around z axis
        background_warped, _ = get_patch_image(background, bbpx,
                                          (ENCODER_DECODER_PARAMS.encoder_decoder.im_size,
                                           ENCODER_DECODER_PARAMS.encoder_decoder.im_size),
                                          rotation_angle)
        return imwarped, background_warped, trans

    def extract_all_info(self, metadata, background,s, act, subact, ca, fno, rotation_angle=None):

        im, joints_world, R, T, f, c, background = self.extract_info(metadata, background,s, act, subact,ca, fno)
        bbpx = bounding_box_pixel(joints_world,H36M_CONF.joints.root_idx, R, T, f,c)
        plt.figure()
        plt.imshow(im)
        im, background, trans = self.patch_images(im,background,bbpx, rotation_angle)
        R_centre = cam_pointing_root(joints_world, H36M_CONF.joints.root_idx, H36M_CONF.joints.number, R, T)
        if rotation_angle is not None:
            R_centre = np.dot(rotate_z(rotation_angle), R_centre)
        R_pointing_centre = np.dot( R_centre, R)
        self.testing(joints_world, im, background, R, T, f, c, trans)

        return im, R_pointing_centre, background, joints_world


    def extract_all_info_memory_background(self,s, act, subact, ca, fno):
        metadata = self.all_metadata[s][act][subact][ca]
        im, R, background, joints = self.extract_all_info(metadata, self.previous_background, s, act, subact, ca,fno)
        return im, R, background, joints

    def update_stored_info(self,s, act, subact, ca, fno):
        self.previous_chache = s, act, subact, ca, fno


    def increase_iteration(self, current_camera):
        self.index_camera += 1
        if self.index_camera == len(self.camera_list):
            self.index_camera = 0
            self.increase_fno()
        if self.index_camera == current_camera:
            return self.increase_iteration(current_camera)
        else:
            return self.camera_list[self.index_camera]


    def return_view(self):
        return self.camera_list[self.index_camera]

    def return_next_view(self, ca):
        self.index_camera+=1
        if self.index_camera >= len(self.camera_list):
            self.index_camera = 0
        if self.camera_list[self.index_camera] == ca:
            return self.return_next_view(ca)
        else:
            return self.return_view()




    def return_random_from_list(self,lst, element_to_compare = None):

        new_index = randint(len(lst))
        assert len(lst)>1
        if element_to_compare is not None:
            while lst[new_index] == element_to_compare:
                new_index = randint(len(lst))
        return lst[new_index]




    def return_apperance_contents(self,s,act,ca):

        act_list = list(self.index_file[s].keys())
        new_act = self.return_random_from_list(act_list, act)
        subact_list = list(self.index_file[s][new_act].keys())
        new_subact = self.return_random_from_list(subact_list)
        ca_list = list(self.index_file[s][new_act][new_subact].keys())
        new_ca =  self.return_random_from_list(ca_list, ca)
        new_ca2 = self.return_random_from_list(ca_list, new_ca)
        fno_list = list(self.index_file[s][new_act][new_subact][new_ca].keys())
        new_fno = self.return_random_from_list(fno_list)
        return new_act, new_subact, new_ca, new_ca2, new_fno

    def return_main_batch(self):

        s, act, subact, ca, fno = self.return_current_file()
        same_backgrounds = self.check_previous_image(s)
        self.load_memory_backgrounds_image(s, act, subact, ca, fno,same_backgrounds)
        print("1",ca)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, act, subact, ca, fno)
        ca_view = self.return_next_view(ca)
        print("2",ca_view)
        imT, RT, backgroundT, jointsT = self.extract_all_info_memory_background(s, act, subact, ca_view, fno)
        return im1, R1, background1, joints1, imT, RT, backgroundT, jointsT

    def return_secondary_batch(self):
        s, act, subact, ca, fno = self.return_current_file()
        new_act, new_subact, new_ca, new_ca2, new_fno = self.return_apperance_contents(s,act,ca)
        print("3",new_ca)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca, new_fno)
        print("4",new_ca2)
        imT, R2T, background1T, jointsT = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca2, new_fno)
        return im1, R1, background1, joints1, imT, R2T, background1T, jointsT

    def testing_all(self):

        if self._current_epoch is None:
            self.iteration_start()
        if self._current_epoch >= self._max_epochs:
            self._logger.info("max epochs reached")
            return None
        s, act, subact, ca, fno = self.return_current_file()
        print(self.return_current_file())
        if s == 1 and act == 2 and subact == 1 and ca == 3 and fno == 65:
            self.return_main_batch()
            self.return_secondary_batch()
        self.update_stored_info(s, act, subact, ca, fno)
        print(self.increase_iteration(ca))



if __name__=="__main__":
    a=Data_Encoder_Decoder()
    for i in range(10000):
        a.testing_all()

