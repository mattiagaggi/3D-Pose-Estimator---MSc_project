
import numpy as np
from numpy.random import randint

from utils.utils_H36M.common import H36M_CONF
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from utils.utils_H36M.transformations import bounding_box_pixel, get_patch_image, cam_pointing_root, rotate_z, transform_2d_joints, world_to_pixel, world_to_camera
from utils.utils_H36M.visualise import Drawer
from matplotlib import pyplot as plt



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



    # extracxt apperance image

    def patch_images(self,im,background, bbpx, rotation_angle):
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
        im, background, trans = self.patch_images(im,background,bbpx, rotation_angle)
        R_centre = cam_pointing_root(joints_world, H36M_CONF.joints.root_idx, H36M_CONF.joints.number, R, T)
        if rotation_angle is not None:
            R_centre = np.dot(rotate_z(rotation_angle), R_centre)
        R_pointing_centre = np.dot( R_centre, R)
        #self.testing(joints_world, im, background, R, T, f, c, trans)

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
        if self.camera_list[self.index_camera] == current_camera:
            return self.increase_iteration(current_camera)


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

    def return_main_data(self):

        s, act, subact, ca, fno = self.return_current_file()
        same_backgrounds = self.check_previous_image(s)
        self.load_memory_backgrounds_image(s,same_backgrounds)
        #print("1",ca)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, act, subact, ca, fno)
        ca_view = self.return_next_view(ca)
        #print("2",ca_view)
        imT, RT, backgroundT, jointsT = self.extract_all_info_memory_background(s, act, subact, ca_view, fno)

        return im1, R1, background1, joints1, imT, RT, backgroundT, jointsT

    def return_apperance_data(self):
        s, act, subact, ca, fno = self.return_current_file()
        new_act, new_subact, new_ca, new_ca2, new_fno = self.return_apperance_contents(s,act,ca)
        #print("3",new_ca)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca, new_fno)
        #print("4",new_ca2)
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
            self.return_main_data()
            self.return_apperance_data()
        self.update_stored_info(s, act, subact, ca, fno)
        print(self.increase_iteration(ca))

    def process_data(self,data):
        im1, R1, background1, joints1, imT, RT, backgroundT, jointsT = data
        rot1 = np.dot(RT, R1.T)
        return im1, rot1, backgroundT, imT, joints1


    def return_batch(self):
        if self._current_epoch is None:
            self.iteration_start()
        if self._current_epoch >= self._max_epochs:
            self._logger.info("max epochs reached")
            return None
        batch_1=self.return_main_data()
        batch_2=self.return_apperance_data()
        s, act, subact, ca, fno = self.return_current_file()
        self.update_stored_info(s, act, subact, ca, fno)
        self.increase_iteration(ca)
        return self.process_data(batch_1), self.process_data(batch_2)


    def group_batches(self,n_batches,im1, rot1, backgroundT, imT, joints1,
                      im2, rot2, backgroundT2, imT2, joints2):
        assert n_batches == len(im1)
        assert n_batches == len(im2)
        dic = {
                'im_in': np.transpose(np.stack(im1+im2,axis=0), axes=[0,3,1,2]),
                'im_target' : np.transpose(np.stack(imT+imT2,axis=0), axes=[0,3,1,2]),
                'background_target' : np.transpose(np.stack(backgroundT+backgroundT2,axis=0), axes=[0,3,1,2]),
                'rot_im': np.stack(rot1 + rot2,axis=0),
                'joints_im': np.stack(joints1+joints2,axis=0),
                'invert_segments': list(range(n_batches,n_batches*2)) + list(range(n_batches)),
            }
        return dic


    def process_batches(self, n_batches):
        im1_tot, rot1_tot, backgroundT_tot, imT_tot, joints1_tot, \
        im2_tot, rot2_tot, backgroundT2_tot, imT2_tot, joints2_tot = \
            [], [], [], [], [], [], [], [], [], []
        for i in range(n_batches):
            all_b = self.return_batch()
            if all_b is None: # epochs are over
                return None
            else:
                im1, rot1, backgroundT, imT, joints1 = all_b[0]
                im2, rot2, backgroundT2, imT2, joints2 = all_b[1]
                im1_tot.append(im1)
                im2_tot.append(im1)
                rot1_tot.append(rot1)
                rot2_tot.append(rot2)
                backgroundT_tot.append(backgroundT)
                backgroundT2_tot.append(backgroundT2)
                imT_tot.append(imT)
                imT2_tot.append(imT2)
                joints1_tot.append(joints1)
                joints2_tot.append(joints2)
        dic = self.group_batches(n_batches,im1_tot, rot1_tot, backgroundT_tot, imT_tot, joints1_tot,
                                 im2_tot, rot2_tot, backgroundT2_tot, imT2_tot, joints2_tot )
        return dic




if __name__=="__main__":
    def check_data_feed():
        d = Data_Encoder_Decoder()
        oo = 0
        for i in range(10000):
            print(oo)
            oo += 1
            o, e = d.return_batch()
            print(d.previous_chache)
            if i == 500:
                for el in range(len(o)):
                    if o[el].shape[0] == 256:
                        plt.figure()
                        plt.imshow(o[el])

                        plt.show()
                for el in range(len(e)):
                    if e[el].shape[0] == 256:
                        plt.figure()
                        plt.imshow(e[el])
                        plt.show()

    def final_check():
        d = Data_Encoder_Decoder()
        oo = 0
        for i in range(300):
            print(oo)
            oo += 1
            dic = d.process_batches(10)
            print(d.previous_chache)
            if i == 250:
                for k in dic.keys():
                    print(k, dic[k].shape)
                    if dic[k].shape[2] == 256:
                        plt.imshow(np.transpose(dic[k][0, ...], axes=[1, 2, 0]))
                        plt.show()

    final_check()
