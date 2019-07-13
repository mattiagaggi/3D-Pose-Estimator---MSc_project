
import numpy as np
from numpy.random import randint
from numpy.random import normal
from random import shuffle

from utils.utils_H36M.common import H36M_CONF
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from dataset_def.trans_numpy_torch import encoder_dictionary_to_pytorch
from utils.utils_H36M.transformations import bounding_box_pixel, get_patch_image, cam_pointing_root, rotate_z, transform_2d_joints, world_to_pixel, world_to_camera
from utils.utils_H36M.visualise import Drawer
from matplotlib import pyplot as plt



class Data_Encoder_Decoder(Data_Base_class):

    def __init__(self,
                 sampling =10,
                 batch_size = 10,
                 index_file_content=['s'],
                 index_file_list=[[1]],
                 randomise=True,
                 get_intermediate_frames = False):

        super().__init__(sampling,get_intermediate_frames=get_intermediate_frames)

        self.batch_size = batch_size
        assert batch_size % 2 == 0

        self.create_index_file(index_file_content, index_file_list)
        self.index_file_content = index_file_content
        self.index_file_list = index_file_list
        self.randomise= randomise
        if self.randomise:
            self.index_file_cameras =[]
        self._logger.info("Warning you changed cameras here")
        for i in self.index_file:
            s,act,subact,ca,fno = i
            for ca2 in range(1,5):
                #if ca2 != ca :###############
                if ca==1 and ca2==2:
                    self.index_file_cameras.append([s,act,subact,ca,fno,ca2])

        if self.randomise:
            shuffle(self.index_file_cameras)
        self.elements_taken=0
        self._current_epoch=0


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


    def get_angle(self):
        rad = normal(scale=np.pi/10)
        return rad
    def extract_all_info(self, metadata, background,s, act, subact, ca, fno, rotation_angle=None):
        #rotation_angle = self.get_angle()
        im, joints_world, R, T, f, c, background = self.extract_info(metadata, background, s, act, subact,ca, fno)
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


    def return_random_from_list(self,lst, element_to_compare = None):

        new_index = randint(len(lst))
        assert len(lst)>1
        if element_to_compare is not None:
            while lst[new_index] == element_to_compare:
                new_index = randint(len(lst))
        return lst[new_index]


    def return_apperance_contents(self,s,act):


        act_list = list(self.all_metadata[s].keys())
        new_act = self.return_random_from_list(act_list, act)
        subact_list = list(self.all_metadata[s][new_act].keys())
        new_subact = self.return_random_from_list(subact_list)
        ca_list = list(self.all_metadata[s][new_act][new_subact].keys())
        new_ca =  self.return_random_from_list(ca_list)
        new_ca2 = self.return_random_from_list(ca_list, new_ca)
        fno_number = self.all_metadata[s][new_act][new_subact][new_ca]['joint_world'].shape[0]
        new_fno = np.random.randint(1,fno_number+1)
        return new_act, new_subact, new_ca, new_ca2, new_fno

    def return_main_data(self,index):

        s, act, subact, ca, fno, ca2 = self.index_file_cameras[index]
        #print(s, act, subact, ca, fno, ca2)
        same_backgrounds = self.check_previous_image(s)
        self.load_memory_backgrounds_image(s,same_backgrounds)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, act, subact, ca, fno)
        imT, RT, backgroundT, jointsT = self.extract_all_info_memory_background(s, act, subact, ca2, fno)

        return im1, R1, background1, joints1, imT, RT, backgroundT, jointsT

    def return_apperance_data(self,index):
        s, act, _, _, _,_ = self.index_file_cameras[index]
        new_act, new_subact, new_ca, new_ca2, new_fno = self.return_apperance_contents(s,act)
        #print(s,new_act, new_subact, new_ca, new_ca2, new_fno)
        im1, R1, background1, joints1 = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca, new_fno)
        imT, R2T, background1T, jointsT = self.extract_all_info_memory_background(s, new_act, new_subact, new_ca2, new_fno)
        return im1, R1, background1, joints1, imT, R2T, background1T, jointsT

    def testing_all(self,index):

        s, act, subact, ca, fno,_ = self.index_file_cameras[index]
        if s == 1 and act == 2 and subact == 1 and ca == 3 and fno == 65:
            self.return_main_data(index)
            self.return_apperance_data(index)
        self.update_stored_info(s, act, subact, ca, fno)


    def process_data(self,data):
        im1, R1, background1, joints1, imT, RT, backgroundT, jointsT = data
        rot1 = np.dot(RT, R1.T)
        return im1, rot1, backgroundT, imT, joints1




    def return_batch(self, index):

        batch_1=self.return_main_data(index)
        batch_2=self.return_apperance_data(index)
        s, act, subact, ca, fno, ca2 = self.index_file_cameras[index]
        self.update_stored_info(s, act, subact, ca, fno)

        return self.process_data(batch_1), self.process_data(batch_2)



    def group_batches(self,im1, rot1, backgroundT, imT, joints1,
                      im2, rot2, backgroundT2, imT2, joints2):
        assert self.batch_size//2 == len(im1)
        assert self.batch_size//2 == len(im2)
        dic_in = {
                'im_in': np.transpose(np.stack(im1+im2,axis=0), axes=[0,3,1,2]),
                'background_target' : np.transpose(np.stack(backgroundT+backgroundT2,axis=0), axes=[0,3,1,2]),
                'rot_im': np.stack(rot1 + rot2,axis=0),
                'invert_segments': list(range(self.batch_size//2, self.batch_size)) + list(range(self.batch_size//2))
            }
        dic_out ={'joints_im': np.stack(joints1+joints2,axis=0),
                  'im_target': np.transpose(np.stack(imT + imT2, axis=0), axes=[0, 3, 1, 2])}
        return dic_in,dic_out


    def track_epochs(self):
        self.elements_taken += self.batch_size // 2
        if self.elements_taken == len(self.index_file_cameras):
            self._logger.info("New Epoch reset elements taken")
            self._current_epoch += 1
            self.elements_taken = 0
            if self.randomise:
                shuffle(self.index_file_cameras)


    def __getitem__(self, item):


        index = item * (self.batch_size // 2)
        im1_tot, rot1_tot, backgroundT_tot, imT_tot, joints1_tot, \
        im2_tot, rot2_tot, backgroundT2_tot, imT2_tot, joints2_tot = \
            [], [], [], [], [], [], [], [], [], []
        for i in range(self.batch_size // 2):
            all_b = self.return_batch(index)
            im1, rot1, backgroundT, imT, joints1 = all_b[0]
            im2, rot2, backgroundT2, imT2, joints2 = all_b[1]
            im1_tot.append(im1)
            im2_tot.append(im2)
            rot1_tot.append(rot1)
            rot2_tot.append(rot2)
            backgroundT_tot.append(backgroundT)
            backgroundT2_tot.append(backgroundT2)
            imT_tot.append(imT)
            imT2_tot.append(imT2)
            joints1_tot.append(joints1)
            joints2_tot.append(joints2)
            index += 1
        dic_in, dic_out = self.group_batches(im1_tot, rot1_tot, backgroundT_tot, imT_tot, joints1_tot,
                                 im2_tot, rot2_tot, backgroundT2_tot, imT2_tot, joints2_tot)


        self.track_epochs()

        return encoder_dictionary_to_pytorch(dic_in), encoder_dictionary_to_pytorch(dic_out)

    def __len__(self):
        return (len(self.index_file_cameras) // (self.batch_size//2) ) -1










if __name__=="__main__":
    def check_data_feed():
        d = Data_Encoder_Decoder()
        oo = 0
        for i in range(10000):
            oo += 1
            o, e = d.return_batch()
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
        print(len(d))
        for i in range(len(d)):
            dic1,dic2= d[i]
            if i == 10:
                for k in dic1.keys():
                    if k== 'rot_im':
                        print('ll',np.linalg.det(dic1[k][0,...]))
                    if k != 'invert_segments' and dic1[k].shape[2] == 128:

                        plt.figure()
                        plt.title("k is "+ k)
                        plt.imshow(np.transpose(dic1[k][0, ...], axes=[1, 2, 0]))
                        #print(dic1[k].shape)
                        plt.figure()
                        plt.title("k app is "+k)
                        plt.imshow(np.transpose(dic1[k][5, ...], axes=[1, 2, 0]))

                for k in dic2.keys():
                    if k != 'invert_segments' and dic2[k].shape[2] == 128:
                        plt.figure()
                        plt.title("k is " + k)
                        plt.imshow(np.transpose(dic2[k][0, ...], axes=[1, 2, 0]))
                        plt.figure()
                        plt.title("k app is " + k)
                        plt.imshow(np.transpose(dic2[k][5, ...], axes=[1, 2, 0]))

                plt.show()

    #final_check()
