
import numpy as np

from utils.utils_H36M.common import H36M_CONF, ENCODER_DECODER_PARAMS
from utils.utils_H36M.preprocess import Data_Base_class
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

    def check_previous_image(self,s, act, subact, ca):
        same_backgrounds, same_metadata = False, False
        if self.previous_chache is not None:
            s_p, act_p, subact_p, ca_p, fno_p = self.previous_chache
            if s_p == s:
                same_backgrounds=True
                if act == act_p and subact == subact_p and ca == ca_p:

                    same_metadata = True
        return same_backgrounds, same_metadata

    def testing(self,joints_world,imwarped,background_warped, R, T, f, c, trans):

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

    def extract_all_info(self,metadata, background,s, act, subact, ca, fno, rotation_angle=np.pi/4):

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

    def update_stored_info(self,s, act, subact, ca, fno,background, metadata):
        self.previous_chache = s, act, subact, ca, fno
        self.previous_background, self.previous_metadata = background, metadata




    def naming(self):

        if self._current_epoch is None:
            self.iteration_start()
        if self._current_epoch >= self._max_epochs:
            self._logger.info("max epochs reached")
            return None
        s, act, subact, ca, fno = self.return_current_file()

        same_backgrounds, same_metadata = self.check_previous_image(s, act, subact, ca)
        metadata, background = self.load_metadata_backgrounds_image(s, act, subact, ca, fno, same_metadata, same_backgrounds)
        print(s, act, subact, ca, fno)
        if s == 1 and act ==2 and subact==2 and ca==4 and fno==1601:
            print(same_metadata,same_backgrounds)
            imw, R_pointing_centre, backgroundw, joints_world=\
                self.extract_all_info(metadata, background,s, act, subact, ca, fno)
            #extract apperance image same metadata and background
            #extract image from other camera
        self.update_stored_info(s, act, subact, ca, fno, background, metadata)
        self.increase_fno()



if __name__=="__main__":
    a=Data_Encoder_Decoder()
    for i in range(100000):
        a.naming()

