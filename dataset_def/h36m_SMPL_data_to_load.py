import numpy as np
from random import shuffle
import torch
from utils.utils_H36M.common import H36M_CONF
from sample.config.data_conf import PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from utils.utils_H36M.transformations import get_patch_image, bounding_box_pixel, cam_pointing_root, rotate_z
from utils.trans_numpy_torch import numpy_to_tensor_float, image_numpy_to_pytorch, numpy_to_long, tensor_to_numpy, numpy_to_tensor
import os
class SMPL_Data_Load(Data_Base_class):

    def __init__(self,
                 sampling,
                 index_file_content=['s'],
                 index_file_list=[[1]],
                 get_intermediate_frames = False,
                 subsampling_fno = 0,
                 randomise=True):
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
        #self._logger.error(os.getcwd() )
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
            "mask_image": [],
            "mask_idx_all": [],
            "mask_R": [],
            "mask_T": [],
            "mask_f": [],
            "mask_c": [],
            "mask_trans_crop": [],
            "mask_idx_n": []
        }
        return dic


    def update_dic_with_image(self, dic, s, act, subact, ca, fno, rotation_angle):
        im, joints_world, R = self.extract_image_info(s, act, subact, ca, fno, rotation_angle=rotation_angle)
        dic['image'] = np.transpose(im, (2, 0, 1))
        dic['joints_im'] = joints_world
        dic['R'] = R
        dic['root_pos'] = dic['joints_im'][ H36M_CONF.joints.root_idx, :]
        dic['root_pos'] = dic['root_pos'].reshape(1, 3)
        return dic


    def update_dic_with_mask(self,dic, s, act, subact, mask_number, fno, rotation_angle):
        im, R, T, f, c, trans = self.extract_masks_info(s,act,subact,mask_number,fno,rotation_angle)
        dic['mask_image'].append(np.transpose(np.expand_dims(im, axis=2), (2, 0, 1)))
        dic['mask_R'].append(R)
        dic['mask_T'].append(T)
        dic['mask_f'].append(f)
        dic['mask_c'].append(c)
        dic['mask_trans_crop'].append(trans)
        dic['mask_idx_n'].append(mask_number)
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
                dic = self.update_dic_with_mask(dic, s, act, subact, mask_number, fno, rotation_angle)
        return dic



if __name__ == '__main__':
    a = SMPL_Data_Load(2000,index_file_content=['s'],
                 index_file_list=[[6]],)

    from utils.utils_H36M.visualise import Drawer
    import matplotlib.pyplot as plt

    d = Drawer()
    im = a.load_image(6, 2, 2, 2, 371)
    #im = np.transpose(im, axes=[1, 2, 0])
    metadata = a.all_metadata[6][2][2][2]
    im, joints_world, R, T, f, c = a.extract_info(metadata, 6, 2, 2, 2, 371)
    bbpx = bounding_box_pixel(joints_world, H36M_CONF.joints.root_idx, R, T, f, c)
    im, trans = a.crop_img(im, bbpx, None)
    im2 = im.copy()
    im2[:, :, 0] = im[:, :, 2]
    im2[:, :, 2] = im[:, :, 0]
    plt.axis('off')
    plt.imshow(im2)
    plt.title("ssksk")
    plt.show()
    #[6.   2.   1.   1. 101.   3.]
    #[6. 2. 2. 1. 1. 4.]
    #[6.   2.   2.   2. 371.   4.]
    fig = plt.figure()
    d.pose_3d(b['joints_im'], plot=True, fig=fig, azim=-90, elev=0)
    plt.show()
    from utils.utils_H36M.transformations import world_to_pixel, transform_2d_joints

    pix = []
    for n, i in enumerate(b['mask_image']):
        plt.axis('off')
        plt.imshow(i[0], cmap='gray')
        plt.show()

    from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
    from utils.smpl_torch.display_utils import Drawer
    from utils.trans_numpy_torch import numpy_to_tensor_float, tensor_to_numpy

    from utils.conversion_SMPL_h36m_torch import from_smpl_to_h36m_world_torch,project_vertices_onto_mask
    cuda = False
    batch_size = 1
    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='data/models_smpl')
    d = Drawer(kintree_table=smpl_layer.kintree_table)

    # Generate random pose and shape parameters
    import numpy as np

    pose_params =  torch.rand(batch_size, 72) *0.02
    shape_params = torch.rand(batch_size, 10) *0.02
    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()
    # Forward from the SMPL layer
    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

    root=np.reshape(b['joints_im'][0,:],(1,1,3))
    root=numpy_to_tensor_float(root)
    verts=from_smpl_to_h36m_world_torch(verts,root, from_camera=False, R_world_cam=None)
    dic={}
    for key in b.keys():
        if type(b[key]) == list and "idx" not in key:
            print(key)
            if len(b[key][0].shape) == 2:
                dim1,dim2=b[key][0].shape
                dic[key]=numpy_to_tensor_float(np.reshape(b[key][0],(1,dim1,dim2)))
    pix_vertices_ca = project_vertices_onto_mask(verts, dic)
    px=tensor_to_numpy(pix_vertices_ca)[0]
    plt.scatter(px[:,0], px[:,1],s=1)
    plt.show()
    plt.rcParams['axes.facecolor'] = 'black'
    plt.scatter(px[:, 0], px[:, 1], s=40,c='w')
    plt.show()






    plt.show()






