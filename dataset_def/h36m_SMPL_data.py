
import numpy as np
from random import shuffle
import torch
from utils.utils_H36M.common import H36M_CONF
from sample.config.data_conf import PARAMS
from dataset_def.h36m_preprocess import Data_Base_class
from utils.utils_H36M.transformations import get_patch_image, bounding_box_pixel, cam_pointing_root, rotate_z
from utils.trans_numpy_torch import numpy_to_tensor_float, image_numpy_to_pytorch, numpy_to_long, tensor_to_numpy, numpy_to_tensor

class SMPL_Data(Data_Base_class):
    def __init__(self,
                 args,
                 sampling,
                 index_file_content=['s'],
                 index_file_list=[[1]],
                 randomise=True,
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
        self.batch_size = args.batch_size
        self.create_index_file(index_file_content, index_file_list)
        self.index_file_content = index_file_content
        self.index_file_list = index_file_list
        self.randomise= randomise
        if subsampling_fno==0:
            pass
        elif subsampling_fno == 1:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=True)
        elif subsampling_fno == 2:
            self.index_file = self.subsample_fno(self.index_file, 0.75, lower=False)
        else:
            self._logger.error("Subsampling not understood")
        if self.randomise:
            shuffle(self.index_file)
        self.elements_taken=0
        self._current_epoch=0




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



    def create_mask_dic(self):
        mask_sub_dic={"image":[],
                      "idx":[],
                      "R":[],
                      "T":[],
                      "f":[],
                      "c":[],
                      "trans_crop":[]
                      }
        return mask_sub_dic


    def create_dictionary_data(self):
        dic= {"image": [],
              "joints_im": [],
              "R": [],
              "mask_image": [],
              "mask_idx_all": [],
              "mask_R": [],
              "mask_T": [],
              "mask_f": [],
              "mask_c": [],
              "mask_trans_crop": [],
              "mask_idx_n":[]
              }
        return dic


    def update_dic_with_image(self, dic, s, act, subact, ca, fno, rotation_angle):
        im, joints_world, R = self.extract_image_info(s, act, subact, ca, fno, rotation_angle=rotation_angle)
        dic['image'].append(image_numpy_to_pytorch(im))
        dic['joints_im'].append(numpy_to_tensor_float(joints_world))
        dic['R'].append(numpy_to_tensor_float(R))
        return dic


    def update_dic_with_mask(self,dic, i,  s, act, subact, mask_number, fno, rotation_angle):
        im, R, T, f, c, trans = self.extract_masks_info(s,act,subact,mask_number,fno,rotation_angle)
        dic['mask_idx_all'].append(i)
        dic['mask_image'].append(numpy_to_tensor_float(im))
        dic['mask_R'].append(numpy_to_tensor_float(R))
        dic['mask_T'].append(numpy_to_tensor_float(T))
        dic['mask_f'].append(numpy_to_tensor_float(f))
        dic['mask_c'].append(numpy_to_tensor_float(c))
        dic['mask_trans_crop'].append(numpy_to_tensor_float(trans))
        dic['mask_idx_n'].append(mask_number)

        return dic

    def dic_final_processing(self, dic):
        for key in dic.keys():
            if 'idx' in key:
                dic[key] = numpy_to_long(dic[key])
            else:
                dic[key] = torch.stack(dic[key], dim=0)
        dic['root_pos'] = dic['joints_im'][:, H36M_CONF.joints.root_idx, :]
        dic['root_pos'] = dic['root_pos'].view(-1, 1, 3)
        return dic


    def track_epochs(self):

        self.elements_taken += self.batch_size
        if self.elements_taken //(self.batch_size ) == self.__len__():
            self._logger.info("New Epoch reset elements taken")
            self._current_epoch += 1
            self.elements_taken = 0
            if self.randomise:
                shuffle(self.index_file)

    def __len__(self):

        return len(self.index_file) // self.batch_size


    def __getitem__(self, item):
        idx = item * self.batch_size
        dic = self.create_dictionary_data()
        rotation_angle = 0
        for i in range(self.batch_size):
            s, act, subact, ca, fno = self.index_file[idx+i]
            dic = self.update_dic_with_image(dic,s, act, subact, ca, fno, rotation_angle)
            for mask_number in range(1,5):
                if mask_number in self.all_metadata[s][act][subact].keys():
                    dic = self.update_dic_with_mask(dic, i, s, act, subact, mask_number, fno, rotation_angle)
        dic = self.dic_final_processing(dic)
        self.track_epochs()
        return dic






if __name__== '__main__' :

    import torch
    from sample.parsers.parser_enc_dec import EncParser
    from utils.utils_H36M.transformations_torch import world_to_camera_batch,camera_to_pixels_batch, transform_2d_joints_batch
    import matplotlib.pyplot as plt
    from utils.utils_H36M.visualise import Drawer

    el=4
    idx=0

    ten=(torch.arange(10, dtype=torch.int32) * 3).cuda()[:, None, None]
    print(ten[:,0,0])
    d = Drawer()
    parser= EncParser("pars")
    arg=parser.get_arguments()
    c = SMPL_Data(arg,5)
    dic=c[el]
    el_idx = el * arg.batch_size
    s,act,sub,ca,fno = c.index_file[el_idx+idx]
    print(s,act,sub,ca,fno)
    joints=dic['joints_im']

    im = dic['image']
    im= tensor_to_numpy(im).transpose(0,2,3,1)
    im=im[idx]
    from utils.smpl_torch.pytorch.smpl_layer import SMPL_Layer
    from utils.smpl_torch.display_utils import Drawer as DrawerS
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='data/models_smpl')
    ds = DrawerS(kintree_table=smpl_layer.kintree_table)
    # Generate random pose and shape parameters
    batch_size=1
    pose_params = torch.rand(batch_size, 72)


    pose_params_1 = numpy_to_tensor_float(np.array([ 1.3438e-01,  1.0050e-03,  1.2732e-01,  8.8333e-02, -1.3887e+00,
        -1.5473e-01,  5.2549e-01, -5.0818e-01,  3.5654e-01,  4.1361e-01,
        -3.5565e-03, -2.1754e-01, -1.2578e-01, -2.0454e-01,  3.0172e-01,
        -1.0112e+00,  4.8895e-01,  3.9189e-01,  3.7336e-01,  5.0514e-01,
        -4.0682e-01,  4.8758e-01, -2.8710e-02, -2.0994e-02,  2.2309e-01,
         1.5088e-01, -1.1272e+00, -1.0828e+00,  8.7898e-02, -2.2011e-02,
         2.2926e-02, -2.0094e-01,  1.3080e-01, -1.3582e-01, -7.7722e-02,
        -3.1845e-01,  1.8695e-01, -5.3144e-01,  2.7748e-01,  6.7985e-02,
        -4.3104e-01, -1.5006e-01,  3.8521e-02, -5.5779e-01,  1.4236e-01,
        -1.8640e-01, -8.4205e-02,  6.0946e-01,  1.5840e-01,  1.3520e+00,
        -1.7617e-01, -1.2128e-01, -4.4586e-01, -5.8661e-01,  4.0632e-01,
         8.6259e-01,  3.4038e-01,  1.8232e-01, -3.3726e-01, -5.7435e-01,
         6.5781e-02, -2.7820e-01, -2.8747e-01, -1.7012e-01, -2.2815e-01,
        -2.5051e-01,  5.8695e-01, -9.4217e-02,  6.6951e-02, -1.7207e-01,
         4.9331e-02,  9.0434e-02]))
    shape_params_1 = numpy_to_tensor_float(np.array([-0.8954, -0.0059, -0.3579, -0.2993, -0.4143, -0.3281, -0.1679, -0.1420,
         0.0835,  0.5819]))
    pose_params = torch.stack([pose_params_1]*batch_size, dim=0)
    shape_params = torch.stack([shape_params_1]*batch_size, dim=0)

    pose_params = pose_params.cuda()
    shape_params = shape_params.cuda()
    smpl_layer.cuda()
    import neural_renderer as nr
    import torch.nn as nn
    from skimage.io import imread, imsave
    import tqdm
    import imageio




    verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
    from utils.conversion_SMPL_h36m_torch import from_smpl_to_h36m_world_torch

    verts = from_smpl_to_h36m_world_torch(verts, dic['root_pos'])
    for i in range(1,5):
        mask_dic=dic['masks'][i]

        cam = world_to_camera_batch(joints,17,mask_dic['R'],mask_dic['T'])
        verts_cam = world_to_camera_batch(verts,verts.size()[1],mask_dic['R'],mask_dic['T'])
        pix = camera_to_pixels_batch(cam, 17, mask_dic['f'], mask_dic['c'])
        verts_pix = camera_to_pixels_batch(verts_cam, verts.size()[1], mask_dic['f'], mask_dic['c'], return_z=True)
        tranpi = transform_2d_joints_batch(pix,mask_dic['trans_crop'])
        verts_fin= transform_2d_joints_batch(verts_pix,mask_dic['trans_crop'])
        tranpi = tensor_to_numpy(tranpi)



        faces = np.expand_dims(smpl_layer.th_faces.cpu(), 0)

        #faces = np.repeat(faces,repeats=batch_size, axis=0)
        faces1 = numpy_to_tensor(faces).int()

        from neural_renderer.rasterize import rasterize_silhouettes
        from neural_renderer.vertices_to_faces import vertices_to_faces

        #faces1 = torch.cat((faces1, faces1[:, :, list(reversed(range(faces1.shape[-1])))]), dim=1)
        print(faces1.size(),verts_fin.size())
        print(faces1)

        verts_fin=verts_fin/64
        verts_fin =verts_fin-1
        faces=vertices_to_faces(verts_fin[0].view(1,-1,3), faces1)
        print(faces.size())
        tranpi_SMPL = tensor_to_numpy(verts_fin)
        verts_fin = (verts_fin + 1) * 64
        vc = tensor_to_numpy(verts_fin[0].detach())

        import torch

        #faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)
        image=rasterize_silhouettes(faces, 128, anti_aliasing=True)
        ii = numpy_to_long(np.array(list(reversed(range(image.shape[-1])))))
        image=torch.index_select(image, dim=1, index=ii)
        implotting = tensor_to_numpy( image[0].detach())
        print(implotting[implotting!=0])

        plt.figure()

        plt.imshow(implotting)

        plt.scatter(vc[:, 0], vc[:, 1], alpha=0.1)
        plt.show()

        plt.figure()
        im= implotting.flatten()
        plt.hist(im)
        plt.show()


        mask=tensor_to_numpy( mask_dic['image'][idx])
        plt.figure()
        plt.imshow(mask)
        plt.scatter(vc[:, 0], vc[:, 1])
        plt.show()
        #if i==ca:
            #fig = plt.figure()
            #imjj = d.pose_2d(mask, tranpi[idx],  False)
            #plt.imshow(imjj)
            #plt.figure()
            #plt.imshow(im)
            #plt.scatter(tranpi_SMPL[idx, :, 0],tranpi_SMPL[idx,:,1])
            #plt.show()
            #mask = mask.reshape(128,128,1)*im
        #plt.figure()
        #imjj=d.pose_2d(mask, tranpi[idx],False)
        #plt.imshow(imjj)
        #plt.show()



















