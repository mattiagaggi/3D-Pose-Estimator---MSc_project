

from data.directories_location import h36m_location
from utils.utils_H36M.common import H36M_CONF
import scipy.io as sio
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl





#######################################################################
#test data loadinf
#######################################################################
from utils.utils_H36M.transformations import world_to_pixel,world_to_camera
from utils.utils_H36M.preprocess import Data_Base_class
from utils.io import *
from data.directories_location import backgrounds_location
#
#
# d=Data_Base_class()
# subjlist=[1,2,3,4,5,6,7,8,9,10]
# d.create_index_file('s',[1])
# print(d.index_file)
# path=d.index_file[1][2][1][1][65]
#
# path2=d.index_file[1][5][1][1][65]
# sample_metadata=d.load_metadata(get_parent(path))
# img=d.extract_image(path)
# sample_metadata2=d.load_metadata(get_parent(path2))
# img2=d.extract_image(path2)
#print(sample_metadata['R'])
#print(sample_metadata2['R'])












#######################################################################
#test backgrounds
#######################################################################
from utils.utils_H36M.transformations import bounding_box_pixel
from utils.utils_H36M.visualise import Drawer


#m = pkl.load(open(os.path.join(backgrounds_location, "backgrounds.pkl"), "rb"))
#fig1=plt.figure()
#plt.imshow(m[0]-img)
#fig=plt.figure()
#mask=m[1]-img
#mask[mask<10**-1]=1
#mask[mask!=1]=0
#plt.imshow(mask)
#fig1=plt.figure()
#plt.imshow(img)
#print("L2 error",np.sqrt(np.sum((img-m[0])**2)))
#print("L2 error",np.sqrt(np.sum((img-m[1])**2)))

#plt.show()



#######################################################################
#test cropping
#######################################################################
from utils.utils_H36M.transformations import get_patch_image,transform_2d_joints

# joints_world=sample_metadata['joint_world'][64]
# joint_cam=world_to_camera(joints_world, 17, sample_metadata['R'], sample_metadata['T'])
# joint_px, center= world_to_pixel(
#    joints_world,
#    H36M_CONF.joints.root_idx,
#    H36M_CONF.joints.number,
#    sample_metadata['R'],
#   sample_metadata['T'],
#    sample_metadata['f'],
#    sample_metadata['c']
)

bbpx_px=bounding_box_pixel(joints_world, 0, sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
imwarped,trans = get_patch_image(img, bbpx_px, (256,256), np.pi/4) # in degrees rotation around z axis
trsf_joints, vis = transform_2d_joints(joint_px, trans)
b=Drawer()
ax=plt.subplot()
ax=b.pose_2d(ax,imwarped,trsf_joints[:,:-1])
plt.show()





#######################################################################
#test rotation
#######################################################################

from utils.utils_H36M.transformations import cam_pointing_root
# new_rot = cam_pointing_root(joints_world,0,17,sample_metadata['R'], sample_metadata['T'])
# new_coords = joint_cam @ new_rot.T
# print("root in new coords",new_coords[0,:])

#######################################################################
#test data class
#######################################################################



# from utils.utils_H36M.preprocess import  Data_Base_class
# data=Data_Base_class()
# data.create_index_file('s',[1])
# data.save_index_file()
# data.load_index_file()
# for i in range(1000):
#     print(next(data))







