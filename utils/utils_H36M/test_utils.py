

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

d=Data_Base_class()
subjlist=[1,2,3,4,5,6,7,8,9,10]
d.create_index_file_subject(subjlist,64)
path,name=d.get_name(1,2,2,2,64)
path2,_=d.get_name(1,9,1,2,64)



sample_metadata=d.load_metadata(get_parent(path))
img=d.extract_image(path)

sample_metadata2=d.load_metadata(get_parent(path2))
img2=d.extract_image(path2)

print(sample_metadata['R'])
print(sample_metadata2['R'])






joints_world=sample_metadata['joint_world'][64]



#######################################################################
#test transformations
#######################################################################
joint_px, center= world_to_pixel(
    joints_world,
    H36M_CONF.joints.root_idx,
    H36M_CONF.joints.number,
    sample_metadata['R'],
    sample_metadata['T'],
    sample_metadata['f'],
    sample_metadata['c']
)

joint_cam= world_to_camera(
    joints_world,
    H36M_CONF.joints.number,
    sample_metadata['R'],
    sample_metadata['T']
)


#######################################################################
#test visualisation
#######################################################################
from utils.utils_H36M.transformations import plot_bounding_box,bounding_box_pixel,cam_pointing_root
from utils.utils_H36M.visualise import Drawer
from utils.utils_H36M.transformations import get_patch_image

#m = pkl.load(open(os.path.join(backgrounds_location, "backgrounds.pkl"), "rb"))
#fig=plt.figure()
#a=Drawer()
#fig=a.pose_2d(fig,img-m[3],joint_px[:,:-1])

#fig=plot_bounding_box(fig,joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
#fig=a.pose_3d(fig,joints_world)
#plt.show()






fig=plt.figure()
d=Drawer()
d.plot_image(fig,img)

fig2=plt.figure()
d.plot_image(fig2,img2)
plt.show()




from utils.utils_H36M.transformations import get_patch_image,transform_2d_joints

bbpx_px=bounding_box_pixel(joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
imwarped,trans = get_patch_image(img, bbpx_px, (512,512))
trsf_joints, vis = transform_2d_joints(joint_px,trans)
fig1=plt.figure()
b=Drawer()
plt.imshow(imwarped)
fig1 = b.pose_2d(fig1,imwarped,trsf_joints[:,:-1])
plt.show()



#######################################################################
#test rotation
#######################################################################


#new_rot = cam_pointing_root(joints_world,0,17,sample_metadata['R'], sample_metadata['T'])
#new_coords = joint_cam @ new_rot.T
#print("root in new coords",new_coords[0,:])

#######################################################################
#test data class
#######################################################################



#from utils.utils_H36M.preprocess import  Data_Base_class
#data=Data_Base_class()
#data.create_index_file_subject([1],60)
#print(data.get_name(1,2,1,1,1))
#data.save_index_file()
#data.load_index_file()
#print(data.index_file[0])







