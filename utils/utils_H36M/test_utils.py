

from data.directories_location import h36m_location
from utils.utils_H36M.common import H36M_CONF
import scipy.io as sio
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl

subdirfile="s_01_act_06_subact_02_ca_04/"
file="s_01_act_06_subact_02_ca_04_000010.jpg"



#######################################################################
#test data loadinf
#######################################################################
from utils.utils_H36M.transformations import world_to_pixel,world_to_camera
sample_metadata=sio.loadmat(h36m_location+subdirfile+"h36m_meta.mat")
joints_world=sample_metadata['pose3d_world'][10]
im=cv2.imread(h36m_location+subdirfile+file)
img = im#[:H36M_CONF.max_size, :H36M_CONF.max_size]
img = img.astype(np.float32)
img=img[:H36M_CONF.max_size,:H36M_CONF.max_size,:]
img /= 256



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

m = pkl.load(open(os.path.join(h36m_location, "backgrounds.pkl"), "rb"))
fig=plt.figure()
a=Drawer()
fig=a.pose_2d(fig,img-m[3],joint_px[:,:-1])
print("look this",np.sum(img-m[3]))
fig=plot_bounding_box(fig,joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
#fig=a.pose_3d(fig,joints_world)
#plt.show()



from utils.utils_H36M.transformations import get_patch_image,transform_2d_joints
bbpx_px=bounding_box_pixel(joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
print(bbpx_px)
imwarped,trans = get_patch_image(img, bbpx_px, (512,512))
trsf_joints, vis = transform_2d_joints(joint_px,trans)
fig1=plt.figure()
b=Drawer()
plt.imshow(imwarped)
fig1 = b.pose_2d(fig1,imwarped,trsf_joints[:,:-1])
print(trsf_joints)
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







