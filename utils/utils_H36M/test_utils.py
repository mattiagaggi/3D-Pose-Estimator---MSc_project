

from data.directories_location import h36m_location
from utils.utils_H36M.common import H36M_CONF
import scipy.io as sio
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

subdirfile="s_01_act_05_subact_02_ca_01/"
file="s_01_act_05_subact_02_ca_01_000120.jpg"
from utils.utils_H36M.transformations import world_to_pixel
from utils.utils_H36M.visualise import Drawer




#######################################################################
#test data loadinf
#######################################################################
sample_metadata=sio.loadmat(h36m_location+subdirfile+"h36m_meta.mat")
joints_world=sample_metadata['pose3d_world'][100]
im=cv2.imread(h36m_location+subdirfile+file)
img = im#[:H36M_CONF.max_size, :H36M_CONF.max_size]
img = img.astype(np.float32)
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



from utils.utils_H36M.transformations import plot_bounding_box,bounding_box_pixel
a=Drawer()
fig=plt.figure()
fig=a.pose_2d(fig,img,joint_px[:,:-1])
bbpx_px=bounding_box_pixel(joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
fig=plot_bounding_box(fig,joints_world, 0,sample_metadata['R'], sample_metadata['T'], sample_metadata['f'], sample_metadata['c'])
#fig=a.pose_3d(fig,joints_world)
plt.imshow(img)
plt.show()
print(bbpx_px[0]+bbpx_px[2]/2,bbpx_px[1]+bbpx_px[3]/2)
print(joint_px[0,:])


#######################################################################
#test data class
#######################################################################
from utils.utils_H36M.preprocess import  Data_Base_class
#data=Data_Base_class()
#data.create_index_file_subject([1],60)
#print(data.get_name(1,2,1,1,1))
#data.save_index_file()
#data.load_index_file()
#print(data.index_file[0])




