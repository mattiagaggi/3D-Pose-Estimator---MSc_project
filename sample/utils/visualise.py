

from data.directories_location import  h36m_location
subdirfile="s_01_act_02_subact_01_ca_04/"
file="s_01_act_02_subact_01_ca_04_000001.jpg"


import scipy.io as sio
def load_mat(subdirectory=subdirfile):
    data=sio.loadmat(h36m_location+subdirectory+"h36m_meta.mat")
    points_3D=data['pose3d_world'] #3D numpy array
    cam_r=data['R']
    cam_t=data['T']
    return points_3D,cam_r,cam_t


from PIL import Image
img=Image.open(h36m_location+subdirfile+file)
img.show()

import numpy as np
points_3D,cam_r,cam_t=load_mat()
print(np.linalg.det(cam_r@cam_r))

from matplotlib import pyplot as plt

plt.scatter(points_3D[0,:,0],points_3D[0,:,1])
plt.show()