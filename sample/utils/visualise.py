

from data.directories_location import  h36m_location
subdirfile="s_01_act_02_subact_01_ca_04/"

file="s_01_act_02_subact_01_ca_04_000002.jpg"
gtfile="h36m_meta.mat"

from PIL import Image
import scipy.io as sio


data=sio.loadmat(h36m_location+subdirfile+gtfile)
points_3D=data['pose3d_world'] #3D numpy array
cam_r=data['R']
print(type(cam_r))
print(cam_r)
img=Image.open(h36m_location+subdirfile+file)
img.show()

print()