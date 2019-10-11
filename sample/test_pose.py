from sample.tester.pose_tester import Pose_Tester

from sample.losses.poses import MPJ, Aligned_MPJ,Normalised_MPJ
from sample.models.pose_encoder_decoder import Pose_3D

metrics =[ MPJ(), Normalised_MPJ(), Aligned_MPJ()]
model=Pose_3D()
output="data/checkpoints"
name="enc_dec_S15678_no_rotfinal3D"



pose =Pose_Tester( model, output, name, metrics)

pose.test_on_test()
pose.test_on_train()