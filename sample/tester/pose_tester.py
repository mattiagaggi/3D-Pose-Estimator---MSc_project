
from sample.config.data_conf import PARAMS
from base....

device=PARAMS['data']['device']
sampling_train=PARAMS.data.sampling_train
sampling_test= PARAMS.data.sampling_test
parser= Pose_Parser("Pose Parser")
args_pose = parser.get_arguments()



class Pose_Tester:
    def __init__(self, model,  data_loader,
             batch_size, output, name, no_cuda):

        super().__init__(self, model,  data_loader,
                 batch_size, output, name, no_cuda)


