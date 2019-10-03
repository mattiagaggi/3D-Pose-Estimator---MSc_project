
import numpy as np
import os
import pickle as pkl

from utils.io import ensure_dir, file_exists
from logger.console_logger import ConsoleLogger
from utils.trans_numpy_torch import image_pytorch_to_numpy, tensor_to_numpy

class TrainingLogger:

    """
    Logger, used by save training history
    """

    def __init__(self, dir_path,):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self._logger= ConsoleLogger()
        self.path=os.path.join(self.dir_path, 'training_logger/')
        ensure_dir(self.path)
        self.scalars = {}
        self.scalars_saved = 0

    def record_scalar(self,scalar_type, scalar, idx):

        if scalar_type not in self.scalars.keys():
            self.scalars[scalar_type]=[]
            self.scalars[scalar_type+"_idx"]=[]
        self.scalars[scalar_type].append(scalar)
        self.scalars[scalar_type+"_idx"].append(idx)



    def record_index(self,idx_type, idx):
        if idx_type not in self.scalars.keys():
            self.scalars[idx_type]=[]
        self.scalars[idx_type].append(idx)


    def save_logger(self):
        self.scalars_saved += 1
        path=os.path.join(self.path,'scalars%s.pkl' % self.scalars_saved)
        lst = [self.scalars_saved, self.scalars]
        pkl.dump(lst, open(path, "wb"))
        self.scalars = {}


    def load_logger(self):
        current = 0
        while file_exists(os.path.join(self.path, 'scalars%s.pkl' % current)):
            current += 1
        current -= 1
        self.scalars_saved = current

    def save_dic(self, name, dic_in, idx, extra_str="_"):
        dic ={}
        if extra_str == "_":
            self.record_index(name, idx)
        for key in dic_in.keys():
            dic[key] = dic_in[key].cpu().data.numpy()
        dir_path = os.path.join(self.path, name)
        ensure_dir(dir_path)
        string = 'dic_%s' % idx
        string = extra_str+string
        path_in = os.path.join(dir_path, string)
        pkl.dump(dic, open(path_in, "wb"))

    def get_dic(self,name, idx, extra_str="_"):
        dir_path = os.path.join(self.path, name)
        string = 'dic_%s' % idx
        string = extra_str + string+".pkl"
        path = os.path.join(dir_path, string)
        dic=pkl.load(open(path,'rb'))
        return dic



    def save_dics(self,name, dic_in, dic_out, idx):
        self.save_dic(name, dic_in, idx,"in" )
        self.save_dic(name, dic_out, idx, "out")
        self.record_index(name, idx)



    def save_batch_images(self, name, image, idx, image_pred=None, image_target=None, pose_pred=None, pose_gt=None):

        name = name+"_images"
        self.record_index(name,idx)
        dir_path = os.path.join(self.path, name)
        ensure_dir(dir_path)
        path = os.path.join(dir_path,'%s.npy' % idx)
        image = image_pytorch_to_numpy(image, True)
        np.save(path,image)
        if image_target is not None:
            path = os.path.join(dir_path,'%sT.npy' % idx)
            image_target = image_pytorch_to_numpy(image_target, True)
            np.save(path, image_target)
        if image_pred is not None:
            path = os.path.join(dir_path, '%sT_gt.npy' % idx)
            image_pred = image_pytorch_to_numpy(image_pred, True)
            np.save(path, image_pred)
        if pose_pred is not None:
            path = os.path.join(dir_path, '%spose.npy' %idx)
            pose_pred= tensor_to_numpy(pose_pred)
            np.save(path, pose_pred)
        if pose_gt is not None:
            path = os.path.join(dir_path, '%spose_gt.npy' % idx)
            pose_gt= tensor_to_numpy(pose_gt)
            np.save(path, pose_gt)

    def load_batch_images(self, name, idx):
        dic={}
        dir_path = os.path.join(self.path, name)
        if name not in self.scalars.keys():
            self._logger.error("Key not found")
        if not os.path.isdir(dir_path):
            self._logger.error("Folder not found")
        path = os.path.join(dir_path,'%s.npy' % idx)
        if not file_exists(path):
            self._logger.error("File not found")
        dic['image'] = np.load(path)
        path = os.path.join(dir_path,'%sT.npy' % idx)
        if file_exists(path):
            dic['image_T'] = np.load(path)
        path = os.path.join(dir_path, '%sT_gt.npy' % idx)
        if file_exists(path):
            dic['image_T_gt'] = np.load(path)
        path = os.path.join(dir_path, '%spose.npy' %idx)
        if file_exists(path):
            dic['pose'] = np.load(path)
        path = os.path.join(dir_path, '%spose_gt.npy' % idx)
        if file_exists(path):
            dic['pose_gt'] = np.load(path)
        return dic














