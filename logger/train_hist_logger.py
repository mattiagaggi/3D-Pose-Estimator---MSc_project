
import numpy as np
import os
import pickle as pkl

from utils.io import ensure_dir, file_exists
from logger.console_logger import ConsoleLogger
from dataset_def.trans_numpy_torch import image_pytorch_to_numpy, tensor_to_numpy

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
        self.scalars ={}

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
        path=os.path.join(self.path,'scalars.pkl')
        pkl.dump(self.scalars, open(path, "wb"))

    def load_logger(self):
        path = os.path.join(self.path, 'scalars.pkl')
        if not file_exists(path):
            self._logger.error("Train log not loaded")
        else:
            self.scalars = pkl.load(open(path, "rb"))


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














