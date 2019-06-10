# -*- coding: utf-8 -*-
"""
Read Human3.6M files

This class assumes that the structure of the directory
is as follows:

Dataset
├── s_01_act_02_subact_01_ca_01 - sequence specific
│   ├── h36m_meta.mat - file with camera parameters
│   ├── s_01_act_02_subact_01_ca_01_000001.jpg
│   ├── ...
│   └── s_01_act_16_subact_02_ca_04_001234.jpg
│
├── ...
│
└── s_11_act_16_subact_02_ca_04 - sequence specific
    ├── h36m_meta.mat - file with camera parameters
    ├── s_11_act_16_subact_02_ca_04_000001.jpg
    ├── ...
    └── s_11_act_16_subact_02_ca_04_001234.jpg

If Dataset contains the [ index_train.h5 | index_test.h5 ] file, then this
is loaded automatically, otherwise it's created.

@author: Denis Tome'

"""
import os
import cv2
import scipy.io as sio
import numpy as np
from base import SubSet
from base import BaseDataset
from dataset_def import transformations as trasnf
import utils

__all__ = [
    'Human36M'
]


class Human36M(BaseDataset):
    """Dataset"""

    def __init__(self, path, subset=SubSet.train, transform=None):
        """Initialization

        Arguments:
            path {str} -- path to the data

        Keyword Arguments:
            subset {SubSet} -- [] (default: {SubSet.train})
            transform {FrameworkClass} -- transformation to apply to
                                          the data (default: {None})
        """

        super().__init__(path, subset)

        if not isinstance(subset, SubSet):
            self._logger.error('subset must be of class SubSet...')

        if subset == SubSet.train:
            subj_list = utils.H36M_CONF.train.sid
            sampling = utils.H36M_CONF.train.sampling
            file_path = os.path.join(path, 'index_train.h5')
        if subset == SubSet.test:
            subj_list = utils.H36M_CONF.test.sid
            sampling = utils.H36M_CONF.test.sampling
            file_path = os.path.join(path, 'index_test.h5')
        if subset == SubSet.val:
            subj_list = utils.H36M_CONF.val.sid
            sampling = utils.H36M_CONF.val.sampling
            file_path = os.path.join(path, 'index_val.h5')

        self._logger.info('Loading metadata...')
        self.subset = subset
        self.meta_data = self._load_metadata(path, subj_list)

        # indexing
        if os.path.isfile(file_path):
            self.data_files = self._load_index_file(file_path)
        else:
            self.data_files = self._index_file(path, subj_list, sampling)
            self._save_index_file(file_path)
        self._logger.info('Done...')

        self.transform = transform
        self.batch_idx = 0

    def __len__(self):
        """Get number of elements in the dataset"""

        return len(self.data_files)

    def get_content(self, name, content):
        """Get content from name according to convention

        Arguments:
            name {str} -- file name
            content {str} -- content name

        Returns:
            int -- content value
        """

        res = name.split('_')

        if content == 's':
            return int(res[1])
        if content == 'act':
            return int(res[3])
        if content == 'subact':
            return int(res[5])
        if content == 'ca':
            return int(res[7])
        if content == 'fno':
            return int(res[9])

        self._logger.error('%s has not %s', name, content)

    def _load_metadata(self, root_path, subject_ids):
        """Load metadata

        Arguments:
            root_path {str} -- path to dir containing the dataset
            subject_ids {list} -- list of subject ID

        Returns:
            dict -- metadata
        """

        metadata = {}

        # get list of sequences
        names, paths = utils.get_sub_dirs(root_path)
        for name, path in zip(names, paths):

            # check data to load
            sid = self.get_content(name, 's')
            if sid not in subject_ids:
                continue

            # check metadata file
            metadata_path = os.path.join(path, 'h36m_meta.mat')
            if not os.path.exists(metadata_path):
                self._logger.error('File %s not loaded', metadata_path)
                exit()

            data = sio.loadmat(metadata_path)

            info = {
                'joint_world': data['pose3d_world'],
                'R': data['R'],
                'T': data['T'],
                'f': data['f'],
                'c': data['c'],
                'img_widths': data['img_width'],
                'img_heights': data['img_height'],
            }

            metadata.update({name: info})

        return metadata

    def _index_file(self, root_path, subj, sampling):
        """Create index file

        Arguments:
            root_path {str} -- path to dir containing the dataset
            subj {list} -- list of subjects
            sampling {int} -- sampling

        Returns:
            dict -- index file
        """

        self._logger.info('Indexing dataset...')

        index = []

        # get list of sequences
        names, paths = utils.get_sub_dirs(root_path)
        for name, path in zip(names, paths):

            # check data to load
            sid = self.get_content(name, 's')
            if sid not in subj:
                continue

            f_data, _ = utils.get_files(path, 'jpg')
            for fid, f_path in enumerate(f_data):

                if fid % sampling != 0:
                    continue

                file_details = {
                    'path': f_path.encode('utf8'),
                    'fid_sequence': fid,
                    'sequence_name': name.encode('utf8')
                }
                index.append(file_details)

        return index

    def _save_index_file(self, file_path):
        """Save index file

        Arguments:
            file_path {str} -- path where to save index file
            training {bool} -- training set
        """

        self._logger.info('Saving index file...')

        data = self.data_files.copy()
        data_dict = {}
        for k in data[0].keys():

            curr_list = []
            for sample in data:
                curr_list.append(sample[k])

            data_dict.update({k: curr_list})

        utils.write_h5(file_path, data_dict)

    def _load_index_file(self, file_path):

        self._logger.info('Loading index file...')
        data_dict = utils.read_h5(file_path)

        keys = list(data_dict.keys())
        num_elem = len(data_dict[keys[0]])

        data_files = []
        for fid in range(num_elem):

            sample = {}
            for k in keys:
                sample.update({k: data_dict[k][fid]})

            data_files.append(sample)

        return data_files

    def __getitem__(self, idx):
        """Get sample

        Arguments:
            idx {int} -- sample id

        Returns:
            tensor -- image
            tensor -- joints in pixels
            tensor -- joints in cam coordinates
            tensor -- center
            tensor -- bounding box
        """

        data_dict = self.data_files[idx]
        sample_metadata = self.meta_data[data_dict['sequence_name'].decode()]

        # Add torax to the data
        fid = int(data_dict['fid_sequence'])
        joints_world = sample_metadata['joint_world'][fid]
        l_shoulder = joints_world[utils.H36M_CONF.joints.l_shoulder_idx]
        r_shoulder = joints_world[utils.H36M_CONF.joints.r_shoulder_idx]
        thorax = (l_shoulder + r_shoulder * 0.5).reshape([1, -1])
        joints_world = np.concatenate([joints_world, thorax], axis=0)

        # process joint positions
        joint_px, joint_cam, center, bbox = utils.world_to_camera(
            joints_world,
            utils.H36M_CONF.joints.root_idx,
            utils.H36M_CONF.joints.number,
            sample_metadata['R'],
            sample_metadata['T'].reshape(-1),
            sample_metadata['f'].reshape(-1),
            sample_metadata['c'].reshape(-1),
            utils.H36M_CONF.bbox_3d
        )

        # process image
        img = cv2.imread(data_dict['path'].decode())
        if not isinstance(img, np.ndarray):
            self._logger.error('Failed to load image %s',
                               data_dict['path'].decode())

        img = img[:utils.H36M_CONF.max_size, :utils.H36M_CONF.max_size]
        img = img.astype(np.float32)
        img /= 256
        img -= 0.5

        if self.transform:
            transformed = self.transform({'img': img})
            img = transformed['img']

        img = trasnf.ImageToTensor()(img)
        joint_px = trasnf.NumpyToTensor()(joint_px)
        joint_cam = trasnf.NumpyToTensor()(joint_cam)
        center = trasnf.NumpyToTensor()(center)
        bbox = trasnf.NumpyToTensor()(bbox)

        return img, joint_px, joint_cam, center, bbox
