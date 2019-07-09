# -*- coding: utf-8 -*-
"""
Code for generating heatmaps based on the
predictions made by OpenPose

@author: Denis Tome'

"""
import os
import torch
from tqdm import tqdm
from base import BaseTester
import utils

__all__ = [
    'HMGenerator'
]


class HMGenerator(BaseTester):
    """
    Trainer class, inherited from BaseTrainer
    """

    def __init__(self, model, data_loader, args):

        super().__init__(model, None,
                         data_loader, **vars(args))

        # load model
        self._logger.info('Loading model %s', args.resume)
        self.model_op.load_state_dict(torch.load(args.resume))

        self.global_idx = 0

    def _save_data(self, heatmaps, labels):

        for hm, gt in zip(heatmaps, labels):

            file_path = os.path.join(self.save_dir,
                                     self.output_name,
                                     '{:06d}.h5'.format(self.global_idx))
            data = {
                'hm': hm.tolist(),
                'p3d': gt.tolist()
            }
            utils.write_h5(file_path, data)

            self.global_idx += 1

    def test(self):
        """Generate heatmaps on training data"""

        self.model_op.eval()
        if self.with_cuda:
            self.model_op.cuda()

        pbar = tqdm(self.data_loader)
        for (img, joint_px, _) in pbar:

            img = self._get_var(img)

            # OpenPose predicted heatmaps
            hm = self.model_op(img)

            cpu_hm = hm.data.cpu().numpy()
            labels = joint_px.data.cpu().numpy()

            self._save_data(cpu_hm, labels)
