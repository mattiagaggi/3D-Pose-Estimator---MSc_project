

from data.directories_location import h36m_location
subdirfile="s_01_act_02_subact_01_ca_01/"
file="s_01_act_02_subact_01_ca_01_000100.jpg"


import scipy.io as sio
import numpy as np
from easydict import EasyDict as edict
import socket
import cv2
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt





def load_mat(subdirectory=subdirfile):
    data=sio.loadmat(h36m_location+subdirectory+"h36m_meta.mat")
    points_3D=data['pose3d_world'] #3D numpy array
    cam_r=data['R']
    cam_t=data['T']
    f=data['f']
    c=data['c']
    return points_3D,cam_r,cam_t,f,c










import logging
from enum import Flag
import cv2
import numpy as np
from sample.utils.common import H36M_CONF
from sample.utils.transformations import world_to_pixel
from sample.utils.transformations import bounding_box_pixel













import logging
from enum import Flag
import cv2
import numpy as np
import matplotlib as mpl
import utils

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


__all__ = [
    'Style',
    'Drawer',
]

_C_LIMBS_IDX = [0, 1, 2, 10,
                0, 3, 4,
                5, 6, 7, 7,
                5, 8, 9, 9]

_COLORS = [[0, 0, 255], [0, 100, 0], [0, 255, 0], [0, 165, 255], [0, 255, 255],
           [255, 255, 0], [100, 0, 0], [255, 0, 0], [130, 0, 75], [255, 0, 255], [0, 0, 0]]

import matplotlib as mpl
class Style(Flag):
    """Class defining the drawing style"""

    BG_BLACK = 0b0001
    PLANE_OFF = 0b0010
    SAME_COLOR = 0b0100
    EQ_AXES = 0b1000
    NONE = 0b000000



import matplotlib


#just *before*


import matplotlib.pyplot as plt


class Drawer:
    """Class specifing visualization parameters"""


    def __init__(self, line=1, marker=2):

        self.LIMBS_NAMES=[]
        self._LIMB_COLOR = [0, 1, 2, 0, 3, 4, 5, 6, 7, 7, 5, 5, 8, 9, 5, 5,10,11,7]
        self._LIMBS = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [1, 7], [7, 8],
                  [8, 9], [9, 10], [7, 11], [4, 7], [11, 12], [12, 13], [7,14], [14,11], [14,15],[15,16], [0,7]]

        self._COLORS = [[255, 255, 100], [0, 100, 0], [0, 255, 0], [0, 165, 255],
                   [0, 255, 255], [255, 255, 0], [100, 0, 0], [255, 0, 0],
                   [130, 0, 75], [255, 0, 255], [0, 0, 255],[0, 0, 150]]

        # dual pose plot
        self.col_preds = '#ff0000'
        self.col_gt = '#0000ff'

        self.line = line
        self.marker = marker

    def _get_color(self, limb_id):

        color = self._COLORS[self._LIMB_COLOR[limb_id]]

        return color

    def _scale_plot(self, pose, ax):
        """Scale plot according to data

        Arguments:
            pose {numpy array} -- 2D or 3D pose
            ax {ax} -- ax contained in figure
        """

        val = np.max([np.abs(pose.min()),
                      np.abs(pose.max())])
        smallest = - val
        largest = val
        ax.set_xlim3d(smallest, largest)
        ax.set_ylim3d(smallest, largest)
        ax.set_zlim3d(smallest, largest)
        ax.set_aspect('equal')

    def _hide_planes(self, ax):
        """Hide planes

        Arguments:
            ax {ax} -- ax contained in figure
        """

        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        # Get rid of the panes

        ax.w_xaxis.set_pane_color(bck_color)
        ax.w_yaxis.set_pane_color(bck_color)

        # Get rid of the lines in 3d
        ax.w_xaxis.line.set_color(bck_color)
        ax.w_yaxis.line.set_color(bck_color)
        ax.w_zaxis.line.set_color(bck_color)

    def _clip_to_max(self, image, max_value):
        """Clip image intensities to the maximum
        value defined by max

        Arguments:
            image {numpy array} -- RGB image
            max_value {float | int} -- maximum value

        Returns:
            numpy array -- clipped image
        """

        shape = image.shape
        img = image.flatten()
        idx = np.where(img > max_value)[0]
        img[idx] = max_value

        new_img = img.reshape(shape)
        return new_img

    def pose_2d(self, image, pose, visibility=None):


        # standardize image type
        img = image.copy()
        if img.dtype == np.float32:
            print("not else")
            img = self._clip_to_max(img, max_value=1.0)
            img *= 255
        else:
            print("else")
            img = self._clip_to_max(img, max_value=255)

        ubyte_img = img.astype(np.uint8)
        img = cv2.cvtColor(ubyte_img,
                           cv2.COLOR_BGR2RGB)

        # checking joint visibility
        if visibility is None:
            visibility = [True] * pose.shape[0]

        # plot joints over image
        for lid, (p0, p1) in enumerate(self._LIMBS):
            x0, y0 = pose[p0].astype(np.int)
            x1, y1 = pose[p1].astype(np.int)

            if visibility[p0]:
                cv2.circle(img, (x0, y0), self.marker,
                           self._get_color(lid), -1)

            if visibility[p1]:
                cv2.circle(img, (x1, y1), self.marker,
                           self._get_color(lid), -1)

            if visibility[p0] and visibility[p1]:
                cv2.line(img, (x0, y0), (x1, y1),
                         self._get_color(lid), self.line, 16)
        return img

    def pose_3d(self, pose, plot=False):
        """Plot 3D pose

        Arguments:
            pose {numpy array} -- format (N_JOINTS x 3)

        Keyword Arguments:
            plot {bool} -- plot or return image (default: {False})

        Returns:
            numpy array -- rgb image
        """

        #pose = utils.standardize_pose(pose, dim=3)
        pose = pose.transpose([1, 0])

        # generate figure
        fig = plt.figure(num=None, figsize=(8, 8),
                         dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(111, projection='3d')
        for lid, (p0, p1) in enumerate(self._LIMBS):
            col = self._get_color(lid)
            ax.plot([pose[0, p0], pose[0, p1]],
                    [pose[1, p0], pose[1, p1]],
                    [pose[2, p0], pose[2, p1]], c=col,
                    linewidth=self.line)
            ax.scatter(pose[0, p0], pose[1, p0], pose[2, p0], c=col,
                       marker='o', edgecolor=col, s=self.marker)
            ax.scatter(pose[0, p1], pose[1, p1], pose[2, p1], c=col,
                       marker='o', edgecolor=col, s=self.marker)

        if self.equal_axes:
            self._scale_plot(pose, ax)

        if not self.planes:
            self._hide_planes(ax)

        fig.canvas.draw()

        if not plot:
            w, h = fig.canvas.get_width_height()
            image = np.fromstring(fig.canvas.tostring_rgb(),
                                  dtype='uint8')
            image = image.reshape([h, w, 3])
            plt.close(fig)

            return image

        plt.show()
        return fig

    def poses_3d(self, predicted, gt, plot=False):
        """Plot predicted and ground truth poses

        Arguments:
            predicted {numpy array} -- format undefined
            gt {numpy array} -- format undefined

        Keyword Arguments:
            plot {bool} -- plot or return image (default: {False})

        Returns:
            numpy array -- rgb image
        """

        predicted = utils.standardize_pose(predicted, dim=3)
        predicted = predicted.transpose([1, 0])
        gt = utils.standardize_pose(gt, dim=3)
        gt = gt.transpose([1, 0])

        # generate figure
        fig = plt.figure(num=None, figsize=(8, 8),
                         dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(111, projection='3d')
        for (p0, p1) in self._LIMBS:
            # plotting predicted pose
            ax.plot([predicted[0, p0], predicted[0, p1]],
                    [predicted[1, p0], predicted[1, p1]],
                    [predicted[2, p0], predicted[2, p1]], c=self.col_preds,
                    linewidth=self.line)
            ax.scatter(predicted[0, p0], predicted[1, p0], predicted[2, p0], c=self.col_preds,
                       marker='o', edgecolor=self.col_preds, s=self.marker)
            ax.scatter(predicted[0, p1], predicted[1, p1], predicted[2, p1], c=self.col_preds,
                       marker='o', edgecolor=self.col_preds, s=self.marker)

            # plotting ground truth predicted
            ax.plot([gt[0, p0], gt[0, p1]],
                    [gt[1, p0], gt[1, p1]],
                    [gt[2, p0], gt[2, p1]], c=self.col_gt,
                    linewidth=self.line)
            ax.scatter(gt[0, p0], gt[1, p0], gt[2, p0], c=self.col_gt,
                       marker='o', edgecolor=self.col_gt, s=self.marker)
            ax.scatter(gt[0, p1], gt[1, p1], gt[2, p1], c=self.col_gt,
                       marker='o', edgecolor=self.col_gt, s=self.marker)

        if self.equal_axes:
            self._scale_plot(gt, ax)

        if not self.planes:
            self._hide_planes(ax)

        fig.canvas.draw()

        if not plot:
            w, h = fig.canvas.get_width_height()
            image = np.fromstring(fig.canvas.tostring_rgb(),
                                  dtype='uint8')
            image = image.reshape([h, w, 3])
            plt.close(fig)

            return image

        plt.show()
        return fig








sample_metadata=sio.loadmat(h36m_location+subdirfile+"h36m_meta.mat")
joints_world=sample_metadata['pose3d_world'][100]
im=cv2.imread(h36m_location+subdirfile+file)



#img = im[:H36M_CONF.max_size, :H36M_CONF.max_size]
#img = img.astype(np.float32)
#img /= 256
#img -= 0.5







joint_px, center= world_to_pixel(
    joints_world,
    H36M_CONF.joints.root_idx,
    H36M_CONF.joints.number,
    sample_metadata['R'],
    sample_metadata['T'],
    sample_metadata['f'],
    sample_metadata['c']
)



#jointspixel2cam(pixel_coord, f, c)

#16,15,14 right arm up down

#13, 12, 11 left arm down up


#10,9,8 spine up down


#4,5,6 left leg up down

#1 2 3 right leg up down


#11 14

#7 14

#16 15

#15 14

#11-4 chaneg to 4-7


print(joint_px[0,0],joint_px[0,1])
plt.scatter(joint_px[10,0],joint_px[10,1])
plt.scatter(joint_px[7,0],joint_px[7,1])

print('smnsan',joint_px[1,:])
a=Drawer()
img=a.pose_2d(img,joint_px[:,:-1])

plt.imshow(img)


plt.show()




