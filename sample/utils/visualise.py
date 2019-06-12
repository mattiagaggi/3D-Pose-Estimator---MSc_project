

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




sample_metadata=sio.loadmat(h36m_location+subdirfile+"h36m_meta.mat")
joints_world=sample_metadata['pose3d_world'][100]
print(joints_world.shape)
l_shoulder = joints_world[H36M_CONF.joints.l_shoulder_idx]
r_shoulder = joints_world[H36M_CONF.joints.r_shoulder_idx]
thorax = (l_shoulder + r_shoulder * 0.5).reshape([1, -1])
joints_world = np.concatenate([joints_world, thorax], axis=0)

im=cv2.imread(h36m_location+subdirfile+file)
img = im[:H36M_CONF.max_size, :H36M_CONF.max_size]
img = img.astype(np.float32)
img /= 256
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




print(joint_px[0,0],joint_px[0,1])
plt.scatter(joint_px[:-1,0],joint_px[:-1,1])
print('smnsan',joint_px[1,:])
plt.imshow(img)

bbox=bounding_box_pixel(joints_world,  H36M_CONF.joints.root_idx,sample_metadata['R'],
    sample_metadata['T'],
    sample_metadata['f'],
    sample_metadata['c'])

plt.scatter([bbox[0],bbox[2]+bbox[0]],[bbox[1],bbox[1]])
plt.scatter([bbox[0],bbox[2]+bbox[0]],[bbox[3]+bbox[1],bbox[3]+bbox[1]])

plt.show()







HOST_NAME = socket.gethostname()
MPL_MODE = 'agg'#TkAgg'
if socket.gethostname() == 'training':
    MPL_MODE = 'agg'


import logging
from enum import Flag
import cv2
import numpy as np
import matplotlib as mpl
import utils
mpl.use(MPL_MODE)
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
matplotlib.use('TkAgg')

#just *before*

import numpy as np
import matplotlib.pyplot as plt


class Drawer():
    """Class specifing visualization parameters"""

    _LIMB_COLOR = [0, 1, 2, 0, 3, 4, 5, 6, 7, 7, 5, 5, 8, 9, 9]
    _LIMBS = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [1, 7], [7, 8],
              [8, 9], [9, 10], [7, 11], [4, 11], [11, 12], [12, 13]]

    _COLORS = [[0, 0, 255], [0, 100, 0], [0, 255, 0], [0, 165, 255],
               [0, 255, 255], [255, 255, 0], [100, 0, 0], [255, 0, 0],
               [130, 0, 75], [255, 0, 255], [0, 0, 0]]

    def __init__(self, code=Style.PLANE_OFF, line=1, marker=1):
        """Initialization

        Arguments:
            code {Style} -- style of drawing (e.g. Style.BG_WHITE | Style.PLANE_OFF)

        Keyword Arguments:
            line {int} -- line width (default: {1})
            marker {int} -- joint size (default: {1})
        """

        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.bg_dark = False
        if bool(code & Style.BG_BLACK):
            plt.style.use('dark_background')
            self.bg_dark = True

        self.planes = True
        if bool(code & Style.PLANE_OFF):
            self.planes = False

        self.colors = _COLORS
        if bool(code & Style.SAME_COLOR):
            self.colors = '#ff0000'

        self.equal_axes = False
        if bool(code & Style.EQ_AXES):
            self.equal_axes = True

        # dual pose plot
        self.col_preds = '#ff0000'
        self.col_gt = '#0000ff'

        self.line = line
        self.marker = marker

    def _get_color(self, limb_id):
        """Get color depending on limb id

        Arguments:
            limb_id {int} -- limb ID

        Returns:
            str -- color
        """

        if isinstance(self.colors, str):
            return self.colors

        rgb = self.colors[self._LIMB_COLOR[limb_id]]
        color = '#{:02x}{:02x}{:02x}'.format(*rgb)
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
        if not self.bg_dark:
            bck_color = (1.0, 1.0, 1.0, 0.0)
        else:
            bck_color = (0.0, 0.0, 0.0, 0.0)
            ax.w_zaxis.set_pane_color((0.54, 0.48, 0.48, 1.0))

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
        """Plot pose 2D

        Arguments:
            image {numpy array} -- RGB image
            pose {numpy array} -- format (N_JOINTS x 2)

        Keyword Arguments:
            visibility {list} -- list of bools where True if
                                    joint is visible (default: {None})
        """

        # standardize image type
        img = image.copy()
        if img.dtype == np.float32:
            img = self._clip_to_max(img, max_value=1.0)
            img *= 255
        else:
            img = self._clip_to_max(img, max_value=255)

        ubyte_img = img.astype(np.uint8)
        img = cv2.cvtColor(ubyte_img,
                           cv2.COLOR_BGR2RGB)

        # checking joint visibility
        if visibility is None:
            visibility = [True] * pose.shape[0]

        # plot joints over image
        for lid, (p0, p1) in enumerate(self.LIMBS):
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


a=Drawer()
a.pose_3d(joints_world)










