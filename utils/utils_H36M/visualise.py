

import cv2
import numpy as np
import matplotlib.pyplot as plt
#for 3D plot
from mpl_toolkits.mplot3d import Axes3D


from utils.utils_H36M.common import H36M_CONF


class Drawer:
    """Class specifing visualization parameters"""


    def __init__(self, line=1, marker=2, visibility=None):

        self.limbs_names=['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
        self._limb_color = [0, 1, 2, 0, 3, 4, 5, 6, 7, 7, 5, 5, 8, 9, 5, 5,10,11,7]
        self._limbs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [1, 7], [7, 8],
                  [8, 9], [9, 10], [7, 11], [4, 7], [11, 12], [12, 13], [7,14], [14,11], [14,15],[15,16], [0,7]]

        self._colors = [[255, 255, 100], [0, 100, 0], [0, 255, 0], [0, 165, 255],
                   [0, 255, 255], [255, 255, 0], [100, 0, 0], [255, 0, 0],
                   [130, 0, 75], [255, 0, 255], [0, 0, 255],[0, 0, 150]]

        if visibility is None:
            self.visibility = [True] * H36M_CONF.joints.number
        else:
            self.visibility = visibility

        # dual pose plot
        self.equal_axes = True
        self.planes = False

        self.col_preds = '#ff0000'
        self.col_gt = '#0000ff'

        self.line = line
        self.marker = marker
        #tranbsparency 3D plot
        self.transparency=0.2

    def _get_color(self, limb_id):

        color = self._colors[self._limb_color[limb_id]]

        return color

    def _scale_plot(self, pose, ax):
        """Scale plot according to data

        Arguments:
            pose {numpy array} -- 2D or 3D pose
            ax {ax} -- ax contained in figure
        """
        root=H36M_CONF.joints.root_idx
        box=H36M_CONF.bbox_3d
        box_mean=np.array(box)/2
        smallest=pose[root, :] - box_mean
        largest=pose[root, :] + box_mean
        ax.set_xlim3d(smallest[0], largest[0])
        ax.set_ylim3d(smallest[1], largest[1])
        ax.set_zlim3d(smallest[2], largest[2])


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

    def rgb_to_string(self, rgb):

        color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        return color

    def get_image(self, img):
        if img.dtype == np.float32:
            img = self._clip_to_max(img, max_value=1.0)
            img *= 255
        else:
            img = self._clip_to_max(img, max_value=255)

        ubyte_img = img.astype(np.uint8)
        img = cv2.cvtColor(ubyte_img,
                           cv2.COLOR_BGR2RGB)
        return img

    def pose_2d(self,fig,img, pose):
        img=self.get_image(img)
        # plot joints over image
        for lid, (p0, p1) in enumerate(self._limbs):
            x0, y0 = pose[p0].astype(np.int)
            x1, y1 = pose[p1].astype(np.int)
            if self.visibility[p0]:
                cv2.circle(img, (x0, y0), self.marker,
                           self._get_color(lid), -1)

            if self.visibility[p1]:
                cv2.circle(img, (x1, y1), self.marker,
                           self._get_color(lid), -1)

            if self.visibility[p0] and self.visibility[p1]:
                cv2.line(img, (x0, y0), (x1, y1),
                         self._get_color(lid), self.line, 16)
        plt.imshow(img)
        return fig


    def pose_3d(self, fig, pose):
        assert pose.shape == (H36M_CONF.joints.number,3)
        ax = fig.add_subplot(111, projection='3d')
        for lid, (p0, p1) in enumerate(self._limbs):
            col = self.rgb_to_string(self._get_color(lid))
            ax.plot([pose[p0, 0], pose[p1, 0]],
                    [pose[p0, 1], pose[p1, 1]],
                    [pose[p0, 2], pose[p1, 2]], c=col,
                    linewidth=self.line)
            ax.scatter(pose[p0, 0], pose[p0, 1], pose[p0, 2], c=col,
                       marker='o', edgecolor=col, s=self.marker)
            ax.scatter(pose[p1, 0], pose[p1, 1], pose[p1, 2], c=col,
                       marker='o', edgecolor=col, s=self.marker)
        if self.equal_axes:
            self._scale_plot(pose, ax)
        if not self.planes:
            self._hide_planes(ax)

        fig.canvas.draw()
        return fig

    def poses_3d(self, fig, predicted, gt):

        ax = fig.add_subplot(111, projection='3d')
        for lid,(p0, p1) in enumerate(self._limbs):
            col = self.rgb_to_string(self._get_color(lid))
            # plotting predicted pose
            ax.plot([predicted[p0, 0], predicted[p1, 0]],
                    [predicted[p0, 1], predicted[p1, 1]],
                    [predicted[p0, 2], predicted[p1, 2]], c=col,
                    linewidth=self.line)#,linestyle="dashed")
            ax.scatter(predicted[p0, 0], predicted[p0, 1], predicted[p0, 2], c=col,
                       marker='o', edgecolor=col, s=self.marker)
            ax.scatter(predicted[p1, 0], predicted[p1, 1], predicted[p1, 2], c=col,
                       marker='o', edgecolor=col, s=self.marker)

            # plotting ground truth predicted
            ax.plot([gt[p0, 0], gt[p1, 0]],
                    [gt[p0, 1], gt[p1, 1]],
                    [gt[p0, 2], gt[p1,2]], c='k',
                    linewidth=self.line,alpha=self.transparency)
            ax.scatter(gt[p0, 0], gt[p0, 1], gt[p0, 2], c='k',
                       marker='o', edgecolor='k', s=self.marker,alpha=self.transparency)
            ax.scatter(gt[p1, 0], gt[p1, 1], gt[p1, 2], c='k',
                       marker='o', edgecolor='k', s=self.marker,alpha=self.transparency)
        if self.equal_axes:
            self._scale_plot(gt, ax)

        if not self.planes:
            self._hide_planes(ax)
        fig.canvas.draw()
        return fig





