from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection







class Drawer:

    def __init__(self, kintree_table):
        self.kintree_table = kintree_table


    def display_model(self,
            model_info,
            model_faces=None,
            with_joints=False,
            batch_idx=0,
            plot = False,
            fig = None,
            savepath=None):
        """
        Displays mesh batch_idx in batch of model_info, model_info as returned by
        generate_random_model
        """
        if plot:
            assert fig is not None
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        verts, joints = model_info['verts'][batch_idx], model_info['joints'][
            batch_idx]
        if model_faces is None:
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
        else:
            mesh = Poly3DCollection(verts[model_faces], alpha=0.2) #[model_faces]


            face_color = (141 / 255, 184 / 255, 226 / 255)
            edge_color = (50 / 255, 50 / 255, 50 / 255)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
        if with_joints:
            self.draw_skeleton(joints, ax=ax)
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        #ax.get_xaxis().set_ticklabels([])
        #ax.get_yaxis().set_ticklabels([])
        #ax.set_zticklabels([])

        ax.view_init(azim=-90, elev=100)
        #ax.view_init(azim=0, elev=190)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if savepath:
            print('Saving figure at {}.'.format(savepath))
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        fig.canvas.draw()
        if plot:
            return fig
        w, h = fig.canvas.get_width_height()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape([w, h, 3])
        plt.close()
        return image


    def draw_skeleton(self,joints3D, ax=None, with_numbers=True):

        if ax is None:
            fig = plt.figure(frameon=False)
            ax = fig.add_subplot(111, projection='3d')
        colors = []
        left_right_mid = ['r', 'g', 'b']
        kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
        for c in kintree_colors:
            colors += left_right_mid[c]
        # For each 24 joint
        for i in range(1, self.kintree_table.shape[1]):
            j1 = self.kintree_table[0][i]
            j2 = self.kintree_table[1][i]
            ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                    [joints3D[j1, 1], joints3D[j2, 1]],
                    [joints3D[j1, 2], joints3D[j2, 2]],
                    color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
            #if with_numbers:
                #ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
        return ax
