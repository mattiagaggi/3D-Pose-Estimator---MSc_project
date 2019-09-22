import numpy as np
import torch
from neural_renderer.rasterize import rasterize_silhouettes
from neural_renderer.vertices_to_faces import vertices_to_faces
from utils.trans_numpy_torch import tensor_to_numpy, numpy_to_long, numpy_to_tensor
from sample.base.base_model import BaseModel
from sample.config.data_conf import PARAMS


IM_SIZE = PARAMS.data.im_size

class Rasterizer(BaseModel):

    def __init__(self, faces):

        super().__init__()
        faces = tensor_to_numpy(faces.cpu())
        faces = np.expand_dims(faces, 0)
        faces = np.repeat(faces,repeats=128*4, axis=0)
        self.faces = numpy_to_tensor(faces).int() # needs to be int (see neural renderer)
        self.faces = torch.cat((self.faces, self.faces[:, :, list(reversed(range(self.faces.shape[-1])))]), dim=1)
        self.reverse_rows = numpy_to_long(np.array(list(reversed(range(IM_SIZE)))))



    def pixel_coords_to_rasterization(self, pixel_coords):
        return (pixel_coords / (IM_SIZE/2)) - 1


    def rasterized_to_image(self, out_rasterisation):
        return torch.index_select(out_rasterisation, dim=1, index=self.reverse_rows)


    def forward(self, vertices):

        verts = self.pixel_coords_to_rasterization(vertices)
        n_verts = verts.size()[0]
        faces = vertices_to_faces(verts, self.faces[:n_verts])
        out_rast = rasterize_silhouettes(faces, IM_SIZE, anti_aliasing=True)
        image = self.rasterized_to_image(out_rast)
        return image





