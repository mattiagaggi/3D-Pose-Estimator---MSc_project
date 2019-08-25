import numpy as np
from sample.config.data_conf import PARAMS
import torch
import torch.nn as nn
from neural_renderer.rasterize import rasterize_silhouettes
from neural_renderer.vertices_to_faces import vertices_to_faces
from utils.trans_numpy_torch import tensor_to_numpy, numpy_to_long, numpy_to_tensor
from sample.base.base_model import BaseModel
from sample.config.data_conf import PARAMS


IM_SIZE = PARAMS.data.im_size

class Rasterizer(BaseModel):

    def __init__(self, batch_size, faces):

        super().__init__()
        faces = tensor_to_numpy(faces.cpu())
        faces = np.expand_dims(faces, 0)
        faces = np.repeat(faces,repeats=batch_size, axis=0)
        self.faces = numpy_to_tensor(faces).int() # needs to be int (see neural renderer)
        self.reverse_rows = numpy_to_long(np.array(list(reversed(range(IM_SIZE)))))


    def pixel_coords_to_rasterization(self, pixel_coords):
        return (pixel_coords / (IM_SIZE//2)) - 1


    def rasterized_to_image(self, out_rasterisation):
        return torch.index_select(out_rasterisation, dim=1, index=self.reverse_rows)


    def forward(self, vertices):

        in_rast = self.pixel_coords_to_rasterization(vertices)
        out_rast = rasterize_silhouettes(in_rast, IM_SIZE)
        image = self.rasterized_to_image(out_rast)
        return image





