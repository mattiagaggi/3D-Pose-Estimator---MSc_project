
from sample.config.encoder_decoder import ENCODER_DECODER_PARAMS

import torch.nn as nn
import torch
from sample.base.base_model import BaseModel
from sample.base.base_modules import unetConv2,unetUpNoSKip



#batch x channel x height x width



class Encoder(BaseModel):
    def __init__(self, batch_size,
                 input_im_size, filter_list):

        super().__init__()

        self.batch_size, self.input_im_size = batch_size, input_im_size
        num_encoding_layers = len(filter_list)
        self.filters = filter_list
        self.dimension_L_app = 128 #apperance dimensions
        self.dimension_L_3D = 200*3
        self.latent_dropout = 0.3


        #try encoding one more

        ##################################
        self.encoder_resolution = input_im_size // (2 ** (num_encoding_layers - 1))
        self.encoder_output_features = self.encoder_resolution ** 2 * self.filters[num_encoding_layers - 1]


        self.conv1 = nn.Sequential(unetConv2(3, self.filters[0]), nn.MaxPool2d(kernel_size = 2))
        self.conv2 = nn.Sequential( unetConv2(self.filters[0],self.filters[1]),nn.MaxPool2d(kernel_size = 2))
        self.conv3 = nn.Sequential( unetConv2(self.filters[1],self.filters[2]),nn.MaxPool2d(kernel_size = 2))
        self.conv4 = unetConv2(self.filters[2],self.filters[3])

        self.to_Lapp = nn.Sequential(nn.Linear(self.encoder_output_features, self.dimension_L_app),
                                     nn.Dropout(inplace=True, p=self.latent_dropout),
                                     nn.ReLU(inplace=False))

        self.to_L3d = nn.Sequential(nn.Linear(self.encoder_output_features, self.dimension_L_3D),
                                    nn.Dropout(inplace=True, p=self.latent_dropout)  # removing dropout degrades results

                                    )

    def forward(self, x):

        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)


        out=out.view(self.batch_size, -1)
        L_3d = self.to_L3d(out)
        L_app= self.to_Lapp(out)
        outputs = {'L_3d':L_3d, 'L_app':L_app} #flattened
        return outputs


class Rotation(BaseModel):
    def __init__(self,
                 batch_size,
                 dimensions_3d,
                 rotation_encoding_dimension = 128, implicit_rotation = False):

        super().__init__()

        self.implicit_rotation = implicit_rotation
        self.batch_size = batch_size
        if self.implicit_rotation:
            self.dimension_3d = dimensions_3d
            self.rotation_encoding_dimension = rotation_encoding_dimension
            self.latent_dropout = 0.3



            self.encode_angle = nn.Sequential(nn.Linear(3 * 3, rotation_encoding_dimension // 2),
                                              nn.Dropout(inplace=True, p=self.latent_dropout),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(rotation_encoding_dimension // 2, rotation_encoding_dimension),
                                              nn.Dropout(inplace=True, p=self.latent_dropout),
                                              nn.ReLU(inplace=False),
                                              nn.Linear(rotation_encoding_dimension, rotation_encoding_dimension),
                                              )
            self.rotate_implicitely = nn.Sequential(nn.Linear(self.dimension_3d + rotation_encoding_dimension, self.dimension_3d),
                                                    nn.Dropout(inplace=True, p=self.latent_dropout),
                                                    nn.ReLU(inplace=False))

    def forward(self,dic):
        L_3d=dic["L_3d"]
        rotation_input = dic["R"]
        if self.implicit_rotation:
            rotation_input = rotation_input.view(self.batch_size,9)
            angle = self.encode_angle(rotation_input)
            concatenated = torch.cat((angle,L_3d), dim=1)
            L_3d_rotated = self.rotate_implicitely(concatenated)
        else:
            L_3d_rotated = torch.bmm(L_3d.view(self.batch_size,-1,3), rotation_input.transpose(1,2))
            L_3d_rotated = L_3d_rotated.view_as(L_3d)
        return L_3d_rotated







class Decoder(BaseModel):
    def __init__(self,
                 batch_size,
                 L_3d_input_channels):

        super().__init__()
        self.batch_size = batch_size
        self. L_3d_input_channels =  L_3d_input_channels

        self.decoded_channels_L = 512
        self.decoded_channels_Lapp= 128
        self.encoded_im_size= 16
        self.latent_dropout = 0.3
        self.L3_conv_channels = self.decoded_channels_L-self.decoded_channels_Lapp
        self.feature_map_dimensions = self.L3_conv_channels * self.encoded_im_size**2

        #####################################

        self.full_layer = nn.Sequential(nn.Linear(self. L_3d_input_channels, self.feature_map_dimensions),
                                   nn.Dropout(inplace=True, p=self.latent_dropout),
                                   nn.ReLU(inplace=False))

        self.conv1 = unetUpNoSKip(in_size = self.decoded_channels_L, out_size = self.decoded_channels_L//2 ,is_deconv = True)
        self.conv2 = unetUpNoSKip(in_size = self.decoded_channels_L//2, out_size = self.decoded_channels_L//4,is_deconv = True)
        self.conv3 = unetUpNoSKip(in_size = self.decoded_channels_L//4, out_size = self.decoded_channels_L//8,is_deconv=True)
        self.conv4 = unetConv2(in_size = self.decoded_channels_L//8, out_size = self.decoded_channels_L//8)
        #self.conv4 = unetUpNoSKip(in_size=self.decoded_channels_L // 4, out_size=self.decoded_channels_L // 8, is_deconv=True)

    def forward(self, dic):

        L_3d=dic["L_3d"]
        L_app=dic["L_app"]
        L_app = L_app.view(self.batch_size, self.decoded_channels_Lapp, 1, 1).expand(self.batch_size,
                                                                                        self.decoded_channels_Lapp,
                                                                                        self.encoded_im_size,
                                                                                        self.encoded_im_size)

        L_3d_conv = self.full_layer(L_3d)
        L_3d_conv = L_3d_conv.view(self.batch_size,
                                               self.L3_conv_channels,
                                               self.encoded_im_size,
                                               self.encoded_im_size)
        L = torch.cat((L_app, L_3d_conv), dim=1)



        out = self.conv1(L)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class Encoder_Decoder(BaseModel):
    def __init__(self,
                 batch_size,
                 input_im_size = ENCODER_DECODER_PARAMS.encoder_decoder.im_size):
        super().__init__()

        self.input_im_size = input_im_size #input image shape
        self.batch_size = batch_size



        #encoder_parameters
        self.filters = [64, 128, 256, 512]
        self.encoder = Encoder(batch_size, input_im_size, self.filters)

        self.dimension_L_app = self.encoder.dimension_L_app
        self.dimension_L_3D = self.encoder.dimension_L_3D
        self.encoder_resolution = self.encoder.encoder_resolution #im size
        self.rotation = Rotation(self.batch_size,self.dimension_L_3D)
        self.decoder = Decoder(self.batch_size, self.dimension_L_3D )
        self.final_linear = nn.Conv2d(self.filters[0]+3, 3, 1)

        #self.to_pose = MLP.MLP_fromLatent(d_in=self.dimension_3d, d_hidden=2048, d_out=51, n_hidden=n_hidden_to3Dpose,
        #                                  dropout=0.5)

    def forward(self, dic):


        im, index_invert = dic['im_in'], dic['invert_segments']
        encode = self.encoder(im)
        L_3d = encode['L_3d']
        L_app= encode['L_app']
        dic_rot = {'L_3d' : L_3d,'R': dic['rot_im']}
        L_3d_rotated = self.rotation(dic_rot)
        #self._logger.info("L3d and Lapp not swapped")
        #L_app_swapped=L_app
        L_app_swapped = torch.index_select(L_app, dim=0, index=index_invert)
        background = dic['background_target']
        dic_dec = { "L_3d": L_3d_rotated, "L_app" : L_app_swapped}
        decoded = self.decoder(dic_dec)
        concatenated = torch.cat((decoded,background), dim=1)
        out_image = self.final_linear(concatenated)

        return out_image













