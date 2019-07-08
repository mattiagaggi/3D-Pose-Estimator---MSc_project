
import torch.nn as nn
import torch



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm = False, padding = 1):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding),
                                       nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding),
                                       nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class unetUpNoSKip(nn.Module):

    def __init__(self, in_size, out_size, is_deconv=True, padding=1): # try false
        super(unetUpNoSKip, self).__init__()
        self.conv = unetConv2(out_size, out_size, True, padding) # note, changed to out_size, out_size for no skip
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), #scale factor
                             nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                             nn.BatchNorm2d(out_size),
                             nn.ReLU()
                             )

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(outputs2)






class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, padding):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False, padding)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.up = nn.Sequential(
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(in_size, out_size, 3, stride=1, padding=1),
                             nn.BatchNorm2d(out_size),
                             nn.ReLU()
                             )

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


