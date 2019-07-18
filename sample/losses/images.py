import torch.nn

from sample.models.resnet18 import resnet18_loss


class ImageNetCriterium(torch.nn.Module):
    """
    Computes difference in the feature space of a NN pretrained on ImageNet
    """

    def __init__(self, criterion, device, weight=1, do_maxpooling=True):
        super().__init__()
        self.weight = weight
        self.criterion = criterion

        self.net = resnet18_loss(pretrained=True, num_channels=3, do_maxpooling=do_maxpooling)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.to(device)

    def forward(self, pred, label):
        preds_x = self.net(pred)
        labels_x = self.net(label)
        losses = [self.criterion(p, l) for p, l in zip(preds_x, labels_x)]

        return self.weight * sum(losses) / len(losses)


class L2_Resnet_Loss(torch.nn.Module):

    def __init__(self, device):
        super().__init__()

        self.loss_pixels = torch.nn.MSELoss()
        self.loss_img_net = ImageNetCriterium(self.loss_pixels,device,weight=2)

    def forward(self, out_im, pred):

        return (self.loss_pixels(out_im,pred)+self.loss_img_net(out_im,pred))/2
