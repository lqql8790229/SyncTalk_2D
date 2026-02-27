"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss:
    """VGG19-based perceptual loss.

    Compares feature representations at conv3_3 layer of VGG19.
    """

    def __init__(self, criterion=None, device=None):
        self.criterion = criterion or nn.MSELoss()
        self.device = device or torch.device("cpu")
        self._build_content_func()

    def _build_content_func(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        cnn = cnn.to(self.device)
        model = nn.Sequential()
        model = model.to(self.device)
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        self.content_func = model
        for param in self.content_func.parameters():
            param.requires_grad = False

    def get_loss(self, fake_im, real_im):
        f_fake = self.content_func(fake_im)
        f_real = self.content_func(real_im)
        f_real_no_grad = f_real.detach()
        return self.criterion(f_fake, f_real_no_grad)


_bce_loss = nn.BCELoss()


def cosine_loss(a, v, y):
    """Cosine similarity based BCE loss for SyncNet."""
    d = F.cosine_similarity(a, v)
    return _bce_loss(d.unsqueeze(1), y)
