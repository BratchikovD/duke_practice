from torchreid import models
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


model_urls = {
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, m=1.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = math.sqrt(2)*math.log(out_features-1)
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ResnetNewLosses(models.resnet.ResNet):
    def __init__(self, num_classes, loss, block, layers, **kwargs):
        super().__init__(num_classes, loss, block, layers, **kwargs)
        self.arc_block = ArcMarginProduct(2048, 702, easy_margin=True)

    def forward(self, x, labels=None):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v
        if self.loss == 'arcface':
            v = F.normalize(v, dim=1)
            y = self.arc_block(v, labels)

        y = self.classifier(v)

        if self.loss in ['softmax', 'arcface']:
            return y
        else:
            return y, v


def resnet_arcface(num_classes, loss='arcface', pretrained=True, **kwargs):
    model = ResnetNewLosses(
        num_classes=num_classes,
        loss=loss,
        block=models.resnet.Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        models.resnet.init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResnetNewLosses(
        num_classes=num_classes,
        loss=loss,
        block=models.resnet.Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        models.resnet.init_pretrained_weights(model, model_urls['resnet152'])
    return model
