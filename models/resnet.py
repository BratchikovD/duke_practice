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


class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, inputs, label):
        x = F.normalize(inputs)
        W = F.normalize(self.weight)
        cosine_theta = F.linear(x, W)
        phi_theta = self.m * torch.acos(cosine_theta)
        one_hot = torch.zeros_like(cosine_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi_theta) + ((1.0 - one_hot) * cosine_theta)

        return output


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = 30
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
        self.sphere = SphereFace(2048, 702)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn_after_fc = nn.BatchNorm1d(1024)

    def forward(self, x, labels=None):
        f = self.featuremaps(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[:2])
        x = self.bn2(x)
        x = self.dp(x)
        x = self.fc(x)
        embeddings = self.bn3(x)

        if self.loss == 'arcface':
            y = self.arc_block(embeddings, labels)
        elif self.loss == 'sphere':
            y = self.sphere(embeddings, labels)
        else:
            y = self.classifier(embeddings)

        if self.loss in ['softmax', 'arcface', 'sphere']:
            return y
        else:
            return y, embeddings


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
