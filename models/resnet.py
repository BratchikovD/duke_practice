import torch
import torch.nn as nn
import torchvision
from torchreid import models
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
model_urls = {
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


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
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn_after_fc = nn.BatchNorm1d(1024)
        if self.loss == 'sphere' or self.loss == 'arcface':
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, labels=None):
        f = self.featuremaps(x)

        if self.loss == 'sphere' or self.loss == 'arcface':
            v = F.avg_pool2d(f, f.size()[2:]).view(f.size()[:2])
            v = self.bn2(v)
            v = self.dp(v)
            embeddings = self.fc(v)
            embeddings = self.bn_after_fc(embeddings)
        else:
            v = self.global_avgpool(f)
            v = v.view(v.size(0), -1)
            embeddings = v

        y = self.classifier(embeddings)

        if not self.training:
            return embeddings

        if self.loss in ['softmax']:
            return y
        else:
            return y, embeddings


class NetworkD(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NetworkD, self).__init__(*args, **kwargs)
        resnet50 = torchvision.models.resnet50()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = create_layer(64, 64, 3, stride=1)
        self.layer2 = create_layer(256, 128, 4, stride=2)
        self.layer3 = create_layer(512, 256, 6, stride=2)
        self.layer4 = create_layer(1024, 512, 3, stride=1)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=2048, out_features=1024, bias=True)
        self.bn3 = nn.BatchNorm1d(1024)

        # load pretrained weights and initialize added weight
        pretrained_state = model_zoo.load_url(resnet50_url)
        state_dict = self.state_dict()
        for k, v in pretrained_state.items():
            if 'fc' in k:
                continue
            state_dict.update({k: v})
        self.load_state_dict(state_dict)
        nn.init.kaiming_normal_(self.fc.weight, a=1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[:2])
        x = self.bn2(x)
        x = self.dp(x)
        x = self.fc(x)
        embd = self.bn3(x)
        if not self.training:
            embd_norm = torch.norm(embd, 2, 1, True).clamp(min=1e-12).expand_as(embd)
            embd = embd / embd_norm
        return embd


class Bottleneck(nn.Module):
    def __init__(self, in_chan, mid_chan, stride=1, stride_at_1x1=False, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)

        out_chan = 4 * mid_chan
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1,
                bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample == None:
            residual = x
        else:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def create_layer(in_chan, mid_chan, b_num, stride):
    out_chan = mid_chan * 4
    blocks = [Bottleneck(in_chan, mid_chan, stride=stride),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, mid_chan, stride=1))
    return nn.Sequential(*blocks)


def network_d(num_classes, loss='sphere', pretrained=True, **kwargs):
    return NetworkD()


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
