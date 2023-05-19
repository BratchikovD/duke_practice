from torchreid import models


model_urls = {
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class ResnetNewLosses(models.resnet.ResNet):
    def __init__(self):
        super(ResnetNewLosses, self).__init__()

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet' or self.loss == 'arcface':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


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
