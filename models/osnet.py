from torchreid import models


class OSNetNewLosses(models.osnet.OSNet):
    def __init__(self):
        super(OSNetNewLosses, self).__init__()

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
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


def osnet_x1_0_arcface(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNetNewLosses(
        num_classes,
        blocks=[models.osnet.OSBlock, models.osnet.OSBlock, models.osnet.OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        models.osnet.init_pretrained_weights(model, key='osnet_x1_0')
    return model

