from __future__ import absolute_import

from .resnet import resnet_arcface, resnet152


__model_factory = {
    'resnet_arcface': resnet_arcface,
    'resnet152': resnet152
}


def build_model(name, num_classes, loss='arcface', pretrained=True, use_gpu=True):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )
