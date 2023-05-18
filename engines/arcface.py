from __future__ import division, print_function, absolute_import

from torchreid import engine
from ..losses.arcface import ArcfaceLoss


class ImageArcFaceEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, feature_scale=30, margin=0.5, scheduler=None):
        super(ImageArcFaceEngine, self).__init__(datamanager, True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion = ArcfaceLoss(2048, 702, feature_scale, margin)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        loss = self.compute_loss(self.criterion, features, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
