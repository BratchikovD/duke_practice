from __future__ import division, print_function, absolute_import

from torchreid import engine, losses
from losses import ArcFaceLoss
import torch


class ImageArcFaceEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, margin=0.5, scheduler=None):
        super(ImageArcFaceEngine, self).__init__(datamanager, True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion = ArcFaceLoss(702, datamanager.num_train_pids, margin)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, outputs, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
