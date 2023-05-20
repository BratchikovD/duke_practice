from __future__ import division, print_function, absolute_import

from torchreid import engine, optim, losses
from losses import CenterLoss
import torch


class CenterLossEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, scheduler=None):
        super(CenterLossEngine, self).__init__(datamanager, True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion = CenterLoss()
        self.criterion_soft = losses.CrossEntropyLoss(702)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss_summary = {}

        loss_center = self.compute_loss(self.criterion, features, pids)
        loss_cross = self.compute_loss(self.criterion_soft, outputs, pids)
        loss = loss_center*0.7 + loss_cross
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
