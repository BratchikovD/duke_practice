from __future__ import division, print_function, absolute_import

from torchreid import engine, optim
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
        self.criterion_optimizer = torch.optim.Adam(
            self.criterion.parameters(),
            lr=0.5
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, features, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        self.criterion_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        for param in self.criterion.parameters():
            param.grad.data *= (1. / 1)
            self.criterion_optimizer.step()

        return loss_summary
