from __future__ import division, print_function, absolute_import

from torchreid import engine
from losses import ContrastiveLoss


class ContrastiveEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, scheduler=None):
        super(ContrastiveEngine, self).__init__(datamanager, True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion = ContrastiveLoss()

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
        loss.backward()
        self.optimizer.step()

        return loss_summary
