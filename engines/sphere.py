from torchreid import engine, losses
from losses import ArcFaceLoss
from losses import SphereLoss
import torch


class SphereFaceEngine(engine.Engine):
    def __init__(self, datamanager, model, scheduler=None):
        super(SphereFaceEngine, self).__init__(datamanager, True)

        self.model = model
        self.criterion = SphereLoss(1024, datamanager.num_train_pids)
        self.optimizer = torch.optim.Adam([self.model.parameters(), self.criterion.parameters()], lr=1e-3)
        self.scheduler = scheduler
        self.register_model('model', model, self.optimizer, scheduler)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, features = self.model(imgs, labels=pids)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, features, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary