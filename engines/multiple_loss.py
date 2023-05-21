from __future__ import division, print_function, absolute_import

from torchreid import metrics
from losses.center_loss import CenterLoss
from torchreid import losses, engine
import torch

class TripletCenterEngine(engine.Engine):
    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_triplet=1,
            weight_center=0.9,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(TripletCenterEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_triplet >= 0 and weight_center >= 0
        assert weight_triplet + weight_center > 0
        self.weight_t = weight_triplet
        self.weight_c = weight_center

        self.criterion_t = losses.TripletLoss(margin=margin)
        self.criterion_c = CenterLoss()
        self.optimizer_center = torch.optim.SGD(
            self.criterion_c.parameters(),
            lr=0.5,
            weight_decay=0.0005
        )

    def forward_backward(self, data):
        self.optimizer.zero_grad()
        self.optimizer_center.zero_grad()

        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_c > 0:
            loss_c = self.compute_loss(self.criterion_c, features, pids)
            loss += self.weight_c * loss_c
            loss_summary['loss_c'] = loss_c.item()

        assert loss_summary

        loss.backward()
        self.optimizer.step()
        for param in self.criterion_c.parameters():
            param.grad.data *= (1. / self.weight_c)

        self.optimizer_center.step()

        return loss_summary
