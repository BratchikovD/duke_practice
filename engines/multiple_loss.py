from __future__ import division, print_function, absolute_import

from torchreid import metrics
from losses.center_loss import CenterLoss
from torchreid import losses, engine


class TripletCenterEngine(engine.Engine):
    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_triplet=1,
            weight_center=0.005,
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
        self.criterion_c = CenterLoss( )

    def forward_backward(self, data):
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
            loss_c = self.compute_loss(self.criterion_c, outputs, pids)
            loss += self.weight_c * loss_c
            loss_summary['loss_c'] = loss_c.item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
