from torchreid import engine, losses
from losses import ArcFaceLoss
import torch


class SphereFaceEngine(engine.Engine):
    def __init__(self, datamanager, model, optimizer, scheduler=None):
        super(SphereFaceEngine, self).__init__(datamanager, True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion = losses.CrossEntropyLoss(
            num_classes=datamanager.num_train_pids,
            use_gpu=True,
            label_smooth=True
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs = self.model(imgs, labels=pids)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, outputs, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
