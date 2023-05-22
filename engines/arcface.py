from __future__ import division, print_function, absolute_import

import torchreid
from torchreid import engine, losses
from losses import ArcFaceLoss
import torch


class ImageArcFaceEngine(engine.Engine):
    def __init__(self, datamanager, model):
        super(ImageArcFaceEngine, self).__init__(datamanager, True)

        self.model = model
        self.criterion = ArcFaceLoss(
            in_features=1024,
            out_features=702,
        )
        self.optimizer = self.optimizer = torch.optim.Adam(list(self.model.parameters())
                                          + list(self.criterion.parameters()),
                                          lr=1e-3)
        self.scheduler = torchreid.optim.build_lr_scheduler(
            self.optimizer,
            lr_scheduler='multi_step',
            stepsize=[30, 45],
            max_epoch=60,
            gamma=0.1,
        )
        self.register_model('model', model, self.optimizer, self.scheduler)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        imgs = imgs.cuda()
        pids = pids.cuda()

        outputs, embeddings = self.model(imgs, labels=pids)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, embeddings, pids)
        loss_summary['loss'] = loss

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
