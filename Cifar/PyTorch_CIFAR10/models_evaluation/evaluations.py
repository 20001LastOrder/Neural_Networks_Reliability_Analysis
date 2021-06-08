import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy


class CombinedNetwork(pl.LightningModule):
    def __init__(self, model1, model2, hparams):
        super().__init__()
        self.hparams = hparams
        self.model1 = model1
        self.model2 = model2

        self.accuracy = Accuracy()

    def forward(self, batch):
        pred1 = torch.argmax(self.model1(batch), dim=1)
        pred2 = torch.argmax(self.model2(batch), dim=1)
        return pred1, pred2


    def training_step(self, batch, batch_nb):
        raise RuntimeError

    def validation_step(self, batch, batch_nb):
        raise RuntimeError

    def test_step(self, batch, batch_nb):
        images, labels = batch
        n = labels.size(0)
        pred1, pred2 = self.forward(images)
        t1 = (pred1 == labels).sum().item() / n
        t2 = (pred2 == labels).sum().item() / n
        tt = ((pred1 == labels) * (pred1 == pred2)).sum().item() / n
        tf = ((pred1 == labels) * (pred1 != pred2)).sum().item() / n
        self.log('acc/t1', t1)
        self.log('acc/t2', t2)
        self.log('acc/tt', tt)
        self.log('acc/tf', tf)




