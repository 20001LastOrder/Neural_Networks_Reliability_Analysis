from argparse import ArgumentParser

from pytorch_lightning import Trainer

from models_evaluation.evaluations import CombinedNetwork
from module import CIFAR10Module
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import CIFAR10Data
import copy
import pytorch_lightning as pl
import torch


def main(args):
    model1 = copy.deepcopy(CIFAR10Module.load_from_checkpoint(args.model1_dir).model)
    model2 = copy.deepcopy(CIFAR10Module.load_from_checkpoint(args.model2_dir).model)
    #
    # print(torch.load(Path(args.model1_dir)).keys())
    # model1.load_state_dict(torch.load(Path(args.model1_dir))['state_dict'])
    # model2.load_state_dict(torch.load(Path(args.model2_dir))['state_dict'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = CombinedNetwork(model1, model2, args)
    data = CIFAR10Data(args)
    net.to(device)
    net.eval()
    t1, t2, tt, tf, ft = evaluate_diff(net, data.test_dataloader(), device)

    print('t1:', t1)
    print('t2:', t2)
    print('tt:', tt)
    print('tf:', tf)
    print('ft:', ft)
    # trainer = Trainer(
    #     gpus=-1,
    #     deterministic=False,
    #     weights_summary=None,
    #     log_every_n_steps=1,
    #     precision=32,
    # )
    #
    # data = CIFAR10Data(args)
    # trainer.test(net, data.test_dataloader())

def evaluate_diff(net: pl.LightningModule, dataloader: DataLoader, device):
    total = 0
    tt = 0
    tf = 0
    t1 = 0
    t2 = 0
    ft = 0
    for data in tqdm(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        total += labels.size(0)
        pred1, pred2 = net.forward(images)
        t1 += (pred1 == labels).sum().item()
        t2 += (pred2 == labels).sum().item()
        tt += ((pred1 == labels) * (pred1 == pred2)).sum().item()
        tf += ((pred1 == labels) * (pred1 != pred2)).sum().item()
        ft += ((pred2 == labels) * (pred1 != pred2)).sum().item()
    print(total)
    tt /= total
    tf /= total
    t1 /= total
    t2 /= total
    ft /= total
    return t1, t2, tt, tf, ft


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model1_dir', type=str)
    parser.add_argument('--model2_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    main(args)