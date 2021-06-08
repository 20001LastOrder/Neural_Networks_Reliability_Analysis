import torch
import torch.utils.data as data
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
from itertools import combinations
import torch.nn.functional as F

def split_data_set(dataset, splits, seed=None):
    if (len(dataset) != sum(splits)):
        raise ValueError("The sum of splits must equal to the size of the     \
                          data set");
    generator = torch.Generator()
    if (seed is not None):
        generator.manual_seed(seed) 
    return torch.utils.data.random_split(dataset, splits, generator)


def get_data_loaders(dataset, train_batch_size, 
                     val_batch_size, val_ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dataset_size = len(dataset)
    indexes = np.arange(dataset_size)
    np.random.shuffle(indexes)
    split = int(dataset_size * (1 - val_ratio))
    train_sampler = data.SubsetRandomSampler(indexes[:split])
    val_sampler = data.SubsetRandomSampler(indexes[split:])
    return (data.DataLoader(dataset=dataset, batch_size=train_batch_size, 
                       sampler=train_sampler),
            data.DataLoader(dataset=dataset, batch_size=val_batch_size,
                       sampler=val_sampler))


# score function
def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


# methods for train
def train_model(model, train_loader, val_loader, epoches, criteria, optimizer,
                device, patience=5):
    trainer = create_supervised_trainer(model, optimizer, criteria, 
                                        device=device)
    val_metrics = {
        'acc': Accuracy(),
        'loss': Loss(criteria)
    }
    handler = EarlyStopping(patience=patience, score_function=score_function, 
                            trainer=trainer)
    pbar = ProgressBar()
    pbar.attach(trainer)
    evaluator = create_supervised_evaluator(model, metrics = val_metrics,
                                            device=device)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['acc']:.2f} Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['acc']:.2f} Avg loss: {metrics['loss']:.2f}")

    trainer.run(train_loader, max_epochs=epoches)
    pbar.close()
    return model 

def train_model_diversity(model, train_loader, val_loader, epoches, criteria, optimizer,
                device, target_model):
    val_metrics = {
        'acc': Accuracy(),
        'loss': Loss(criteria)
    }
    evaluator = create_supervised_evaluator(model, metrics = val_metrics,
                                            device=device)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.require_grad = False
    target_loss = torch.nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epoches):
        for i, data in enumerate(tqdm(train_loader, desc=f'training:')):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            away_targets = target_model(inputs)

            loss = criteria(outputs, labels) - \
                    criteria(away_targets, labels) * \
                    target_loss(F.log_softmax(outputs, dim=1), F.softmax(away_targets, dim=1))
            loss.backward()
            optimizer.step()

        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {epoch}  Avg accuracy: {metrics['acc']:.2f} Avg loss: {metrics['loss']:.2f}")

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {epoch}  Avg accuracy: {metrics['acc']:.2f} Avg loss: {metrics['loss']:.2f}")


def eval_acc(model, test_loader, device):
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(test_loader, desc=f'evaluating:')):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return correct / total


# Calculate the probablity that "nagates" nagative events happen
def calculate_prob(probs, negates):
    if negates == 0:
        return np.prod(probs)

    indexes = range(len(probs))
    combs = combinations(indexes, negates)
    prob = 0
    for comb in combs:
        comb = set(comb)
        eventProb = 1
        for i, p in enumerate(probs):
            eventProb *= (1 - p) if i in comb else p
        prob += eventProb
    return prob


def save_models(models, folder_path):
    folder_path.mkdir(parents=True, exist_ok=True)
    for model in models:
        torch.save(model.state_dict, folder_path / f'{model.name}.chkpt')

def load_models(models, folder_path):
    for model in models:
        model.load_state_dict(torch.load(folder_path / f'{model.name}.chkpt')())