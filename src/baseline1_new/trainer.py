import os
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import json

# fixed random state
torch.manual_seed(42)
import math
from ipdb import set_trace as pdb


class Trainer:
    def __init__(self, device, trainData, validData, model, criteria, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria = criteria          # loss_fn
        self.opt = opt
        self.batch_size = batch_size
        self.arch = arch
        self.history = {'train': [], 'valid': []}
        self.scheduler = StepLR(self.opt, step_size=200, gamma=0.5)         # static update lr
        self.min_loss = math.inf

        if not os.path.exists(self.arch):
            os.makedirs(self.arch)

    def run_epoch(self, epoch, training):
        self.model.train(training)

        if training:
            description = '[Train]'
            dataset = self.trainData
            shuffle = True
        else:
            description = '[Valid]'
            dataset = self.validData
            shuffle = False

        # dataloader for train and valid
        dataloader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = shuffle,
            num_workers = 8,
            collate_fn = dataset.collate_fn,
        )

        trange = tqdm(enumerate(dataloader), total = len(dataloader), desc = description)
        loss = 0
        accuracy = Accuracy()
        for i, (x, y) in trange:  # (x,y) = 128*128
            o_labels, batch_loss = self.run_iter(x, y)
            if training:
                self.opt.zero_grad()  # reset gradient to 0
                batch_loss.backward()  # calculate gradient
                self.opt.step()  # update parameter by gradient

            loss += batch_loss.item()  # .item() to get python number in Tensor
            accuracy.update(o_labels.cpu(), y)

            trange.set_postfix(accuracy=accuracy.print_score(), loss=loss / (i + 1))

        if training:
            self.history['train'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})
            self.scheduler.step()
        else:
            self.history['valid'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})
            if loss < self.min_loss:
                self.save_best_model(epoch)
            self.min_loss = loss

        self.save_hist()

    def run_iter(self, x, y):

        features = x.to(self.device)
        labels = y.to(self.device)  # (b)
        o_labels = self.model(features)  # (b, 12)
        l_loss = self.criteria(o_labels, labels)

        return o_labels, l_loss

    def save_best_model(self, epoch):

        torch.save(self.model.state_dict(), f'{self.arch}/model.pkl')

    def save_hist(self):

        with open(f'{self.arch}/history.json', 'w') as f:
            # dump() : dict to str
            json.dump(self.history, f, indent=4)


