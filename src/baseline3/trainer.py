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
    def __init__(self, device, trainData, validData, model, criteria1, criteria2, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria1 = criteria1          # loss_fn
        self.criteria2 = criteria2
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
        for i, (x, y,miss) in trange:  # (x,y) = 128*128
            o_labels, batch_loss = self.run_iter(x,y,miss)
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

    def run_iter(self, x, y,miss):
        features = x.to(self.device)    # (256, 8)
        labels = y.to(self.device)      # (256)
        miss = miss.to(self.device)

        o_miss, o_labels = self.model(features)  # (256, 1)

        missing_loss = self.criteria1(o_miss, miss)
        l_loss = self.criteria2(o_labels, labels)
        batch_loss = missing_loss + l_loss


        return o_labels, batch_loss

    def save_best_model(self, epoch):

        torch.save(self.model.state_dict(), f'{self.arch}/model.pkl')

    def save_hist(self):

        with open(f'{self.arch}/history.json', 'w') as f:
            # dump() : dict to str
            json.dump(self.history, f, indent=4)


