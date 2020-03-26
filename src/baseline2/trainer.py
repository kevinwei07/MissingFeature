import os
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import json

import math

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
            collate_fn = dataset.collate_fn_f1,
        )
        dataloader_f1 = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle = shuffle,
            num_workers = 8,
            collate_fn = dataset.collate_fn_f1,
        )

        trange = tqdm(enumerate(dataloader_f1), total = len(dataloader_f1), desc = description)
        loss = 0
        accuracy = Accuracy()

        for i, (x, y) in trange:  # (x,y) = 128*128
            f1, batch_loss = self.run_iter_f1(x, y)

            if training:
                self.opt.zero_grad()  # reset gradient to 0
                batch_loss.backward()  # calculate gradient
                self.opt.step()  # update parameter by gradient

            loss += batch_loss.item()  # .item() to get python number in Tensor
            accuracy.update(f1.cpu(), y)

            trange.set_postfix(accuracy=accuracy.print_score(), loss=loss / (i + 1))
        pass

    def run_iter_f1(self,x,y):

        # TODO : predict f1

        features = x.to(self.device)
        f1 = y.to(self.device)  # (b)
        o_f1 = self.model(features)  # (b, 12)
        l_loss = self.criteria(o_f1, f1)

        return o_f1, l_loss

    def save_best_model(self, epoch):

        torch.save(self.model.state_dict(), f'{self.arch}/model.pkl')

    def save_hist(self):

        with open(f'{self.arch}/history.json', 'w') as f:
            # dump() : dict to str
            json.dump(self.history, f, indent=4)
