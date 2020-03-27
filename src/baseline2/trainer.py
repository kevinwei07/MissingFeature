import os
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ipdb import set_trace as pdb
import json

import math

class Trainer:
    def __init__(self, device, trainData, validData, model1, model2, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model1 = model1
        self.model2 = model2
        self.criteria1 = torch.nn.MSELoss()         # loss_fn
        self.criteria2 = torch.nn.CrossEntropyLoss()
        self.opt = opt
        self.batch_size = batch_size
        self.arch = arch
        self.history = {'train': [], 'valid': []}
        self.scheduler = StepLR(self.opt, step_size=200, gamma=0.5)         # static update lr
        self.min_loss = math.inf

        if not os.path.exists(self.arch):
            os.makedirs(self.arch)

    def run_epoch(self, epoch, training, stage1):
        if stage1:
            self.model1.train(training)
        else:
            self.model1.train(False)
            self.model2.train(training)

        if training:
            description = '[Stage1 Train]' if stage1 else '[Stage2 Train]'
            dataset = self.trainData
            shuffle = True
        else:
            description = '[Stage1 Valid]' if stage1 else '[Stage2 Valid]'
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
        loss2 = 0
        accuracy = Accuracy()

        if stage1:
            for i, (x, y, miss) in trange:  # (x,y) = b*b
                pdb()
                o_f1, batch_loss = self.run_iter_stage1(x, miss)

                if training:
                    self.opt.zero_grad()  # reset gradient to 0
                    batch_loss.backward()  # calculate gradient
                    self.opt.step()  # update parameter by gradient

                loss += batch_loss.item()  # .item() to get python number in Tensor
                trange.set_postfix(loss=loss / (i + 1))

        else:
            for i, (x, y, miss) in trange: # (x,y) = b*b
                o_labels, batch_loss, missing_loss = self.run_iter_stage2(x, miss, y)       # x=(256, 8),  y=(256)
                loss2 += missing_loss.item()

                if training:
                    self.opt.zero_grad() # reset gradient to 0
                    batch_loss.backward() # calculate gradient
                    self.opt.step() # update parameter by gradient

                loss += batch_loss.item() #.item() to get python number in Tensor
                accuracy.update(o_labels.cpu(), y)

                trange.set_postfix(accuracy=accuracy.print_score(), loss=loss / (i + 1), missing_loss=loss2 / (i + 1))


    def run_iter_stage1(self, x, miss):
        # TODO : predict f1

        features = x.to(self.device)
        miss = miss.to(self.device)  # (b)
        o_miss = self.model1(features)  # (b, 12)
        l_loss = self.criteria1(o_miss, miss)

        return o_miss, l_loss

    def run_iter_stage2(self, x, miss, y):

        features = x.to(self.device)    # (256, 8)
        labels = y.to(self.device)      # (256)
        miss = miss.to(self.device)

        o_miss = self.model1(features)  # (256, 1)
        missing_loss = self.criteria1(o_miss, miss)

        full_fts = torch.cat((features, o_miss), dim=1)     # (256, 9)
        o_labels = self.model2(full_fts)

        l_loss = self.criteria2(o_labels, labels)

        return o_labels, l_loss, missing_loss

    def save_best_model(self, epoch):

        torch.save(self.model.state_dict(), f'{self.arch}/model.pkl')

    def save_hist(self):

        with open(f'{self.arch}/history.json', 'w') as f:
            # dump() : dict to str
            json.dump(self.history, f, indent=4)
