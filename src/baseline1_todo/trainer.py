import os
import json
import torch
from tqdm import tqdm
from metrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
torch.manual_seed(42)
from ipdb import set_trace as pdb


class Trainer:
    def __init__(self, device, trainData, validData, model, criteria, opt, batch_size, arch):
        self.device = device
        self.trainData = trainData
        self.validData = validData
        self.model = model
        self.criteria = criteria
        self.opt = opt
        self.batch_size = batch_size
        self.arch = arch
        self.history = {'train': [], 'valid': []}
        self.scheduler = StepLR(self.opt, step_size=200, gamma=0.5)

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
        
        # TODO: create dataloader for train and valid.
        # You can set batch_size as `self.batch_size` here, and `collate_fn=dataset.collate_fn`.
        # DO NOT shuffle for valid
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=8,
            collate_fn=dataset.collate_fn,
        )

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        #pdb()
        loss = 0
        accuracy = Accuracy()

        for i, (x, y) in trange: # (x,y) = 128*128
            o_labels, batch_loss = self.run_iter(x, y)
            if training:
                self.opt.zero_grad() # reset gradient to 0
                batch_loss.backward() # calculate gradient
                self.opt.step() # update parameter by gradient

            loss += batch_loss.item() #.item() to get python number in Tensor
            accuracy.update(o_labels.cpu(), y)

            trange.set_postfix(accuracy=accuracy.print_score(), loss=loss / (i + 1))

        if training:
            self.history['train'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})
            self.scheduler.step()
        else:
            self.history['valid'].append({'accuracy': accuracy.get_score(), 'loss': loss / len(trange)})

    def run_iter(self, x, y):
        features = x.to(self.device)
        labels = y.to(self.device)                  # (b)
        o_labels = self.model(features)             # (b, 12)
        l_loss = self.criteria(o_labels, labels)
        return o_labels, l_loss

    def save(self, epoch):
        if not os.path.exists(self.arch):
            os.makedirs(self.arch)
        if epoch % 10 == 0:
            torch.save(self.model.state_dict(), f'{self.arch}/model.pkl')
            with open(f'{self.arch}/history.json', 'w') as f:
                json.dump(self.history, f, indent=4)
