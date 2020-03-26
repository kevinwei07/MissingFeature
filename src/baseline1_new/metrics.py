import torch
from ipdb import set_trace as pdb

class Accuracy:
    def __init__(self):
        self.n_corrects = 0
        self.n_total = 0
        self.name = 'Accuracy'

    def reset(self):
        self.n_corrects = 0
        self.n_total = 0

    def update(self, predicts, groundTruth):
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n_total, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        # max(,0) return the max element and its index (column-based)
        # max(,1) return the max element and its index (row-based)
        self.n_total += len(groundTruth)
        pdb()
        pred = torch.max(predicts,1)[1]
        self.n_corrects += (pred == groundTruth).sum()
        #train_acc = self.n_corrects.item()
        #return train_acc

    def print_score(self):
        acc = float(self.n_corrects) / self.n_total
        return '{:.5f}'.format(acc)

    def get_score(self):
        acc = float(self.n_corrects) / self.n_total
        return acc