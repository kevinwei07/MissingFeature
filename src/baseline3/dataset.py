from torch.utils.data import Dataset
import torch
from ipdb import set_trace as pdb

class FeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):

        batch_feature = []
        batch_label = []
        batch_missing = []
        for data in datas:
            batch_feature.append(data['Features'])
            batch_missing.append(data['Missing'])
            if 'Label' in data:
                batch_label.append(data['Label'])

        return torch.FloatTensor(batch_feature), torch.LongTensor(batch_label), torch.FloatTensor(batch_missing)
