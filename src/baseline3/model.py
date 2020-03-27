import torch
import torch.nn as nn
from ipdb import set_trace as pdb

class simpleNet(nn.Module):

    def __init__(self, hidden_size, input_size, output_size):
         super(simpleNet, self).__init__()

         self.input = nn.Linear(input_size, hidden_size)
         self.bn0 = nn.BatchNorm1d(hidden_size)
         self.l1 = nn.Linear(hidden_size, hidden_size)
         self.bn1 = nn.BatchNorm1d(hidden_size)

         # self.l2 = nn.Linear(hidden_size, hidden_size)
         # self.bn2 = nn.BatchNorm1d(hidden_size)
         # self.l3 = nn.Linear(hidden_size, hidden_size)
         # self.bn3 = nn.BatchNorm1d(hidden_size)
         self.out = nn.Linear(hidden_size, output_size)

         self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        #pdb()
        x = torch.relu(self.input(x))   # (b, 8) > (b, h) > (b, h)
        x = self.bn0(x)
        x = self.dropout(x)       # (b, h)

        x = torch.relu(self.l1(x)) # (b, h)
        x = self.bn1(x)
        x = self.dropout(x)

        # x = torch.relu(self.l2(x)) # (b, h)
        # x = self.bn2(x)
        # x = self.dropout(x)

        # x = torch.relu(self.l3(x)) # (b, h)
        # x = self.bn3(x)
        # x = self.dropout(x)

        x = self.out(x)            # (b, 12)
        return x