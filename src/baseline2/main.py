import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from ipdb import set_trace as pdb
import torch
import torch.nn as nn

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from trainer import Trainer
from model import simpleNet

torch.backends.cudnn.enabled = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required = True)
    parser.add_argument('--do_train', action = 'store_true')
    parser.add_argument('--do_predict', action = 'store_true')
    parser.add_argument('--data_dir', default = '../../data/')
    parser.add_argument('--cuda', default = 0)
    parser.add_argument('--hidden_size', default = 256 , type = int )
    parser.add_argument('--batch_size', default = 256, type = int)
    parser.add_argument('--max_epoch', default=1500, type = int)
    parser.add_argument('--lr', default = 1e-3 , type = float)
    parser.add_argument('--wd', default = 1e-2, type = float)
    parser.add_argument('--do_plot', action = 'store_true')
    args = parser.parse_args()

    missing_list = ['F1']


    if args.do_train:

        data = pd.read_csv(args.data_dir + 'train.csv')
        # axis = 0 for row ; axis = 1 for column
        # inplace = if modify the origin data
        data.drop("Id", axis = 1, inplace = True)
        # for drop in missing_list:
        #    data.drop(drop, axis = 1, inplace = True)

        train_set, valid_set = train_test_split(data, test_size=0.1, random_state=73)
        train = preprocess_samples(train_set, missing_list)
        valid = preprocess_samples(valid_set, missing_list)
        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size,missing_list)
        model.to(device)
        batch_size = args.batch_size
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
        # MSELoss is loss function for
        loss_function = torch.nn.MSELoss()
        max_epoch = args.max_epoch

        trainer = Trainer(device, trainData, validData, model, loss_function, optimizer, batch_size, args.arch)

        for epoch in range(max_epoch):
            print('Epoch: {}'.format(epoch))
            # True for training ; False for validation
            trainer.run_epoch(epoch, True)
            trainer.run_epoch(epoch, False)
        pass

    if args.do_predict:
        pass

    if args.do_plot:
        pass



if __name__ == '__main__':
    main()
