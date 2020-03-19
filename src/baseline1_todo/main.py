import argparse
import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ipdb import set_trace as pdb
from torch.optim.lr_scheduler import StepLR

from preprocessor import preprocess_samples
from dataset import FeatureDataset
from model import simpleNet
from trainer import Trainer
from utils import SubmitGenerator, plot_history

torch.backends.cudnn.enabled = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, help='architecture (model_dir)')
    parser.add_argument('--data_dir', default='../../data/', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--do_plot', action='store_true')
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_epoch', default=1500, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--cuda', default=1, type=int)
    args = parser.parse_args()

    missing_list = []

    if args.do_train:

        dataset = pd.read_csv(args.data_dir + "train.csv")
        dataset.drop("Id", axis=1, inplace=True)
        for drop in missing_list:
            dataset.drop(drop, axis=1, inplace=True)

        train_set, valid_set = train_test_split(dataset, test_size=0.1, random_state=73)
        train = preprocess_samples(train_set, missing=missing_list)
        valid = preprocess_samples(valid_set, missing=missing_list)
        trainData = FeatureDataset(train)
        validData = FeatureDataset(valid)

        device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size,args.batch_size,len(missing_list))
        model.to(device)
        batch_size = args.batch_size
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)
        
        # TODO: choose proper loss function for multi-class classfication 
        criteria = torch.nn.CrossEntropyLoss()
        
        trainer = Trainer(device, trainData, validData, model, criteria, opt, batch_size, args.arch)

        max_epoch = args.max_epoch
        min_loss = 100

        for epoch in range(max_epoch):
            #if epoch >= 10:
                #plot_history(args.arch, plot_acc=True)
            print('Epoch: {}'.format(epoch))
            trainer.run_epoch(epoch, True) # True for training
            loss = trainer.run_epoch(epoch, False)
            if loss<min_loss:
                min_loss = loss
                trainer.save(epoch)

    if args.do_predict:

        dataset = pd.read_csv(args.data_dir + "test.csv")
        dataset.drop("Id", axis=1, inplace=True)
        dataset.drop("F1", axis=1, inplace=True)
        test = preprocess_samples(dataset, missing=["F1"])
        testData = FeatureDataset(test)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = simpleNet(args.hidden_size)
        
        # TODO: Load saved model here
        pass
        
        model.train(False)
        model.to(device)

        # TODO: create dataloader for testData.
        # You can set batch_size as `args.batch_size` here, and `collate_fn=testData.collate_fn`.
        # DO NOT shuffle for testing
        dataloader = None

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
        prediction = []
        for i, (x, y) in trange:
            o_labels = model(x.to(device))
            o_labels = torch.argmax(o_labels, axis=1)
            prediction.append(o_labels.to('cpu'))

        prediction = torch.cat(prediction).detach().numpy().astype(int)
        SubmitGenerator(prediction, args.data_dir + 'sampleSubmission.csv')

    if args.do_plot:
        plot_history(args.arch,args.max_epoch,args.do_plot)


if __name__ == '__main__':
    main()
