import pandas as pd
import json
import os
from matplotlib import pyplot as plt
from ipdb import set_trace as pdb


def SubmitGenerator(prediction, sampleFile):
    """
    Args:
        prediction (numpy array)
        sampleFile (str)
    """
    if not os.path.exists('result'):
        os.makedirs('result')

    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
    key_list = list(label_dict.keys())

    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['Id'] = list(sample.Id.values)
    prediction_class = [key_list[p] for p in prediction]
    submit['Class'] = list(prediction_class)
    df = pd.DataFrame.from_dict(submit)
    df.to_csv('result/prediction.csv', index=False)


def plot_history(arch, max_epoch, plot_acc=True):
    """
    Ploting training process
    """
    plot_list = [arch, 'missing0', 'missing0+bn','missing0+bn+lr','missing0+bn+lr+l2']
    plt.figure(figsize=(7, 5))
    plt.grid()
    plt.xlim(0, max_epoch)
    line_style = ['-','--','-.',':','.']

    for id, history_path in enumerate(plot_list):
        path = f'{history_path}/history.json'
        with open(path, 'r') as f:
            history = json.loads(f.read())
            train_loss = [loss['loss'] for loss in history['train']]
            valid_loss = [loss['loss'] for loss in history['valid']]
            ls = line_style[id]
            # plt.plot(train_loss, label='train_'+history_path)
            plt.plot(valid_loss,ls, label='valid_'+history_path)
            plt.xlabel("Lowest Loss : " + str(min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ history_path)
    plt.legend()
    plt.title('Loss')

    if not os.path.exists(arch):
        os.makedirs(arch)
    plt.savefig(f'{arch}/loss.png')


    print('Lowest Loss ', str(min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ history_path)

    if plot_acc:
        plt.figure(figsize=(7, 5))
        plt.grid()
        plt.xlim(0,max_epoch)
        for id, history_path in enumerate(plot_list):
            path = f'{history_path}/history.json'
            with open(path, 'r') as f:
                history = json.loads(f.read())
                train_f1 = [l['accuracy'] for l in history['train']]
                valid_f1 = [l['accuracy'] for l in history['valid']]
                plt.title('Accuracy')
                ls = line_style[id]
                #plt.plot(train_f1, label='train')
                plt.plot(valid_f1, ls, label='valid_'+history_path)
                plt.xlabel("Best acc : " + str(max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ history_path)
        plt.legend()
        plt.savefig(f'{arch}/accuracy.png')

        print('Best acc', str(max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ history_path)