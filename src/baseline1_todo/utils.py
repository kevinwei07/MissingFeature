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


def plot_history(arch, plot_acc=True):
    """
    Ploting training process
    """
    plot_list=[arch,]

    history_path = f'{arch}/history.json'
    with open(history_path, 'r') as f:
        history = json.loads(f.read())

    #pdb()
    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.xlabel("Lowest Loss : " + str(min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])])))
    plt.legend()
    plt.savefig(f'{arch}/loss.png')

    print('Lowest Loss ', min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))

    if plot_acc:
        train_f1 = [l['accuracy'] for l in history['train']]
        valid_f1 = [l['accuracy'] for l in history['valid']]
        plt.figure(figsize=(7, 5))
        plt.title('Accuracy')
        plt.plot(train_f1, label='train')
        plt.plot(valid_f1, label='valid')
        plt.xlabel("Best acc : " + str(max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])])))
        plt.legend()
        plt.savefig(f'{arch}/accuracy.png')

        print('Best acc', max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])]))