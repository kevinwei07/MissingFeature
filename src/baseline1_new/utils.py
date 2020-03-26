import pandas as pd
import json
import os
from matplotlib import pyplot as plt
from ipdb import set_trace as pdb


def SubmitGenerator(prediction, sampleFile):

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
    df.to_csv('result/prediction.csv', index = False)


def plot_history(arch, max_epoch, plot_acc = True):
    """
    Ploting training process
    """

    arch_list = ['missing0', 'missing0+bn', 'missing0+bn+lr', 'missing0+bn+lr+l2']
    plt.figure(figsize=(7, 5))
    plt.grid()
    line_styles = ['-','--','-.',':','.']
    min_loss = 10000
    min_arch = ''
    min_id = []

    for arc, line_style in zip(arch_list, line_styles):
        path = f'{arc}/history.json'
        with open(path, 'r') as f:
            history = json.loads(f.read())
            train_loss = [loss['loss'] for loss in history['train']]
            valid_loss = [(i, loss['loss']) for i, loss in enumerate(history['valid']) if i%10 == 0]
            valid_loss = [*zip(*valid_loss)]
            x, y = valid_loss[0], valid_loss[1]
            # plt.plot(train_loss, label='train_'+history_path)

            plt.plot(x, y, line_style, label='valid_'+arch)

            loss = min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])])

            if loss[0] < min_loss:
                min_loss = loss[0]
                min_id = loss
                min_arch = arc

    plt.xlabel("Lowest Loss : " + str(min_id)+ " in "+ min_arch)
    plt.legend()
    plt.title('Loss')

    if not os.path.exists(arch):
        os.makedirs(arch)
    plt.savefig(f'{arch}/loss.png')


    print('Lowest Loss ' + str(min_id) + " in "+ min_arch)

    if plot_acc and False:
        plt.figure(figsize=(7, 5))
        plt.grid()
        plt.xlim(0,max_epoch)
        for id, arch in enumerate(arch_list):
            path = f'{arch}/history.json'
            with open(path, 'r') as f:
                history = json.loads(f.read())
                train_f1 = [l['accuracy'] for l in history['train']]
                valid_f1 = [l['accuracy'] for l in history['valid']]
                plt.title('Accuracy')
                ls = line_style[id]
                #plt.plot(train_f1, label='train')
                plt.plot(valid_f1, ls, label='valid_'+arch)
                plt.xlabel("Best acc : " + str(max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ arch)
        plt.legend()
        plt.savefig(f'{arch}/accuracy.png')

        print('Best acc', str(max([[l['accuracy'], idx + 1] for idx, l in enumerate(history['valid'])]))+ " in "+ arch)