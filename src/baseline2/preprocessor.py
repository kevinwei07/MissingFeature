from tqdm import tqdm
from ipdb import set_trace as pdb


def preprocess_samples(data, missing):

    processed_list = []
    for sample in tqdm(data.iterrows(), total=len(data), desc='[Preprocess]'):
        # sample[0] is id, sample[1] is data F1-F9 & class
        processed_list.append(preprocess_sample(sample[1], missing))

    return processed_list

def preprocess_sample(data,missing):

    features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
    if missing:
        for m in missing:
            features.remove(m)
    processed_dict = {}
    processed_dict['Features'] = [data[feature] for feature in features]
    processed_dict['F1'] = [data['F1']]
    if 'Class' in data:
        processed_dict['Label'] = label_to_idx(data['Class'])

    # ex {'Features': [0.269287421, 0.5376664210000001, 0.222336421, -0.19022157899999997, 0.04662542099999999, 0.277113421, 0.420988421, -0.15465757900000002], 'Label': 0}
    return processed_dict

def label_to_idx(labels):

    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

    return label_dict[labels]