from tqdm import tqdm
from ipdb import set_trace as pdb

def preprocess_samples(dataset, missing=None):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset), desc='[Preprocess]'):
        #pdb() # sample[0] is id, sample[1] is data F2-F9 & class
        processed.append(preprocess_sample(sample[1], missing))

    return processed


def preprocess_sample(data, missing):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    features = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
    if missing:
        for m in missing:
            features.remove(m)
    processed = {}
    processed['Features'] = [data[feature] for feature in features]
    if 'Class' in data:
        processed['Label'] = label_to_idx(data['Class'])

    return processed


def label_to_idx(labels):
    """
    Args:
        labels (string): data's labels.
    Return:
        outputs (int): index of data's label 
    """
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}
    return label_dict[labels]