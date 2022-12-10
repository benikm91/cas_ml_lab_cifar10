import pickle
from typing import Tuple

import numpy as np


def _load(file: str) -> Tuple[np.array, np.array]:
    with open(file, 'rb') as fo:
        train_data = pickle.load(fo, encoding='bytes')
        return train_data['images'].astype(np.ubyte), train_data['labels'].flatten()


def load_data(data_dir='../../data'):
    return _load(f'{data_dir}/train.pkl')


def load_test(data_dir='../../data'):
    return _load(f'{data_dir}/test.pkl')


def load_data_and_test(data_dir='../../data'):
    return load_data(data_dir), load_test(data_dir)
