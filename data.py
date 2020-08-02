# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to load data from folders and augment it"""

import os
import itertools
import logging
import torch

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


LOG = logging.getLogger('main')
NO_LABEL = -1


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in  zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def normalization_processing(data):

    data_mean = data.mean()
    data_std = data.std()

    data = data - data_mean
    data = data / data_std

    return data

# def wgn2D(data, snr=6, boxSideLen=80):

#     imgSideLen = data.shape[0]
#     halfSideLen = boxSideLen // 2
#     x = np.random.randint(halfSideLen, imgSideLen - halfSideLen)
#     y = np.random.randint(halfSideLen, imgSideLen - halfSideLen)
#     mask = np.zeros((imgSideLen, imgSideLen))
#     mask[y - halfSideLen: y + halfSideLen, x - halfSideLen: x + halfSideLen] = 1

#     Ps = np.sum(abs(data)**2) / np.size(data)
#     Pn = Ps / (10**((snr / 10)))
#     noise = np.random.randn(imgSideLen, imgSideLen) * np.sqrt(Pn) * mask
#     signal_add_noise = data + noise
    
#     return signal_add_noise

def wgn2D(data, snr=6):
    Ps = np.sum(abs(data)**2) / np.size(data)
    Pn = Ps / (10**((snr / 10)))
    noise = np.random.randn(data.shape[0], data.shape[0]) * np.sqrt(Pn)
    signal_add_noise = data + noise
    return signal_add_noise


class SemiSet(Dataset):
    def __init__(self, path, aug):
        self.data_path = os.path.join(path, "data/")
        self.lab_path = os.path.join(path, "label/")
        self.aug = aug

        data_num = len(os.listdir(self.data_path))
        lab_num = len(os.listdir(self.lab_path))

        self.unlabeled_idx = data_num - lab_num
    
    def __len__(self):
        return len(os.path.listdir(self.data_path))

    def __getitem__(self, id):
        data = np.load(self.data_path + str(id) + ".npy").astype('float32')
    
        if id > self.unlabeled_idx:
            label = np.load(self.lab_path + str(id) + ".npy").astype('float32')
        else:
            label = np.zeros_like(data) - 1

        data = normalization_processing(data)
        
        # data augmentation
        if self.aug:
            data = wgn2D(data)
    
        data = torch.FloatTensor(data).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        return data, label


class OriSet(Dataset):
    def __init__(self, path):
        self.data_path = path + "/data/"
        self.lab_path = path + "/label/"

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, id):
        data = np.load(self.data_path + str(id) + ".npy").astype('float32')
        label = np.load(self.lab_path + str(id) + ".npy").astype('float32')
        data = normalization_processing(data)
        data = torch.FloatTensor(data).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)
        return data, label





    

