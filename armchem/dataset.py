# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import os
import random
import csv
import numpy as np
from tqdm import tqdm

from transformers import create_transformers
from utils import Messenger, Monitor

from sklearn.model_selection import KFold, train_test_split


# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def create_dataset(source, mirror, separator=','):
    if isinstance(source, list):
        return MemoryDataset(source, mirror)
    elif isinstance(source, str):
        return DiskDataset(source, mirror)
    else:
        raise Exception('Unsupported source type: %s' % str(type(source)))


def is_in_memory(dataset):
    return isinstance(dataset, MemoryDataset)


def is_on_disk(dataset):
    return isinstance(dataset, DiskDataset)


def read_records(path):
    with open(path, 'r') as file:
        total = os.path.getsize(path)
        with tqdm(desc='Computes Score', total=total) as pbar:
            for record in file:
                pbar.update(len(record))

                fields = []
                for field in record.split(','):
                    if len(field.strip()) > 0 and len(field.strip()) <= 150:
                        fields.append(field.strip())
                    else:
                        fields = None
                        break

                if fields is None:
                    continue
                else:
                    yield fields


def split_dataset(src):
    filename, filext = os.path.splitext(src)

    dest_train = filename + '-train' + filext
    dest_val = filename + '-val' + filext
    dest_eval = filename + '-eval' + filext

    lookup = {
        'train': {
            'count': 0,
            'file': open(dest_train, 'w')
        },
        'val': {
            'count': 0,
            'file': open(dest_val, 'w')
        },
        'eval': {
            'count': 0,
            'file': open(dest_eval, 'w')
        }
    }

    for fields in read_records(src):
        dice = random.random()
        if dice < 0.8:
                lookup['train']['file'].write(','.join(fields) + '\n')
                lookup['train']['count'] += 1
        elif dice >= 0.8 and dice < 0.9:
                lookup['val']['file'].write(','.join(fields) + '\n')
                lookup['val']['count'] += 1
        elif dice >= 0.9:
                lookup['eval']['file'].write(','.join(fields) + '\n')
                lookup['eval']['count'] += 1

    lookup['train']['file'].close()
    lookup['val']['file'].close()
    lookup['eval']['file'].close()

    print 'Splitting summary:'
    print '   - Stored %d train samples -> %s;' % \
        (lookup['train']['count'], lookup['train']['file'].name)
    print '   - Stored %d validation samples -> %s;' % \
        (lookup['val']['count'], lookup['val']['file'].name)
    print '   - Stored %d evaluation samples -> %s;' % \
        (lookup['eval']['count'], lookup['eval']['file'].name)

    return dest_train, dest_val, dest_eval

# -----------------------------------------------------
# Dataset Handlers
# -----------------------------------------------------


class MemoryDataset:

    def __init__(self, data, mirror):
        self.__data = data
        self.__mirror = mirror

    def __len__(self):
        return len(self.__data)

    def X(self, metadata):
        X = list()
        for feature in self.iterator(metadata, None):
            X.append(feature)
        return np.array(X)

    def y(self, metadata):
        y = list()
        for target in self.iterator(None, metadata):
            y.append(target)
        return np.array(y)

    def partition(self, mode, **kvargs):
        eval_ratio = kvargs['eval_ratio'] if 'eval_ratio' in kvargs else 0.3
        val_ratio = kvargs['val_ratio'] if 'val_ratio' in kvargs else 0.2

        if mode == 'tve':
            learn_data, eval_data = train_test_split(
                self.__data, test_size=eval_ratio
            )
            train_data, val_data = train_test_split(
                learn_data, test_size=val_ratio
            )

            print 'Splitting summary:'
            print '   - Loaded %d train samples -> memory;' % \
                len(train_data)
            print '   - Loaded %d validation samples -> memory' % \
                len(val_data)
            print '   - Loaded %d evaluation samples -> memory' % \
                len(eval_data)

            return \
                MemoryDataset(train_data, self.__mirror),\
                MemoryDataset(val_data, self.__mirror),\
                MemoryDataset(eval_data, self.__mirror)
        elif mode == 'tv':
            train_data, val_data = train_test_split(
                learn_data, test_size=val_ratio
            )
            return \
                MemoryDataset(val_data, self.__mirror),\
                MemoryDataset(eval_data, self.__mirror)
        elif mode == '3f':
            return FoldsIterator(self.__data, self.__mirror, 3, val_ratio)
        elif mode == '5f':
            return FoldsIterator(self.__data, self.__mirror, 5, val_ratio)
        elif mode == '10f':
            return FoldsIterator(self.__data, self.__mirror, 10, val_ratio)
        else:
            raise Exception('Unsupported partitioning mode: %s' % mode)

    def iterator(self, feature_metadata, target_metadata):
        return DatasetIterator(
            iter(self.__data),
            self.__mirror,
            feature_metadata,
            target_metadata
        )

    def __str__(self):
        return 'Memory Dataset( mirror=[%s], size=%d )' % \
            (self.__mirror, 0)


class DiskDataset:

    def __init__(self, path, mirror):
        self.__path = path
        self.__mirror = mirror
        self.__size = 0
        with open(path, 'r') as file:
            for line in file:
                self.__size += 1

    def __len__(self):
        return self.__size

    def iterator(self, feature_metadata, target_metadata):
        return DatasetIterator(
            csv.reader(open(self.__path, 'rb'), delimiter=','),
            self.__mirror,
            feature_metadata,
            target_metadata
        )

    def G(self, batch_size, feature_metadata, target_metadata):
        while True:
            X, y = list(), list()
            for feature, target in self.iterator(
                feature_metadata, target_metadata
            ):
                X.append(feature)
                y.append(target)

                if len(X) >= batch_size:
                    X_yield, y_yield = np.array(X), np.array(y)
                    X, y = list(), list()

                    yield X_yield, y_yield

    def n(self, batch_size):
        return int(np.ceil(self.__size/batch_size))

    def __str__(self):
        return \
            'Disk Dataset( path=\'%s\', size=%d )' % \
            (self._source, self.__size)


# -----------------------------------------------------
# Dataset Iterator
# -----------------------------------------------------

class SourceIterator:

    def __init__(self, source, mirror):
        self.__source = source
        self.__mirror = mirror

    def __iter__(self):
        return self

    def next(self):
        values = self.__source.next()

        if self.__mirror:
            feature = values[0]
            target = values[0]
        else:
            feature = values[0]
            target = values[1]

        return feature, target


class DatasetIterator:

    def __init__(self, source, mirror, feature_metadata, target_metadata):
        self.__dataset = SourceIterator(source, mirror)

        if feature_metadata is None:
            self.__transform_feature = None
        else:
            self.__transform_feature = lambda value: \
                create_transformers(feature_metadata).encode(value)

        if target_metadata is None:
            self.__transform_target = None
        else:
            self.__transform_target = lambda value: \
                create_transformers(target_metadata).encode(value)

    def __iter__(self):
        return self

    def next(self):
        feature, target = self.__dataset.next()
        if all([
            self.__transform_feature is not None,
            self.__transform_target is not None
        ]):
            return \
                self.__transform_feature(feature), \
                self.__transform_target(target)
        elif all([
            self.__transform_feature is not None,
            self.__transform_target is None
        ]):
            return self.__transform_feature(feature)
        elif all([
            self.__transform_feature is None,
            self.__transform_target is not None
        ]):
            return self.__transform_target(target)
        else:
            raise Exception(
                'Neither feature or target transformation are specified.'
            )


# -----------------------------------------------------
# Folds Iterator
# -----------------------------------------------------

class FoldsIterator:
    def __init__(self, source, mirror, n_splits, val_ratio):
        self.__source = source
        self.__mirror = mirror
        self.__folds = KFold(
            n_splits=n_splits, shuffle=True, random_state=42
        ).split(source)
        self.__val_ratio = val_ratio

    def __iter__(self):
        return self

    def next(self):
        tdx, edx = self.__folds.next()
        learn_ds = [self.__source[idx] for idx in tdx]
        eval_ds = [self.__source[idx] for idx in edx]
        train_ds, val_ds = train_test_split(
            learn_ds, test_size=self.__val_ratio
        )
        return \
            MemoryDataset(train_ds, self.__mirror),\
            MemoryDataset(val_ds, self.__mirror),\
            MemoryDataset(eval_ds, self.__mirror)
