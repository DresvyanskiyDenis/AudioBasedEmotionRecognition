#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import os
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

class AudioFixedChunksGenerator(tf.keras.utils.Sequence):

    num_chunks:int
    window_length:float
    load_mode:str
    data:Optional[np.ndarray]
    load_path: Optional[str]
    data_preprocessing_mode:str
    labels:Dict[str,np.ndarray]

    def __init__(self, sequence_max_length:float, window_length:float, load_mode:str='path',
                 data:Optional[np.ndarray]=None, load_path:Optional[str]=None, data_preprocessing_mode:str='raw',
                 labels:Dict[str, np.ndarray]=None, batch_size:int=32):
        """Assigns basic values for data cutting and providing, loads labels, defines how data will be loaded
            and check if all the provided values are in appropriate format

        :param sequence_max_length: float
                    max length of sequence in seconds. Can be decimal.
        :param window_length: float
                    the length of window for data cutting, in seconds. Can be decimal.
        :param load_mode: str

        :param data:
        :param load_path:
        :param data_preprocessing_mode:
        :param labels:
        :param batch_size:
        """
        self.num_chunks=int(np.floor(sequence_max_length/window_length))
        self.window_length=window_length
        self.batch_size=batch_size
        # check if load mode has an appropriate value
        if load_mode=='path':
            self.load_mode=load_mode
            if isinstance(load_path,str):
                self.load_path=load_path
                self.data_filenames=self._load_data_filenames(load_path)
            else:
                raise AttributeError('load_path must be a string path to the directory with data and label files. Got %s'%(load_path))

        elif load_mode=='data':
            if isinstance(data, np.ndarray):
                self.data=data
            else:
                raise AttributeError('With \'data\' load mode the data should be in np.ndarray format. Got %s.'%(type(data)))

        else:
            raise AttributeError('load_mode can be either \'path\' or \'data\'. Got %s.'%(load_mode))
        # check if data_preprocessing_mode has an appropriate value
        if data_preprocessing_mode in ('raw', 'LLD', 'EGEMAPS'):
            self.data_preprocessing_mode =data_preprocessing_mode
        else:
            raise AttributeError('data_preprocessing_mode can be either \'raw\' or \'LLD\', or \'EGEMAPS\'. Got %s.'%(data_preprocessing_mode))
        # check if labels are provided in an appropriate way
        if isinstance(labels, dict):
            if len(labels.keys())==0:
                raise AttributeError('Provided labels are empty.')
            elif not isinstance(list(labels.keys())[0], str):
                raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')
            elif not isinstance(list(labels.values())[0], np.ndarray):
                raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')
            else:
                self.labels = labels
        else:
            raise AttributeError('Labels should be a dictionary in str->np.ndarray format.')

    def _load_data_filenames(self, path:str) -> List[str]:
        data_filenames=os.listdir(path)
        return data_filenames




    def __len__(self):
        """Calculates how many batches per one epoch will be.

        :return: int
                how many batches will be per epoch.
        """
        if self.load_mode=='path':
            num_batches=int(np.ceil(len(self.data_filenames)/self.batch_size))
            return num_batches
        else:
            num_batches=int(np.ceil(self.data.shape[0]/self.batch_size))
            return num_batches

    def __getitem__(self, index):
        pass

    def on_epoch_end(self):
        pass









if __name__=="__main__":
    pass