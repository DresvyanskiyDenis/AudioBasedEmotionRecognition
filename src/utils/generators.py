#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import random
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import os
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from src.utils.audio_preprocessing_utils import cut_data_on_chunks

Data_type_format=Dict[str, Tuple[np.ndarray, int]]


class AudioFixedChunksGenerator(tf.keras.utils.Sequence):

    num_chunks:int
    window_length:float
    load_mode:str
    data:Optional[np.ndarray]
    load_path: Optional[str]
    data_preprocessing_mode:str
    labels:Dict[str,np.ndarray]

    def __init__(self, sequence_max_length:float, window_length:float, load_mode:str='path',
                 data:Optional[Data_type_format]=None, load_path:Optional[str]=None, data_preprocessing_mode:str='raw',
                 labels:Dict[str, np.ndarray]=None, batch_size:int=32):
        """Assigns basic values for data cutting and providing, loads labels, defines how data will be loaded
            and check if all the provided values are in appropriate format

        :param sequence_max_length: float
                    max length of sequence in seconds. Can be decimal.
        :param window_length: float
                    the length of window for data cutting, in seconds. Can be decimal.
        :param load_mode: str
                    the mode of loading data. It can be either 'data', if data in np.array is provided
                    or 'path', if data will be loaded from files located in specified path.
        :param data: Optional[Data_type_format=Dict[str, Tuple[np.ndarray, int]]]
                    if load_mode is 'data', then the data in almost the same format as labels must be
                    provided (besides np.ndarray it contains a sample_rate int number also)
        :param load_path: Optional[str]
                    if load_mode is 'path', then the path to the files must be provided
        :param data_preprocessing_mode: str
                    can be either 'raw', 'LLD', or 'EGEMAPS'.
        :param labels: Dict[str, np.ndarray]
                    labels for data. dictionary represents mapping str -> np.ndarray, where
                    str denotes filename and np.ndarray denotes label on each timestep, or 1 label per whole filename,
                    thus, shape of np.ndarray will be (1,1)
        :param batch_size: int
        """
        self.num_chunks=int(np.ceil(sequence_max_length/window_length))
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
            if isinstance(data, dict):
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


    def __len__(self) -> int:
        """Calculates how many batches per one epoch will be.

        :return: int
                how many batches will be per epoch.
        """
        if self.load_mode=='path':
            num_batches=int(np.ceil(len(self.data_filenames)/self.batch_size))
            return num_batches
        else:
            num_batches=int(np.ceil(self._calculate_overall_size_of_data()/self.batch_size))
            return num_batches

    def __getitem__(self, index):
        pass

    def on_epoch_end(self):
        if self.load_mode=='path':
            random.shuffle(self.data_filenames)
        elif self.load_mode=='data':
            # TODO: so, what? How to shuffle it?
            pass
        pass

    def _calculate_overall_size_of_data(self):
        """Calculates the overall size of data.
           It is assumed that self.data is in the format dict(str, np.ndarray), where
           np.ndarray has shape (Num_batches, num_chunks, window_size, num_features)
                    num_chunks is evaluated in self.__init__() function
                    window_size is evaluated in self.__init__() function
                    num_features is provided by data initially supplied
            The reason that data has 4 dimensions is that it was cut on slices for algorithm

        :return: int
            the overall size of data, evaluated across all entries in dict
        """
        sum=0
        for key, value in self.data:
            sum+=value[0].shape[0]
        return sum

    def _load_data_filenames(self, path:str) -> List[str]:
        """Load filenames of data located in directory with path.

        :param path: str
                    path to directory with data files
        :return: List[str]
                    list with filenames in path
        """
        data_filenames=os.listdir(path)
        return data_filenames

    def _cut_sequence_on_slices(self, sequence:np.ndarray, sample_rate:int) -> np.ndarray:
        """cut provided sequence on fixed number of slices. The cutting process carries out on axis=0
           The number of chunks is fixed and equals self.num_chunks.
           For example, 2-d sequence becomes an 3-d
           (num_timesteps, num_features) -> (num_chunks, window_size, num_features)

        :param sequence: np.ndarray
                    all dimensions starting from 2 is supported.
        :param sample_rate: int
                    sample rate of sequence
        :return: np.ndarray
                    cut on chunks array
        """
        # evaluate window length and step in terms of units (indexes of arrays).
        # self.window_length is presented initially  in seconds
        window_length_in_units=int(self.window_length*sample_rate)
        window_step_in_units=int((sequence.shape[0]-window_length_in_units)/(self.num_chunks-1))
        # cut data with special function in audio_preprocessing_utils.py
        cut_data=cut_data_on_chunks(data=sequence, chunk_length=window_length_in_units, window_step=window_step_in_units)
        # check if we got as much chunks as we wanted
        if len(cut_data)!=self.num_chunks:
            raise ValueError('Function _cut_sequence_on_slices(). The number of cut chunks is not the same as '
                             'was computed in __init__() function. cut_data.shape[0]=%i, should be: %i'
                             %(len(cut_data), self.num_chunks))
        # concatenate cut chunks in np.ndarray
        cut_data=np.concatenate(cut_data, axis=0)
        return cut_data






if __name__=="__main__":
    pass