#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import random
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import os
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from src.utils.audio_preprocessing_utils import cut_data_on_chunks, load_wav_file, \
    extract_opensmile_features_from_audio_sequence, extract_mfcc_from_audio_sequence

Data_type_format=Dict[str, Tuple[np.ndarray, int]]
data_preprocessing_types=('raw', 'LLD', 'EGEMAPS', 'MFCC')
labels_types=('sequence_to_one',)

class AudioFixedChunksGenerator(tf.keras.utils.Sequence):

    num_chunks:int
    window_length:float
    load_mode:str
    data:Data_type_format
    load_path: Optional[str]
    data_preprocessing_mode:Optional[str]
    labels:Dict[str,np.ndarray]
    labels_type:str
    num_mfcc:Optional[int]

    def __init__(self, sequence_max_length:float, window_length:float, load_mode:str='path',
                 data:Optional[Data_type_format]=None, load_path:Optional[str]=None,
                 data_preprocessing_mode:Optional[str]='raw', num_mfcc:Optional[int]=128,
                 labels:Dict[str, np.ndarray]=None, labels_type:str='sequence_to_one', batch_size:int=32):
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
                    thus, shape of np.ndarray will be (1,)
        :param batch_size: int
        """
        self.num_chunks=int(np.ceil(sequence_max_length/window_length))
        self.window_length=window_length
        self.batch_size=batch_size
        self.num_mfcc=num_mfcc
        # check if load mode has an appropriate value
        if load_mode=='path':
            self.load_mode=load_mode
            if isinstance(load_path,str):
                self.load_path=load_path
                self.data_filenames=self._load_data_filenames(load_path)
            else:
                raise AttributeError('load_path must be a string path to the directory with data and label files. Got %s'%(load_path))

            # check if data_preprocessing_mode has an appropriate value
            if data_preprocessing_mode in data_preprocessing_types:
                self.data_preprocessing_mode = data_preprocessing_mode
            else:
                raise AttributeError(
                    'data_preprocessing_mode can be either \'raw\', \'LLD\', \'MFCC\'or \'EGEMAPS\'. Got %s.' % (
                        data_preprocessing_mode))

        elif load_mode=='data':
            if isinstance(data, dict):
                self.data=data
                # cut provided data
                self.data=self._cut_data_in_dict(self.data)
            else:
                raise AttributeError('With \'data\' load mode the data should be in np.ndarray format. Got %s.'%(type(data)))

        else:
            raise AttributeError('load_mode can be either \'path\' or \'data\'. Got %s.'%(load_mode))
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
        # check if labels_type is provided in appropriate way
        if labels_type in labels_types:
            self.labels_type = labels_type
        else:
            raise AttributeError('Labels_type can be either \'sequence_to_one\' or \'sequence_to_sequence\'. Got %s.'%(labels_type))

        # form indexes for batching then
        self.indexes=self._form_indexes(self.load_mode)



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
        if self.load_mode=='path':
            loaded_data, labels=self._form_batch_with_path_load_mode(index)
            return loaded_data, labels
        elif self.load_mode=='data':
            data, labels = self._form_batch_with_data_load_mode(index)
            return data, labels

    def on_epoch_end(self):
        """Do some actions at the end of epoch.
           We use random.shuddle function to shuffle list of indexes presented via self.indexes
        :return: None
        """
        self.indexes= random.shuffle(self.indexes)



    def _form_batch_with_path_load_mode(self, index:int)-> Tuple[np.ndarray, np.ndarray]:
        """Forms batch, which consists of data and labels chosen according index from shuffled self.indexes.
           The data and labels slice depends on index and defined as index*self.batch_size:(index+1)*self.batch_size
           Thus, index represents the number of slice.
        :param index: int
                    the number of slice
        :return: Tuple[np.ndarray, np.ndarray]
                    data and labels slice
        """
        # select batch_size indexes from randomly shuffled self.indexes
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size if (index + 1) * self.batch_size <= len(self.data_filenames) \
            else len(self.data_filenames)
        file_indexes_to_load = self.indexes[start_idx:end_idx]
        # load every file and labels for it, preprocess them and concatenate
        loaded_data=[]
        labels=[]
        for file_index in file_indexes_to_load:
            # load
            filename_path=os.path.join(self.load_path, self.data_filenames[file_index])
            wav_file, sample_rate = self._load_wav_audiofile(filename_path)

            # cut
            wav_file = self._cut_sequence_on_slices(wav_file, sample_rate)

            # preprocess (extracting features)
            wav_file = self._preprocess_cut_audio(wav_file, sample_rate, self.data_preprocessing_mode, num_mfcc=self.num_mfcc)
            # to concatenate wav_files we need to add new axis
            wav_file=wav_file[np.newaxis, ...]
            loaded_data.append(wav_file)
            # append labels to labels list for next concatenation
            corresponding_labels=self.labels[self.data_filenames[file_index]]
            labels.append(corresponding_labels)
        # concatenate elements in obtained lists
        loaded_data=np.concatenate(loaded_data, axis=0)
        labels=np.concatenate(labels, axis=0)
        return loaded_data, labels

    def _form_batch_with_data_load_mode(self, index:int)-> Tuple[np.ndarray, np.ndarray]:
        """Forms batch, which consists of data and labels chosen according index from shuffled self.indexes.
           The data and labels slice depends on index and defined as index*self.batch_size:(index+1)*self.batch_size
           Thus, index represents the number of slice.

        :param index: int
                    the number of slice
        :return: Tuple[np.ndarray, np.ndarray]
                    data and labels slice
        """
        # select batch_size indexes from randomly shuffled self.indexes
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size if (index + 1) * self.batch_size <= len(self.indexes) \
            else len(self.indexes)
        indexes_to_load=self.indexes[start_idx:end_idx]
        # form already loaded data and labels in batch
        batch_data=[]
        batch_labels=[]
        for chosen_idx in indexes_to_load:
            filename, chunk_idx=chosen_idx
            # self.data[filename][0][np.newaxis,...] :
            # > self.data[filename] provides us (np.ndarray, sample_rate) according to filename
            # > self.data[filename][0] provides us np.ndarray with shape (num_chunks, window_size, num_features)
            # > self.data[filename][0][np.newaxis,...] expand array to add new axis for success concatenation further
            current_data=self.data[filename][0][np.newaxis,...]
            current_labels=self.labels[filename][np.newaxis,...]
            batch_data.append(current_data)
            batch_labels.append(current_labels)
        # concatenate extracted data and labels
        batch_data=np.concatenate(batch_data, axis=0)
        batch_labels=np.concatenate(batch_labels, axis=0)
        return batch_data, batch_labels


    def _load_wav_audiofile(self, path) -> Tuple[np.ndarray, int]:
        """Loads wav file according path

        :param path: str
                path to wav file
        :return: Tuple[int, np.ndarray]
                sample rate of file and values
        """
        sample_rate, wav_file=load_wav_file(path)
        return wav_file, sample_rate

    def _preprocess_raw_audio(self, raw_audio:np.ndarray, sample_rate:int,
                              preprocess_type:str, num_mfcc:Optional[int]=None) -> np.ndarray:
        """Preprocesses raw audio with 1 channel according chosen preprocess_type

        :param raw_audio: np.ndarray
                    raw audio in numpy format. Can be with shape (n_samples,) or (n_samples,1)
        :param sample_rate: int
                    sample rate of audio
        :param preprocess_type: str
                    which algorithm to choose to preprocess? The types are presented in variable data_preprocessing_types
                    at the start of the script (where imports)
        :param num_mfcc: int
                    if preprocess_type=='MFCC', defines how much mfcc are needed.
        :return: np.ndarray
                    extracted features from audio
        """
        # check if raw_audio have 2 dimensions
        if len(raw_audio.shape)!=2 and len(raw_audio.shape) ==1:
            raw_audio=raw_audio[..., np.newaxis]
        else:
            raise AttributeError('raw_audio should be 1- or 2-dimensional. Got %i dimensions.'%(len(raw_audio.shape)))

        if preprocess_type=='LLD':
            preprocessed_audio=extract_opensmile_features_from_audio_sequence(raw_audio, sample_rate, preprocess_type)
        elif preprocess_type=='MFCC':
            preprocessed_audio=extract_mfcc_from_audio_sequence(raw_audio, sample_rate, num_mfcc)
        elif preprocess_type=='EGEMAPS':
            # TODO: implement EGEMAPS features extraction
            raise Exception('EGEMAPS are not implemented yet.')
        else:
            raise AttributeError('preprocess_type should be either \'LLD\', \'MFCC\' or \'EGEMAPS\'. Got %s.'%(preprocess_type))
        return preprocessed_audio

    def _preprocess_cut_audio(self, cut_audio:np.ndarray, sample_rate:int,
                              preprocess_type:str, num_mfcc:Optional[int]=None) -> np.ndarray:
        """ Extract defined in preprocess_type features from cut audio.

        :param cut_audio: np.ndarray
                    cut audio with shape (num_chunks, window_length, 1) or (num_chunks, window_length)
        :param sample_rate: int
                    sample rate of audio
        :param preprocess_type: str
                    which algorithm to choose to preprocess? The types are presented in variable data_preprocessing_types
                    at the start of the script (where imports)
        :param num_mfcc: int
                    if preprocess_type=='MFCC', defines how much mfcc are needed.
        :return: np.ndarray
                    extracted from each window features. The output shape is (num_chunks, window_length, num_features)
        """
        chunks=[]
        for chunk_idx in range(cut_audio.shape[0]):
            chunk_extracted_features=[]
            for window_idx in range(cut_audio.shape[1]):
                extracted_features=self._preprocess_raw_audio(cut_audio[chunk_idx, window_idx], sample_rate,
                                                              preprocess_type, num_mfcc)
                chunk_extracted_features.append(extracted_features)
            chunk_extracted_features=np.concatenate(chunk_extracted_features, axis=0)[np.newaxis,...]
            chunks.append(chunk_extracted_features)
        chunks=np.concatenate(chunks, axis=0)
        return chunks

    def _form_indexes(self, data_mode:str) -> List[Union[int, List[str, int]]]:
        """Forms random indexes depend on data type was loaded.

        :param data_mode: str
                    can be either 'path' or 'data'.
                    'path' means that files will be loaded on-the-fly according self.data_filenames array
                    'data' means that files are already loaded and cut. However, they are located in
                    dict[str, np.ndarray], so, to do batching fair, we will construct list[list[str, int]],
                    where str - filename, which contains n number of cut samples and int - the index of this
                    cut samples. Thus, we will provide randomly shuffled samples independent from filenames
                    (means all samples of filenames will have the same chance to be included in batch).

        :return: indexes
                list[str] or list[list[str, int]], see the explanation above.
        """
        # if data_mode=='path', we simply make permutation of indexes of self.data_filenames
        if data_mode=='path':
            indexes=np.random.permutation(len(self.data_filenames)).tolist()
            # if not (means that load mode is data), we make procedure described above
        else:
            indexes=[]
            for key, value in self.data:
                # extract values
                data_array, sample_rate = value
                filename = key
                # make permutation for np.ndarray of particular filename
                num_indexes=data_array.shape[0]
                permutations=np.random.permutation(num_indexes)
                # append indexes randomly permutated
                for i in range(num_indexes):
                    indexes.append([filename, permutations[i]])

        return indexes


    def _calculate_overall_size_of_data(self) -> int:
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

    def _cut_data_in_dict(self, data:Data_type_format)->Data_type_format:
        for key, value in data:
            array, sample_rate=value
            array = self._cut_sequence_on_slices(array, sample_rate)
            data[key]=(array, sample_rate)
        return data




if __name__=="__main__":
    pass