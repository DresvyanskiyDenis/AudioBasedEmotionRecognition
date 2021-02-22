#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import List, Union, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
import librosa

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"



def load_wav_file(path:str) -> Tuple[int,np.ndarray]:
    """Loads .wav file according provided path and returns sample_rate of audio and the data

    :param path: str
                path to .wav file
    :return: tuple
                sample rate of .wav file and data
    """
    sample_rate, data = wavfile.read(path)
    return sample_rate, data

def get_trained_minmax_scaler(data:np.ndarray, feature_range:Tuple[float, float]=(-1,1)) -> object:
    """Trains and returns MinMaxScaler from sklearn library.

    :param data: np.ndarray
            data, on which scaler will be learnt
    :param feature_range: tuple(int, int)
            range of the future features
    :return: sklearn.preprocessing.MinMaxScaler
            trained on data scaler
    """
    normalizer = MinMaxScaler(feature_range=feature_range)
    normalizer = normalizer.fit(data)
    return normalizer

def transform_data_with_scaler(data:np.ndarray, scaler:object) -> np.ndarray:
    """Transforms data by passed scaler object (from sklearn.preprocessing).

    :param data: np.ndarray
            data to trasform
    :param scaler: sklearn.preprocessing object
            scaler, which will apply transformation operation to data
    :return: np.ndarray
            transformed data
    """
    transformed_data=scaler.transform(data)
    return transformed_data

def normalize_min_max_data(data:np.ndarray, return_scaler:bool=False) -> Tuple[np.ndarray,Optional[object]] or np.ndarray:
    """Normalize data via minmax normalization with the help of sklearn.preprocessing.MinMaxScaler.
       Normalization will use last dimension.

    :param data: numpy.ndarray
                data to normalize
    :param return_scaler: bool
                return MinMaxScaler object, if you need it for further using
    :return: (numpy.ndarray, object) or numpy.ndarray
                return either data or data with scaler
    """
    normalizer=get_trained_minmax_scaler(data)
    transformed_data=transform_data_with_scaler(data, normalizer)
    if return_scaler:
        return transformed_data, normalizer
    else:
        return transformed_data

def extract_mfcc_from_audio_sequence(data:np.ndarray, sample_rate:int, num_mfcc:int,
                                     length_fft:int, length_fft_step:int) -> np.ndarray:
    """Extracts features from given data ndarray with the help of librosa library.
    https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

    :param data: ndarray
                represents audio file (from example, extracted from .wav file)
                the shape should be (data_length,)
    :param sample_rate: int
                sample rate of the provided data
    :param num_mfcc: int
                desirable number of mfcc features to extract
    :param length_fft: int
                the length of the window, in which the foyer transformation will be applied
    :param length_fft_step: int
                the length of the step, on which the former window will be shifted
    :return: ndarray
                extracted features with the shape (?,num_mfcc)
                the first dimension is computed by librosa and depends on length_fft and length_fft_step
    """
    mfcc_features=librosa.feature.mfcc(data, sr=sample_rate, n_mfcc=num_mfcc,
                                       n_fft=length_fft, hop_length=length_fft_step)
    return mfcc_features


def cut_data_on_chunks(data:np.ndarray, chunk_length:int, window_step:int) -> List[np.ndarray]:
    """Cuts data on chunks according to supplied chunk_length and windows_step.
        Example:
        data=|123456789|, chunk_length=4, window_step=3
        result= [1234, 4567, 6789] -> last element (6789)

    :param data: ndarray
                sequence to cut
    :param chunk_length: int
                length of window/chunk. It is calculated before function as seconds*sample_rate
    :param window_step: int
                length of shift of the window/chunk.
    :return: list of np.ndarrays
                cut on windows data
    """
    if data.shape[0]<chunk_length:
        raise AttributeError("data length should be >= chunk length. Got data length:%i, chunk length:%i"%(data.shape[0], chunk_length))
    if data.shape[0]<window_step:
        raise AttributeError("data length should be >= window_step. Got data length:%i, window_step:%i"%(data.shape[0], window_step))

    cut_data=[]
    num_chunks=int(np.ceil((data.shape[0]-chunk_length)/window_step+1))

    for chunk_idx in range(num_chunks-1):
        start=chunk_idx*window_step
        end=chunk_idx*window_step+chunk_length
        chunk=data[start:end]
        cut_data.append(chunk)

    last_chunk=data[-chunk_length:]
    cut_data.append(last_chunk)

    return cut_data






if __name__=='__main__':
    path=r'D:\Databases\SEWA\Original\audio\SEW1101.wav'
    sr, data=load_wav_file(path)
    cut_data=cut_data_on_chunks(data, chunk_length=16000,window_step=8000)
