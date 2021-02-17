#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import List, Union, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler

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
    normalizer=MinMaxScaler()
    normalizer=normalizer.fit(data)
    transformed_data=np.array(normalizer.transform(data))
    if return_scaler:
        return transformed_data, normalizer
    else:
        return transformed_data

def cut_data_on_chunks(data:np.ndarray, chunk_length:int, window_step:int) -> List[np.ndarray]:
    """

    :param data: ndarray
                sequence to cut
    :param chunk_length: int

    :param window_step:
    :return:
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
