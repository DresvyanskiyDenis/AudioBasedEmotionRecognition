#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# TODO: write description of the file
"""
import math
from typing import List, Union, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa
import opensmile

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


def extract_mfcc_from_audio_sequence(data:np.ndarray, sample_rate:int, num_mfcc:int,
                                     length_fft:int=512, length_fft_step:int=256) -> np.ndarray:
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
                extracted features with the shape (?, num_mfcc)
                the first dimension is computed by librosa and depends on length_fft and length_fft_step
    """
    # check if data is monotonic, but have 2 dimensions
    if len(data.shape)==2 and data.shape[1]==1:
        data=data.reshape((-1,))
    mfcc_features=librosa.feature.mfcc(data, sr=sample_rate, n_mfcc=num_mfcc,
                                       n_fft=length_fft, hop_length=length_fft_step)
    return mfcc_features.T

def extract_opensmile_features_from_audio_sequence(data:Union[np.ndarray, str], sample_rate:Optional[int]=None,
                                                   feature_type:str='LLD') -> np.ndarray:
    """Extracts opensmile ComParE_2016 features from audio sequence represented either by ndarray or path.
    https://github.com/audeering/opensmile-python

    :param data: np.ndarray or str
                Can be ndarray - already loaded sound data or str - path to data
    :param sample_rate: Optional[int]
                Sample rate is needed if data in ndarray format is provided
    :return: np.ndarray
                extracted features
    """
    supported_feature_types=('LLD', 'Compare_2016_functionals','EGEMAPS')
    if not feature_type in supported_feature_types:
        raise AttributeError('feature_type must the value from %s. Got %s.' % (supported_feature_types,feature_type))
    # configure opensmile to extract desirable features
    if feature_type == 'LLD':
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        feature_set=opensmile.FeatureSet.ComParE_2016
    elif feature_type == 'Functionals':
        feature_level=opensmile.FeatureLevel.Functionals
        feature_set = opensmile.FeatureSet.ComParE_2016
    elif feature_type == 'EGEMAPS':
        feature_level=opensmile.FeatureLevel.Functionals
        feature_set = opensmile.FeatureSet.eGeMAPSv02
    # create opensmile Extractor
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
    # check if data is a valid type and if yes, extract features properly
    if isinstance(data,str):
        extracted_features = smile.process_file(data).values
    elif isinstance(data, np.ndarray):
        # check if audio data is one-channel, then reshape it to 1D array (the requirement of opensmile)
        if len(data.shape)==2 and data.shape[1]==1:
            data=data.reshape((-1,))
        extracted_features = smile.process_signal(data, sampling_rate=sample_rate).values
    else:
        raise AttributeError('Data should be either ndarray or str. Got %s.'%(type(data)))
    return extracted_features

def extract_HLDs_from_LLDs(LLDs:np.ndarray, window_size:float, window_step:float, required_HLDs:Tuple[str,...]) -> np.ndarray:
    """Extracts High level discriptors (functionals) from low-level discriptors. The output is in format List[np.ndarray] -
       each np.ndarray is functionals extracted from window calculated with the help of window_size and window_step params

    :param LLDs: np.ndarray
            low-level discriptors with shape (n_timesteps, n_features)
    :param window_size: float
            the size of window for extracting high-level discriptors (functionals)
            It is fraction, which denotes the percentage of full length of sequence
            for example, 0.1 means 10% of full length of sequence
    :param window_step: float
            the size of shift of window every step. The float number means percentages of
            full length of sequence as well.
    :param required_HLDs: Tuple[str,...]
            The names of functionals needed to be extracted (in str). Currently the supported types are:
            ('min', 'max', 'mean', 'std')
    :return: List[np.ndarray]
            The list of np.ndarrays. Each np.ndarray corresponds to HLDs (functionals) extracted from window.
    """
    supported_HLDs=('min', 'max', 'mean', 'std')
    # check if required_HLDs are all supported by this function
    if not set(required_HLDs).issubset(supported_HLDs):
        raise AttributeError('Required HLDs contain some unsupported HLDs. Possible HLDs: %s. Got: %s.'%(supported_HLDs, required_HLDs))
    # calculate sizes of step and window in units (indexes)
    window_size_in_units=int(round(window_size*LLDs.shape[0]))
    window_step_in_units = int(round(window_step * LLDs.shape[0]))
    # cut LLDs on windows
    cut_LLDs=cut_data_on_chunks(LLDs, window_size_in_units, window_step_in_units)
    # extract HLDs for each window
    result_array=[]
    for window_idx in range(len(cut_LLDs)):
        calculated_HLDs=[]
        for HLD_type in required_HLDs:
            if HLD_type=='min':
                calculated_HLDs.append(cut_LLDs[window_idx].min(axis=0))
            elif HLD_type == 'max':
                calculated_HLDs.append(cut_LLDs[window_idx].max(axis=0))
            elif HLD_type == 'mean':
                calculated_HLDs.append(cut_LLDs[window_idx].mean(axis=0))
            elif HLD_type == 'std':
                calculated_HLDs.append(cut_LLDs[window_idx].std(axis=0))
        calculated_HLDs=np.concatenate(calculated_HLDs, axis=0)[np.newaxis,...]
        result_array.append(calculated_HLDs)
    result_array=np.concatenate(result_array, axis=0)
    return result_array

def extract_subwindow_EGEMAPS_from_audio_sequence(sequence:np.ndarray, sample_rate:int,
                                                      subwindow_size:float, subwindow_step:float) -> np.ndarray:
    """

    :param sequence: np.ndarray
                represents the secuence of raw audio
    :param sample_rate: int
                the sample rate of provided audio
    :param subwindow_size: float
            the size of window for extracting EGEMAPS with the help of opensmile lib
            It is fraction, which denotes the percentage of full length of sequence
            for example, 0.1 means 10% of full length of sequence
    :param subwindow_step: float
            the size of shift of window every step. The float number means percentages of
            full length of sequence as well.
    :return: np.ndarray
            the numpy array with shape (num_windows, num_EGEMAPS_features). It can be further interpreted as
            (timesteps, num_EGEMAPS_features) as well.
    """
    # calculate sizes of step and window in units (indexes)
    window_size_in_units = int(subwindow_size * sequence.shape[0])
    window_step_in_units = int(subwindow_step * sequence.shape[0])
    # cut LLDs on windows
    cut_sequence = cut_data_on_chunks(sequence, window_size_in_units, window_step_in_units)
    # extract HLDs for each window
    result_array = []
    for subwindow_idx in range(len(cut_sequence)):
        calculated_subwindow_egemaps=extract_opensmile_features_from_audio_sequence(cut_sequence[subwindow_idx],
                                                                                    sample_rate, feature_type='EGEMAPS')
        result_array.append(calculated_subwindow_egemaps)
    result_array=np.concatenate(result_array, axis=0)
    return result_array


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
    path=r'E:\Databases\SEWA\Original\audio\SEW1101.wav'
    sr, data=load_wav_file(path)
    extracted_LLDs=cut_data_on_chunks(data,48000, 24000)
    print(extracted_LLDs)
    extracted_HLDs=extract_subwindow_EGEMAPS_from_audio_sequence(extracted_LLDs[0],sr, 0.1, 0.05)

