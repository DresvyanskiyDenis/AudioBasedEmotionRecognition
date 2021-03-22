#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add file description and functions list
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os
import scipy.signal as sps

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from src.utils.audio_preprocessing_utils import load_wav_file

"""
Angry		(A)
Sad		    (S)
Happy		(H)
Surprise	(U)
Fear		(F)
Disgust		(D)
Contempt	(C)
Neutral		(N)
Other		(O)
No agreement(X)
"""
emotion_to_int_mappping={
    'A': 0,
    'S': 1,
    'H': 2,
    'U': 3,
    'F': 4,
    'D': 5,
    'C': -1,
    'N': 7,
    'O': -1,
    'X': -1
}

def load_and_preprocess_labels_MSP_podcast(path:str)->Dict[str, np.ndarray]:
    """Loads and preprocesses labels of MSP-podcast dataset. The all labels are contained in one .txt file.
       The typical format of line in file is
       MSP-PODCAST_0001_0008.wav; N; A:float_value; V:float_value; D:float_value;
       There are also some other lines, which should be ignored.

    :param path: str
            path to the file with labels
    :return: Dict[str, int]
            loaded labels in format Dict[filename->emotion_category]
    """
    labels={}
    # open file
    with open(path) as file_object:
        for line in file_object:
            # check if we read the needed line. It always shoul starts from 'MSP-PODCAST'
            if line[:11]=='MSP-PODCAST':
                # split line by ';' and throw out last element of array, because it will be always empty string
                splitted_line=line.split(';')[:-1]
                # save extracted label and filename
                filename=splitted_line[0]
                class_category=splitted_line[1].strip()
                labels[filename]=np.array(emotion_to_int_mappping[class_category]).reshape((-1,))
    return labels

def delete_instances_with_class(labels:Dict[str, int], class_to_delete:int=-1)-> Dict[str, int]:
    # TODO: add description
        keys_to_delete=[]
        for key, item in labels.items():
            if item==class_to_delete:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            labels.pop(key)
        return labels



if __name__=='__main__':
    path_to_labels='D:\\Downloads\\MSP_podcast\\Labels.txt'
    path_to_data='D:\\Downloads\\Audios\\Audios'
    one_hot_labeling=True
    num_classes=7
    normalization=False
    data_preprocessing_mode='raw'
    num_mfcc=128
    sequence_max_length=14
    window_length=1
    labels=load_and_preprocess_labels_MSP_podcast(path_to_labels)
    labels=delete_instances_with_class(labels, class_to_delete=-1)
    a=1+2
    max=0
    lengths=np.zeros((len(labels),))
    i=0
    for filename in labels:
        sr, data=load_wav_file(os.path.join(path_to_data, filename))
        length=data.shape[0]/sr
        lengths[i]=length
        i+=1
    a=1+2