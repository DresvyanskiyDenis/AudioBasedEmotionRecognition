#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add file description and functions list
"""
import contextlib
import wave
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os
import scipy.signal as sps
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from src.utils.audio_preprocessing_utils import load_wav_file
from src.utils.generator_loader import FixedChunksGenerator_loader
from src.utils.sequence_to_one_models import chunk_based_1d_cnn_attention_model

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
    'N': 6,
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

def delete_instances_with_class(labels:Dict[str, np.ndarray], class_to_delete:int=-1)-> Dict[str, np.ndarray]:
    # TODO: add description
        keys_to_delete=[]
        for key, item in labels.items():
            if item==class_to_delete:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            labels.pop(key)
        return labels

def get_train_dev_separation_in_dict(path:str)-> Dict[str, List[str]]:
    #TODO: write description
    labels_partition_dict={}
    partition_df=pd.read_csv(path, sep=';', header=None)
    labels_partition_dict['train']=partition_df[partition_df.iloc[:,0]=='Train'].iloc[:,1].to_list()
    labels_partition_dict['dev'] = partition_df[partition_df.iloc[:,0]=='Validation'].iloc[:,1].to_list()
    return labels_partition_dict

def get_labels_according_to_filenames(labels:Dict[str, np.ndarray],
                                                  filenames:List[str])-> Dict[str, np.ndarray]:
    # TODO: write description
    extracted_labels=dict((filename.strip(), labels[filename.strip()]) for filename in filenames)
    return extracted_labels


def main():
    path_to_labels = 'D:\\Databases\\MSP_podcast\\labels\\Labels.txt'
    path_to_data = 'D:\\Databases\\MSP_podcast\\Audios'
    path_to_labels_partition='D:\\Databases\\MSP_podcast\\Partitions.txt'
    one_hot_labeling = True
    num_classes = 7
    normalization = False
    data_preprocessing_mode = 'raw'
    num_mfcc = 128
    sequence_max_length = 14
    window_length = 1
    # load and preprocess labels
    labels = load_and_preprocess_labels_MSP_podcast(path_to_labels)
    # get labels partition (on train and dev) and split labels on train and dev
    labels_partition=get_train_dev_separation_in_dict(path_to_labels_partition)
    train_labels= get_labels_according_to_filenames(labels, labels_partition['train'])
    dev_labels=get_labels_according_to_filenames(labels, labels_partition['dev'])

    # delete instance with inappropriate class (-1)
    train_labels = delete_instances_with_class(train_labels, class_to_delete=-1)
    dev_labels = delete_instances_with_class(dev_labels, class_to_delete=-1)
    # create generator-loader
    train_generator=FixedChunksGenerator_loader(sequence_max_length=sequence_max_length, window_length=window_length, load_path= path_to_data,
                 data_preprocessing_mode= data_preprocessing_mode, num_mfcc = num_mfcc,
                 labels = train_labels, labels_type= 'sequence_to_one', batch_size= 8,
                 normalization= normalization, one_hot_labeling = one_hot_labeling,
                 num_classes = num_classes)

    input_shape=(14, 16000,1)

    model = chunk_based_1d_cnn_attention_model(input_shape=input_shape, num_output_neurons=num_classes,
                                               filters_per_layer=(128, 128, 128, 256, 256),
                                               filter_sizes=(20, 15, 10, 6, 5),
                                               pooling_sizes=(8, 4, 2, 2, 2),
                                               pooling_step=1,
                                               need_regularization=True,
                                               dropout=True,
                                               dropout_rate=0.3
                                               )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0005, decay=1e-6), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Recall()])
    hist = model.fit(train_generator, epochs=50, use_multiprocessing=True)


if __name__=='__main__':
    main()
    path_to_labels='D:\\Databases\\MSP_podcast\\labels\\Labels.txt'
    path_to_data='D:\\Databases\\MSP_podcast\\Audios'
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
        with contextlib.closing(wave.open(os.path.join(path_to_data, filename), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        lengths[i]=duration
        i+=1
        print(i)
    a=1+2
    print(a+5)