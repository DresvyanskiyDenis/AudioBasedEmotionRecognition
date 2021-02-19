#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""


from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os

from src.utils.data_preprocessing_utils import load_wav_file, get_trained_minmax_scaler, transform_data_with_scaler, \
    cut_data_on_chunks
from src.utils.label_preprocessing_utils import load_gold_shifted_labels, split_labels_dataframe_according_filenames
from src.utils.tf_utils import create_1d_cnn_model_classification, create_1d_cnn_model_regression, ccc_loss, CCC_loss_tf


def load_and_split_labels(path:str) -> Dict[str,pd.DataFrame]:
    labels = load_gold_shifted_labels(path_to_labels)
    splitted_labels=split_labels_dataframe_according_filenames(labels)
    return splitted_labels

def load_all_wav_files(path:str)-> Dict[str,Tuple[np.ndarray, int]]:
    loaded_wav_files={}
    filenames=os.listdir(path)
    for filename in filenames:
        sample_rate,wav_file=load_wav_file(os.path.join(path,filename))
        if len(wav_file.shape)<2: wav_file=wav_file[..., np.newaxis]
        loaded_wav_files[filename.split(".")[0]]=(wav_file, sample_rate)
    return loaded_wav_files

def split_data_on_train_dev_test(data:dict,dev_filenames:List[str],test_filenames:List[str])->\
        Tuple[Dict[str,np.ndarray],Dict[str,np.ndarray],Dict[str,np.ndarray]]:

    # separate dev data
    dev_data = {}
    for dev_filename in dev_filenames:
        dev_data[dev_filename] = data.pop(dev_filename)

    # separate test data
    test_data = {}
    for test_filename in test_filenames:
        test_data[test_filename] = data.pop(test_filename)

    return data, dev_data, test_data

def split_labels_on_train_dev_test(labels:dict, dev_filenames:List[str],test_filenames:List[str])->\
        Tuple[Dict[str,pd.DataFrame],Dict[str,pd.DataFrame],Dict[str,pd.DataFrame]]:
    # separate dev labels
    dev_labels = {}
    for dev_filename in dev_filenames:
        dev_labels[dev_filename] = labels.pop(dev_filename)

    # separate test data
    test_labels = {}
    for test_filename in test_filenames:
        test_labels[test_filename] = labels.pop(test_filename)

    return labels, dev_labels, test_labels

def cut_train_data_and_labels_on_chunks(train_data:Dict[str,np.ndarray],train_labels:Dict[str,pd.DataFrame], chunk_length:float, chunk_step:float) -> Tuple[np.ndarray, np.ndarray]:
    result_data =[]
    result_labels=[]
    filenames=list(train_data.keys())
    for filename in filenames:
        # data cutting
        current_data, sample_rate=train_data[filename]
        current_chunk_length=int(chunk_length*sample_rate)
        current_chunk_step=int(chunk_step*sample_rate)
        cut_data=cut_data_on_chunks(current_data, current_chunk_length,current_chunk_step)
        result_data.append(cut_data)
        # labels cutting
        current_label=train_labels[filename]
        sample_rate_labels=1./(current_label['timestamp'].iloc[1]-current_label['timestamp'].iloc[0])
        current_chunk_length=int(chunk_length*sample_rate_labels)
        current_chunk_step=int(chunk_step*sample_rate_labels)
        cut_labels=cut_data_on_chunks(current_label.values, current_chunk_length,current_chunk_step)
        result_labels.append(cut_labels)
    # concatenate all
    result_data=np.concatenate(result_data, axis=0)
    result_labels=np.concatenate(result_labels, axis=0)
    return result_data, result_labels

def align_audio_and_labels(data:Dict[str, Tuple[np.ndarray, int]], labels:Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Tuple[np.ndarray, int]], Dict[str, pd.DataFrame]]:
    filenames=list(data.keys())
    for filename in filenames:
        current_data, sample_rate_data=data[filename]
        current_labels=labels[filename]
        sample_rate_labels = np.round(1. / (current_labels['timestamp'].iloc[1] - current_labels['timestamp'].iloc[0]),2)
        align_to_idx=int(current_labels.shape[0]/sample_rate_labels*sample_rate_data)
        current_data=current_data[:align_to_idx]
        data[filename]=(current_data, sample_rate_data)
    return data, labels


def normalize_minmax_train_dev_test(data:Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray], Dict[str,np.ndarray]], scaler) -> Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray], Dict[str,np.ndarray]]:
    train_data, dev_data, test_data=data
    for key, value in train_data.items():
        sample_rate=train_data[key][1]
        values=train_data[key][0]
        train_data[key]=(transform_data_with_scaler(values, scaler).astype('float32'), sample_rate)
    for key, value in dev_data.items():
        sample_rate=dev_data[key][1]
        values=dev_data[key][0]
        dev_data[key]=(transform_data_with_scaler(values, scaler).astype('float32'), sample_rate)
    for key, value in test_data.items():
        sample_rate=test_data[key][1]
        values=test_data[key][0]
        test_data[key] = (transform_data_with_scaler(values, scaler).astype('float32'), sample_rate)
    return train_data, dev_data, test_data


def shuffle_data_labels(data:np.ndarray, labels:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    permutations=np.random.permutation(data.shape[0])
    data, labels = data[permutations], labels[permutations]
    return data, labels

if __name__=="__main__":
    # params
    path_to_data=r"D:\Databases\SEWA\Original\audio"
    path_to_labels=r"D:\Databases\SEWA\SEW_labels_arousal_100Hz_gold_shifted.csv"
    test_filenames=["SEW1123","SEW1124","SEW2223","SEW2224"]
    dev_filenames=["SEW1119","SEW1120","SEW1121","SEW1122",
                          "SEW2219","SEW2220","SEW2221","SEW2222"]
    # lengths are in seconds
    chunk_length=1
    chunk_step=0.5
    # load labels and data
    labels=load_and_split_labels(path_to_labels)
    data=load_all_wav_files(path_to_data)
    # preprocess data
        # align data to labels
    data, labels= align_audio_and_labels(data, labels)
        # separate dev, test data
    train_data, dev_data, test_data=split_data_on_train_dev_test(data, dev_filenames, test_filenames)
    train_lbs, dev_lbs, test_lbs=split_labels_on_train_dev_test(labels, dev_filenames, test_filenames)
        # normalize data
    concatenated_train_data=np.concatenate([x[0] for x in train_data.values()])
    train_scaler=get_trained_minmax_scaler(data=concatenated_train_data)
    train_data, dev_data, test_data = normalize_minmax_train_dev_test((train_data, dev_data, test_data), train_scaler)
        # cut train data and labels
    cut_train_data, cut_train_lbs=cut_train_data_and_labels_on_chunks(train_data, train_lbs, chunk_length, chunk_step)
        # shuffle train data and labels
    cut_train_data, cut_train_lbs = shuffle_data_labels(cut_train_data, cut_train_lbs)
    cut_train_lbs=cut_train_lbs[:,::5,2]
    # create model
    model=create_1d_cnn_model_regression(input_shape=cut_train_data.shape[1:],num_output_neurons=20,
                                       kernel_sizes=(15,15,12,12,10,10,5,5,4,3,3),
                                       filter_numbers=(16,32,64,64,128,128,256,256,512,512,1024),
                                       pooling_step=1, need_regularization=False)
    model.compile(optimizer='Adam', loss=CCC_loss_tf)
    model.summary()
    model.fit(cut_train_data, cut_train_lbs.astype('float32')[..., np.newaxis], batch_size=1, epochs=1)