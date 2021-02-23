#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os
import scipy.signal as sps
import tensorflow as tf

from src.utils.data_preprocessing_utils import load_wav_file, get_trained_minmax_scaler, transform_data_with_scaler, \
    cut_data_on_chunks, extract_mfcc_from_audio_sequence, extract_opensmile_features_from_audio_sequence
from src.utils.label_preprocessing_utils import load_gold_shifted_labels, split_labels_dataframe_according_filenames
from src.utils.tf_utils import create_1d_cnn_model_classification, create_1d_cnn_model_regression, ccc_loss, \
    CCC_loss_tf, create_simple_RNN_network


def load_and_split_labels(path:str) -> Dict[str,pd.DataFrame]:
    labels = load_gold_shifted_labels(path)
    splitted_labels=split_labels_dataframe_according_filenames(labels)
    return splitted_labels

def load_all_wav_files(path:str, resample:bool=False, new_sample_rate:int=16000)-> Dict[str,Tuple[np.ndarray, int]]:
    loaded_wav_files={}
    filenames=os.listdir(path)
    for filename in filenames:
        sample_rate,wav_file=load_wav_file(os.path.join(path,filename))
        # Resample data
        if resample:
            number_of_samples = round(len(wav_file) * new_sample_rate / sample_rate)
            wav_file = sps.resample(wav_file, number_of_samples)
            sample_rate=new_sample_rate

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

def normalize_data(data:np.ndarray, scaler:object)->np.ndarray:
    for i in range(data.shape[0]):
        data[i]=scaler.transform(data[i])
    return data

def shuffle_data_labels(data:np.ndarray, labels:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    permutations=np.random.permutation(data.shape[0])
    data, labels = data[permutations], labels[permutations]
    return data, labels

def delete_data_with_percent_zeros(data:np.ndarray, labels:np.ndarray, percent:int)->Tuple[np.ndarray, np.ndarray]:
    new_data=[]
    new_labels=[]
    threshold_amount_of_zeros=int(data.shape[1]*percent/100)
    for i in range(data.shape[0]):
        values=data[i]
        if not (values==0).sum()>=threshold_amount_of_zeros:
            new_data.append(data[i][np.newaxis,...])
            new_labels.append(labels[i][np.newaxis,...])
    new_data=np.concatenate(new_data, axis=0)
    new_labels=np.concatenate(new_labels, axis=0)
    return new_data, new_labels

def extract_mfcc_from_cut_data(data:np.ndarray, sample_rate:int=16000, num_mfcc:int=256,
                                     length_fft:int=512, length_fft_step:int=256)-> np.ndarray:
    extracted_mfcc=[]
    for i in range(data.shape[0]):
        extracted_mfcc_audio=extract_mfcc_from_audio_sequence(data[i].reshape((-1,)), sample_rate=sample_rate, num_mfcc=num_mfcc,
                                                              length_fft=length_fft, length_fft_step=length_fft_step)
        extracted_mfcc.append(extracted_mfcc_audio.T[np.newaxis,...])
    extracted_mfcc=np.concatenate(extracted_mfcc, axis=0)
    return extracted_mfcc

def extract_opensmile_features_from_cut_data(data:np.ndarray, sample_rate:int=16000)-> np.ndarray:
    extracted_features=[]
    for i in range(data.shape[0]):
        extracted_opensmile_features=extract_opensmile_features_from_audio_sequence(data[i],sample_rate=sample_rate)
        extracted_features.append(extracted_opensmile_features[np.newaxis,...])
    extracted_features = np.concatenate(extracted_features, axis=0)
    return extracted_features


if __name__=="__main__":
    # params
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    path_to_data=r"D:\Databases\SEWA\Original\audio"
    path_to_labels=r"D:\Databases\SEWA\SEW_labels_arousal_100Hz_gold_shifted.csv"
    test_filenames=["SEW1123","SEW1124","SEW2223","SEW2224"]
    dev_filenames=["SEW1119","SEW1120","SEW1121","SEW1122",
                          "SEW2219","SEW2220","SEW2221","SEW2222"]
    # lengths are in seconds
    chunk_length=2
    chunk_step=0.5
    # load labels and data
    labels=load_and_split_labels(path_to_labels)
    data=load_all_wav_files(path_to_data, resample=True, new_sample_rate=16000)
    # preprocess data
        # align data length according to labels
    data, labels= align_audio_and_labels(data, labels)
        # separate dev, test data
    train_data, dev_data, test_data=split_data_on_train_dev_test(data, dev_filenames, test_filenames)
    train_lbs, dev_lbs, test_lbs=split_labels_on_train_dev_test(labels, dev_filenames, test_filenames)
    del dev_data
    del test_data

        # cut train data and labels
    cut_train_data, cut_train_lbs=cut_train_data_and_labels_on_chunks(train_data, train_lbs, chunk_length, chunk_step)
        # delete data with zeros more than 50%
    cut_train_data, cut_train_lbs = delete_data_with_percent_zeros(cut_train_data, cut_train_lbs, percent=50)
        #extract mfcc from cut data
    cut_train_data = extract_opensmile_features_from_cut_data(cut_train_data)
        # train normalizer
    concatenated_train_data=np.concatenate([x for x in cut_train_data])
    train_scaler=get_trained_minmax_scaler(data=concatenated_train_data)
        # normalize train data
    cut_train_data=cut_train_data.astype('float32')
    cut_train_data = normalize_data(cut_train_data, train_scaler)
        # shuffle train data and labels
    cut_train_data, cut_train_lbs = shuffle_data_labels(cut_train_data, cut_train_lbs)
    cut_train_lbs=cut_train_lbs[:,::4,2]
    # create model
    """    model=create_1d_cnn_model_regression(input_shape=cut_train_data.shape[1:],num_output_neurons=cut_train_lbs.shape[1],
                                       kernel_sizes=(10,8,6,5),
                                       filter_numbers=(128,128,256,256),
                                       pooling_sizes=(10,4,4,4),
                                       pooling_step=1, need_regularization=False, dropout=True)"""
    model=create_simple_RNN_network(input_shape=cut_train_data.shape[1:],num_output_neurons=50,
                              neurons_on_layer = (128, 128),
                              rnn_type='LSTM',
                              dnn_layers=(256,),
                              need_regularization = True,
                              dropout= False)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=ccc_loss, metrics=['mse'])
    model.summary()
    model.fit(cut_train_data.astype('float32'), cut_train_lbs.astype('float32')[..., np.newaxis], batch_size=128, epochs=200)