#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Contains script of training the model on MSP-podcast dataset. Dataset description: https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html
    Model is based on dynamic adjusting steps of cut window to always get the same number of chunks.
    THe idea of model is taken from: https://indico2.conference4me.psnc.pl/event/35/contributions/3415/attachments/531/557/Wed-1-9-1.pdf

List of functions:
    * load_and_preprocess_labels_MSP_podcast - laods and preprocess labels of original MSP-podcast database.
    * delete_instances_with_class - deletes instances from dataset with undesirable class (usually -1)
    * get_train_dev_separation_in_dict - loads partition file and returns dict, in which the training and development
    files are defined
    * get_labels_according_to_filenames - returns the subset of provided labels according proviede filenames in list
    * custom_recall_validation_with_generator - recall score, which calculates recall on one full iteration of generator
    * validation_callback - class for model.fit() function. Calculates recall score at the end of each epoch and save
    model weights if they are better all the former (thus, at the end of training you will have the best weights). Moreover,
    it assign at the end of training saved best weights to the model.
    * separate_train_dev_test_audion_on_different_dirs - moves files located in one directory to separate directories.
    Thus, separates train, dev and test sets apart each other.
"""

import contextlib
import shutil
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

from sklearn.metrics import recall_score

from keras_datagenerators.generator_loader import FixedChunksGenerator_loader
from preprocessing.sequence_to_one_models import chunk_based_rnn_model

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
                filename=splitted_line[0].strip()
                class_category=splitted_line[1].strip()
                labels[filename]=np.array(emotion_to_int_mappping[class_category]).reshape((-1,))
    return labels

def delete_instances_with_class(labels:Dict[str, np.ndarray], class_to_delete:int=-1)-> Dict[str, np.ndarray]:
    """Deletes instances with label defined in class_to_delete from labels.

    :param labels: Dict[str, np.ndarray]
                labels in format Dict[filename->np.ndarray with shape (1,1)]
    :param class_to_delete: int
                class shoud to be deleted
    :return: Dict[str, np.ndarray]
                labels in format Dict[filename->np.ndarray with shape (1,1)] without deleted instances
    """
    keys_to_delete=[]
    for key, item in labels.items():
        if item==class_to_delete:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        labels.pop(key)
    return labels

def get_train_dev_separation_in_dict(path:str)-> Dict[str, List[str]]:
    """Loads train and dev partition according to file provided via path.

    :param path: str
            path to the file with partitions on train and dev sets. SHould be .txt file
    :return: Dict[str, List[str]]
            dict in format ['train'->List[filenames], 'dev'->List[filenames]]
    """
    labels_partition_dict={}
    partition_df=pd.read_csv(path, sep=';', header=None)
    labels_partition_dict['train']=partition_df[partition_df.iloc[:,0]=='Train'].iloc[:,1].to_list()
    labels_partition_dict['dev'] = partition_df[partition_df.iloc[:,0]=='Validation'].iloc[:,1].to_list()
    return labels_partition_dict

def get_labels_according_to_filenames(labels:Dict[str, np.ndarray],
                                                  filenames:List[str])-> Dict[str, np.ndarray]:
    """Returns subset of labels according to provided filenames in List.

    :param labels:Dict[str, np.ndarray]
                labels in format DIct[filename->np.ndarray with shape (1,1)]
    :param filenames: List[str]
                filenames, which should be included in new formed subset
    :return:Dict[str, np.ndarray]
                subset of labels according to provided filenames
    """
    extracted_labels=dict((filename, labels[filename]) for filename in filenames)
    return extracted_labels

def custom_recall_validation_with_generator(generator:FixedChunksGenerator_loader, model:tf.keras.Model)->float:
    total_predictions=np.zeros((0,))
    total_ground_truth=np.zeros((0,))
    for x,y in generator:
        predictions=model.predict(x)
        predictions=predictions.argmax(axis=-1).reshape((-1,))
        total_predictions=np.append(total_predictions,predictions)
        total_ground_truth=np.append(total_ground_truth,y.argmax(axis=-1).reshape((-1,)))
    return recall_score(total_ground_truth, total_predictions,average='macro')

class validation_callback(tf.keras.callbacks.Callback):
    """Calculates the recall score at the end of each training epoch and saves the best weights across all the training
        process. At the end of training process, it will set weights of the model to the best found ones.

    """
    def __init__(self, val_generator:FixedChunksGenerator_loader):
        super(validation_callback, self).__init__()
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator=val_generator

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        current_recall=custom_recall_validation_with_generator(self.val_generator, self.model)
        print('current_recall:', current_recall)
        if np.greater(current_recall, self.best):
            self.best = current_recall
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)

def separate_train_dev_test_audion_on_different_dirs(path_to_data:str, path_to_partitions:str, destination_path:str) -> None:
    # read partitions from file
    partitions=pd.read_csv(path_to_partitions, sep=';', header=None)
    # create destination directory
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
    # create train, val, test directories
    if not os.path.exists(os.path.join(destination_path, 'train')):
        os.makedirs(os.path.join(destination_path, 'train'), exist_ok=True)
    if not os.path.exists(os.path.join(destination_path, 'dev')):
        os.makedirs(os.path.join(destination_path, 'dev'), exist_ok=True)
    if not os.path.exists(os.path.join(destination_path, 'test')):
        os.makedirs(os.path.join(destination_path, 'test'), exist_ok=True)
    # separate train, dev, test
    train_filenames=partitions[partitions[0]=='Train'].iloc[:,1].values
    val_filenames=partitions[partitions[0]=='Validation'].iloc[:,1].values
    test_filenames=partitions[(partitions[0]=='Test1') | (partitions[0]=='Test2')].iloc[:,1].values
    # moving train files
    for filename in train_filenames:
        strip_filename=filename.strip()
        old_path=os.path.join(path_to_data, strip_filename)
        new_path=os.path.join(destination_path, 'train',strip_filename)
        shutil.move(old_path, new_path)
    # moving val files
    for filename in val_filenames:
        strip_filename=filename.strip()
        old_path=os.path.join(path_to_data, strip_filename)
        new_path=os.path.join(destination_path, 'dev',strip_filename)
        shutil.move(old_path, new_path)
    # moving test files
    for filename in test_filenames:
        strip_filename=filename.strip()
        old_path=os.path.join(path_to_data, strip_filename)
        new_path=os.path.join(destination_path, 'test',strip_filename)
        shutil.move(old_path, new_path)




def main():
    path_to_labels = 'C:\\MSP_podcast\\labels\\Labels.txt'
    path_to_train_data = 'C:\\MSP_podcast\\Sep_audios\\train'
    path_to_val_data='C:\\MSP_podcast\\Sep_audios\\dev'
    path_to_labels_partition='C:\\MSP_podcast\\Partitions.txt'
    path_to_save_model_weights='best_model'
    one_hot_labeling = True
    num_classes = 7
    normalization = True
    data_preprocessing_mode = 'MFCC'
    num_mfcc = 128
    sequence_max_length = 14
    window_length = 0.5
    batch_size=64
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
    train_generator=FixedChunksGenerator_loader(sequence_max_length=sequence_max_length, window_length=window_length,
                                                load_path= path_to_train_data, resample=16000,
                 data_preprocessing_mode= data_preprocessing_mode, num_mfcc = num_mfcc,
                 labels = train_labels, labels_type= 'sequence_to_one', batch_size=batch_size ,
                 normalization= normalization, one_hot_labeling = one_hot_labeling,
                 num_classes = num_classes)
    # define dev generato and callback to evaluate model performance at the end of epoch
    dev_generator=FixedChunksGenerator_loader(sequence_max_length=sequence_max_length, window_length=window_length,
                                              load_path= path_to_val_data,resample=16000,
                 data_preprocessing_mode= data_preprocessing_mode, num_mfcc = num_mfcc,
                 labels = dev_labels, labels_type= 'sequence_to_one', batch_size=batch_size ,
                 normalization= normalization, one_hot_labeling = one_hot_labeling,
                 num_classes = num_classes)
    dev_callback=validation_callback(dev_generator)
    # create and compile model
    input_shape = (28, 32, 128)
    model = chunk_based_rnn_model(input_shape=input_shape, num_output_neurons=num_classes,
                          neurons_on_layer = (256, 512),
                          rnn_type = 'LSTM',
                          bidirectional = False,
                          need_regularization = True,
                          dropout = True,
                          dropout_rate=0.3
                          )

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Recall()])
    model.summary()
    # fit model
    hist = model.fit(train_generator, epochs=15, use_multiprocessing=True, callbacks=[dev_callback])
    # save model. The model will be saved with best weights evaluated on devel set (since the dev_callback will set
    # weights of model at the end of training process)
    if not os.path.exists(path_to_save_model_weights):
        os.mkdir(path_to_save_model_weights)
    model.save_weights(os.path.join(path_to_save_model_weights,'best_model_weights.h5'))




if __name__=='__main__':
    main()
    '''separate_train_dev_test_audion_on_different_dirs(path_to_data='E:\\Databases\\MSP_podcast\\Audios',
                                                     path_to_partitions='E:\\Databases\\MSP_podcast\\Partitions.txt',
                                                     destination_path='E:\\Databases\\MSP_podcast\\Sep_audios')'''