#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple, Union, Optional

import tensorflow as tf
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def create_1d_cnn_model_classification(*,input_shape:Tuple[int,...],num_classes:int,kernel_sizes:Tuple[int,...]=(15,15,12,12,10,10,5,5,4,3),
                        filter_numbers:Tuple[int,...]=(16,32,64,64,128,128,256,256,512,512), pooling_step:Optional[int]=2, need_regularization:bool=False) -> tf.keras.Model:
    """ Creates 1D CNN model according to provided parameters

    :param input_shape:tuple
                    input shape for tensrflow.keras model
    :param num_classes: int
    :param kernel_sizes: list
                    list of kernel sizes
    :param filter_numbers:list
                    list of number of filters, usually in descending order
    :param pooling_step: int
                    the step after which the pooling operation will be used
                    e. g. poling_step=2 means that every 2 layers the pooling operation will be applied
    :param need_regularization: bool
                    use regularization in conv layers or not
                    if true, tf.keras.regularizers.l2(1e-5) will be applied
    :return:
    """
    # if length of kernel_sizes and filter_numbers are not equal, raise exception
    if len(kernel_sizes)!=len(filter_numbers):
        raise AttributeError('lengths of kernel_sizes and filter_numbers must be equal! Got kernel_sizes:%i, filter_numbers:%i.'%(len(kernel_sizes), len(filter_numbers)))
    # create input layer for model
    input=tf.keras.layers.Input(input_shape)
    x=input

    for layer_idx in range(len(kernel_sizes)):
        # define parameters for convenience visibility
        filters=filter_numbers[layer_idx]
        kernel_size=kernel_sizes[layer_idx]
        regularization=tf.keras.regularizers.l2(1e-5) if need_regularization else None
        # create Conv1D layer
        x = tf.keras.layers.Conv1D(filters, kernel_size,padding='same', activation='relu', kernel_regularizer=regularization)(x)
        # if now is a step of pooling layer, then create and add to model AveragePooling layer
        if (layer_idx+1)%pooling_step==0:
            x=tf.keras.layers.AveragePooling1D(2)(x)
    # apply GlobalAveragePoling to flatten output of conv1D
    x=tf.keras.layers.GlobalAveragePooling1D()(x)
    # apply Dense layer and then output softmax layer with num_class numbers neurons
    x=tf.keras.layers.Dense(512, activation='relu')(x)
    output=tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    # create a model and return it as a result of function
    model=tf.keras.Model(input=input, output=output)
    return model