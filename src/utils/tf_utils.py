#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import Tuple, Union, Optional
import tensorflow as tf
import tensorflow.keras.backend as K
__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


def create_1d_cnn_model_classification(*,input_shape:Tuple[int,...],num_classes:int,
                                       kernel_sizes:Tuple[int,...]=(15,15,12,12,10,10,5,5,4,3),
                                       filter_numbers:Tuple[int,...]=(16,32,64,64,128,128,256,256,512,512),
                                       pooling_step:Optional[int]=2, need_regularization:bool=False) -> tf.keras.Model:
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
    model=tf.keras.Model(inputs=input, outputs=output)
    return model



def create_1d_cnn_model_regression(*,input_shape:Tuple[int,...],num_output_neurons:int,
                                       kernel_sizes:Tuple[int,...]=(15,15,12,12,10,10,5,5,4,3),
                                       filter_numbers:Tuple[int,...]=(16,32,64,64,128,128,256,256,512,512),
                                       pooling_sizes:Tuple[int,...]=(1,1,1,1,1,1,1,1),
                                       pooling_step:Optional[int]=2, need_regularization:bool=False,
                                       dropout:bool=False) -> tf.keras.Model:
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
    if len(kernel_sizes)/pooling_step!=len(pooling_sizes):
        raise AttributeError('length of kernel_sizes must be pooling_step time more than length of pooling_sizes. '
                             'Got kernel_sizes:%i, pooling_step:%i, pooling_sizes:%i.'%(len(kernel_sizes), pooling_step,len(pooling_sizes)))
    # create input layer for model
    input=tf.keras.layers.Input(input_shape)
    x=input
    pooling_idx=0
    for layer_idx in range(len(kernel_sizes)):
        # define parameters for convenience visibility
        filters=filter_numbers[layer_idx]
        kernel_size=kernel_sizes[layer_idx]
        regularization=tf.keras.regularizers.l2(1e-5) if need_regularization else None
        # create Conv1D layer
        x = tf.keras.layers.Conv1D(filters, kernel_size,padding='same', activation='relu', kernel_regularizer=regularization)(x)
        # add dropout
        if dropout: x=tf.keras.layers.Dropout(0.2)(x)
        # if now is a step of pooling layer, then create and add to model AveragePooling layer
        if (layer_idx+1)%pooling_step==0:
            x=tf.keras.layers.MaxPool1D(pooling_sizes[pooling_idx])(x)
            pooling_idx+=1
    # apply GlobalAveragePoling to flatten output of conv1D
    x=tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    # apply Dense layer and then output softmax layer with num_class numbers neurons
    #x=tf.keras.layers.Dense(128, activation='tanh')(x)
    output=tf.keras.layers.Dense(1, activation='tanh')(x)
    # create a model and return it as a result of function
    model=tf.keras.Model(inputs=input, outputs=output)
    return model



def ccc_loss(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.epsilon())
    ccc_loss   = K.constant(1.) - ccc
    return ccc_loss

def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """
    #tf.print('y_true_shape:',tf.shape(y_true))
    #tf.print('y_pred_shape:',tf.shape(y_pred))

    y_true_mean = K.mean(y_true, axis=-2, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-2, keepdims=True)

    y_true_var = K.mean(K.square(y_true-y_true_mean), axis=-2, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred-y_pred_mean), axis=-2, keepdims=True)

    cov = K.mean((y_true-y_true_mean)*(y_pred-y_pred_mean), axis=-2, keepdims=True)

    ccc = tf.math.multiply(2., cov) / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    ccc_loss=1.-K.mean(K.flatten(ccc))
    #tf.print('ccc:', tf.shape(ccc_loss))
    #tf.print('ccc_loss:',ccc_loss)
    return ccc_loss



if __name__=="__main__":
    pass