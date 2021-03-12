#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# TODO: write description of the file
"""
from typing import Tuple, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler


__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"


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



if __name__=="__main__":
    pass