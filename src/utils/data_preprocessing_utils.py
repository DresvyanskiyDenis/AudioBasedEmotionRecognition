#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from typing import List, Union, Tuple
import pandas as pd
import numpy as np
from scipy.io import wavfile

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
    sample_rate, data = wavfile.read('./output/audio.wav')
    return sample_rate, data



if __name__=='main':
    path=r'C:\Users\Dresvyanskiy\Downloads\SEW1101.wav'