import pandas as pd
import numpy as np

import os
import sys


import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
import keras
from keras.models import load_model
import pickle

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)



## FastAPI Imports
from fastapi import FastAPI
from typing import List,Dict,Any
from pydantic import BaseModel
import uvicorn


app = FastAPI() ## App starts from here


## Data Augmentation

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)



## Feature Extraction
def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    # Power Spec
    M = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    M_db = np.mean(librosa.power_to_db(M).T, axis=0)
    result = np.hstack((result, M_db))  # stacking horizontally

    # Tempo
    chroma = librosa.feature.chroma_cqt(y=data, sr=sample_rate)
    tempo, beats = librosa.beat.beat_track(y=data, sr=sample_rate)
    result = np.hstack((result, tempo))

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    res3 = extract_features(new_data)
    result = np.vstack((result, res3))  # stacking vertically

    data_stretch_pitch = pitch(data, sample_rate)
    res4 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res4))

    return result


def res_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res1 = extract_features(data)
    result = np.array(res1)

    return result

Y=["angry","neutral","unhappy","happy"]
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

scaler = StandardScaler()



## LOADING THE MODEL

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Emotion_Voice_Dete q ction_Model.h5")
print("Loaded model from disk")


@app.post('/emotion') # API CALL
## Predicting Result
def result(fname):
    La=get_features(fname)
    scaler = StandardScaler()
    La = scaler.fit_transform(La)
    La = np.expand_dims(La, axis=2)
    pred_test = loaded_model.predict(La)
    m=[]
    for i in pred_test:
        m.append(np.max(i))
    k=max(m)
    n=m.index(k)
    y_pred = encoder.inverse_transform(pred_test)
    p=y_pred[n]
    return p

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)












