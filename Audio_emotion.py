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
import requests
import traceback
import wave
from uuid import uuid4
from config import ES_INDEX, ES_DOC
from constant import AUDIO_DIR
from elastic import get_from_elastic
from statistics import mean
from scipy.io import wavfile
from util import access_log, error_log, execution_time
from audioread.exceptions import NoBackendError



import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


## LOADING THE MODEL

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")



scaler=StandardScaler()
Y=["angry","neutral","unhappy","happy"]
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()



class AudioEmotion:
    valid_ext = '.wav'

    def __init__(self, conversation_id) -> None:
        self.conversation_id = conversation_id
        self.url = self.__url()
        self.path = None
        self.emotion = {}

    def __url(self):
        try:
            es_response = get_from_elastic(ES_INDEX, ES_DOC, self.conversation_id)
            if es_response['es_status'] == 'success':
                url = es_response.get('_source', {}).get('url')
                self.is_stereo = es_response.get('_source', {}).get('channel', False)
                self.channel_count = es_response.get('_source', {}).get('config', {}).get('audio_channel_count', 1)
                return url
        except Exception:
            traceback.print_exc()

    def __retrieve_audio(self):
        filepath = None
        try:
            audio_dir = os.path.join(os.getcwd(), AUDIO_DIR)
            if not os.path.exists(audio_dir):
                os.makedirs(AUDIO_DIR, mode=0o777, exist_ok=False)
            if self.url:
                response = requests.get(self.url)
                if response.status_code == 200:
                    filename = str(uuid4()) + self.valid_ext
                    filepath = os.path.join(audio_dir, filename)
                    with open(filepath, 'wb') as file:
                        file.write(response.content)
        except Exception:
            traceback.print_exc()
        return filepath

    def eval_audio_emotion(self):
        try:
            # Download audio to dir
            self.path = self.__retrieve_audio()
            if self.path:
                filepath_list = [self.path]
            for channel, filepath in enumerate(filepath_list, start=1):
                self.emotion = self.result(filepath)
        except wave.Error:
            exception = 'Not a wav file'
            error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
        except NoBackendError as exception:
            exception = 'Not a wav file'
            error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
        except:
            traceback.print_exc()

    @staticmethod
    def noise(self,data):
        try:
            noise_amp = 0.035 * np.random.uniform() * np.amax(data)
            data = data + noise_amp * np.random.normal(size=data.shape[0])
        except:
            traceback.print_exc()
        return data

    def stretch(self,data, rate=0.8):
        try:
            stretch_data= librosa.effects.time_stretch(data, rate)
        except:
            traceback.print_exc()
        return stretch_data

    def shift(self,data):
        try:
            shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
            shift_data= np.roll(data, shift_range)
        except:
            traceback.print_exc()
        return shift_data

    def pitch(self,data, sampling_rate, pitch_factor=0.7):
        try:
            pitch_data=librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
        except:
            traceback.print_exc()
        return pitch_data

    def __extract_features(self,data,sample_rate):
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

    def get_features(self,path):
        try:
            # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
            data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

            # without augmentation
            res1 = self.extract_features(data,sample_rate)
            result = np.array(res1)

            # data with noise
            noise_data = self.noise(data)
            res2 = self.extract_features(noise_data,sample_rate)
            result = np.vstack((result, res2))  # stacking vertically

            # data with stretching and pitching
            new_data = self.stretch(data)
            res3 = self.extract_features(new_data,sample_rate)
            result = np.vstack((result, res3))  # stacking vertically

            data_stretch_pitch = self.pitch(data, sample_rate)
            res4 = self.extract_features(data_stretch_pitch,sample_rate)
            res = np.vstack((result, res4))
        except:
            traceback.print_exc()

        return res

    ## Predicting Result
    def result(self,fname):
        try:
            La = self.get_features(fname)
            La = scaler.fit_transform(La)
            La = np.expand_dims(La, axis=2)
            pred_test = loaded_model.predict(La)
            m = []
            for i in pred_test:
                m.append(np.max(i))
            k = max(m)
            if max(m) >= 0.80:
                n = m.index(k)
                # pred_test=np.ndarray.max(pred_test)
                y_pred = encoder.inverse_transform(pred_test)

                p = y_pred[n]
            else:
                p = np.array('Neutral')

        except:
            traceback.print_exc()
        return p

    def remove(self):
        if self.path and os.path.isfile(self.path):
            os.remove(self.path)
        if hasattr(self, 'channelwise_filepaths') and self.channelwise_filepaths:
            for filepath in self.channelwise_filepaths:
                if os.path.isfile(filepath):
                    os.remove(filepath)







