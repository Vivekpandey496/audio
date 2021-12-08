"""
This file contains the class to evaluate the audio features
viz. Pitch, Amplitude, Frequency, Temp etc.
"""
import crepe
import librosa
import numpy as np
import os
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


class Audio:
    valid_ext = '.wav'

    def __init__(self, conversation_id) -> None:
        self.conversation_id = conversation_id
        self.url = self.__url()
        self.path = None
        self.amplitude = {}
        self.frequency = {}
        self.pitch = {}
        self.db = {}
        self.tempo = {}
        self.stft = {}
        self.mfcc = {}
        self.mel_spectogram = {}
        self.power_spectogram = {}
        self.chroma = {}

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

    def __segregate_channels(self, wav):
        '''
        Segregate a wav audio file into two channelwise splitted separate audio files.
        '''
        channelwise_filepaths = []
        try:
            for channel in range(self.channel_count):
                # Read data
                total_channels = wav.getnchannels()
                depth = wav.getsampwidth()
                wav.setpos(0)
                sdata = wav.readframes(wav.getnframes())

                # Extract channel data (24-bit data not supported)
                typ = {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(depth)
                if not typ:
                    exception = ValueError("sample width {} not supported".format(depth))
                    error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
                if channel >= total_channels:
                    exception = ValueError("cannot extract channel {} out of {}".format(channel+1, total_channels))
                    error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
                access_log.info(
                    "Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, total_channels, depth*8))
                data = np.fromstring(sdata, dtype=typ)
                channel_data = data[channel::total_channels]
                # Save channel to a separate file
                dirpath = '/'.join(self.path.split('/')[:-1])
                filename = self.path.split('/')[-1]
                segregated_filename = f"ch{channel + 1}_{filename}"
                filepath = os.path.join(dirpath, segregated_filename)
                with wave.open(filepath, 'w') as segregated_file:
                    segregated_file.setparams(wav.getparams())
                    segregated_file.setnchannels(1)
                    segregated_file.writeframes(channel_data.tostring())
                channelwise_filepaths.append(filepath)
        except:
            traceback.print_exc()
        self.channelwise_filepaths = channelwise_filepaths
        return channelwise_filepaths

    @execution_time
    def eval_audio_features(self):
        try:
            # Download audio to dir
            self.path = self.__retrieve_audio()
            if self.path:
                filepath_list = [self.path]
                if self.is_stereo:
                    wave_obj = wave.open(self.path)
                    filepath_list = self.__segregate_channels(wave_obj)
                for channel, filepath in enumerate(filepath_list, start=1):
                    # Load the audio as a waveform `y` & Store the sampling rate as `sr`
                    waveform, sampling_rate = librosa.load(filepath)
                    self.amplitude[channel] = self.get_amplitude(waveform, sampling_rate)
                    self.frequency[channel], self.pitch[channel] = self.__frequency_and_pitch(filepath)
                    self.db[channel] = self.get_db(waveform, sampling_rate)
                    self.tempo[channel] = self.get_tempo(waveform, sampling_rate)
                    self.stft[channel] = self.get_stft(waveform, sampling_rate)
                    self.mfcc[channel] = self.get_mfcc(waveform, sampling_rate)
                    self.mel_spectogram[channel] = self.get_mel_spectogram(waveform, sampling_rate)
                    self.power_spectogram[channel] = self.get_power_spectogram(waveform, sampling_rate)
                    self.chroma[channel] = self.get_chroma(waveform, sampling_rate)
        except wave.Error:
            exception = 'Not a wav file'
            error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
        except NoBackendError as exception:
            exception = 'Not a wav file'
            error_log.exception("Error occurred in eval_audio_features method %s \n", exception)
        except:
            traceback.print_exc()

    @staticmethod
    def get_amplitude(waveform, sampling_rate):
        amplitude = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            n = int(len(waveform)/sampling_rate)
            a = []
            d = {}

            for i in range(n):
                b = []
                for j in waveform[i*sampling_rate:(i+1)*sampling_rate]:
                    b.append(j)
                m = max(b)
                a.append((m, i))

            for i in a:
                p, q = i
                d[q] = p
            t = []
            for k in d:
                t.append(d[k])
            mean_val = mean(t)
            min_val = min(t)
            max_val = max(t)
            amplitude = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return amplitude

    def __frequency_and_pitch(self, filepath):
        frequency = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        pitch = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            sr, audio = wavfile.read(filepath)
            time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
            a = np.array((time, frequency)).T
            frequency_mean_val = np.mean(frequency)
            frequency_min_val = np.min(frequency)
            frequency_max_val = np.max(frequency)
            frequency = (str(frequency_min_val), str(frequency_max_val), str(frequency_mean_val))
            pitch_mean_val = np.mean(confidence)
            pitch_min_val = np.min(confidence)
            pitch_max_val = np.max(confidence)
            pitch = (str(pitch_min_val), str(pitch_max_val), str(pitch_mean_val))
        except:
            traceback.print_exc()
        return frequency, pitch

    @staticmethod
    def get_tempo(waveform, sampling_rate):
        # we can't find min,max or mean of tempo, this is only singular value
        tempo = 'N/A'
        try:
            tempo, beats = librosa.beat.beat_track(y=waveform, sr=sampling_rate)
            tempo = str(tempo)
        except:
            traceback.print_exc()
        return tempo

    @staticmethod
    def get_stft(waveform, sampling_rate):
        stft = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            D = librosa.stft(waveform)
            b = np.abs(D)
            max_val = np.max(b)
            min_val = np.min(b)
            mean_val = np.mean(b)
            stft = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return stft

    @staticmethod
    def get_db(waveform, sampling_rate):
        db = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            D = librosa.stft(waveform)  # STFT of y
            S_db = librosa.amplitude_to_db(np.abs(D))
            max_val = np.max(S_db)
            min_val = np.min(S_db)
            mean_val = np.mean(S_db)
            db = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return db

    @staticmethod
    def get_mfcc(waveform, sampling_rate):
        mfcc = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            mfcc = librosa.feature.mfcc(y=waveform, sr=sampling_rate, n_mfcc=13)
            mean_val = np.mean(mfcc)
            min_val = np.min(mfcc)
            max_val = np.max(mfcc)
            mfcc = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return mfcc

    @staticmethod
    def get_mel_spectogram(waveform, sampling_rate):
        mel_spectogram = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            M = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
            mean_val = np.mean(M)
            min_val = np.min(M)
            max_val = np.max(M)
            mel_spectogram = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return mel_spectogram

    @staticmethod
    def get_power_spectogram(waveform, sampling_rate):
        power_spectogram = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            M = librosa.feature.melspectrogram(y=waveform, sr=sampling_rate)
            M_db = librosa.power_to_db(M)
            mean_val = np.mean(M_db)
            min_val = np.min(M_db)
            max_val = np.max(M_db)
            power_spectogram = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return power_spectogram

    @staticmethod
    def get_chroma(waveform, sampling_rate):
        chroma = {'min': 'N/A', 'max': 'N/A', 'mean': 'N/A'}
        try:
            chroma = librosa.feature.chroma_cqt(y=waveform, sr=sampling_rate)
            mean_val = np.mean(chroma)
            min_val = np.min(chroma)
            max_val = np.max(chroma)
            chroma = (str(min_val), str(max_val), str(mean_val))
        except:
            traceback.print_exc()
        return chroma

    def get_features_dict(self):
        try:
            feature_list = ['amplitude', 'frequency', 'pitch', 'db', 'tempo', 'stft',
                            'mfcc', 'mel_spectogram', 'power_spectogram', 'chroma', ]
            feature_dict = {}
            for feat in self.__dict__:
                if feat not in feature_list or not self.__dict__[feat]:
                    continue
                feature_dict[feat] = {}
                for channel in range(1, self.channel_count + 1):
                    if channel not in self.__dict__[feat]:
                        continue
                    if isinstance(self.__dict__[feat][channel], tuple):
                        feature_dict[feat][channel] = {}
                        feature_dict[feat][channel]['min'] = self.__dict__[feat][channel][0]
                        feature_dict[feat][channel]['max'] = self.__dict__[feat][channel][1]
                        feature_dict[feat][channel]['mean'] = self.__dict__[feat][channel][2]
                    else:
                        feature_dict[feat][channel] = self.__dict__[feat][channel]
        except:
            traceback.print_exc()
        return feature_dict

    def remove(self):
        if self.path and os.path.isfile(self.path):
            os.remove(self.path)
        if hasattr(self, 'channelwise_filepaths') and self.channelwise_filepaths:
            for filepath in self.channelwise_filepaths:
                if os.path.isfile(filepath):
                    os.remove(filepath)
