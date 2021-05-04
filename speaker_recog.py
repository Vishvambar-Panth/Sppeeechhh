import os
import wave
import time
import pickle
import sounddevice as sd
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from scipy.io.wavfile import write

warnings.filterwarnings("ignore")

def calculate_delta(array):
    rows, cols = array.shape
    #print(rows)
    #print(cols)
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=2048, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    #print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def record_audio_train():
    Name = (input("Please Enter Your Name:"))
    for count in range(5):
        FORMAT = "float32"
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 2

        print("recording started")
        
        recording = sd.rec(samplerate=RATE, channels=CHANNELS,
                           dtype=FORMAT, frames=RATE*RECORD_SECONDS)
        sd.wait()

        print("recording stopped")
        if not os.path.exists('training_set'):
            os.makedirs('training_set')
        OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME + "\n")
        write(WAVE_OUTPUT_FILENAME, RATE, recording)



def record_audio_test():
    FORMAT = "float32"
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 2

    print("recording started")

    recording = sd.rec(samplerate=RATE, channels=CHANNELS,
                           dtype=FORMAT, frames=RATE*RECORD_SECONDS)
    sd.wait()
   
    print("recording stopped")
    
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    write(WAVE_OUTPUT_FILENAME, RATE, recording)

def train_model():
    source = "./training_set/"
    if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
    dest = "./trained_models/"
    train_file = "./training_set_addition.txt"
    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(n_components=20, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1

def test_model(audio_data, sr):
    modelpath = "./trained_models/"

    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
   
    vector = extract_features(audio_data, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    print(log_likelihood)
    winner = np.argmax(log_likelihood)
    if np.max(log_likelihood) > -35:
        print("\tdetected as -", speakers[winner].split("/")[2])
        return True
    else:
        return False
