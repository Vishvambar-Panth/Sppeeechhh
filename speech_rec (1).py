import numpy as np
import speaker_recog as spr
import python_speech_features
from tensorflow import keras
import sounddevice as sd
import speech_recognition as sr
import RPi.GPIO as GPIO 


def audio2feature(audio):
    features = python_speech_features.base.mfcc(audio,
                                                samplerate=16000,
                                                winlen=0.025,
                                                winstep=0.01,
                                                numcep=20,
                                                nfilt=20,
                                                nfft=2048,
                                                preemph=0,
                                                ceplifter=0,
                                                appendEnergy=False,
                                                winfunc=np.hanning)
    return features.transpose()


model = keras.models.load_model('cmd_recog_model.h5')
targets = ['on', 'two', 'one', 'three', 'off']

GPIO.setmode(GPIO.BCM)

off = 1
off_pin = 27
GPIO.setup(off_pin, GPIO.OUT, initial=1)
on = 0
on_pin = 17
GPIO.setup(on_pin, GPIO.OUT, initial=0)
speed_1 = 0
speed_1_pin = 22
GPIO.setup(speed_1_pin, GPIO.OUT, initial=0)
speed_2 = 0
speed_2_pin = 10
GPIO.setup(speed_2_pin, GPIO.OUT, initial=0)
speed_3 = 0
speed_3_pin = 9
GPIO.setup(speed_3_pin, GPIO.OUT, initial=0)

def switch_leds():
    if off == 1:
        GPIO.output(off_pin, 1) 
    else:
        GPIO.output(off_pin, 0) 
    if on == 1:
        GPIO.output(on_pin, 1) 
    else:
        GPIO.output(on_pin, 0) 
    if speed_1 == 1:
        GPIO.output(speed_1_pin, 1) 
    else:
        GPIO.output(speed_1_pin, 0) 
    if speed_2 == 1:
        GPIO.output(speed_2_pin, 1) 
    else:
        GPIO.output(speed_2_pin, 0) 
    if speed_3 == 1:
        GPIO.output(speed_3_pin, 1) 
    else:
        GPIO.output(speed_3_pin, 0) 


def predict_cmd():
    print("Listening for command...")
    cmd_rec = sd.rec(samplerate=16000,
                     channels=1,
                     dtype="float32",
                     frames=16000 * 1)
    sd.wait()
    print("Recording stopped")
    recorded_feature = audio2feature(cmd_rec)
    recorded_feature = np.float32(recorded_feature.reshape(1, recorded_feature.shape[0], recorded_feature.shape[1], 1))
    prediction = model.predict(recorded_feature).reshape((5,))
    prediction /= prediction.sum()
    best_candidate_index = prediction.argmax()
    best_candidate_probability = prediction[best_candidate_index]
    word = targets[best_candidate_index]
    speak_verif = spr.test_model(cmd_rec, 16000)
    if speak_verif == True:
        return word
    else:
        return None


r = sr.Recognizer()
mic = sr.Microphone()


def speech_to_text():
    global on
    global off
    global speed_1
    global speed_2
    global speed_3
    while True:
        switch_leds()
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, phrase_time_limit=1)
        audioText = r.recognize_google(audio)
        if audioText == "motor":
            print(audioText)
            cmd = predict_cmd()
            print(cmd)
            if cmd == 'on':
                on = 1
                off = 0
                speed_1 = 1
                speed_2 = 0
                speed_3 = 0
            elif cmd == 'off':
                on = 0
                off = 1
                speed_1 = 0
                speed_2 = 0
                speed_3 = 0
            elif cmd == 'one':
                if on == 1:
                    speed_1 = 1
                    speed_2 = 0
                    speed_3 = 0
            elif cmd == 'two':
                if on == 1:
                    speed_1 = 0
                    speed_2 = 1
                    speed_3 = 0
            elif cmd == 'three':
                if on == 1:
                    speed_1 = 0
                    speed_2 = 0
                    speed_3 = 1
            else:
                print("Invalid User or Command.")
        else:
            print("Wakeword not detected!")


def main_func():
    try:
        speech_to_text()
    except Exception as e:
        print(str(e))
        print("Sry,we couldnt get your voice")
        print("Try again")
        main_func()


if __name__ == "__main__":
    while True:
        opr = input("Select Operation to perform:\n1) Train Speaker\n2) Activate Voice Recognition\n")
        if opr == '1':
            spr.record_audio_train()
            spr.train_model()
        elif opr == '2':
            main_func()
        else:
            print("Invalid Option")
