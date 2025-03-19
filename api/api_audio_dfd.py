
import pandas as pd
import numpy as np
import librosa
import os
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage

from nos_paquets.sound_prep.params import *
from nos_paquets.sound_prep.preprocess import *




def load_model():
    """download the model from GCS"""

    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_CHECKPOINT).list_blobs(prefix="DEMO"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_MODEL_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = tf.keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model

    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_CHECKPOINT}")

        return None


app = FastAPI()
#app.state.model = load_model()
app.state.model = tf.keras.models.load_model(LOCAL_PATH_TO_MODEL)


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/predict")
async def predict(audio):
    """makes a prediction :
            - it will first download locally the audio file
            - it will process the audio file and convert it into mel-spectrogramm
            - returns if it is AI generated or not"""

    audio_file_path=LOCAL_PATH_TO_RAW_DATA

    conf = Conf()
    conf.sampling_rate=16000
    conf.n_mels=128
    conf.duration=10
    conf.hop_length=347*conf.duration
    conf.fmin=20
    conf.fmax=conf.sampling_rate//2
    conf.n_fft=conf.n_mels*20
    conf.samples=conf.sampling_rate * conf.duration

    X = read_as_melspectrogram(conf=conf,
                               pathname=audio_file_path)

    X = X.flatten()
    X = X.reshape((128,47))
    X = np.expand_dims(np.stack(X), axis=-1)
    X = np.expand_dims(X, axis=0)

    y_pred = app.state.model.predict(X)

    if y_pred == 1:
        print("This sound has been AI generated")
        return "This sound has been AI generated"
    else:
        print("This sound has been created by real humans")
        return "This sound has been created by real humans"


@app.get("/")
async def root():
    return {"message": "Ne pas avoir toutes ses frites dans le même sachet = ne pas avoir toute sa tête"}
