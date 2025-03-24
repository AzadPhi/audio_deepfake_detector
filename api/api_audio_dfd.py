
import pandas as pd
import numpy as np
import librosa
import os
import tensorflow as tf
# from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import tempfile
import io
# from click import File
import shutil
from pydantic import BaseModel

from nos_paquets.params import *
from nos_paquets.sound_prep.preprocess import *


def load_model():
    """download the model from GCS"""

    client = storage.Client()

    bucket = client.bucket(BUCKET_CHECKPOINT)
    blob = bucket.blob("DEMO/DEMO_checkpoint.model.keras")

    temp_dir = tempfile.mkdtemp()  # Create a temp directory
    model_local_path = os.path.join(temp_dir, os.path.basename("DEMO/DEMO_checkpoint.model.keras"))

    blob.download_to_filename(model_local_path)

    model = tf.keras.models.load_model(model_local_path)

    print("✅ Model loaded successfully")

    return model


app = FastAPI()
app.state.model = load_model()
#app.state.model = tf.keras.models.load_model(LOCAL_PATH_TO_MODEL)


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# @app.post("/upload-audio")
# async def upload_audio(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)

#     # Save the uploaded file locally
#     with open(file_path, "wb") as buffer:
#         buffer.write(await file.read())

#     return {"file_path": file_path, "message": "File successfully uploaded"}



@app.post("/predict")
#async def predict(audio: AudioInput):
async def predict(file: UploadFile = File(...)):
    """makes a prediction :
            - it will first download locally the audio file
            - it will process the audio file and convert it into mel-spectrogramm
            - returns if it is AI generated or not"""

    tmp_filename = f"./temp/{file.filename}"


    with open(tmp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conf = Conf()
    conf.sampling_rate=16000
    conf.n_mels=128
    conf.duration=10
    conf.hop_length=347*conf.duration
    conf.fmin=20
    conf.fmax=conf.sampling_rate//2
    conf.n_fft=conf.n_mels*20
    conf.samples=conf.sampling_rate * conf.duration

    print(tmp_filename)

    X = read_as_melspectrogram(conf=conf, pathname=tmp_filename)

    # os.remove(tmp_filename)

    # X = read_as_melspectrogram(conf=conf, pathname=io.BytesIO(await file.read()))


    X = X.flatten()
    X = X.reshape((128,47))
    X = np.expand_dims(np.stack(X), axis=-1)
    X = np.expand_dims(X, axis=0)

    y_pred = app.state.model.predict(X)

    print(y_pred)

    if y_pred[0][0] > 0.5:
        print("This sound has been AI generated")
        return "This sound has been AI generated"
    else:
        print("This sound has been created by real humans")
        return "This sound has been created by real humans"


# class AudioInput(BaseModel):
#     file_path: str  # Receives the file path

# audio = AudioInput(file_path=LOCAL_PATH_TO_RAW_DATA)

@app.get("/predict_get")
async def predict_get(audio):
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

    if y_pred[0][0] > 0.5:
        print("This sound has been AI generated")
        return "This sound has been AI generated"
    else:
        print("This sound has been created by real humans")
        return "This sound has been created by real humans"

@app.get("/")
async def root():
    return {"message": "Ne pas avoir toutes ses frites dans le même sachet = ne pas avoir toute sa tête"}
