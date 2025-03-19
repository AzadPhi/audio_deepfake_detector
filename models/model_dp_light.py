import pandas as pd
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import keras
import IPython.display as ipd
import ast
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import (Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple
from nos_paquets.sound_prep.params import *
from models.reshaping import *
from google.cloud import storage


### ------------ Etape 4: 1er Modèle CNN léger ------------
def model_cnn_light(input_shape, use_global_pooling=True):
    model = models.Sequential()

    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape)) # Number of filters in the layer + Size of each filter
    model.add(BatchNormalization()) ## Rend le modèle plus rapide
    model.add(Activation('relu')) ## Activation recommandée
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    if use_global_pooling:
        model.add(GlobalAveragePooling2D()) # Reduces size by taking the max value in 2x2 regions (apparently good for CNN)
    else:
        model.add(Flatten())

    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid')) #for binary choice

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

### ------------ Etape 5 : Compile le modèle ------------
# Compile the model
def compile_model_cnn_light(model: models.Model, learning_rate=0.05):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', #For binary choix (0 or 1)
        metrics=['accuracy']
    )
    print("🎉​🎉​ MODEL COMPILED 🎉​🎉​")
    return model

### ------------ Etape 6 : Test le modèle ------------
# Define the function and input parameters
def train_model_cnn_light(
        model: models.Model,  # The CNN model to be trained
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=256,  # Number of samples per batch
        validation_data=None,  # Overrides validation_split if provided
        validation_split=0.3  # Percentage of training data for validation
    ):

    # Always save localy
    checkpoint_path = LOCAL_PATH_SAVE_WEIGHT

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
        save_freq="epoch",
    )

    history = model.fit(
        X_train, y_train,
        epochs=30,  # Train the model for 30 epochs
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0.0,
        callbacks=[early_stopping, checkpoint]  # Always save locally
    )

    # If running in Google Cloud, upload the model after training
    if TARGET == 'gcloud':
        cloud_checkpoint_path = CLOUD_PATH_SAVE_WEIGHT_LIGHT
        upload_to_gcloud_light(checkpoint_path, cloud_checkpoint_path)

    print("🏋️​🏋️​ MODEL TRAINED 🏋️​🏋️​")
    if TARGET == 'gcloud':
        print(f"☁️ Model uploaded in Cloud ☁️")

    return model, history.history
### ------------ Etape 6.1 : Évaluer le modèle sur les données de test ------------
def evaluate_model_light(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"🎯 FINAL MODEL PERFORMANCE 🎯")
    print(f"💢💢 Loss : {test_loss:4f} 💢💢")
    print(f"✅​✅​ Accuracy : {test_acc:.4%}✅​✅​")

    return test_loss, test_acc

### ------------ Step 7: Google Cloud Upload Function ------------
# Checks if the model exists locally before uploading, connects to Google Cloud Storage, uploads the file to the specified cloud bucket, prints confirmation with the file’s GCS path. Not in the main as doesn't run the modle
def upload_to_gcloud_light(local_model_path, destination_blob_name):
    """Uploads a model file to Google Cloud Storage."""

    if not os.path.exists(local_model_path):
        print(f"File not found: {local_model_path}")
        return

    client = storage.Client()
    bucket = client.bucket(BUCKET_CHECKPOINT)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_model_path)
    print("DATA UPLOADED IN THE CLOUD")
