import pandas as pd
import numpy as np
import librosa.display
import tensorflow as tf
import IPython.display as ipd
import ast
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.layers import (Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple
from nos_paquets.sound_prep.params import *
from google.cloud import storage


### ------------ Etape 1: Récuperer le CSV ------------
def load_data_heavy(csv_path):
# Load dataset from a CSV file
    df = pd.read_csv(csv_path)
    print("DATA LOADED")
    return df

### ------------ Etape 2: Reshape dataframe ------------

def reshape_spectrograms_heavy(df: pd.DataFrame, array_col="music_array", shape_col="shape_arr"):
    # Transform the music array value into a Tuple so that it can be read by the Model
    reshaped_arrays = []  # To store reshaped spectrograms
    valid_indices = []  # Track valid indices for potential filtering

    for i in range(len(df)):
        try:
            value = df.iloc[i][array_col]
            shape_value = df.iloc[i][shape_col]

            # Ensure proper conversion
            if isinstance(value, str):
                array_values = np.array(ast.literal_eval(value))  # Convert string to list, then NumPy array
            else:
                array_values = np.array(value)

            original_shape = ast.literal_eval(shape_value) if isinstance(shape_value, str) else shape_value  # Ensure tuple format
            reshaped_array = array_values.reshape(original_shape)  # Reshape to its correct shape
            reshaped_arrays.append(reshaped_array)  # Store the reshaped spectrogram
            valid_indices.append(i)

        except Exception as e:
            print(f"Error processing row {i}: {e}")  # If an error occurs, print the issue

    df = df.iloc[valid_indices].copy()  # Filter out invalid rows (optional, if you want to remove them)
    df[array_col] = reshaped_arrays  # Replace the original column (music_array) with reshaped data

    print("DATA RESHAPED")
    return df

### ------------ Etape 3: Définir les X et y ------------
# Define the X and y, initiate the train test split
def preprocess_data_heavy(df: pd.DataFrame):

    df = df.sample(frac=1) #mélange les données

    X = np.array(df["music_array"].values) #sélectionne le X
    y = np.array(df["is_generated"].values) #sélectionne le y

    X = np.expand_dims(np.stack(X), axis=-1)  ## Ensure correct shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Train test split

    return X_train, X_test, y_train, y_test

### ------------ Etape 4: 1er Modèle CNN léger ------------
# CNN Model

def model_cnn_heavy(input_shape, use_global_pooling=True):
    model = models.Sequential()

    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))


    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    if use_global_pooling:
        model.add(GlobalAveragePooling2D())  #Reduces size by taking the max value in 2x2 regions (apparently good for CNN)
    else:
        model.add(Flatten())

    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

### ------------ Etape 5 : Compile le modèle ------------
def compile_model_cnn_heavy(model: models.Model, learning_rate=0.001):
# Compile the model
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', #For binary choix (0 or 1)
        metrics=['accuracy']
    )
    print("MODEL COMPILED")
    return model

### ------------ Etape 6 : Test le modèle ------------
# Define the function and input parameters
def train_model_cnn_heavy(
        model: models.Model, # The CNN model to be trained
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=256, # Number of samples per batch
        validation_data=None, # Overrides validation_split if yes
        validation_split=0.3, # Percentage of training data for validation
    ):

    if TARGET == TARGET: #checkpoint to save the weight (if local, then local file, if not, then in Google bucket)
        checkpoint_path = LOCAL_PATH_SAVE_WEIGHT
    else:
        checkpoint_path = checkpoint.model.keras

    early_stopping = EarlyStopping(monitor="val_loss", patience = 5, restore_best_weights=True) #stop training if val_loss doesn't improve, but goes anyway until 5 epochs (patience)

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,   # File where model is saved
        monitor="val_loss",         # Save la validation loss (checkpoint based on this, saves the best model)
        save_best_only=True,        # Save if it's the best pr le moment
        mode="min",
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=30, #train the model during 30 epochs
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0.0,
        callbacks=[early_stopping, checkpoint] # Use early stopping and save best model
    )

    if TARGET == 'gcloud': #to save the model in the g bucket if Target = cloud. Needs to be at the end as the model needs to be trained before saving
        upload_to_gcloud_heavy(checkpoint_path, "checkpoint_result", "checkpoint.model.keras")

    print("MODEL TRAINED")

    return model, history.history

### ------------ Etape 6.1 : Évaluer le modèle sur les données de test ------------
#print the result
def evaluate_model_heavy(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"💢💢 Loss : {test_loss:.4%} 💢💢")
    print(f"✅​✅​ Accuracy : {test_acc:.4%}✅​✅​")

    return test_loss, test_acc

### ------------ Step 7: Google Cloud Upload Function ------------
# Checks if the model exists locally before uploading, connects to Google Cloud Storage, uploads the file to the specified cloud bucket, prints confirmation with the file’s GCS path. Not in the main as doesn't run the modle
def upload_to_gcloud_heavy(local_model_path, bucket_name, destination_blob_name):
    """Uploads a model file to Google Cloud Storage."""

    if not os.path.exists(local_model_path):
        print(f"File not found: {local_model_path}")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_model_path)
    print(f"Upload complete - file saved at: gs://{bucket_name}/{destination_blob_name}")


### ------------ Etape 8: Execution ------------
if __name__ == "__main__":
    if TARGET == 'local':  # Fix the comparison operator
        csv_path = LOCAL_PATH_TO_RAW_DATA  # Use the correct variable depending on the environment
    else:
        csv_path = PATH_PROCESSED_DATA
    df = load_data_heavy(csv_path) ## Load the data

    if df is not None:
        df_reshaped = reshape_spectrograms_heavy(df, array_col="music_array", shape_col="shape_arr")

        X_train, X_test, y_train, y_test = preprocess_data_heavy(df)

        model = model_cnn_heavy(X_train.shape[1:])

        model_compiled = compile_model_cnn_heavy(model)

        model_trained, history = train_model_cnn_heavy(model, X_train, y_train, validation_data=(X_test, y_test))

        evaluate_model_heavy(model_trained, X_test, y_test)
    else:
        print("Heavy not working - Retry")
