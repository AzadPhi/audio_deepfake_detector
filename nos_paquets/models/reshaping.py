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
from google.cloud import storage

from nos_paquets.sound_prep.params import *
# from models.model_dp_light import *
# from models.model_dp_heavy import *
from models.models_ml.models_ml import *

def load_data(csv_path):
    print(type(csv_path))
    df = pd.read_csv(csv_path)
    print("🔥​🔥​ DATA LOADED 🔥​🔥​")
    return df

def reshape_spectrograms(df: pd.DataFrame, array_col="music_array", shape_col="shape_arr"):
# Transform the music array value into a Tuple so that it can be read by the Model
    reshaped_arrays = [] #to store reshaped spectrograms
    valid_indices = []

    for i in range(len(df)):
        try:
            value = df.iloc[i][array_col]
            shape_value = df.iloc[i][shape_col]

            # Ensure proper conversion
            if isinstance(value, str):
                array_values = np.array(ast.literal_eval(value)) #Read the spectrogram from music_array, converts the string into a Python list, transforms it into a NumPy array
            else:
                array_values = np.array(value)

            original_shape = ast.literal_eval(shape_value) if isinstance(shape_value, str) else shape_value # Read the original shape of the spectrogram and convert to a Tuple
            reshaped_array = array_values.reshape(original_shape) # Reshape the spectrogram to its correct shape
            reshaped_arrays.append(reshaped_array) # Store the reshaped spectrogram in a list
            valid_indices.append(i)

        except Exception as e:
            print(f"Error processing row {i}: {e}") # If an error occurs, print the issue

    # Keep only valid rows
    df = df.iloc[valid_indices].copy()
    df[array_col] = reshaped_arrays #Replace the original column (music_array) with reshaped data

    print(reshaped_arrays[0].shape)
    print(type(df[array_col][0])) # ICI C'EST BIEN UN NP ARRAY

    print("✅​✅​ DATA RESHAPED ✅​✅​")
    return df

def preprocess_data(df: pd.DataFrame):

    df = reshape_spectrograms(df, array_col="music_array", shape_col="shape_arr")

    df = df.sample(frac=1) #mélange les données

    print(type(df["music_array"][0])) # ICI C'EST UNE STRING
    print(df["music_array"][0])

    X = np.array(df["music_array"].values) #sélectionne le X
    y = np.array(df["is_generated"].values) #sélectionne le y

    X = np.expand_dims(np.stack(X), axis=-1) ## Ensure correct shape mais ca doit changer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Train test split

    return X_train, X_test, y_train, y_test
