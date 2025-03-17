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
from models.model_dp_light import *
from models.model_dp_heavy import *
from google.cloud import storage
from models.models_ml.models_ml import *

if __name__ == "__main__":
    if MODEL == "CNN_heavy":
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

    elif MODEL == "CNN_light":
        if TARGET == 'local':
            csv_path = LOCAL_PATH_TO_RAW_DATA
        else:
            csv_path = PATH_PROCESSED_DATA
        df = load_data_light(csv_path)

        if df is not None:

            df_reshaped = reshape_spectrograms_light(df, array_col="music_array", shape_col="shape_arr")

            X_train, X_test, y_train, y_test = preprocess_data_light(df)

            model = model_cnn_light(X_train.shape[1:])

            model_compiled = compile_model_cnn_light(model)

            model_trained, history = train_model_cnn_light(model, X_train, y_train, validation_data=(X_test, y_test))

            evaluate_model_light(model_trained, X_test, y_test)

        else:
            print("Light not working - Retry")

### -----------------Modèles Machine Learning--------------------
    elif MODEL == "SVCModel_poly":
        if TARGET == 'local':  # Fix the comparison operator
            csv_path = LOCAL_PATH_TO_RAW_DATA_PAU  # Use the correct variable depending on the environment
        else:
            csv_path = PATH_PROCESSED_DATA

        df = load_data(csv_path)

        if df is not None :
            df_reshaped = reshape_spectrograms_light(df, array_col="music_array", shape_col="shape_arr")
            X = df_reshaped["music_array"]
            y = df_reshaped["is_generated"]
            X_flattened = np.array([x.flatten() for x in X.values])
            X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=42)

            model = SVCModel_poly(kernel='poly', degree=3, C=5)
            model_trained = train_models(model, X_train, y_train)
            model_evaluation = test_models(model,X_test, y_test, scoring = 'accuracy')
        else:
            print("SVC Poly - Light not working - Retry")

    elif MODEL == "SVCModel_linear":
        if TARGET == 'local':  # Fix the comparison operator
            csv_path = LOCAL_PATH_TO_RAW_DATA_PAU  # Use the correct variable depending on the environment
        else:
            csv_path = PATH_PROCESSED_DATA
        df = load_data(csv_path)

        if df is not None :
            df_reshaped = reshape_spectrograms_light(df, array_col="music_array", shape_col="shape_arr")
            X = df_reshaped["music_array"]
            y = df_reshaped["is_generated"]
            X_flattened = np.array([x.flatten() for x in X.values])
            X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=42)

            model = SVCModel_linear(kernel='linear', C=0.5)
            model_trained = train_models(model, X_train, y_train)
            model_evaluation = test_models(model,X_test, y_test, scoring = 'accuracy')
        else:
            print("SVC Linear -Light not working - Retry")

    elif MODEL == "KNN_model":
        if TARGET == 'local':  # Fix the comparison operator
            csv_path = LOCAL_PATH_TO_RAW_DATA_PAU  # Use the correct variable depending on the environment
        else:
            csv_path = PATH_PROCESSED_DATA
        df = load_data(csv_path)

        if df is not None :
            df_reshaped = reshape_spectrograms_light(df, array_col="music_array", shape_col="shape_arr")
            X = df_reshaped ["music_array"]
            y = df_reshaped ["is_generated"]
            X_flattened = np.array([x.flatten() for x in X.values])
            X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.3, random_state=42)

            model = KNeighborsClassifier(n_neighbors = 10)
            model_trained = train_models(model, X_train, y_train)
            model_evaluation = test_models(model,X_test, y_test, scoring = 'accuracy')
        else:
            print("KNN -Light not working - Retry")
