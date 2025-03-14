import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple
import os
import ast

### ------------ Etape 1: Récuperer le CSV ------------
def load_data_light(csv_path): # Load dataset from a CSV file
    #df = pd.read_csv(f"/home/{os.environ.get('USER_NAME')}/audio_deepfake_detector/processed_data/music_preprocessed.csv")
    df = pd.read_csv(csv_path)
    return df

### ------------ Etape 2: Reshape dataframe ------------
def reshape_spectrograms(df: pd.DataFrame, array_col="music_array", shape_col="shape_arr"):
    reshaped_arrays = []
    valid_indices = []

    for i in range(len(df)):
        try:
            value = df.iloc[i][array_col]
            shape_value = df.iloc[i][shape_col]

            # Ensure proper conversion
            if isinstance(value, str):
                array_values = np.array(ast.literal_eval(value))
            else:
                array_values = np.array(value)

            original_shape = ast.literal_eval(shape_value) if isinstance(shape_value, str) else shape_value
            reshaped_array = array_values.reshape(original_shape)
            reshaped_arrays.append(reshaped_array)
            valid_indices.append(i)

        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # Keep only valid rows
    df = df.iloc[valid_indices].copy()
    df[array_col] = reshaped_arrays
    return df

### ------------ Etape 3: Définir les X et y ------------
def preprocess_data_light(df: pd.DataFrame):
    X = np.array(df["music_array"].values)  ## Preprocess the dataset: extract X and y, reshape X, and split into train/test sets.
    y = np.array(df["is_generated"].values)

    X = np.expand_dims(np.stack(X), axis=-1) ## Ensure correct shape mais ca doit changer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Train test split

    return X_train, X_test, y_train, y_test

### ------------ Etape 4: 1er Modèle CNN léger ------------
def model_cnn_light(input_shape, use_global_pooling=True):
    model = models.Sequential()

    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
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
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

### ------------ Etape 5 : Compile le modèle ------------

def compile_model_cnn_light(model: models.Model, learning_rate=0.0005):
## Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

### ------------ Etape 6 : Test le modèle ------------

def train_model_cnn_light(
        model: models.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=256,
        validation_data=None,  # Overrides validation_split
        validation_split=0.3
    ):

    early_stopping = EarlyStopping(monitor="val_loss", restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0.0,
        callbacks=[early_stopping]
    )

    return model, history.history

### ------------ Etape 6.1 : Évaluer le modèle sur les données de test ------------
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print(f"Loss : {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4%}")

    return test_loss, test_acc


### ------------ Etape 7 : Execution ------------
if __name__ == "__main__":
    csv_path = "/home/nicolast/code/NicoTerli/99-Perso/data_processed_1000 (1).csv"
    df = load_data_light(csv_path) ## Load the data

    #df = load_data_light(f"/home/{os.environ.get('USER_NAME')}/audio_deepfake_detector/processed_data/music_preprocessed.csv") ## Load the data
    if df is not None:

        df_reshaped = reshape_spectrograms(df, array_col="music_array", shape_col="shape_arr")

        X_train, X_test, y_train, y_test = preprocess_data_light(df) ## Preprocess the df & define X & y

        model = model_cnn_light(X_train.shape[1:]) ## Create the model

        model_compiled = compile_model_cnn_light(model) ## Compile the model

        model_trained, history = train_model_cnn_light(model, X_train, y_train, validation_data=(X_test, y_test))  ## Test the model

        evaluate_model(model_trained, X_test, y_test)

    else:
        print("Light not working - Retry")
