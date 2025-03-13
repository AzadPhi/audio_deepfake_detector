import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple

### ------------ Etape 1: Récuperer le CSV ------------
def load_data_light(csv_path): # Load dataset from a CSV file
    df = pd.read_csv(csv_path)
    return df

### ------------ Etape 2: Définir les X et y ------------
def preprocess_data_light(df: pd.DataFrame):
    X = np.array(df["music_array"].tolist())  ## Preprocess the dataset: extract X and y, reshape X, and split into train/test sets.
    y = np.array(df["is_generated"])

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))  ## Ensure correct shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Train test split

    return X_train, X_test, y_train, y_test

### ------------ Etape 3: 1er Modèle CNN léger ------------
def initialize_model_light(input_shape: tuple):

    model = models.Sequential() ### Initialize the model light

    model.add(layers.Conv2D(16, (3,1), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,1)))

    model.add(layers.Conv2D(32, (3,1), activation='relu'))
    model.add(layers.MaxPooling2D((2,1)))

    model.add(layers.Conv2D(64, (3,1), activation='relu'))
    model.add(layers.MaxPooling2D((2,1)))

    model.add(layers.Conv2D(128, (3,1), activation='relu'))
    model.add(layers.MaxPooling2D((2,1)))

    model.add(layers.Conv2D(256, (3,1), activation='relu'))
    model.add(layers.MaxPooling2D((2,1)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    return model

### ------------ Etape 5 : Compile le modèle ------------

def compile_model_light(model: models.Model, learning_rate=0.0005):
## Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

### ------------ Etape 6 : Test le modèle ------------

def train_model_light(
        model: models.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=256,
        validation_data=None,  # Overrides validation_split
        validation_split=0.3
    ) -> Tuple[models.Model, dict]:

    early_stopping = EarlyStopping(monitor="val_loss", restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split if validation_data is None else 0.0,
        callbacks=[early_stopping]
    )

    return model, history.history

### ------------ Etape 7 : Execution ------------
if __name__ == "__main__":
    df = load_data_light("XXX.csv") ## Load the data

    X_train, X_test, y_train, y_test = preprocess_data_light(df) ## Preprocess the df & define X & y

    model = initialize_model_light(X_train.shape[1:]) ## Create the model

    model = compile_model_light(model) ## Compile the model

    model, history = train_model_light(model, X_train, y_train, validation_data=(X_test, y_test)) ## Test the model
