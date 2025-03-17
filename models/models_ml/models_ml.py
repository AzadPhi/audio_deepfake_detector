from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import ast
from nos_paquets.sound_prep.params import *

### ------------ Etape 1: Récuperer le CSV ------------
def load_data(csv_path):
# Load dataset from a CSV file
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
# def preprocessed_ml(df: pd.DataFrame):
#     X = np.array(df["music_array"].values)  ## Preprocess the dataset: extract X and y, reshape X, and split into train/test sets.
#     y = np.array(df["is_generated"].values)

#     X = np.expand_dims(np.stack(X), axis=-1) ## Ensure correct shape mais ca doit changer

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Train test split

#     return X_train, X_test, y_train, y_test

### ------------ Etape 4: Modele SVCPoly ------------

def SVCModel_poly(kernel='poly', degree=3, C=5):
    model_poly = SVC(kernel=kernel, degree=degree, C=C)
    return model_poly

### ------------ Etape 4 bis: Modele SVCLinear ------------
def SVCModel_linear(kernel='linear', C=0.5):
     model_linear = SVC(kernel=kernel, C=C)
     return model_linear

### ------------ Etape 4 ter: Modele KNN ------------

def KNN_model(n_neighbors = 10):
    model_KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

### ------------ Etape 5: Train models------------

def train_models(model, X_train, y_train):
    model.fit(X_train,y_train)
    return model

### ------------ Etape 5: Test model ------------

def test_models(model,X_test, y_test, scoring = 'accuracy'):
    score = model.score(X_test, y_test)
    return score
