
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
import pandas as pd
from nos_paquets.sound_prep.params import *

### ------------ Etape 1: Definition des paramètres ------------

class Conf:
    sampling_rate = SAMPLING_RATE  # Fréquence d'échantillonnage (Hz) # on l'a fixé à 16000
    duration = DURATION  # Durée cible en secondes
    n_mels = N_MELS  # Nombre de bandes de Mel

    hop_length = HOP_LENGTH  # Détermine le nombre de frames temporelles
    fmin = FMIN  # Fréquence minimale
    fmax = FMAX  # Fréquence maximale (Nyquist)
    n_fft = N_FFT  # Taille de la FFT
    samples = SAMPLES  # Nombre total d'échantillons

conf = Conf()

### ------------ Etape 2: Première fonction pour Lecture et nettoyage de l'audio ------------

def read_audio(conf, pathname, trim_long_data=True):

    print(f"🍒​🍒​🍒​ {pathname} 🍒​🍒​🍒​")

    if "fma" not in pathname:
        y, sr = librosa.load(pathname,
                             sr=conf.sampling_rate,
                             duration=conf.duration,
                             mono=True)
        # Charge le fichier audio pathname avec une fréquence d'échantillonnage définie dans conf.sampling_rate (44100 Hz dans ce cas)
        # y est le signal audio sous forme d'un tableau NumPy
        # sr est la fréquence d'échantillonnage

        if 0 < len(y): # Évite une erreur en cas de fichier audio vide
            y, _ = librosa.effects.trim(y) # Supprime les silences au début et à la fin
        # La fonction librosa.effects.trim(y) supprime les parties silencieuses en début et fin d'audio

        # Si l'audio est plus long que la durée cible, on coupe l'excédent.
        # Si l'audio est trop court, on ajoute du padding (remplissage avec des zéros) des deux côtés pour uniformiser la taille.

        if len(y) > conf.samples: # Si l'audio est trop long
            if trim_long_data:
                y = y[0:0+conf.samples] # On garde seulement la partie nécessaire

        else: # Si l'audio est trop court, on ajoute du padding
            padding = conf.samples - len(y)    # Nombre d'échantillons à ajouter
            offset = padding // 2 # Ajout équilibré à gauche et à droite
            y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

        return y

    else: ## quand l'extrait provient de FMA, il fait 30sec : on le split en 3x10sec y_1, y_2, y_3
        print("🍒​1 split🍒​")
        y_1, sr_1 = librosa.load(pathname,
                                 sr=conf.sampling_rate,
                                 duration=conf.duration,
                                 mono=True,
                                 offset=0.0)
        if 0 < len(y_1): # Évite une erreur en cas de fichier audio vide
            y_1, _ = librosa.effects.trim(y_1)

        if len(y_1) > conf.samples: # Si l'audio est trop long
            if trim_long_data:
                y_1 = y_1[0:0+conf.samples]

        else: # Si l'audio est trop court, on ajoute du padding
            padding = conf.samples - len(y_1)
            offset = padding // 2
            y_1 = np.pad(y_1, (offset, conf.samples - len(y_1) - offset), 'constant')

        print("🍒​2 split🍒​")
        y_2, sr_1 = librosa.load(pathname,
                                 sr=conf.sampling_rate,
                                 duration=conf.duration,
                                 mono=True,
                                 offset=10.0)
        if 0 < len(y_2): # Évite une erreur en cas de fichier audio vide
            y_2, _ = librosa.effects.trim(y_2)

        if len(y_2) > conf.samples: # Si l'audio est trop long
            if trim_long_data:
                y_2 = y_2[0:0+conf.samples]

        else: # Si l'audio est trop court, on ajoute du padding
            padding = conf.samples - len(y_2)
            offset = padding // 2
            y_2 = np.pad(y_2, (offset, conf.samples - len(y_2) - offset), 'constant')

        print("🍒​3 split🍒​")
        y_3, sr_1 = librosa.load(pathname,
                                 sr=conf.sampling_rate,
                                 duration=conf.duration,
                                 mono=True,
                                 offset=20.0)
        if 0 < len(y_3): # Évite une erreur en cas de fichier audio vide
            y_3, _ = librosa.effects.trim(y_3)

        if len(y_3) > conf.samples: # Si l'audio est trop long
            if trim_long_data:
                y_3 = y_3[0:0+conf.samples]

        else: # Si l'audio est trop court, on ajoute du padding
            padding = conf.samples - len(y_3)
            offset = padding // 2
            y_3 = np.pad(y_3, (offset, conf.samples - len(y_3) - offset), 'constant')

        return (y_1, y_2, y_3)


### ------------ Etape 3: Conversion en spectrogramme ------------

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram    # type(spectrogram)=numpy.ndarray

    # Cette fonction génère un mel-spectrogram (une version transformée du spectrogramme classique) dans un premier temps
    # Puis conversion en dB (plus représentatif de la perception humaine du son)
    # Enfin, conversion en float32 pour optimiser la mémoire et la compatibilité

### ------------ Etape 4: Réunion des deux fonctions ------------

def read_as_melspectrogram(conf, pathname, trim_long_data=False):

    try:
        x = read_audio(conf, pathname, trim_long_data)
        print(f"🌷​ file processed: {pathname} 🌷​")

    except:
        print(f"⚠️ file ignored: {pathname} 💩​💩​")


    if type(x)==tuple: # dans le cas où x est un tuple (x1, x2, X3), cad quand le fichier audio provient de fma
        prep_results_arr_1 = audio_to_melspectrogram(conf, x[0])
        prep_results_arr_2 = audio_to_melspectrogram(conf, x[1])
        prep_results_arr_3 = audio_to_melspectrogram(conf, x[2])

        return (prep_results_arr_1, prep_results_arr_2, prep_results_arr_3)

    else:
        prep_results_arr = audio_to_melspectrogram(conf, x)
        return prep_results_arr # type(spectrogram)=numpy.ndarray

# -> la fonction permet d'obtenir des np array
# exemple array_test = read_as_melspectrogram(conf,music_1_path,trim_long_data=False)

### ------------ Etape 5 (OPTIONNEL): Fonction pour plotter notre résultat ressorti par l'étape 4 ------------

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

# -> la fonction permet d'obtenir des plots
# exemple array_test = plot_spectrogram(array_test,sr=conf.sampling_rate,hop_length=conf.hop_length,y_axis="log")

### ------------ Etape 6 : Convertit le np array et l'ajoute à un Dataframe ------------

def create_spectrogram_dataframe(conf, pathnames : list, trim_long_data=False):

    """cette fonction prend en entrée la liste des paths des fichiers audio et rend un dataframe:
    le dataframe contient :
                - l'id de la musique (plusieurs musique ont peut-être la même id, notamment dans le cas de fma),
                - le nom du folder et des sous-folders dans lesquels le son est storé,
                - l'array FLATTENED du spectrogramme,
                - la shape d'origine du spectrogramme pour pouvoir reshaper l'array avant de l'entrer dans les modèles,
                - une dernière colonne "is_generated" : 1 si la musique est générée / 0 si la musique n'est pas générée
    """

    data = []

    count = 0
    for pathname in pathnames:

        print(f"number of files processed: {count}")
        count += 1

        music_id = pathname.split('/')[-1] # Extrait le nom de la musique
        folder_name = "/".join(pathname.split("/")[:-1]) # Extrait le lien / path
        prep_results_arr = read_as_melspectrogram(conf, pathname)  # retourne l'array du spec

        if type(prep_results_arr)==tuple: # si c'est un tuple, il faut faire 3 fois :
            for i in range(3):
                arr_shape = prep_results_arr[i].shape
                array_flatten = prep_results_arr[i].flatten()
                if "fake" in folder_name.lower():
                    is_generated=1
                else:
                    is_generated=0
                data.append([music_id, folder_name, array_flatten, arr_shape, is_generated])

        else:
            arr_shape = prep_results_arr.shape # pour garder la shape de l'array
            array_flatten = prep_results_arr.flatten() # on flatten l'array

            if "fake" in folder_name.lower():
                is_generated=1

            else:
                is_generated=0

            data.append([music_id, folder_name, array_flatten, arr_shape, is_generated])


    df = pd.DataFrame(data, columns=["music_id", "folder_name", "music_array", "shape_arr", "is_generated"])

    df["music_array"] = df["music_array"].apply(lambda x: x.tolist())

    print('❤️​🩷​💛​💚​💙​ all data converted to df ❤️​🩷​💛​💚​💙​')

    return df

def create_csv(df):
    df.to_csv(PATH_PROCESSED_DATA,
              index=True)
    print('❤️​🩷​💛​💚​💙 all data saved as csv ❤️​🩷​💛​💚​💙​')
