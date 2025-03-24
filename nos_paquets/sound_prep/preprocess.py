import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import IPython.display as ipd
import pandas as pd
from nos_paquets.params import *
from datetime import datetime


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

def process_audio(conf, pathname, offset=0.0, trim_long_data=True):
    """
    Cette fonction lit les fichiers audios et les retourne sous forme de np array
    params:
            - conf : avec sampling_rate, duration et samples définis dans params.py
            - pathname : string
            - offset: par défaut à 0, mais il faut le décaler de 10 sec pour les fichiers FMA
            - trim_long_data : bool, si True, coupe les audios trop longs
    returns:
            - y : np.array (signal audio traité)
    """

    try:
        y, sr = librosa.load(pathname,
                                sr=conf.sampling_rate,
                                duration=conf.duration,
                                offset=offset,
                                mono=True)
    except Exception as e:
        print(f"🔕​ Ignoring {pathname}: {e} 💩​")
        return None

    if 0 < len(y): # Évite une erreur en cas de fichier audio vide
        y, _ = librosa.effects.trim(y) # Supprime les silences au début et à la fin

    # Si l'audio est plus long que la durée cible, on coupe l'excédent.
    # Si l'audio est trop court, on ajoute du padding (remplissage avec des zéros) des deux côtés pour uniformiser la taille.

    if len(y) > conf.samples: # Si l'audio est trop long
        if trim_long_data:
            y = y[0:0+conf.samples] # On garde seulement la partie nécessaire

    elif len(y) < conf.samples: # Si l'audio est trop court, on ajoute du padding
        padding = conf.samples - len(y)    # Nombre d'échantillons à ajouter
        offset = padding // 2 # Ajout équilibré à gauche et à droite
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')

    return y

def read_audio(conf, pathname, trim_long_data=True):
    """
    Cette fonction lit les fichiers audios et les retourne sous forme de np array

    params:
            - conf : avec sampling_rate, duration et samples définis dans params.py
            - pathname : string
            - trim_long_data : bool, si True, coupe les audios trop longs
    returns:
            - y : np.array SI LE FICHIER NE VIENT PAS DE FMA
            - (y1, y2, y3) tuple de np.array si le fichier vient de FMA
    """
    print(f"🎶​🎶​ Lecture de​ {pathname} 🎶​🎶​​")

    if "fma" not in pathname:
        return process_audio(conf, pathname, trim_long_data=trim_long_data)

    else:
        print("🔔​🔔​ audio file from FMA: splitting it in three parts")
        y_1, y_2, y_3 = tuple(process_audio(conf, pathname, offset=i*10.0, trim_long_data=trim_long_data) for i in range(3))
        return y_1, y_2, y_3



### ------------ Etape 3: Conversion en spectrogramme ------------

def audio_to_melspectrogram(conf, audio):

    """
    Cette fonction convertit un signal audio en mel-spectrogramme.

    Params :
        - conf (params.py)
        - audio: signal audio sous forme de np.array

    Returns: spectrogramme sous forme de np.array
    """

    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram) # conversion en dB (plus représentatif de la perception humaine du son
    spectrogram = spectrogram.astype(np.float32) # to save memory
    return spectrogram    # type(spectrogram)=numpy.ndarray

### ------------ Etape 4: Réunion des deux fonctions ------------

def read_as_melspectrogram(conf, pathname, trim_long_data=True):
    # le trim_long_data était par défaut en FALSE / en TRUE dans la function read_audio : j'ai mis TRUE par défaut dans les deux

    try:
        x = read_audio(conf, pathname, trim_long_data)
        print(f"🌷​ file processed: {pathname} 🌷​")

    except:
        print(f"⚠️ file ignored: {pathname} 💩​💩​")

    if x is None:
        print(f"⚠️ Error reading file: {pathname}")
        return None


    if type(x)==tuple: # dans le cas où x est un tuple (x1, x2, X3), cad quand le fichier audio provient de fma
        prep_results_arr_1 = audio_to_melspectrogram(conf, x[0])
        prep_results_arr_2 = audio_to_melspectrogram(conf, x[1])
        prep_results_arr_3 = audio_to_melspectrogram(conf, x[2])

        return (prep_results_arr_1, prep_results_arr_2, prep_results_arr_3)

    else:
        prep_results_arr = audio_to_melspectrogram(conf, x)
        return prep_results_arr # type(spectrogram)=numpy.ndarray

# -> la fonction permet d'obtenir des np array

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

def create_spectrogram_dataframe(conf, pathnames : list, batch_size, trim_long_data=False):

    """cette fonction prend en entrée la liste des paths des fichiers audio et rend un dataframe storé sous format .csv:
    le dataframe contient :
                - l'id de la musique (plusieurs musique ont peut-être la même id, notamment dans le cas de fma),
                - le nom du folder et des sous-folders dans lesquels le son est storé,
                - l'array FLATTENED du spectrogramme,
                - la shape d'origine du spectrogramme pour pouvoir reshaper l'array avant de l'entrer dans les modèles,
                - une dernière colonne "is_generated" : 1 si la musique est générée / 0 si la musique n'est pas générée

    le paramètre batch_size permet de gérer le cas où les fichiers sont trop nombreux
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{PATH_PROCESSED_DATA}/music_processed_{DURATION}_{timestamp}.csv"

    data = [] # liste pour storer temporairement les lignes du df
    df = pd.DataFrame(data, columns=["music_id", "folder_name", "music_array", "shape_arr", "is_generated"])
    df.to_csv(file_name, index=True) # fichier csv vide, avec les headers


    for count, pathname in enumerate(pathnames, start=1):
        print(f"Processign file : {count} / {len(pathnames)}")

        music_id = pathname.split('/')[-1] # Extrait le nom de la musique
        folder_name = "/".join(pathname.split("/")[:-1]) # Extrait le lien / path
        prep_results_arr = read_as_melspectrogram(conf, pathname)  # retourne l'array du spec

        # if prep_results_arr is None:
        #     print(f"⚠️ Skipping file due to error: {pathname}")
        #     continue

        if isinstance(prep_results_arr, tuple): # si c'est un tuple, il faut faire 3 fois :
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


        # when batchsize is reached, we store the list in the df and the df in the csv
        if len(data) >= batch_size:
            print(f"{batch_size} files have been processed, writing the results in the csv.")
            df_batch = pd.DataFrame(data, columns=["music_id", "folder_name", "music_array", "shape_arr", "is_generated"])
            df_batch["music_array"] = df_batch["music_array"].apply(lambda x: x.tolist())
            df_batch.to_csv(file_name, index=True, mode='a', header=False)
            data = []

    if data:
        print("Writing the remaining data to the .csv")
        df_batch = pd.DataFrame(data, columns=["music_id", "folder_name", "music_array", "shape_arr", "is_generated"])
        df_batch["music_array"] = df_batch["music_array"].apply(lambda x: x.tolist())
        df_batch.to_csv(file_name, index=True, mode='a', header=False)

    print('❤️​🩷​💛​💚​💙​ all data converted to df and stored as .csv ❤️​🩷​💛​💚​💙​')

    return pd.read_csv(file_name)
