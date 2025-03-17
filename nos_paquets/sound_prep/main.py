from nos_paquets.sound_prep.params import *
from nos_paquets.sound_prep.preprocess import *
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
import concurrent.futures
import os


def download_data_from_cloud(bucket_name, raw_data_directory, data_size):
    """this function download the data stored on a bucket on gcloud, it stores is in the chosen directory
    if data_size=all, it downloads all the data"""

    # initialize client
    client = Client()

    bucket = client.bucket(bucket_name)

    # get the list of blob names
    if data_size=="all":
        blob_names = [blob.name for blob in bucket.list_blobs()]
        # when no argument max_results is passed, it takes all the blobs
        print(f"✨✨​​ Downloading {len(blob_names)} files from Gcloud ✨​✨​")

    else:
        blob_names = [blob.name for blob in bucket.list_blobs(max_results=int(data_size))]
        print(f"✨✨​​ Downloading {len(blob_names)} files from Gcloud ✨​✨​")

    # starts downloading
    results = transfer_manager.download_many_to_path(bucket,
                                                     blob_names,
                                                     destination_directory=raw_data_directory)

    print("❤️​🩷​💛​💚​💙​ The data has been downloaded! ❤️​🩷​💛​💚​💙​")


def get_paths_as_list(raw_data_directory):
    """
    cette fonction permet de récupérer une liste contenant tous les noms des fichiers audio sous forme de liste de strings
    """
    files_path = []
    for dirpath, _, filenames in os.walk(raw_data_directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.getsize(file_path) >= 400000:
                files_path.append(file_path)

    wanted_files = (".mp3", ".wav")
    files_path = [p for p in files_path if p.lower().endswith(wanted_files)] # pour exclure les fichiers non audio

    return files_path

def upload_data_processed_on_gcloud(bucket_processed_data,
                                    csv_path):
    """cette fonction upload la data processed au format .csv dans un bucket sur gcloud:
    bucket_processed_data: nom du bucket de destination
    csv_path: chemin du fichier .csv à uploader
    blob_name: nom du blob dans le bucket
    """

    storage_client = Client()
    bucket = storage_client.bucket(bucket_processed_data)
    blob_name = csv_path.strip('.csv').split('/')[-1]
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(csv_path)

    print("❤️​🩷​💛​💚​💙​ The data processed has been uploaded on the Gcloud bucket! ❤️​🩷​💛​💚​💙​")


if __name__ == '__main__':

    # download_data_from_cloud(bucket_name=BUCKET_NAME_RAW_DATA,
    #                          raw_data_directory=PATH_TO_RAW_DATA,
    #                          data_size=DATA_SIZE)

    paths_to_audio_files = get_paths_as_list(PATH_TO_RAW_DATA)

    print("🎉​ First step done: now we will convert the audiofiles into mel-spectrogram 🤓​")

    df = create_spectrogram_dataframe(conf,
                                      paths_to_audio_files,
                                      batch_size=250,
                                      trim_long_data=False)

    print("🎉​ Second step done: now we will store the results into a csv 🤓​​")

    # create_csv(df)

    print("🚀​ And one last thing: we need to store the csv on gcloud 😎​")

    upload_data_processed_on_gcloud(bucket_processed_data=BUCKET_PROCESSED_DATA,
                                    csv_path=PATH_PROCESSED_DATA)

    print("🎉​ THE END OF PREPROCESSING 🏁​🏁​🏁​🏁​​​")
