from nos_paquets.sound_prep.params import *
from nos_paquets.sound_prep.preprocess import *
from pathlib import Path
from google.cloud.storage import Client, transfer_manager


def download_data_from_cloud(bucket_name, path_to_raw_data, data_size):
    """this function download the data stored on a bucket on gcloud, it stores is in the chosen directory
    if data_size=all, it downloads all the data"""

    client = Client()
    workers=2


    bucket = client.bucket(bucket_name)

    if data_size=="tout":
        blob_names = [blob.name for blob in bucket.list_blobs()]
        # when no argument max_results is passed, it takes all the blobs

    else:
        blob_names = [blob.name for blob in bucket.list_blobs(max_results=int(data_size))]

    results = transfer_manager.download_many_to_path(bucket,
                                                     blob_names,
                                                     destination_directory=path_to_raw_data,
                                                     max_workers=workers)

    print("❤️​🩷​💛​💚​💙​ The data has been downloaded! ❤️​🩷​💛​💚​💙​")

    directory = Path(PATH_TO_RAW_DATA)
    files_path = [p for p in directory.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    string_paths = [str(path) for path in files_path]

    return string_paths

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
    string_paths = download_data_from_cloud(bucket_name=BUCKET_NAME_RAW_DATA,
                                            path_to_raw_data=PATH_TO_RAW_DATA,
                                            data_size=DATA_SIZE)

    df = create_spectrogram_dataframe(conf, string_paths, trim_long_data=False)
    create_csv(df)
    upload_data_processed_on_gcloud(bucket_processed_data=BUCKET_PROCESSED_DATA,
                                    csv_path=PATH_PROCESSED_DATA)
