import os
import zipfile
import requests
from path import Path


def downloadData(dataset_directory='../input/'):
    """
    Downloads Google Speech Commands dataset (version0.01)
    :param data_path: Path to download dataset
    :return: None
    """


    urls = [
        'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
    ]

    for url in urls:
        # Check if we need to extract the dataset
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)

        # Check if the dataset has been downloaded, else download it
        file_name = dataset_directory + 'dataset.zip'
        if os.path.isfile(file_name):
            print("{} already downloaded. Skipping download.".format(file_name))
        else:
            print("Downloading '{}' into '{}' file".format(url, file_name))

            data_request = requests.get(url)
            with open(file_name, 'wb') as file:
                file.write(data_request.content)

            # Extract downloaded file
            print("Extracting {} into {}".format(file_name, dataset_directory))

            if file_name.endswith("zip"):
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(dataset_directory)
            else:
                print("Unknown format.")

    print("Input data setup successful.")
