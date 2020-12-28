import os
import zipfile
import requests


def downloadData(dataset_directory='../input/'):
    """
    Downloads Flicker8k dataset
    :param data_path: Path to download dataset
    :return: None
    """

    urls = [
        'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
    ]

    for url in urls:
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)

        # Check if the dataset has been downloaded, else download and extract it
        file_name = dataset_directory + 'dataset.zip'
        if os.path.isfile(file_name):
            print("{} already downloaded. Skipping download.".format(file_name))
        else:
            print("Downloading '{}' into '{}' file".format(url, file_name))

            data_request = requests.get(url)
            with open(file_name, 'wb') as file:
                file.write(data_request.content)

            print("Extracting {} into {}".format(file_name, dataset_directory))
            if file_name.endswith("zip"):
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(dataset_directory)
            else:
                print("Unknown format.")

    print("Input data setup successful.")
