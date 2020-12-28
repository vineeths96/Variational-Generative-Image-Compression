import os
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from parameters import *


class ImageTrainData(Dataset):
    """
    PyTorch Dataset for train data
    """

    def __init__(self, image_dir, image_list, train_split):
        """
        Train Dataset initialization
        :param image_dir: Directory path to image dataset
        :param image_list: List of images in the dataset
        :param train_split: Train Test split ratio
        """

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.CenterCrop((HEIGHT, WIDTH)),
                                             transforms.Normalize(MEAN, STD),])
        self.image_dir = image_dir
        self.image_list = image_list
        self.train_split = train_split

    def __len__(self):
        return int(len(self.image_list) * self.train_split)

    def __getitem__(self, index):
        image = io.imread(self.image_dir + self.image_list[index])
        image = self.transform(image)

        return image


class ImageTestData(Dataset):
    """
    PyTorch Dataset for test data
    """

    def __init__(self, image_dir, image_list, train_split):
        """
        Test Dataset initialization
        :param image_dir: Directory path to image dataset
        :param image_list: List of images in the dataset
        :param train_split: Train Test split ratio
        """

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.CenterCrop((HEIGHT, WIDTH)),
                                             transforms.Normalize(MEAN, STD),])
        self.image_dir = image_dir
        self.image_list = image_list
        self.train_split = train_split

    def __len__(self):
        return int(len(self.image_list) * (1 - self.train_split))

    def __getitem__(self, index):
        image = io.imread(self.image_dir + self.image_list[-index])
        image = self.transform(image)

        return image


def train_dataloader():
    """
    Train dataloader with custom train dataset
    :return: Train Dataloader
    """

    image_list = os.listdir(IMG_DIR)
    train_dataset = ImageTrainData(IMG_DIR, image_list, SPLIT)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader


def test_dataloader():
    """
    Test dataloader with custom test dataset
    :return: Tes Dataloader
    """

    image_list = os.listdir(IMG_DIR)
    test_dataset = ImageTrainData(IMG_DIR, image_list, SPLIT)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return test_dataloader
