import os
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from parameters import *


class ImageTrainData(Dataset):
    def __init__(self, image_dir, image_list, train_split):
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
    def __init__(self, image_dir, image_list, train_split):
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
    image_list = os.listdir(IMG_DIR)
    train_dataset = ImageTrainData(IMG_DIR, image_list, SPLIT)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader


def test_dataloader():
    batch_size = BATCH_SIZR
    image_list = os.listdir(IMG_DIR)
    test_dataset = ImageTrainData(IMG_DIR, image_list, SPLIT)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_dataloader
