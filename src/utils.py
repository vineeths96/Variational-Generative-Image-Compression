import os
import datetime
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from parameters import *


def init_weights(m):
    """
    Weight initialization to layers
    :param m: layer
    :return: None
    """

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)


def plot_image(test_batch, reconstructed_image, num_images):
    """
    Plot images during validation
    :param test_batch: Test images set
    :param reconstructed_image: Reconstructed images set
    :param num_images: Number of images to plot
    :return: None
    """

    f, ax = plt.subplots(num_images, 2)

    for i in range(num_images):
        test_image = (test_batch[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
        rec_image = (reconstructed_image[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN

        if num_images == 1:
            ax[0].imshow(test_image)
            ax[1].imshow(rec_image)
        else:
            ax[i, 0].imshow(test_image)
            ax[i, 1].imshow(rec_image)

        f.set_figheight(20)
        f.set_figwidth(20)

    plt.show()


def plot_image_grid(test_batch, reconstructed_images, num_images):
    """
    Plot image grid during validation
    :param test_batch: Test images set
    :param reconstructed_images: Reconstructed images set
    :param num_images: Number of images to plot
    :return: None
    """

    f, ax = plt.subplots(num_images, len(reconstructed_images) + 1)

    ax[0, 0].title.set_text("Original \n Image")
    ax[0, 1].title.set_text("Reconstructed with \n 96% Compression")
    ax[0, 2].title.set_text("Reconstructed with \n 92% Compression")
    ax[0, 3].title.set_text("Reconstructed with \n 84% Compression")
    ax[0, 4].title.set_text("Reconstructed with \n 68% Compression")
    ax[0, 5].title.set_text("Reconstructed with \n 43% Compression")

    for i in range(num_images):
        test_image = (test_batch[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
        ax[i, 0].imshow(test_image)
        for ind, channel in enumerate(reconstructed_images.keys(), 1):
            rec_image = (reconstructed_images[channel][i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
            ax[i, ind].imshow(rec_image)

        f.set_figheight(20)
        f.set_figwidth(20)

    plt.savefig("../results/result.png")
    plt.show()


def save_images(test_batch, reconstructed_images):
    """
    Plot image grid during validation
    :param test_batch: Test images set
    :param reconstructed_images: Reconstructed images set
    :return: None
    """

    PATH = f"../results/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(PATH, exist_ok=True)

    for i in range(len(test_batch)):
        test_image = (test_batch[i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
        plt.imsave(f"{PATH}/original_image.png", test_image.numpy())

        for channel in reconstructed_images.keys():
            rec_image = (reconstructed_images[channel][i].cpu().detach().permute(1, 2, 0) * STD) + MEAN
            plt.imsave(f"{PATH}/image_{channel}.png", rec_image.numpy())


def metrics(firstImage, secondImage):
    """
    Calculate the evaluation metrics for the images
    :param firstImage: First image
    :param secondImage: Second image
    :return: Metrics dictionary
    """

    ssim = structural_similarity(
        firstImage, secondImage, data_range=firstImage.max() - firstImage.min(), multichannel=True
    )
    psnr = peak_signal_noise_ratio(firstImage, secondImage, data_range=firstImage.max() - firstImage.min())

    image_metrics = {"SSIM": ssim, "PSNR": psnr}

    return image_metrics
