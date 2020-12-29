import os
import torch
from skimage import io
from models import Encoder, Generator
from data_loader import test_dataloader
from utils import plot_image_grid, save_images, metrics
from parameters import *


def calculate_metric(channels):
    """
    Calculates the evaluation metrics for the original and reconstructed images
    :param channels: List of channels
    :return: None
    """

    folders = os.walk("../results")
    num_folders = 0

    SSIM = {channel: [] for channel in channels}
    PSNR = {channel: [] for channel in channels}

    next(folders)
    for folder in folders:
        num_folders += 1
        firstImage = io.imread(f"{folder[0]}/original_image.png")
        for channel in channels:
            secondImage = io.imread(f"{folder[0]}/image_{channel}.png")
            image_metrics = metrics(firstImage, secondImage)

            SSIM[channel].append(image_metrics["SSIM"])
            PSNR[channel].append(image_metrics["PSNR"])

    with open("../results/avg_ssim.txt", "w") as file:
        for channel in SSIM.keys():
            file.write(f"SSIM for {channel} Channel : {sum(SSIM[channel]) / len(SSIM[channel])}\n")

    with open("../results/avg_psnr.txt", "w") as file:
        for channel in PSNR.keys():
            file.write(f"PSNR for {channel} Channel : {sum(PSNR[channel]) / len(PSNR[channel])}\n")


def validate_models(channels):
    """
    Validate trained models
    :param channels: List of compressed channels used
    :return: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = test_dataloader()
    test_batch = next(iter(test_loader)).to(device)

    reconstructed_images = {}

    for channel in channels:
        NUM_CHANNELS = channel
        encoder = Encoder(NUM_CHANNELS).to(device)
        generator = Generator(NUM_CHANNELS).to(device)

        encoder.load_state_dict(
            torch.load(f"../models/encoder_{NUM_CHANNELS}.model", map_location=torch.device("cpu"))
        )
        generator.load_state_dict(
            torch.load(f"../models/generator_{NUM_CHANNELS}.model", map_location=torch.device("cpu"))
        )

        encoder.eval()
        generator.eval()

        reconstructed_image = generator(encoder(test_batch))
        reconstructed_images[NUM_CHANNELS] = reconstructed_image

    plot_image_grid(test_batch, reconstructed_images, NUM_IMAGES_GRID)
    save_images(test_batch, reconstructed_images)
    calculate_metric(channels)
