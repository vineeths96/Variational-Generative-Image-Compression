import torch
import torch.nn as nn
from datetime import datetime as datetime
from models import Encoder, Generator, Discriminator
from data_loader import train_dataloader, test_dataloader
from utils import init_weights, plot_image
from parameters import *


def models(channels):
    """
    Creates and initializes the models
    :return: Encoder, Generator, Discriminator
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(channels).to(device)
    encoder.apply(init_weights)

    generator = Generator(channels).to(device)
    generator.apply(init_weights)

    discriminator = Discriminator(channels).to(device)
    discriminator.apply(init_weights)

    return encoder, generator, discriminator


def train(channels):
    """
    Trains the VAE-GAN model
    :return: Encoder, Generator
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, generator, discriminator = models(channels)

    # Setup the dataloaders
    train_loader = train_dataloader()
    test_loader = test_dataloader()
    test_batch = next(iter(test_loader)).to(device)

    # Setup the loss functions
    bce_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()

    # Setup Optimizers
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Loss tracker
    losses = {model: [] for model in ["encoder", "generator", "discriminator"]}

    # Training Loop
    print(f"Starting Training at {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    for epoch in range(NUM_EPOCHS):
        for i, images in enumerate(train_loader):
            # Set models to train mode
            encoder.train()
            generator.train()
            discriminator.train()

            # Create real images and fake (reconstructed) images
            images = images.to(device)
            fake_images = generator(encoder(images))

            z_real = {"image": images, "encoded_image": encoder(images)}
            z_fake = {"image": fake_images, "encoded_image": encoder(images)}

            ##############################
            ###  Disriminator training ###
            ##############################
            discriminator.zero_grad()

            # Real batch with label smoothing
            label = torch.empty(images.size(0), device=device).uniform_(1 - SMOOTH, 1)
            output = discriminator(z_fake).view(-1)
            discriminator_loss_real = bce_criterion(output, label)

            # Fake batch with label smoothing
            label = torch.empty(images.size(0), device=device).uniform_(0, SMOOTH)
            output = discriminator(z_fake).view(-1)
            discriminator_loss_fake = bce_criterion(output, label)

            # Update weights
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            ##############################
            ###   Generator training   ###
            ##############################
            generator.zero_grad()

            # Fake batch with label smoothing
            label = torch.empty(images.size(0), device=device).uniform_(1 - SMOOTH, 1)
            output = discriminator(z_fake).view(-1)

            # Update gradiente
            generator_loss = bce_criterion(output, label) + 2 * l1_criterion(images, fake_images)
            generator_loss.backward(retain_graph=True)

            ##############################
            ###    Encoder training    ###
            ##############################
            encoder.zero_grad()

            # Fake batch with label smoothing
            label = torch.empty(images.size(0), device=device).uniform_(1 - SMOOTH, 1)
            output = discriminator(z_fake).view(-1)

            # Update gradiente
            encoder_loss = bce_criterion(output, label) + 2 * l1_criterion(images, fake_images)
            encoder_loss.backward(retain_graph=True)

            # Update weights
            generator_optimizer.step()
            encoder_optimizer.step()

            ##############################
            ###   Training Statistics  ###
            ##############################
            if i % LOG_FREQUENCY == 0:
                print(
                    f"Epoch: {epoch}, Iteration: {i}, Discriminator Loss: {discriminator_loss.item()}, "
                    f"Generator Loss: {generator_loss.item()}, Encoder Loss: {encoder_loss.item()}"
                )

            # Track losses
            losses["encoder"].append(encoder_loss.item())
            losses["generator"].append(generator_loss.item())
            losses["discriminator"].append(discriminator_loss.item())

            # Plot the original and reconstructed image for test dataset
            if i % VAL_FREQUENCY == 0:
                encoder.eval()
                generator.eval()

                encoded_image = encoder(test_batch)
                reconstructed_image = generator(encoded_image)
                plot_image(test_batch, reconstructed_image, NUM_IMAGES)

    print(f"Completed Training at {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")

    return encoder, generator
