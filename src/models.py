import torch
import torch.nn as nn
from parameters import *


class Encoder(nn.Module):
    """
    Encoder model
    """

    def __init__(self, num_channels=NUM_CHANNELS):
        super(Encoder, self).__init__()

        self.num_channels = num_channels
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=num_channels, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.Tanh()
        )

    def forward(self, x):
        ec_1 = self.e_conv_1(x)
        ec_2 = self.e_conv_2(ec_1)
        eblock_1 = self.e_block_1(ec_2) + ec_2
        eblock_2 = self.e_block_2(eblock_1) + eblock_1
        eblock_3 = self.e_block_3(eblock_2) + eblock_2
        ec_3 = self.e_conv_3(eblock_3)

        return ec_3


class Generator(nn.Module):
    """
    Generator - Decoder Model
    """

    def __init__(self, num_channels=NUM_CHANNELS):
        super(Generator, self).__init__()

        self.num_channels = num_channels
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        uc_1 = self.d_up_conv_1(x)
        dblock_1 = self.d_block_1(uc_1) + uc_1
        dblock_2 = self.d_block_2(dblock_1) + dblock_1
        dblock_3 = self.d_block_3(dblock_2) + dblock_2
        uc_2 = self.d_up_conv_2(dblock_3)
        dec = self.d_up_conv_3(uc_2)

        return dec


class Discriminator(nn.Module):
    """
    Discriminator Model
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis_upconv_1 = nn.Sequential(
            nn.ConvTranspose2d(NUM_CHANNELS, 12, (3, 3), stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.dis_upconv_2 = nn.Sequential(
            nn.ConvTranspose2d(12, 16, (3, 3), stride=1, padding=2, output_padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.dis_upconv_3 = nn.Sequential(
            nn.ConvTranspose2d(16, 24, (3, 3), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.dis_upconv_4 = nn.Sequential(
            nn.ConvTranspose2d(24, 36, (5, 5), stride=2, padding=0, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.dis_upconv_5 = nn.Sequential(
            nn.ConvTranspose2d(36, 3, (3, 3), stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Tanh(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Tanh(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 54 * 44, 2000),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2000, 100),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        y = z['encoded_image']
        y = self.dis_upconv_1(y)
        y = self.dis_upconv_2(y)
        y = self.dis_upconv_3(y)
        y = self.dis_upconv_4(y)
        y = self.dis_upconv_5(y)

        x = z['image']
        x = torch.cat((x, y), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
