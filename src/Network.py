import torch
import torchvision

"""
This script has the network architecture for the autoencoder.
"""

class ConvBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, filter_shape):
        super(ConvBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, out_dim, filter_shape, padding='same'),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.ReLU())

    def forward(self, x):
        return self.block(x)


class AttentionBlock(torch.nn.Module):

    def __init__(self, embed_dim, heads=8, dropout=0.1):
        """
        :param embed_dim: Embedding dimension.
        :param heads: Number of attention heads.
        :param dropout: Dropout probability.
        """
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.mha = torch.nn.MultiheadAttention(
            embed_dim, heads, batch_first=True, dropout=dropout)
        self.batch_norm = torch.nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        """
        Forward pass of the attention block.
        :param x: Input tensor.
        """
        x_shape = x.shape
        y = self.batch_norm(x)
        y = y.reshape(-1, self.embed_dim,  x_shape[2] * x_shape[3])
        y = torch.transpose(y, 1, 2)
        y, _ = self.mha(y, y, y)
        y = torch.transpose(y, 1, 2)
        y = y.reshape(-1, self.embed_dim, x_shape[2], x_shape[3])
        return x + y


class ResNetBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, filter_shape, dropout_rate=0.1,
                 use_attention_out=False, use_attention_in=False):
        """
        :param in_dim: Input dimension.
        :param out_dim: Output dimension.
        :param filter_shape: Filter shape.
        :param dropout_rate: Dropout rate.
        :param use_attention_out: Use attention in the output.
        :param use_attention_in: Use attention in the input.
        """
        super(ResNetBlock, self).__init__()
        self.conv_1 = ConvBlock(in_dim, out_dim, filter_shape)
        self.conv_2 = ConvBlock(out_dim, out_dim, filter_shape)

        self.dropout = torch.nn.Dropout2d(dropout_rate)

        if use_attention_in:
            self.attention_in = AttentionBlock(in_dim)
        else:
            self.attention_in = torch.nn.Identity()

        if use_attention_out:
            self.attention_out = AttentionBlock(out_dim)
        else:
            self.attention_out = torch.nn.Identity()

        if in_dim != out_dim:
            self.conv_res = torch.nn.Conv2d(in_dim, out_dim, 1,
                                            padding='same')
        else:
            self.conv_res = torch.nn.Identity()

    def forward(self, x):
        y = self.attention_in(x)
        y = self.conv_1(y)
        y = self.dropout(y)
        y = self.conv_2(y)
        y = self.attention_out(y)
        return y + self.conv_res(x)


class AutoEncoder(torch.nn.Module):

    def __init__(self, channels, strides, linear_dims, latent_dim):
        super(AutoEncoder, self).__init__()
        encode_block_1 = ResNetBlock(channels[0], channels[1], 3)
        encode_block_2 = ResNetBlock(channels[1], channels[2], 3)
        encode_block_3 = ResNetBlock(channels[2], channels[3], 3)
        encode_block_4 = ResNetBlock(channels[3], channels[4], 3,
                                     use_attention_out=True)
        encoder_linear_1 = LinearBlock(linear_dims[0], linear_dims[1])
        encoder_linear_2 = LinearBlock(linear_dims[1], linear_dims[2])
        encoder_latent_layer = torch.nn.Linear(linear_dims[2], latent_dim)

        encoder_pool_1 = torch.nn.Conv2d(channels[1], channels[1], 3,
                                         strides[0], padding=strides[0]//2)
        encoder_pool_2 = torch.nn.Conv2d(channels[2], channels[2], 3,
                                         strides[1], padding=strides[1]//2)
        encoder_pool_3 = torch.nn.Conv2d(channels[3], channels[3], 3,
                                         strides[2], padding=strides[2]//2)
        encoder_pool_4 = torch.nn.Conv2d(channels[4], channels[4], 3,
                                         strides[3], padding=strides[3]//2)

        decoder_latent_layer = torch.nn.Linear(latent_dim, linear_dims[2])
        decoder_linear_1 = LinearBlock(linear_dims[2], linear_dims[1])
        decoder_linear_2 = LinearBlock(linear_dims[1], linear_dims[0])

        decoder_block_1 = ResNetBlock(channels[4], channels[3], 3,
                                      use_attention_in=True)
        decoder_block_2 = ResNetBlock(channels[3], channels[2], 3)
        decoder_block_3 = ResNetBlock(channels[2], channels[1], 3)
        decoder_block_4 = ResNetBlock(channels[1], channels[0], 3)

        decoder_unpool_1 = torch.nn.ConvTranspose2d(
            channels[4], channels[4], 3, strides[3],
            padding=strides[3]//2, output_padding=1)
        decoder_unpool_2 = torch.nn.ConvTranspose2d(
            channels[3], channels[3], 3, strides[2],
            padding=strides[2]//2, output_padding=1)
        decoder_unpool_3 = torch.nn.ConvTranspose2d(
            channels[2], channels[2], 3, strides[1],
            padding=strides[1]//2, output_padding=1)
        decoder_unpool_4 = torch.nn.ConvTranspose2d(
            channels[1], channels[1], 3, strides[0],
            padding=strides[0]//2, output_padding=1)

        self.encoder_block = torch.nn.Sequential(
            encode_block_1, encoder_pool_1,
            encode_block_2, encoder_pool_2,
            encode_block_3, encoder_pool_3,
            encode_block_4, encoder_pool_4,
            torch.nn.Flatten(),
            encoder_linear_1, encoder_linear_2,
            encoder_latent_layer)

        self.decoder_block = torch.nn.Sequential(
            decoder_latent_layer, decoder_linear_1,
            decoder_linear_2, torch.nn.Unflatten(1, (channels[4], 8, 8)),
            decoder_unpool_1, decoder_block_1,
            decoder_unpool_2, decoder_block_2,
            decoder_unpool_3, decoder_block_3,
            decoder_unpool_4, decoder_block_4)

    def encoder(self, x):
        return self.encoder_block(x)

    def decoder(self, x):
        return self.decoder_block(x)

    def forward(self, x):
        return self.encoder(x)


class LinearBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GroupNorm(4, out_dim),
            torch.nn.LeakyReLU())

    def forward(self, x):
        return self.block(x)


class FullyConnected(torch.nn.Module):

    def __init__(self, in_dim=64, out_dim=32):
        super(FullyConnected, self).__init__()
        self.fc = torch.nn.Sequential(
            LinearBlock(in_dim, 128),
            LinearBlock(128, 128),
            LinearBlock(128, 128),
            LinearBlock(128, 128),
            torch.nn.Linear(128, out_dim))

    def forward(self, x):
        return self.fc(x)
