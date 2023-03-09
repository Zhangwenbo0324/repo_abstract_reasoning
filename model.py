import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ARII(nn.Module):

    def __init__(self, image_size=160):

        super(ARII, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(1, 16, kernel_size=3, stride=2, padding=1),
            ConvLayer(16, 16, kernel_size=3, padding=1),
            ConvLayer(16, 32, kernel_size=3, padding=1),
            ConvLayer(32, 32, kernel_size=3, padding=1)
        )
        conv_dimension =  80 * 80

        self.conv_projection = nn.Sequential(
            nn.Linear(conv_dimension, 80),
            nn.ReLU(inplace=True)
        )
        self.cnn_feature = FeedForwardResidualBlock(80)

        self.divide = division()

        self.image_encode = nn.Sequential(
            nn.Linear(32 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8)
        )
        self.image_feature = FeedForwardResidualBlock(80)

        self.rule_encode = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        self.rule_feature = FeedForwardResidualBlock(5 * 80)

        self.vq_layer = VectorQuantizerEMA(num_embeddings=512,
                                        embedding_dim=5,
                                        beta=0.25)



        ########## decoder  ###########
        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(80*6+80*5, 64 * self.init_size ** 2))
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )


    def forward(self, x: torch.Tensor):
        batch_size, num_panels, height, width = x.size()

        origin_image = x
        x = x.view(batch_size * num_panels, 1, height, width)
        x = self.ConvBlock(x)
        x = x.view(batch_size, num_panels, 32, -1)
        x = self.conv_projection(x)
        x = self.cnn_feature(x)

        x = self.divide(x, num_groups=10)
        x = self.image_encode(x)
        x = x.view(batch_size, num_panels, 10 * 8)
        x = self.image_feature(x)
        x_for_recon = x[:, :6]

        blank_for_recon = torch.ones(batch_size, 1, height, width).cuda()
        blank_for_recon = self.ConvBlock(blank_for_recon)
        blank_for_recon = blank_for_recon.view(batch_size, 32, -1)
        blank_for_recon = self.conv_projection(blank_for_recon)
        blank_for_recon = self.cnn_feature(blank_for_recon)

        blank_for_recon = self.divide(blank_for_recon, num_groups=10)
        blank_for_recon = self.image_encode(blank_for_recon)
        blank_for_recon = blank_for_recon.view(batch_size, 10 * 8)
        blank_for_recon = self.image_feature(blank_for_recon)


        row3 = x[:, 6:8, :].unsqueeze(1)

        row3_candidates = torch.cat((row3.repeat(1, 8, 1, 1), x[:, 8:16, :].unsqueeze(2)), dim=2)

        row1 = x[:, 0:3, :].unsqueeze(1)

        row2 = x[:, 3:6, :].unsqueeze(1)


        tworow_13 = torch.cat((row1.repeat(1,8,1,1), row3_candidates), dim=2)

        tworow_23 = torch.cat((row2.repeat(1, 8, 1, 1), row3_candidates), dim=2)

        tworow_12 = torch.cat((row1, row2), dim=2)





        tworow_all = torch.cat((tworow_12, tworow_13, tworow_23), dim=1)



        x = self.divide(tworow_all, num_groups=80)

        x = self.rule_encode(x)
        x = x.view(batch_size, 17, 80 * 5)
        x = self.rule_feature(x)


        ###### VQ #######
        quantized_x, vq_loss, _, _ = self.vq_layer(x.view(batch_size, 17, 80, 5))

        x = quantized_x.view(batch_size, 17, 80 * 5)

        ###### VQ #######

        r12 = x[:,0]
        r13 = x[:,1:9]
        r23 = x[:,9:] #


        recon_loss = 0

        for i in range(6):
            recon_input = torch.cat((x_for_recon[:, :i].view(batch_size,-1), blank_for_recon, x_for_recon[:, i+1:].view(batch_size,-1), r12),dim=1)
            recon_input = self.l1(recon_input).view(batch_size, 64, self.init_size, self.init_size)
            image_recon = self.decoder(recon_input)
            recon_loss += F.mse_loss(image_recon.squeeze(1), origin_image[:,i])

        recon_loss = recon_loss / 6




        logit1 = (r13 * r12.unsqueeze(1).repeat(1, 8, 1)).sum(-1)
        logit2 = (r23 * r12.unsqueeze(1).repeat(1, 8, 1)).sum(-1)



        return logit1, logit2, vq_loss, recon_loss



class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = beta

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):

        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)


        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)


            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encodings


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projection(x)


class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim, expansion_multiplier=1):
        super(FeedForwardResidualBlock, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim)
        )

    def forward(self, x: torch.Tensor):
        return x + self.projection(x)


class division(nn.Module):
    def forward(self, x, num_groups):

        shape = x.shape[:-1] + (num_groups,) + (x.shape[-1] // num_groups,)
        x = x.view(shape)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)