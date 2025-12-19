import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import BinarizeSTE, ShiftedLeakyReLU, conv_output_shape, BinarizeLayer, BinaryConv2d
import torchhd


class ValueVectorInputLayer(nn.Module):
    def __init__(self,
                D = 10000, # hypervector dimension
                in_channels = 1, # number of input channels
                n_value_vectors = 10, # number of vectors used to encode values
                groups = 1, # number groups to use for the conv layers
                image_shape = 28, # side length of input images (sqaure)
                ):
        super(ValueVectorInputLayer, self).__init__()

        self.D = D
        self.in_channels = in_channels 
        self.n_value_vectors = n_value_vectors
        self.values = nn.Embedding(n_value_vectors, D)

        # convolution layers to apply seperately to each input channel
        self.convs = nn.ModuleList([
            BinaryConv2d(D,  D, kernel_size=5, stride=2, padding=0, groups=groups) for _ in range(in_channels)
        ])

        self.out_shape = conv_output_shape(image_shape, self.convs[0])
    
    # returns the value hypervector for the values
    def encode_values(self, x):
        # quantize x and convert it to int
        x = (x * (self.n_value_vectors - 1)).long()

        # split x by channels
        splits = []
        for i in range(self.in_channels):
            splits.append(x[:, i:i+1, :, :])
        
        # assign the value hypervectors to the values
        split_embeddings = [
            F.embedding(split, BinarizeSTE.apply(self.values.weight)) for split in splits
        ]

        # rearange the dimensions so that the hypervector dimension is the channel dimension
        split_embedding_perumted = [
            split.squeeze(1).permute(0, 3, 1, 2) for split in split_embeddings
        ]

        return split_embedding_perumted
    
    def forward(self, x):
        batch_size = x.size(0)

        # tensor to accumulate results for each channel
        split_out = x.new_zeros(batch_size, self.D, self.out_shape, self.out_shape)

        split_embeddings = self.encode_values(x)

        # apply the convolutions to the value hypervctors
        for split, conv in zip(split_embeddings, self.convs):
            split_out += conv(split)
        out = BinarizeSTE.apply(split_out)

        return out