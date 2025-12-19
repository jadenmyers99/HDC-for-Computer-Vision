import torch
import torch.nn as nn
from .utils import BinarizeSTE, ShiftedLeakyReLU, conv_output_shape, BinarizeLayer, BinaryConv2d


# class for a ConvHDC block with residual connections
class ResBlock(nn.Module):
    def __init__(self,
                D = 10000, # hypervector dimension
                affine_bn = True, # use affine for batch norm
                groups = 1, # groups for layers in the block
                input_layer = False, # for whether this is the first block or a hidden one
                in_channels = 1, # number of input channels for when this is the first block

                # When true, uses weighted bundling for input layer
                # When false maps input values to value hypervectors for the input layer
                weighted_bundling = True,
                ):
        super(ResBlock, self).__init__()
        activation = ShiftedLeakyReLU

        # set up first layer
        if input_layer:
            self.conv_input = BinaryConv2d(in_channels,  D, kernel_size=5, stride=2, padding=1)
        else:
            self.conv_input = BinaryConv2d(D,  D, kernel_size=3, stride=2, padding=1, groups=groups)
        self.bn_input = nn.BatchNorm2d(D, affine=affine_bn)
        self.activ_input = activation()
        self.binarize_input = BinarizeLayer()

        self.conv_hidden = BinaryConv2d(D,  D, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn_hidden = nn.BatchNorm2d(D, affine=affine_bn)
        self.activ_hidden = activation()
        self.binarize_hidden = BinarizeLayer()
        
        self.conv_final = BinaryConv2d(D,  D, kernel_size=3, stride=1, padding=1, groups=groups)
        self.bn_final = nn.BatchNorm2d(D, affine=affine_bn)
        self.activ_final = activation()
        self.binarize_final = BinarizeLayer()
    
    def forward(self, x):
        # apply first layer
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.activ_input(x)
        x = self.binarize_input(x)

        # save identity for residual connection
        y = x

        # apply hidden layer
        x = self.conv_hidden(x)
        x = self.bn_hidden(x)
        x = self.activ_hidden(x)
        x = self.binarize_hidden(x)

        # apply final layer and add residual connection
        x = self.conv_final(x)
        x = self.bn_final(x)
        x += y # residual connection
        x = self.activ_final(x)
        x = self.binarize_final(x)

        return x