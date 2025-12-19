import torch
import torch.nn as nn
from .layers.utils import BinarizeSTE, ShiftedLeakyReLU, conv_output_shape, BinarizeLayer, BinaryConv2d
from .layers.ValueVector import ValueVectorInputLayer
import torchhd


class ConvHDC(nn.Module):
    def __init__(self,
                D = 10000, # hypervector dimension
                n_classes = 10, # number of classes
                in_channels = 1, # number of input channels
                n_hidden = 1, # number of hidden convolution layers
                image_shape = 28, # side length of input images (sqaure)
                affine_bn = True, # use affine for batch norm
                n_value_vectors = 10, # number of vectors used to encode values (if uses value hypervectors)

                # When true, uses weighted bundling for input layer
                # When false maps input values to value hypervectors for the input layer
                weighted_bundling = True,

                # When false, uses binding in convolutions (i.e. grouped convolutions with D groups)
                # When true, uses fully connected hypervector mapping in place of binding (i.e. standard non-grouped convolutions)
                fully_connected_mapping = False,

                # When false, uses global position keys to map feature hypervector map to a single hypervector 
                # When true, uses multiple prototypes to classify each feature hypervector
                multiple_prototype = False
                ):
        super(ConvHDC, self).__init__()

        self.D = D
        self.n_classes = n_classes
        self.image_shape = image_shape
        self.n_hidden = n_hidden
        self.weighted_bundling = weighted_bundling
        self.fully_connected_mapping = fully_connected_mapping
        self.multiple_prototype = multiple_prototype

        self.activation = ShiftedLeakyReLU
        encoder_layers = []

        groups = 1 if fully_connected_mapping else D

        # set up input layer
        if weighted_bundling:
            encoder_layers.append(BinaryConv2d(in_channels,  D, kernel_size=5, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm2d(D, affine=affine_bn))
            encoder_layers.append(self.activation())
            encoder_layers.append(BinarizeLayer())
        else:
            encoder_layers.append(ValueVectorInputLayer(D=D, in_channels=in_channels, n_value_vectors=n_value_vectors, groups=groups, image_shape=image_shape))
            encoder_layers.append(nn.BatchNorm2d(D, affine=affine_bn))
            encoder_layers.append(self.activation())
            encoder_layers.append(BinarizeLayer())

        # set up hidden layers
        for i in range(n_hidden):
            encoder_layers.append(BinaryConv2d(D, D, kernel_size=3, stride=2, groups=groups))
            encoder_layers.append(nn.BatchNorm2d(D, affine=affine_bn))
            encoder_layers.append(self.activation())
            encoder_layers.append(BinarizeLayer())
        
        # calculate the shape of the final hypervector feature map
        output_shape = image_shape
        for layer in encoder_layers:
            if isinstance(layer, ValueVectorInputLayer):
                output_shape = layer.out_shape
            elif isinstance(layer, BinaryConv2d):
                output_shape = conv_output_shape(output_shape, layer)
        
        # set up global position hypervectors if not using multiple prototype classification
        if not multiple_prototype:
            encoder_layers.append(nn.Conv2d(D, D, kernel_size=output_shape, stride=1, groups=groups)) 
            encoder_layers.append(nn.BatchNorm2d(D, affine=affine_bn))
            encoder_layers.append(self.activation())
            encoder_layers.append(BinarizeLayer())
        
        self.encoder = nn.Sequential(*encoder_layers)

        if multiple_prototype:
            self.final_D = D * output_shape * output_shape
        else:
            self.final_D = D

        self.classifier = torchhd.models.Centroid(self.final_D, n_classes, requires_grad=True)

        # initial weights between -1 and 1
        with torch.no_grad():
            for layer in encoder_layers:
                if isinstance(layer, BinaryConv2d):
                    nn.init.uniform_(layer.weight, -1, 1)
            nn.init.uniform_(self.classifier.weight, -1, 1)
    
    # returns the final hypervector representation or final hypervector feature map for the input samples (depending on if using multiple prototype or not)
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        # encode the input
        x = self.encode(x)

        # compute similarities with binarized prototype weights
        similarities = torch.matmul(x, BinarizeSTE.apply(self.classifier.weight).t())
        
        # scale and return siliarities
        return similarities / self.final_D ** 0.5