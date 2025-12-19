import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.utils import BinarizeSTE, ShiftedLeakyReLU, conv_output_shape, BinarizeLayer, BinaryConv2d
from .layers.ValueVector import ValueVectorInputLayer
import torchhd


def generate_random_bipolar(size, dtype=None):
    out = torch.where(torch.randint(low=0, high=2, size=size) >= 1, 1.0, -1.0)
    if dtype is not None:
        out = out.to(dtype)
    return out


class StaticHDC(nn.Module):
    def __init__(self,
                D = 10000, # hypervector dimension
                n_classes = 10, # number of classes
                in_channels = 1, # number of input channels
                n_value_vectors = 10, # number of vectors used to encode values
                image_shape = 28, # side length of input images (sqaure)
                mode = 'orthogonal', # mode for generating encoding hypervectors (orthogonal, linear or local_linear)
                S = 5 # number of hypervectors in each split (for local linear encoding)
                ):
        super().__init__()

        self.D = D
        self.value_input_layer = ValueVectorInputLayer(n_value_vectors=n_value_vectors)
        self.position_vectors = BinaryConv2d(D, D, kernel_size=image_shape, stride=1, padding=0, groups=D)
        self.classifier = torchhd.models.Centroid(D, n_classes, requires_grad=False)

        # encoding hypervectors stay static
        for param in self.value_input_layer.parameters():
            param.requires_grad = False
        for param in self.position_vectors.parameters():
            param.requires_grad = False

        # initialize position and value hypervectors with uniform random values
        position_weights = generate_random_bipolar(self.position_vectors.weight.shape, self.position_vectors.weight.dtype)
        value_weights = generate_random_bipolar(self.value_input_layer.values.weight.shape, self.value_input_layer.values.weight.dtype)

        if mode != 'orthogonal':
            if mode == 'linear':
                # generate hypervectors for the values with linear encoding
                values_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                value_weights = self.generate_linear_encoding(values_initial_hypervector, n_value_vectors)

                # generate axis hypervectors with linear encoding for the positions
                X_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                X = self.generate_linear_encoding(X_initial_hypervector, image_shape)
                Y_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                Y = self.generate_linear_encoding(Y_initial_hypervector, image_shape)
            elif mode == 'local_linear':
                # generate hypervectors for the values with linear encoding
                values_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                value_weights = self.generate_local_linear_encoding(values_initial_hypervector, S, n_value_vectors)

                # generate axis hypervectors with linear encoding for the positions
                X_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                X = self.generate_local_linear_encoding(X_initial_hypervector, S, image_shape)
                Y_initial_hypervector = generate_random_bipolar((D,), self.value_input_layer.values.weight.dtype)
                Y = self.generate_local_linear_encoding(Y_initial_hypervector, S, image_shape)
            
            # combine X and Y with binding and assign the positions vectors
            for i in range(image_shape):
                for j in range(image_shape):
                    position_weights[:, 0, i, j] = X[i] * Y[j]

        self.position_vectors.weight.copy_(position_weights)
        self.value_input_layer.values.weight.copy_(value_weights)

    def generate_linear_encoding(self,
                                initial, # the initial hypervector to start the linear encoding from
                                N # the number of hypervectors to encode
                                ):
        D = initial.shape[0]
        linear_hypervectors = torch.zeros((N, D), dtype=initial.dtype)
        linear_hypervectors[0] = initial
        
        flip_n = int((D / 2) / (N - 1)) # number of entries to filp for each hypervector
        flipped_positions = set() # keep track of which positions have already been flipped

        for i in range(1, N):
            previous = linear_hypervectors[i-1]
            new = previous.clone()

            # randomly select bits to flip (make sure they haven't already been flipped)
            available_positions = [i for i in range(D) if i not in flipped_positions]
            flip_indices = torch.randperm(len(available_positions))[:flip_n]
            flip_positions = [available_positions[idx] for idx in flip_indices.tolist()]

            for j in flip_positions:
                new[j] *= -1
                flipped_positions.add(j)

            linear_hypervectors[i] = new
    
        return linear_hypervectors
    
    def generate_local_linear_encoding(self,
                                        initial, # the initial hypervector to start the local linear encoding from
                                        S, # number of vectors in each split
                                        N # the number of hypervectors to encode
                                        ):
        D = initial.shape[0]
        local_linear_hypervectors = torch.zeros((N, D), dtype=initial.dtype)

        # generate the first split with linear encoding
        initial_linear_encodings = self.generate_linear_encoding(initial, S)
        local_linear_hypervectors[:S] = initial_linear_encodings

        # generate the rest of the splits with linear encoding
        initial = initial_linear_encodings[len(initial_linear_encodings)-1:len(initial_linear_encodings)].squeeze(0)
        for i in range(S, N, S-1):
            end = i + (S - 1) if i + (S - 1) <= N else N
            linear_encodings = self.generate_linear_encoding(initial, S)
            initial = linear_encodings[len(linear_encodings)-1:len(linear_encodings)].squeeze(0)
            local_linear_hypervectors[i:end] = linear_encodings[1:(N-i)+1] if end == N else linear_encodings[1:len(linear_encodings)]
        
        return local_linear_hypervectors
    
    def encode(self, x):
        batch_size = x.size(0)

        # get value encodings
        value_encoded = self.value_input_layer.encode_values(x)

        out = x.new_zeros(batch_size, self.D, 1, 1)

        # apply the position hypervectors to each channel seperately
        for channel in value_encoded:
            out += self.position_vectors(channel)
        
        # binarize
        out = torch.sign(out)

        # remove uneeded dimensions
        out = out.squeeze(-1).squeeze(-1)

        return out
    
    def classify(self, x):
        # compute similarities with binarized prototype weights
        similarities = torch.matmul(x, BinarizeSTE.apply(self.classifier.weight).t())
        
        # get the maximum similarities
        if len(similarities.shape) == 1:
            preds = torch.argmax(similarities)
        else:
            preds = torch.argmax(similarities, dim=1)

        return preds
    
    def forward(self, x):
        x = self.encode(x)
        x = self.classify(x)
        return x
