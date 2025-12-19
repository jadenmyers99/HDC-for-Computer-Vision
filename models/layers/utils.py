import torch
import torch.nn as nn
import torch.nn.functional as F


# function to calculate the output shape of a conv layer
def conv_output_shape(image_shape, # image side length (square)
                        conv, # binary convolution layer
                        ):
    side_length_out = int((image_shape + 2 * conv.padding - conv.dilation * (conv.kernel_size - 1) - 1)/conv.stride + 1)
    return side_length_out


# binary convolution layer
class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BinaryConv2d, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, int(in_channels/groups), kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.kernel_size = kernel_size
    
    def forward(self, x):
        return F.conv2d(x, BinarizeSTE.apply(self.weight), stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
    

# straight through estimator class for binarization during training
class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (1 - torch.pow(torch.tanh(input), 2)) * grad_output
        return grad_input, None, None


# nn.Module for binarization
class BinarizeLayer(nn.Module):
    def forward(self, x):
        return BinarizeSTE.apply(x)


# class for shifted leaky relu activation function
class ShiftedLeakyReLU(nn.LeakyReLU):
    def __init__(self, shift=0.5):
        super(ShiftedLeakyReLU, self).__init__()
        self.shift = shift
    
    def forward(self, x):
        return super().forward(x) - self.shift
