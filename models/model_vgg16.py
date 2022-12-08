import torch
import torch.nn as nn
import math
import time

# define CNN layers
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers.append(nn.MaxPool2d(kernel_size=pooling_k,
                             stride=pooling_s))
    # print(layers)
    return layers

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

# VGG16 model
class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.conv_layers = nn.Sequential(*layer1,*layer2,*layer3,*layer4,*layer5)
        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        in_tensor = x[0]
        layer_num = x[1]
        if layer_num < 18:
            output = self.conv_layers[layer_num](in_tensor)
        elif layer_num == 18:
            output = in_tensor.view(in_tensor.size()[0], -1)
            output = self.layer6(output)
        elif layer_num == 19:
            output = self.layer7(in_tensor)
        else:
            output = self.layer8(in_tensor)
        return output

VGG16()