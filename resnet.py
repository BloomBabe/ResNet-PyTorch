import torch
import torch.nn as nn

def conv3x3(input_channels, output_channels, stride = 1, 
            groups = 1, dilation = 1):
    return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(input_channels, output_channels, stride = 1):
    return nn.Conv2d(input_channels, output_channels, kernel_size=1, 
                    stride=stride, bias=False)

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01,inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    """Abstract class (interface) of ResNet Block"""
    expansion = 1
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride = 1,
                 downsample = 1,
                 activation='relu',
                 *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsample = downsample
        self.activation = activation

        self.blocks = nn.Sequential(
                conv3x3(self.input_channels, self.output_channels, 
                                     self.downsample),
                nn.BatchNorm2d(self.output_channels),
                activation_func(self.activation),
                conv3x3(self.output_channels, self.expanded_channels),
                nn.BatchNorm2d(self.output_channels)
        )
        
        self.shortcut = nn.Sequential(
                conv1x1(self.input_channels, self.expanded_channels, stride=self.downsample),
                nn.BatchNorm2d(self.expanded_channels)
                ) if self.should_apply_shortcut else None
            

    def forward(self, x):
        identity = x
        out = self.blocks(x)

        if self.shortcut is not None: 
            identity = self.shortcut(identity)

        out += identity 
        out = activation_func(self.activation)(out)

        return out

    @property
    def should_apply_shortcut(self):
        return self.input_channels != self.output_channels

    @property
    def expanded_channels(self):
        return self.output_channels * self.expansion


class BottleNeck(ResidualBlock):
    expansion = 4
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride = 1,
                 *args, **kwargs):
        super(BottleNeck, self).__init__(input_channels,
                                         output_channels)
        self.blocks = nn.Sequential(
            conv1x1(self.input_channels, self.output_channels),
            nn.BatchNorm2d(self.output_channels),
            activation_func(self.activation),
            conv3x3(self.output_channels, self.output_channels, stride=self.downsample),
            nn.BatchNorm2d(self.output_channels),
            activation_func(self.activation),
            conv1x1(self.output_channels, self.expanded_channels),
            nn.BatchNorm2d(self.expanded_channels),
        )

class ResNet(nn.Module):
    
    def __init__(self,
                 num_classes,
                 input_channels = 3,
                 pretrained = False,
                 block = ResidualBlock,
                 blocks_sizes = [64, 128, 256, 512],
                 layers = [2, 2, 2, 2],
                 activation = 'relu',
                 *args, **kwargs):
        super(ResNet, self).__init__()
        self.blocks_sizes = blocks_sizes
        self.input_channels = input_channels
        self.layers = layers
        self.activation = activation

        self.gate = nn.Sequential(
            nn.Conv2d(self.input_channels, self.blocks_sizes[0], kernel_size=7, 
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(self.activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.blocks = [
            *self._make_layer(self.blocks_sizes[0], self.blocks_sizes[0], block, depth=self.layers[0]),
        ]
        for (inputs, outs), depth in zip(list(zip(self.blocks_sizes, self.blocks_sizes[1:])),self.layers[1:]):
            for block_layer in self._make_layer(inputs * block.expansion, outs, block, depth=depth):
                self.blocks.append(block_layer) 
        self.blocks = nn.Sequential(*self.blocks)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(blocks_sizes[-1] * block.expansion, num_classes)

    def forward(self, x):
        out = self.gate(x)
        out = self.blocks(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(self, input_channels, output_channels, block, depth=1):
        downsample = 2 if input_channels != output_channels else 1
        layer = []
        layer.append(block(input_channels , output_channels, downsample=downsample))
        input_channels = output_channels * block.expansion
        for _ in range(1, depth):
            layer.append(block(input_channels, output_channels, downsample=1))

        return layer




def resnet18(num_classes=1000, in_channels=3, block=ResidualBlock, *args, **kwargs):
    model = ResNet(num_classes, in_channels,  block=block, deepths=[2, 2, 2, 2], *args, **kwargs)    
    return model

def resnet34(num_classes=1000, in_channels=3,  block=ResidualBlock, *args, **kwargs):
    model = ResNet(num_classes, in_channels,  block=block, deepths=[3, 4, 6, 3], *args, **kwargs)
    """ Not implement: matching my layers with orinal ones for loading pytorch 
    resnet weights """
    # if pretrained:
    #     if num_classes != 1000:
    #         raise ValueError("Number classes should equals 1000 if model's pretrained")
    #     model = copy_weights(model, 'resnet34')    
    return model

def resnet50(num_classes=1000, in_channels=3, block=BottleNeck, *args, **kwargs):
    model = ResNet(num_classes, in_channels,  block=block, deepths=[3, 4, 6, 3], *args, **kwargs)    
    return model

def resnet101(num_classes=1000, in_channels=3, block=BottleNeck, *args, **kwargs):
    model = ResNet(num_classes, in_channels,  block=block, deepths=[3, 4, 23, 3], *args, **kwargs)    
    return model

def resnet152(num_classes=1000, in_channels=3, block=BottleNeck, *args, **kwargs):
    model = ResNet(num_classes, in_channels,  block=block, deepths=[3, 8, 36, 3], *args, **kwargs)    
    return model











