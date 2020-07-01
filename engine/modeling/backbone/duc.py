import torch.nn as nn
from engine.modeling import registry


@registry.BACKBONE.register("DUC")
class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(self, inplanes, planes,
                 upscale_factor=2, norm_layer=nn.BatchNorm2d):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


@registry.BACKBONE.register("DUCnx")
class DUCnx(nn.Module):
    def __init__(self, in_channels, stages, upscale_factor=2,
                 norm_layer=None):
        super(DUCnx, self).__init__()
        layers = list()
        for stage in stages:
            layers.append(self._make_layers(in_channel=in_channels,
                                            out_channel=stage,
                                            upscale_factor=upscale_factor,
                                            norm_layer=norm_layer)
                          )
            in_channels = stage

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _make_layers(in_channel,
                     out_channel,
                     upscale_factor=2,
                     norm_layer=None):

        return DUC(in_channel,
                   out_channel,
                   upscale_factor=upscale_factor,
                   norm_layer=norm_layer)
