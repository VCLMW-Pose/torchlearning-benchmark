# Copyright (c) MW-Pose Group, 2020

from collections import OrderedDict

import torch
import torch.nn as nn

from engine.modeling import registry


@registry.BACKBONE.register("SimpleMLP")
class SimpleMLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        """
        Args:
            input_dims (int): Input dimension for fully connected network
            n_hiddens (int): Hidden layer dimension
            n_class (int): Output layer dimension
        """
        super(SimpleMLP, self).__init__()
        assert isinstance(input_dims, int), "Expect int for input_dims"
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i + 1)] = nn.ReLU()
            layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.models = nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.models(input)


@registry.BACKBONE.register("MLP")
class MLP(nn.Module):
    def __init__(self, in_channel, n_hiddens, out_channel, p_dropout=0.2):
        """
        :param in_channel:
        :param n_hiddens:
        :param out_channel:
        :param p_dropout:
        """
        super(MLP, self).__init__()
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers["fc{}".format(i + 1)] = nn.Linear(in_channel, n_hidden)
            layers["relu{}".format(i + 1)] = nn.ReLU()
            layers["drop{}".format(i + 1)] = nn.Dropout(p_dropout)
            in_channel = n_hidden
        layers["out"] = nn.Linear(in_channel, out_channel)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
