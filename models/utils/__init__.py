import math

import torch
import torch.nn as nn

# import e2cnn.nn as enn
# from e2cnn.nn import init


def weights_init_cnn(module):
    classname = module.__class__.__name__
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if "linear" in classname.lower():
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if "norm" in classname.lower():
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    return


def weights_init_dsf(module):
    classname = module.__class__.__name__
    if classname == "GConv2d":
        w_shape = module.weight.size()
        Q = w_shape[2]  # nr basis filters
        fan_out = w_shape[-1]
        std = math.sqrt(2 / fan_out * Q)
        nn.init.normal_(module.weight, mean=0.0, std=std)

    if isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    return


# def weights_init_e2cnn(module):
#     classname = module.__class__.__name__
#     if isinstance(module, nn.Conv2d):
#         nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

#     if isinstance(module, enn.R2Conv):
#         init.deltaorthonormal_init(module.weights, module.basisexpansion)
#     elif isinstance(module, torch.nn.BatchNorm2d):
#         module.weight.data.fill_(1)
#         module.bias.data.zero_()
#     elif isinstance(module, torch.nn.Linear):
#         module.bias.data.zero_()
#     return
