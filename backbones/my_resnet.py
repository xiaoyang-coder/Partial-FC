from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
import symbol_utils
import memonger
import sklearn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from default import config


def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    act_type = kwargs.get('version_act', 'prelu')
    # print('in unit3')
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
    conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                    no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    if dim_match:
        shortcut = data
    else:
        conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                        workspace=workspace, name=name + '_conv1sc')
        shortcut = mx.sym.BatchNorm(data=conv1sc, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_sc')
    return bn3 + shortcut




def resnet(units, num_stages, filter_list, num_classes, bottle_neck):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {'version_se': config.net_se,
              'version_input': config.net_input,
              'version_output': config.net_output,
              'version_unit': config.net_unit,
              'version_act': config.net_act,
              'bn_mom': bn_mom,
              'workspace': workspace,
              }
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)
    body = mx.sym.Variable(name='data')
    body = Conv(data=body, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (2, 2), False, bottle_neck=bottle_neck, **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True,bottle_neck=bottle_neck, **kwargs)
                                 
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    body = mx.symbol.Dropout(data=body, p=0.4)
    fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    return fc1


def get_symbol():
    num_classes = 512
    num_layers = 50
    filter_list = [64, 64, 128, 256, 512]
    bottle_neck = False
    num_stages = 4
    units = [3, 4, 14, 3]
    net = resnet(units=units,
                 num_stages=num_stages,
                 filter_list=filter_list,
                 num_classes=num_classes,
                 bottle_neck=bottle_neck)

    return net
if __name__ == "__main__":
    import torchvision
    torchvision.models