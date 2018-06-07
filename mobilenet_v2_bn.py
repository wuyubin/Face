import mxnet as mx


def inverted_residual_unit(data, num_filter_input, num_filter_output, name, use_shortcut=True, stride=1,
                           expansion_rate=1, bn_mom=0.9, workspace=256):
    conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter_input * expansion_rate), kernel=(1, 1),
                               stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_pointwise_kernel_in')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter_input * expansion_rate), kernel=(3, 3),
                               num_group=int(num_filter_input * expansion_rate), stride=(stride, stride), pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_depthwise_kernel')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter_output, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_pointwise_kernel_out')
    bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')

    if use_shortcut:
        return bn3 + data
    else:
        return bn3


def get_symbol(num_classes=256, **kwargs):
    filter_num_inputs = [16, 24, 32, 64, 96, 160]
    filter_num_outputs = [24, 32, 64, 96, 160, 320]
    units_num = [2, 3, 4, 3, 3, 1]
    stride_list = [2, 2, 1, 2, 2, 1]

    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=32, pad=(1, 1), kernel=(3, 3), stride=(2, 2),
                                  no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1, fix_gamma=False, eps=2e-5)
    relu1 = mx.symbol.Activation(name='relu1', data=conv1_bn, act_type='relu')
    body = inverted_residual_unit(relu1, 32, 16, 'stage_1', use_shortcut=False)
    for stage_num in range(0, 6):
        for level_num in range(units_num[stage_num]):
            if level_num == 0:
                body = inverted_residual_unit(body, filter_num_inputs[stage_num], filter_num_outputs[stage_num], \
                                              'stage%d_level%d' % (stage_num + 2, level_num + 1), use_shortcut=False,
                                              stride=stride_list[stage_num], expansion_rate=6)
            else:
                body = inverted_residual_unit(body, filter_num_inputs[stage_num], filter_num_outputs[stage_num], \
                                              'stage%d_level%d' % (stage_num + 2, level_num + 1), expansion_rate=6)
    stage8 = mx.sym.Convolution(data=body, num_filter=1280, kernel=(1, 1), no_bias=True, name='stage8_pointwise_kernel')
  

    gpool = mx.symbol.Pooling(data=stage8, pool_type='avg', kernel=(7, 7),
                              global_pool=True, name='global_pool')
    flat = mx.symbol.Flatten(data=gpool)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='pre_fc1', no_bias=True)



    return fc1
