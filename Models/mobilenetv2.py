"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

from Models.CBAM import CBAM
from torch import nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

__all__ = [
    "MobileNetV2",
    "build_mnv2_backbone",
    "build_mobilenetv2_fpn_backbone",
    "build_fcos_mobilenetv2_fpn_backbone",
]


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        FrozenBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2d(inp, oup, 1, 1, 0, bias=False),
        FrozenBatchNorm2d(oup),
        #CBAM(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio)) #Fugi insert int()
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                    ),
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                FrozenBatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

# TODO: create a class for adding CBAM to the neural network
class AddCBAM(nn.Module):
    def __init__(self,output_channel):
        super(AddCBAM, self).__init__()
        self.cbam = nn.Sequential
        (
            CBAM(output_channel),
        )
        
    def forward(self, x):
        
        return x + self.cbam(x)
        

class MobileNetV2(Backbone):
    def __init__(self, cfg, num_classes=5, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2], #original s = 2
            [6,  32, 3, 2], #original s = 2
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        
        # Variable insert by Fugi
        self.return_features_indices = [3, 6, 13, 17]
        self.return_features_num_channels = []       
        
        self.features = nn.ModuleList([conv_3x3_bn(3, input_channel, 2)]) #original 2
        # building inverted residual blocks
        block = InvertedResidual
        blockCBAM = AddCBAM
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                self.features.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
                if len(self.features) - 1 in self.return_features_indices:
                    self.return_features_num_channels.append(output_channel)
        self.features.append(blockCBAM(output_channel))
        input_channel = output_channel
        if len(self.features) - 1 in self.return_features_indices:
            self.return_features_num_channels.append(output_channel)
        # building last several layers
        #output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        return {"res{}".format(i + 2): r for i, r in enumerate(res)}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
@BACKBONE_REGISTRY.register()
def build_mnv2_classCBAM_backbone(cfg, input_shape):
    """
    Create a MobileNetV2 instance from config.
    Returns:
        MobileNetV2: a :class:`MobileNetV2` instance.
    """
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    out_feature_channels = {"res2": 24, "res3": 32, "res4": 96, "res5": 320}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = MobileNetV2(cfg)
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model

'''
@BACKBONE_REGISTRY.register()
def build_mobilenetv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mnv2_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
'''