import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Mlp

# Explicit import (ensure the filename matches your directory structure)
from .atlasmom import AtlasMoM

__all__ = ['BasicBlock', 'Bottleneck', 'AtlasNet']


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def downsample_basic_block(x, planes, stride):
    """Downsamples the input block and pads channels with zeros."""
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)

    # Device-agnostic zero padding
    pad_channels = planes - out.size(1)
    zero_pads = torch.zeros(
        (out.size(0), pad_channels, out.size(2), out.size(3), out.size(4)),
        device=x.device,
        dtype=x.dtype
    )

    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.InstanceNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AtlasNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            shortcut_type='B',
            num_classes=2,
            num_heads=4,
            dim_latent=256,  # Note: Default adjusted to 256 to match layer2 output
            num_atlas=56,
            lambda_init=0.7
    ):
        super(AtlasNet, self).__init__()

        self.inplanes = 48
        self.num_classes = num_classes

        # Initial layers
        self.conv1 = nn.Conv3d(
            1,
            self.inplanes,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, stride=2)

        # Atlas-specific layers
        self.init_ln = nn.LayerNorm(dim_latent)
        self.MoM = AtlasMoM(
            input_dim=dim_latent,
            num_atlas=num_atlas,
            num_heads=num_heads,
            lambda_init=lambda_init
        )

        self.fc = Mlp(
            in_features=dim_latent * 2,
            hidden_features=int(dim_latent),
            out_features=num_classes,
            act_layer=nn.GELU,
            drop=0.1
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.InstanceNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, atlas):
        # Backbone processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  # Shape: (B, C, D, H, W)

        # Atlas RoI Pooling (Unrolled for readability and numerical stability)
        # Flatten spatial dimensions: (B, C, D*H*W) -> (B, D*H*W, C)
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)

        # Flatten atlas spatial dimensions: (B, Num_Atlas, D*H*W)
        atlas_flat = atlas.flatten(start_dim=2)

        # Compute regional feature representations
        numerator = atlas_flat.matmul(x_flat)  # Shape: (B, Num_Atlas, C)
        denominator = atlas_flat.sum(dim=2).unsqueeze(2).clamp_min(1e-6)  # Prevent zero-division

        x = self.init_ln(numerator / denominator)

        # Atlas Mixture of Models (MoM) routing
        (x,
         expert_outputs,
         shared_outputs,
         expert_inputs,
         recon_outputs,
         routing_weights) = self.MoM(x)

        # Final classification
        logit = self.fc(x)

        return logit, x, expert_outputs, shared_outputs, expert_inputs, recon_outputs, routing_weights