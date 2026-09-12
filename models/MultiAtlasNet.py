import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Mlp

# Explicit import replacing wildcard
from .atlasmom import AtlasMoM

__all__ = ['BasicBlock', 'Bottleneck', 'MultiAtlasNet']


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

    # Device-agnostic and dtype-agnostic zero padding
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


class MultiAtlasNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            shortcut_type='B',
            num_classes=2,
            num_heads=4,
            dim_latent=128,
            num_atlas=56,
            lambda_init=0.7
    ):
        super(MultiAtlasNet, self).__init__()
        self.num_classes = num_classes

        self.MoM = AtlasMoM(
            input_dim=dim_latent * 2,
            num_atlas=num_atlas,
            num_heads=num_heads,
            lambda_init=lambda_init,
            mode='single'
        )

        # ---------- MRI Extractor ----------
        self.inplanes = 48  # State reset for MRI layers
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.InstanceNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, dim_latent // 2, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, dim_latent, layers[1], shortcut_type, stride=2)
        self.init_ln = nn.LayerNorm(dim_latent)

        # ---------- PET Extractor ----------
        self.inplanes = 48  # State reset for PET layers
        self.pconv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.pbn1 = nn.InstanceNorm3d(self.inplanes)
        self.prelu = nn.ReLU(inplace=True)
        self.pmaxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.player1 = self._make_layer(block, dim_latent // 2, layers[0], shortcut_type)
        self.player2 = self._make_layer(block, dim_latent, layers[1], shortcut_type, stride=2)
        self.pinit_ln = nn.LayerNorm(dim_latent)

        # ---------- Classification Head ----------
        self.fc = Mlp(
            in_features=dim_latent * 4,
            hidden_features=int(dim_latent * 2),
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

    def forward(self, x, px, atlas):
        # Pre-compute atlas flatten and denominator to save redundant operations
        atlas_flat = atlas.flatten(start_dim=2)
        # Add epsilon to prevent division by zero
        atlas_denom = atlas_flat.sum(dim=2).unsqueeze(2).clamp_min(1e-6)

        # ---------- Process MRI Data ----------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Atlas RoI Pooling
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.init_ln(atlas_flat.matmul(x_flat) / atlas_denom)

        # ---------- Process PET Data ----------
        px = self.pconv1(px)
        px = self.pbn1(px)
        px = self.prelu(px)
        px = self.pmaxpool(px)
        px = self.player1(px)
        px = self.player2(px)

        # Atlas RoI Pooling
        px_flat = px.flatten(start_dim=2).permute(0, 2, 1)
        px = self.pinit_ln(atlas_flat.matmul(px_flat) / atlas_denom)

        # ---------- Fusion & Routing ----------
        # Concatenate multi-modal features
        x = torch.cat((x, px), dim=2)

        # Self-attention for the atlas network (MoM)
        (x,
         expert_outputs,
         shared_outputs,
         expert_inputs,
         recon_outputs,
         routing_weights) = self.MoM(x)

        logit = self.fc(x)

        return logit, x, expert_outputs, shared_outputs, expert_inputs, recon_outputs, routing_weights