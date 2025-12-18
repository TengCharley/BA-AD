import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class DynamicContrastAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.wl = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.wg = nn.Parameter(torch.zeros(in_ch, 1, 1, 1))
        self.norm = nn.InstanceNorm3d(in_ch, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.ac = HardSwish()

    def forward(self, x):
        local_mean = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        local_contrast = x - local_mean
        local_out = x + self.wl * local_contrast

        global_max = torch.amax(x, dim=(2, 3, 4), keepdim=True)
        global_min = torch.amin(x, dim=(2, 3, 4), keepdim=True)
        global_contrast = global_max - global_min
        global_attn = self.sigmoid(self.wg * global_contrast)

        out = local_out * global_attn
        out = x + out
        return self.ac(self.norm(out))


class ChannelAdaptiveConvolution(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.dw1 = nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, groups=in_ch, dilation=1)
        self.dw3 = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=2, groups=in_ch, dilation=2)
        self.conv = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.wc = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.bn = nn.BatchNorm3d(in_ch)
        self.ac = HardSwish()

    def forward(self, x):
        out1 = self.dw1(x)
        out2 = self.dw3(x)
        out3 = self.ac(self.bn(x + out1 + out2))

        out4 = self.wc * self.conv(out3)
        out = out3 + out4
        return self.ac(self.bn(out))


class GroupIntegratedAttention(nn.Module):
    def __init__(self, in_ch, squeeze_groups=16):
        super().__init__()
        assert in_ch % squeeze_groups == 0, "channel num must be divided by squeeze groups"
        self.squeeze_groups = squeeze_groups
        self.dw_encode = nn.Conv3d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.dw_decode = nn.Conv3d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.group_conv = nn.Conv3d(in_ch, in_ch, 3, 1, 1, groups=squeeze_groups)
        self.pw_squeeze = nn.Conv3d(in_ch, in_ch // squeeze_groups, 1)
        self.pw_restore = nn.Conv3d(in_ch // squeeze_groups, in_ch, 1)
        self.bn = nn.BatchNorm3d(in_ch)
        self.bn_squeeze = nn.BatchNorm3d(in_ch // squeeze_groups)
        self.ac = HardSwish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.shape
        identity = x
        x = self.dw_encode(x)

        out_com = self.bn_squeeze(self.pw_squeeze(x))
        com_attn = self.sigmoid(out_com)

        out_het = self.bn(self.group_conv(x))
        out_het = out_het.reshape(b, self.squeeze_groups, c // self.squeeze_groups, d, h, w).mean(dim=1)

        out = self.ac(self.bn_squeeze(com_attn * out_het))
        out = self.bn(self.pw_restore(out))
        out = self.dw_decode(out)
        return self.ac(self.bn(identity + out))


class DCG(nn.Module):
    def __init__(self, in_ch, squeeze_groups=16, drop_path=0.):
        super().__init__()
        self.stage1 = DynamicContrastAttention(in_ch)
        self.stage2 = ChannelAdaptiveConvolution(in_ch)
        self.stage3 = GroupIntegratedAttention(in_ch, squeeze_groups)
        self.drop = DropPath(drop_prob=drop_path)
        self.bn = nn.BatchNorm3d(in_ch)
        self.ac = HardSwish()

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.drop(out)
        return self.ac(self.bn(x + out))


if __name__ == '__main__':
    x = torch.rand((2, 4, 6, 8, 10), dtype=torch.float32)

    # block = DynamicContrastAttention(4)
    # block = ChannelAdaptiveConvolution(4)
    # block = GroupIntegratedAttention(4, squeeze_groups=4)
    block = DCG(4, 4)

    y = block(x)
    print(y.shape)

