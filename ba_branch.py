import torch
import torch.nn as nn
from dcg import HardSwish, DCG
from guided_fusion import SEF
from kan import KAN


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-4),
            HardSwish(),
            nn.Conv3d(in_dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-4),
            HardSwish()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv3d(dim, dim_out, 3, 2, 1, bias=False)

    def forward(self, x):
        x = self.reduction(x)
        return x


class BrainAgeBranch(nn.Module):
    def __init__(self, in_channel, in_dim, dim, num_classes, drop_path_rate, depth, group):
        super().__init__()
        dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.patch_embed = PatchEmbed3D(in_chans=in_channel, in_dim=in_dim, dim=dim)

        cur_layer = 0
        layer_dim = dim
        block = DCG
        self.layers = nn.ModuleList()
        for idx, repeat in enumerate(depth):
            level = nn.Sequential(
                *[block(
                    in_ch=dim * 2 ** idx,
                    squeeze_groups=group[idx],
                    drop_path=dprs[cur_layer]
                ) for _ in range(repeat)]
            )
            if idx < 3:
                level.append(Downsample3D(dim=dim * 2 ** idx))
                layer_dim *= 2
            self.layers.append(level)
            cur_layer += repeat

        self.layers.append(SEF(dim=layer_dim))
        self.layers.append(KAN([layer_dim, dim, num_classes]))

    def forward(self, f_age, sex):
        f_age = self.patch_embed(f_age)
        
        features = []
        for layer in self.layers:
            if isinstance(layer, SEF):
                f_age = layer(f_age, sex)
            else:
                f_age = layer(f_age)
            features.append(f_age.clone().detach())

        return f_age, features


def create_model(num_classes=1):
    return BrainAgeBranch(
        in_channel=1,
        in_dim=32,
        dim=64,
        num_classes=num_classes,
        drop_path_rate=0.,
        depth=[1, 1, 2, 1],
        group=[8, 16, 32, 64]
    )

if __name__ == '__main__':
    x = torch.rand(1, 1, 96, 114, 96).to(torch.float32)
    sex = torch.tensor([0]).to(torch.int64)
    model = create_model(num_classes=1)
    y, feats = model(x, sex)
    for i in feats:
        print(i.shape)

    from torchinfo import summary
    summary(model, input_data=(x, sex))
