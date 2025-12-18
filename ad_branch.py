import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from dcg import DCG, HardSwish
from guided_fusion import SSMGT, SAMCGA, SEF
from kan import KAN


def window_partition_3d(x, window_size):
    B, C, D, H, W = x.shape
    x = x.view(B,
               C,
               D // window_size, window_size, 
               H // window_size, window_size, 
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(-1, window_size*window_size*window_size, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / window_size / window_size / window_size))
    x = windows.reshape(
        B, D // window_size, H // window_size, W // window_size, 
        window_size, window_size, window_size, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, windows.shape[2], D, H, W)
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


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm3d(in_dim, eps=1e-5),
            HardSwish(),
            nn.Conv3d(in_dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm3d(dim, eps=1e-5),
            HardSwish()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class Mamba(nn.Module):
    """
    hidden_states: (B, L, D)
    Returns: same shape as hidden_states
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MTBlock(nn.Module):
    def __init__(self, 
                 dim,
                 type,
                 num_heads=4,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 ):
        super().__init__()
        assert type in ['Transformer', 'Mamba']
        if type == 'Transformer':
            self.block = Transformer(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        elif type == 'Mamba':
            self.block = Mamba(d_model=dim, d_state=8, d_conv=3, expand=1)

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_block(in_features=dim, hidden_features=dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.block(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WMT(nn.Module):
    def __init__(self,
                 in_ch,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.
                 ):
        super().__init__()
        self.window_size = window_size

        self.mt = nn.ModuleList([
            MTBlock(dim=in_ch, type='Mamba', mlp_ratio=mlp_ratio, drop_path=drop_path),
            MTBlock(dim=in_ch, type='Transformer', num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path)
        ])

    def forward(self, x):
        _, _, D, H, W = x.shape

        pad_d = (self.window_size - D % self.window_size) % self.window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            _, _, Dp, Hp, Wp = x.shape
        else:
            Dp, Hp, Wp = D, H, W

        x = window_partition_3d(x, self.window_size)

        for _, blk in enumerate(self.mt):
            x = blk(x)

        x = window_reverse_3d(x, self.window_size, Dp, Hp, Wp)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :, :D, :H, :W].contiguous()

        return x


class ADBranch(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 groups,
                 drop_path_rate=0.2,
                 drop_rate=0.,
                 in_chans=1,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
    ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim)
        self.memory_embed = nn.Sequential(
            PatchEmbed3D(in_chans=in_chans, in_dim=in_dim, dim=dim),
            Downsample3D(dim=dim)
        )
        self.stages = nn.ModuleList()
        self.ssm_gt = nn.ModuleList()

        cur_dim = dim
        cur_layer = 0
        for sta, dep in enumerate(depths):
            layers = nn.Sequential()
            for _ in range(dep):
                if sta < 2:
                    layers.append(
                        DCG(
                            in_ch=cur_dim,
                            squeeze_groups=groups[sta],
                            drop_path=dpr[cur_layer]
                        )
                    )
                else:
                    layers.append(
                        WMT(
                            in_ch=cur_dim,
                            num_heads=num_heads[sta],
                            window_size=window_size[sta],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=dpr[cur_layer],
                            drop_path=dpr[cur_layer]
                        )
                    )
                cur_layer += 1
            if sta < 3:
                layers.append(Downsample3D(dim=cur_dim))
                cur_dim *= 2
                self.ssm_gt.append(SSMGT(in_ch=cur_dim, mem_squeeze=(sta < 2)))

            self.stages.append(layers)

        self.norm = nn.BatchNorm3d(dim * 2 ** (len(depths) - 1))
        self.gender_emb = nn.Embedding(num_embeddings=2, embedding_dim=512)
        self.fuse_emb = nn.Linear(cur_dim * 2, cur_dim)

        self.sef = SEF(dim=cur_dim)
        self.sam_cga = SAMCGA(dim=cur_dim, num_heads=16, qkv_bias=False)
        self.head = KAN([cur_dim, dim, num_classes])

    def forward(self, f_ad, age_feats, sex, age):
        mem = self.memory_embed(f_ad)
        f_ad = self.patch_embed(f_ad)

        features = []
        for i in range(3):
            f_ad = self.stages[i](f_ad)
            f_ad, mem = self.ssm_gt[i](f_ad, age_feats[i], mem)
            features.append(f_ad.clone().detach())

        f_ad = self.stages[-1](f_ad)
        features.append(f_ad.clone().detach())

        f_ad = self.sef(f_ad, sex)
        features.append(f_ad.clone().detach())

        age_gap = age_feats[-1] - age
        f_ad = self.sam_cga(f_ad, age_feats[-2], mem, age_gap)
        out = self.head(f_ad)

        return out, features


def create_model(num_classes=2):
    model = ADBranch(
        in_chans=1,
        num_classes=num_classes,
        depths=[1, 1, 2, 1],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 15, 8],
        groups=[8, 16, 32, 64],
        dim=64,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.25
    )
    return model


if __name__ == '__main__':
    device = 'cuda:7'
    feas_list = [
        torch.rand(1, 128, 24, 29, 24),
        torch.rand(1, 256, 12, 15, 12),
        torch.rand(1, 512, 6, 8, 6),
        torch.rand(1, 512, 6, 8, 6),
        torch.rand(1, 512),
        torch.rand(1, 1)
    ]
    feas_on_device = [f.to(device) for f in feas_list]

    age = torch.tensor([1.], dtype=torch.float32).to(device)
    x = torch.rand(1, 1, 96, 114, 96).to(device)
    sex = torch.tensor([1], dtype=torch.int64).to(device)

    model = create_model(num_classes=2).to(device)
    y, features = model(x, feas_on_device, sex, age)

    print(y.shape)
    for f in features:
        print(f.shape)

    from torchinfo import summary
    summary(model, input_data=(x, feas_on_device, sex, age))
