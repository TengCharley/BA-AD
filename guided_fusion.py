import torch
from torch import nn
from torch.nn import functional as F


def cosine_similarity_attn(q, k, eps=1e-8):
    b, c, d, h, w = q.shape
    q = q.reshape(b, c, -1)
    k = k.reshape(b, c, -1)
    dot = (q * k).sum(dim=-1)
    k_norm = torch.norm(q, p=2, dim=-1)
    v_norm = torch.norm(k, p=2, dim=-1)
    cos_sim = dot / (k_norm * v_norm + eps)
    attn = torch.sigmoid(cos_sim).reshape(b, c, 1, 1, 1)
    return attn


class SSMGT(nn.Module):
    def __init__(self, in_ch, mem_squeeze=True):
        super().__init__()
        self.pw_sel = nn.Conv3d(2 * in_ch, in_ch, kernel_size=1)
        self.pw_k = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.pw_v = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.pw_q = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.ln = nn.LayerNorm(in_ch)
        self.wa = nn.Parameter(torch.ones(in_ch, 1, 1, 1))
        self.wb = nn.Parameter(torch.zeros(in_ch, 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.ac = nn.SiLU()
        self.bn = nn.BatchNorm3d(in_ch)

        if mem_squeeze:
            self.squeeze = nn.Sequential(
                nn.Conv3d(in_ch, 2 * in_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(2 * in_ch),
                nn.SiLU()
            )
        else:
            self.squeeze = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_ch),
                nn.SiLU()
            )

    def forward(self, f_ad, f_age, mem):
        gate_sel = torch.cat((f_ad, f_age), dim=1)
        gate_sel = self.bn(self.pw_sel(gate_sel))
        mem = mem * self.sigmoid(gate_sel)

        q = self.pw_q(f_ad)
        k = self.pw_k(f_age)
        g_sim = cosine_similarity_attn(q, k)
        mem = mem + g_sim

        gate_ret = self.wa * f_age + self.wb * f_ad
        gate_ret = self.sigmoid(self.bn(gate_ret))

        v = self.pw_v(f_ad)
        v = mem * v * gate_ret
        return self.ac(self.bn(f_ad + v)), self.squeeze(mem)


class SEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=dim)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(dim * 2, dim, 1, bias=False)
        self.mlp = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f, s):
        f = self.avg_pool(f).flatten(1)
        s = self.emb(s)
        out = torch.cat([f, s], dim=1).unsqueeze(dim=-1)

        out = self.conv(out).squeeze(dim=-1)
        attn = self.sigmoid(self.ln(out))
        out = self.mlp(f) * attn
        return out


class AGF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduce = nn.Linear(dim, 1)
        self.expand = nn.Linear(2, dim)

    def forward(self, f_age, age_gap):
        out = self.reduce(f_age)
        out = torch.cat((out, age_gap), dim=1)
        out = f_age + self.expand(out)
        return out


class SAMCGA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ln = nn.LayerNorm(self.head_dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ac = nn.GELU(approximate='tanh')

        self.qv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.k1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.agf = AGF(dim)
        self.reduce = nn.Linear(dim, 1, bias=qkv_bias)
        self.expand = nn.Linear(2, dim, bias=qkv_bias)

        self.conv1 = nn.Conv1d(self.num_heads, self.num_heads, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv1d(self.num_heads, self.num_heads, 3, 1, 1, bias=False)

    def forward(self, f_ad, f_age, mem, age_gap):
        k1 = self.k1(mem.mean(dim=(2, 3, 4)))
        k1 = k1.reshape(-1, self.num_heads, self.head_dim)
        k1_norm = self.ln(k1)

        k2 = self.agf(f_age, age_gap)
        k2 = k2.reshape(-1, self.num_heads, self.head_dim)
        k2_norm = self.ln(k2)

        qv = self.qv(f_ad)
        q, v = torch.chunk(qv, chunks=2, dim=1)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        v = v.reshape(-1, self.num_heads, self.head_dim)
        q_norm = self.ln(q)

        out1 = F.scaled_dot_product_attention(q_norm, k1_norm, v)
        out2 = F.scaled_dot_product_attention(q_norm, k2_norm, v)

        out1 = self.ln(self.conv1(out1))
        out2 = self.ln(self.conv2(out2))
        out = self.ac(self.ln(out1 + out2 + v))
        out = out.reshape(-1, self.num_heads * self.head_dim)
        return self.ac(self.ln2(out))


if __name__ == '__main__':
    f_age = torch.rand(2, 3, 4, 5, 6)
    f_ad = torch.rand(2, 3, 4, 5, 6)
    mem = torch.rand(2, 3, 4, 5, 6)
    age_gap = torch.rand(2, 1)
    sex = torch.tensor([0, 1], dtype=torch.int64)
    # model = SSMGT(in_ch=3)
    # f_ad_out, mem_out = model(f_ad, f_age, mem)
    # print(f_ad_out.shape)
    # print(mem_out.shape)

    model = SEF(dim=3)
    y = model(f_age, sex)
    print(y.shape)

    # f_age = torch.mean(f_age, dim=(2, 3, 4))
    # f_ad = torch.mean(f_ad, dim=(2, 3, 4))
    # mem = torch.mean(mem, dim=(2, 3, 4))

    # model = AGF(dim=3)
    # y = model(f_age, age_gap)

    # model = SAMCGA(dim=3, num_heads=1)
    # y = model(f_ad, f_age, mem, age_gap)

    # print(y.shape)
