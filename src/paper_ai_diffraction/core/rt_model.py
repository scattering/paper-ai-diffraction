"""
Regular Transformer (RT) for XRD spectra — supplemental provenance.

Adapted from the user's `flash_attn_version/model.py` (the working copy used to
train the supplemental RT checkpoints). Differences from the package's
`core.model.VIT`:
  - Per-point linear embedding + average-pool head (no patches, no CLS token).
  - Optional RoPE positional encoding via a custom rotary module.
  - Dual attention paths: Flash-Attention 2 (training) or eager attention
    (inference / interpretability — exposes `last_attn_weights`).

For attention visualisation, instantiate with `use_flash_attn=False`; Flash
Attention never materialises the full attention matrix.
"""

import math
import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class CustomRotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device)
        freqs = torch.outer(position, self.inv_freq.to(x.dtype))
        cos = torch.cos(freqs).unsqueeze(0)
        sin = torch.sin(freqs).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        return x_rot.flatten(start_dim=-2)


class MultiHeadFlashAttentionWithRoPE(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.k_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.v_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout_p = dropout

    def forward(self, x, rope_module=None):
        B, T, C = x.shape
        q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(B, T, self.num_heads, self.head_dim)

        if rope_module is not None:
            qs, ks = [], []
            for h in range(self.num_heads):
                qs.append(rope_module(q[:, :, h, :]))
                ks.append(rope_module(k[:, :, h, :]))
            q = torch.stack(qs, dim=2)
            k = torch.stack(ks, dim=2)

        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=None, causal=False,
        )
        return self.out_linear(out.reshape(B, T, C))


class MultiHeadAttentionWithRoPE(nn.Module):
    """Eager attention with RoPE — stores `last_attn_weights` for interpretability."""

    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.out_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, rope_module=None):
        B, T, C = x.shape
        q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_module is not None:
            qs, ks = [], []
            for h in range(self.num_heads):
                qs.append(rope_module(q[:, h, :, :]))
                ks.append(rope_module(k[:, h, :, :]))
            q = torch.stack(qs, dim=1)
            k = torch.stack(ks, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.last_attn_weights = attn.detach()

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_linear(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout, use_rope=False, use_flash_attn=True):
        super().__init__()
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        if use_rope:
            if self.use_flash_attn:
                self.attention = MultiHeadFlashAttentionWithRoPE(embed_size, heads, dropout)
            else:
                self.attention = MultiHeadAttentionWithRoPE(embed_size, heads, dropout)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True
            )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_module=None):
        if self.use_rope and rope_module is not None:
            attn_out = self.attention(x, rope_module)
        elif not self.use_rope:
            attn_out, _ = self.attention(x, x, x)
        else:
            attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, ff_dim, num_layers,
                 mlp_units, dropout, n_classes, pos_encoding="sinusoidal",
                 use_flash_attn=True):
        super().__init__()
        self.embedding = nn.Linear(1, embed_size)

        self.pos_encoding_type = pos_encoding
        if pos_encoding == "rope":
            head_dim = embed_size // num_heads
            assert head_dim % 2 == 0
            self.pos_encoding = CustomRotaryPositionalEmbedding(head_dim, max_seq_len=input_dim)
        else:
            self.pos_encoding = None

        use_rope = (pos_encoding == "rope")
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_dim, dropout,
                             use_rope=use_rope, use_flash_attn=use_flash_attn)
            for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, mlp_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_units, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        if self.pos_encoding_type == "rope":
            for block in self.transformer_blocks:
                x = block(x, self.pos_encoding)
        else:
            for block in self.transformer_blocks:
                x = block(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.mlp(x)


def transformer_model(spec_length=2251, num_output=1, ff_dim=64, embed_dim=40,
                      depth=12, num_heads=2, mlp_units=64, dropout=0.0,
                      pos_encoding="sinusoidal", use_flash_attn=True):
    """Construct the supplemental Regular Transformer.

    Defaults match the user's training rig. For interpretability, pass
    `use_flash_attn=False` so attention weights are materialised.
    """
    return TimeSeriesTransformer(
        input_dim=spec_length,
        embed_size=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=depth,
        mlp_units=mlp_units,
        dropout=dropout,
        n_classes=num_output,
        pos_encoding=pos_encoding,
        use_flash_attn=use_flash_attn,
    )
