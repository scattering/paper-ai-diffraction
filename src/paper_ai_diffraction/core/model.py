import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

try:
    from torchtune.modules import RotaryPositionalEmbeddings
except Exception:
    RotaryPositionalEmbeddings = None

class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_rope=False):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, use_rope=use_rope)
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    # def forward(self, x):
    #     x = x + self.drop_path(self.attn(self.norm1(x)))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x
    
    def forward(self, x, return_attn=False):
        # compute attention output
        # attn_out = self.attn(self.norm1(x))

        if return_attn:
            attn_out, attn_map = self.attn(self.norm1(x), return_attn=True)
        else:
            attn_out = self.attn(self.norm1(x))
            attn_map = None

        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attn:
            return x, attn_map  # return the attention map, not projected output
        else:
            return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 use_rope=False):
        super(Attention, self).__init__()
        self.use_rope = use_rope

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        if self.use_rope:
            if RotaryPositionalEmbeddings is None:
                raise ImportError("use_rope=True requires torchtune with torchao available")
#            print("Using RoPE")
            self.rope = RotaryPositionalEmbeddings(self.head_dim)

    def forward(self, x, return_attn=False):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        if self.use_rope:
#            print(f"[RoPE] Rotary embedding applied")
            q, k = self.rope(q), self.rope(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # for inference
        if return_attn:
            # Return both the projected output and the raw attention weights
            return x, attn
        
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed1D(nn.Module):
    def __init__(self, spec_length, patch_size, embed_dim, in_chans=1):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = spec_length // patch_size

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        # Pad to multiple of patch size
        pad_len = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)

        x = self.proj(x)  # (B, embed_dim, num_patches)
        return x.transpose(1, 2)  # (B, num_patches, embed_dim)


'''
def Spectra_Embedding(x, spec_length, embed_dim):
    batch_size = x.shape[0]
    new_spec_length = (spec_length // embed_dim) * embed_dim
    x = x[:, :new_spec_length]
    x = torch.reshape(x, (batch_size, spec_length // embed_dim, embed_dim))
    return x

def Spectra_Embedding_old(x, spec_length, embed_dim):

    batch_size = x.shape[0]
    x = torch.reshape(x, (batch_size, spec_length // embed_dim, embed_dim))
    return x

def Spectra_Embedding_enlong(x, spec_length, embed_dim):
    batch_size = x.shape[0]
    remainder = spec_length % embed_dim
    if remainder != 0:
        pad = embed_dim - remainder
        x = F.pad(x, (0, pad))
    x = torch.reshape(x, (batch_size, -1, embed_dim))
    return x
'''
class MLPHead(nn.Module):
    """Optional non-linear head for ViT fine-tuning."""
    def __init__(self, embed_dim, num_classes, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    
class VIT(nn.Module):
    def __init__(self, spec_length=2000, patch_size=10, num_output=1,
                 embed_dim=40, depth=12, num_heads=2, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None, use_rope=False, use_mlp_head=False, mlp_head_hidden_dim=None,
                 use_physics_pe=False, physics_pe_mode="sin2theta", two_theta_min=5.0, two_theta_max=90.0,
                 physics_pe_scale=1.0, use_coordinate_channel=False, coordinate_mode="sin2theta"):

        # MSTransformer
        super(VIT, self).__init__()
        self.num_classes = num_output
        self.spec_length = spec_length
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.use_mlp_head = use_mlp_head
        self.mlp_head_hidden_dim = mlp_head_hidden_dim
        self.use_physics_pe = use_physics_pe
        self.physics_pe_mode = physics_pe_mode
        self.two_theta_min = two_theta_min
        self.two_theta_max = two_theta_max
        self.physics_pe_scale = physics_pe_scale
        self.use_coordinate_channel = use_coordinate_channel
        self.coordinate_mode = coordinate_mode

        in_chans = 2 if self.use_coordinate_channel else 1
        self.patch_embed = PatchEmbed1D(spec_length, patch_size, embed_dim, in_chans=in_chans)
        #self.num_patches = spec_length // patch_size
        # Calculate padded spec length
        padded_spec_length = (spec_length + patch_size - 1) // patch_size * patch_size
        self.num_patches = padded_spec_length // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for cls
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.use_rope = use_rope

        self.blocks = nn.Sequential(*[
            EncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                        norm_layer=norm_layer, act_layer=act_layer, use_rope=use_rope)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if self.use_physics_pe:
            self.physics_pe_mlp = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            patch_centers = self._build_patch_centers()
            self.register_buffer("physics_patch_coords", patch_centers, persistent=False)
        else:
            self.physics_pe_mlp = None
            self.register_buffer("physics_patch_coords", torch.empty(0), persistent=False)

        if self.use_coordinate_channel:
            point_coords = self._build_point_coords(self.num_patches * self.patch_size)
            self.register_buffer("coordinate_channel_coords", point_coords, persistent=False)
        else:
            self.register_buffer("coordinate_channel_coords", torch.empty(0), persistent=False)



        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        if self.use_mlp_head:
            self.head = MLPHead(embed_dim, num_output, hidden_dim=mlp_head_hidden_dim, drop=drop_ratio)
        else:
            self.head = nn.Linear(embed_dim, num_output)

    def _build_patch_centers(self):
        padded_spec_length = self.num_patches * self.patch_size
        two_theta = torch.linspace(self.two_theta_min, self.two_theta_max, padded_spec_length, dtype=torch.float32)
        patch_centers = []
        for idx in range(self.num_patches):
            start = idx * self.patch_size
            end = start + self.patch_size
            center_two_theta = two_theta[start:end].mean()
            theta_rad = torch.deg2rad(center_two_theta / 2.0)
            if self.physics_pe_mode == "q":
                coord = 4.0 * math.pi * torch.sin(theta_rad)
            elif self.physics_pe_mode == "theta":
                coord = center_two_theta
            else:
                coord = torch.sin(theta_rad) ** 2
            patch_centers.append(coord * self.physics_pe_scale)
        return torch.stack(patch_centers).unsqueeze(-1)

    def _build_point_coords(self, padded_spec_length):
        two_theta = torch.linspace(self.two_theta_min, self.two_theta_max, padded_spec_length, dtype=torch.float32)
        theta_rad = torch.deg2rad(two_theta / 2.0)
        if self.coordinate_mode == "q":
            coords = 4.0 * math.pi * torch.sin(theta_rad)
        elif self.coordinate_mode == "theta":
            coords = two_theta
        else:
            coords = torch.sin(theta_rad) ** 2
        coord_min = coords.min()
        coord_max = coords.max()
        if torch.isclose(coord_max, coord_min):
            return torch.zeros_like(coords)
        coords = 2.0 * (coords - coord_min) / (coord_max - coord_min) - 1.0
        return coords

    def _augment_input_channels(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if not self.use_coordinate_channel:
            return x

        pad_len = (self.patch_size - x.shape[-1] % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)
        coord_len = x.shape[-1]
        coords = self.coordinate_channel_coords
        if coords.numel() != coord_len:
            coords = self._build_point_coords(coord_len).to(device=x.device, dtype=x.dtype)
        else:
            coords = coords.to(device=x.device, dtype=x.dtype)
        coords = coords.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((x, coords), dim=1)

    def _get_position_embedding(self, x):
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        if not self.use_physics_pe:
            return pos_embed

        patch_count = x.size(1) - 1
        coords = self.physics_patch_coords
        if patch_count != coords.size(0):
            coords = F.interpolate(
                coords.transpose(0, 1).unsqueeze(0),
                size=patch_count,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        physics_patch_embed = self.physics_pe_mlp(coords.to(x.device, x.dtype)).unsqueeze(0)
        physics_cls_embed = torch.zeros((1, 1, self.embed_dim), device=x.device, dtype=x.dtype)
        physics_pos_embed = torch.cat((physics_cls_embed, physics_patch_embed), dim=1)
        return pos_embed + physics_pos_embed


    def forward(self, x, return_cls_embedding=False):
        x = self.patch_embed(self._augment_input_channels(x))  # (B, num_patches, embed_dim)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        x = x + self._get_position_embedding(x)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]

        return cls_out if return_cls_embedding else self.head(cls_out)

    def forward_with_attn(self, x, return_cls_embedding=False):
        """
        Forward pass that returns attention weights for CAM-style visualization.
        """
        x = self.patch_embed(self._augment_input_channels(x))  # (B, num_patches, embed_dim)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        x = x + self._get_position_embedding(x)
        x = self.pos_drop(x)

        attn_weights_list = []
        for blk in self.blocks:
            if hasattr(blk, "attn"):
                x, attn = blk(x, return_attn=True)  # block must support return_attn
                attn_weights_list.append(attn)
            else:
                x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]

        return (cls_out if return_cls_embedding else self.head(cls_out), attn_weights_list)


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def VIT_model(spec_length=2251,num_output: int = 1, patch_size=10, embed_dim=40, depth=12, num_heads=2, mlp_ratio=4.0, drop_ratio = 0., local_rank=0, use_rope=False, use_mlp_head=False, mlp_head_hidden_dim=None, use_physics_pe=False, physics_pe_mode="sin2theta", two_theta_min=5.0, two_theta_max=90.0, physics_pe_scale=1.0, use_coordinate_channel=False, coordinate_mode="sin2theta"):
    model = VIT(spec_length=spec_length,
                              patch_size=patch_size,
                              embed_dim=embed_dim,
                              depth=depth,
                              num_heads=num_heads,
                              num_output=num_output,
                              mlp_ratio=mlp_ratio,
                              drop_ratio=drop_ratio,
                              use_rope=use_rope,
                              use_mlp_head=use_mlp_head,
                              mlp_head_hidden_dim=mlp_head_hidden_dim,
                              use_physics_pe=use_physics_pe,
                              physics_pe_mode=physics_pe_mode,
                              two_theta_min=two_theta_min,
                              two_theta_max=two_theta_max,
                              physics_pe_scale=physics_pe_scale,
                              use_coordinate_channel=use_coordinate_channel,
                              coordinate_mode=coordinate_mode)

    return model


def adapt_patch_embed_input_channels(state_dict, model, zero_init_new_channels=True):
    """Expand legacy patch-embed weights when a checkpoint has fewer input channels."""
    key = "patch_embed.proj.weight"
    if key not in state_dict:
        return state_dict

    checkpoint_weight = state_dict[key]
    model_weight = model.state_dict().get(key)
    if model_weight is None or checkpoint_weight.shape == model_weight.shape:
        return state_dict

    if checkpoint_weight.dim() != 3 or model_weight.dim() != 3:
        return state_dict

    out_old, in_old, patch_old = checkpoint_weight.shape
    out_new, in_new, patch_new = model_weight.shape
    if out_old != out_new or patch_old != patch_new or in_old >= in_new:
        return state_dict

    expanded = model_weight.clone()
    expanded[:, :in_old, :] = checkpoint_weight
    if not zero_init_new_channels:
        expanded[:, in_old:, :] = checkpoint_weight[:, :1, :].expand(-1, in_new - in_old, -1)

    adapted = dict(state_dict)
    adapted[key] = expanded
    return adapted
