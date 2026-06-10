import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from einops import rearrange, repeat

from timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timesformer.models.helpers import load_pretrained
from timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from timesformer.models.gaze_attention import (
    generate_attention_bias_from_gaze_coords,
    generate_aggregate_weight_from_gaze_coords,
)
from .build import MODEL_REGISTRY


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, attn_bias=None, attn_bias_weight=20.0):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias*attn_bias_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   #20  # Apply bias in log-space before softmax

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Attention_gaze(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)

        self.v2 = nn.Linear(dim, dim,bias=qkv_bias)
        self.proj_merg = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, attn_bias=None, attn_bias_weight=20.0):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        v2 = self.v2(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        v2 = v2[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x2 = (attn @ v2).transpose(1,2).reshape(B,N,C)

        x = torch.cat((x,x2),-1)
        x = self.proj_merg(x)

        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W, attn_bias=None, attn_bias_weight=20.0):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)

            if attn_bias is not None:
                attn_bias_weight = repeat(attn_bias_weight, 'b -> (b t)', t=T)  # [B*T]
            res_spatial = self.drop_path(self.attn(self.norm1(xs), attn_bias=attn_bias, attn_bias_weight=attn_bias_weight))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', gaze_att_weight=20.0, dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.gaze_att_weight = gaze_att_weight
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward_with_time(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        xt = x[:,1:,:]
        xt = rearrange(xt, 'b (h w t) m -> b t (h w) m',b=B,h=W,w=W,t=T)
        xt = torch.mean(xt, 2)  # averaging predictions for every frame

        return xt

    def forward_gaze_att_with_time(self, x, gaze_2D, layer = "all", std=1.0, gaze_weight=20.0):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)


        ## Gaze Attention Bias
        attn_bias = generate_attention_bias_from_gaze_coords(
            gaze_coords=gaze_2D,  # [B, T, 2] in pixel units (x, y)
            num_heads=self.num_heads,
            image_size=self.img_size,  # Assuming square images
            patch_size=16,  # Assuming square patches
            sigma=std  # Standard deviation for Gaussian in patch units
        )

        ## Attention blocks
        for idx, blk in enumerate(self.blocks):
            if layer == "all":
                x = blk(x, B, T, W, attn_bias=attn_bias)
            elif layer == "first":
                if idx == 0:
                    # print( "Vision transformer, gaze weight:", type(gaze_weight), gaze_weight)
                    attn_bias = attn_bias.to(x.device)
                    x = blk(x, B, T, W, attn_bias=attn_bias, attn_bias_weight=gaze_weight)
                else:
                    x = blk(x, B, T, W)
            elif layer == "aggregate":
                x = blk(x, B, T, W)
        
        if layer == "aggregate":
            x = x[:,1:,:]
            x = rearrange(x, 'b (h w t) m -> b t h w m',b=B,h=W,w=W,t=T)
            attn_bias = generate_aggregate_weight_from_gaze_coords(
                gaze_coords=gaze_2D,  # [B, T, 2] in pixel units (x, y)
                grid_num=W,  # Assuming square patches
                sigma=std  # Standard deviation for Gaussian in patch units
            )
            attn_bias = attn_bias.unsqueeze(-1)  # [B, T, H, W, 1]
            x = (x * attn_bias).mean(dim=[1, 2, 3]).unsqueeze(1)  # [B, 1, M]

        # ### Predictions for space-only baseline
        # if self.attention_type == 'space_only':
        #     x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)

        return x

    def forward_gaze_att(self, x, gaze_2D, layer = "all", std=1.0, gaze_weight=20.0):
        x = self.forward_gaze_att_with_time(x, gaze_2D, layer=layer, std=std, gaze_weight=gaze_weight)
        x = self.norm(x)[:, 0]
        return x

    def forward_and_layer_features(self, x, gaze_2D, layer_idx, layer = "all", std=1.0):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)


        ## Gaze Attention Bias
        attn_bias = generate_attention_bias_from_gaze_coords(
            gaze_coords=gaze_2D,  # [B, T, 2] in pixel units (x, y)
            num_heads=self.num_heads,
            image_size=self.img_size,  # Assuming square images
            patch_size=16,  # Assuming square patches
            sigma=std  # Standard deviation for Gaussian in patch units
        )

        ## Attention blocks
        for idx, blk in enumerate(self.blocks):
            if layer == "all":
                x = blk(x, B, T, W, attn_bias=attn_bias)
            elif layer == "first":
                if idx == 0:
                    x = blk(x, B, T, W, attn_bias=attn_bias, attn_bias_weight=self.gaze_att_weight)
                else:
                    x = blk(x, B, T, W)
            elif layer == "aggregate":
                x = blk(x, B, T, W)
            
            if idx == layer_idx:
                layer_feat = x.clone()
        
        
        if layer == "aggregate":
            x = x[:,1:,:]
            x = rearrange(x, 'b (h w t) m -> b t h w m',b=B,h=W,w=W,t=T)
            attn_bias = generate_aggregate_weight_from_gaze_coords(
                gaze_coords=gaze_2D,  # [B, T, 2] in pixel units (x, y)
                grid_num=W,  # Assuming square patches
                sigma=std  # Standard deviation for Gaussian in patch units
            )
            attn_bias = attn_bias.unsqueeze(-1)  # [B, T, H, W, 1]
            x = (x * attn_bias).mean(dim=[1, 2, 3]).unsqueeze(1)  # [B, 1, M]

        # ### Predictions for space-only baseline
        # if self.attention_type == 'space_only':
        #     x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
        x = self.norm(x)[:, 0]
        return x, layer_feat[:,0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

class ImageSeqEncoder(nn.Module):
    def __init__(self, feat_dim):
        super(ImageSeqEncoder, self).__init__()
        self.clip_proj = nn.Linear(feat_dim, 768)  # Project CLIP features to match ViT dimension
        self.pos_embed_clip = nn.Parameter(torch.randn(1, 1024, 768))  # Positional embedding for CLIP features
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            batch_first=True
        )
        self.images_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, clip_features):
        """
        clip_features: Tensor of shape (B, T, 512)
        Returns: Encoded image features of shape (B, T, 256)
        """
        vision_feat_crop = self.forward_feat(clip_features)  # [B, T, 768]
        vision_feat_crop = vision_feat_crop.mean(dim=1, keepdim=True)  # [B, 1, 768]

        return vision_feat_crop

    def forward_feat(self, clip_features):
        """
        clip_features: Tensor of shape (B, T, 512)
        Returns: Encoded image features of shape (B, T, 256)
        """
        B, T, _ = clip_features.shape
        vision_feat_crop = self.clip_proj(clip_features)

        # clip transformer
        pos_embed_clip = self.pos_embed_clip[:, :T, :]              # Shape: [T, D]           # Shape: [1, T, D]
        pos_embed_clip = pos_embed_clip.expand(B, -1, -1)        # Shape: [B, T, D]
        pos_embed_clip = pos_embed_clip.to(clip_features.device)
        vision_feat_crop += pos_embed_clip
        vision_feat_crop = self.images_encoder(vision_feat_crop) 

        return vision_feat_crop


class GazeEncoderClass(nn.Module):
    def __init__(self, length=15, class_num=6, feat_dim=768):
        super(GazeEncoderClass, self).__init__()
        self.gaze_proj = nn.Linear(length, feat_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, feat_dim))
        self.class_tokens = nn.Parameter(torch.randn(class_num, feat_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=8,
            batch_first=True
        )
        self.gaze_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward_features_img(self, gaze, img_tensor, scenario_id=torch.tensor([0])):
        """
        Returns: Encoded output of shape (B, 1024, feat_dim)
        """
        #gaze feat
        B, T, _ = gaze.shape
        gaze_feat = self.gaze_proj(gaze)
        pos_embed = self.pos_embed[:, :T+2, :]              # Shape: [T, D]           # Shape: [1, T, D]
        pos_embed = pos_embed.expand(B, -1, -1)        # Shape: [B, T, D]
        pos_embed = pos_embed.to(gaze_feat.device)

        class_tokens = self.class_tokens[scenario_id].unsqueeze(1)
        gaze_feat = torch.cat([class_tokens, gaze_feat, img_tensor], dim=1)
        gaze_feat += pos_embed

        gaze_feat = self.gaze_encoder(gaze_feat)

        return gaze_feat[:,0]
    
    def forward(self, gaze, scenario_id=torch.tensor([0])):
        """
        gaze_input: Tensor of shape (B, 1024, 15)
        Returns: Encoded output of shape (B, 1024, feat_dim)
        """
        gaze_feat = self.forward_feat(gaze, scenario_id)  # [B, T, 768]

        return gaze_feat[:,0]

    def forward_feat(self, gaze, scenario_id=torch.tensor([0])):
        """
        gaze_input: Tensor of shape (B, 1024, 15)
        Returns: Encoded output of shape (B, 1024, feat_dim)
        """
        #gaze feat
        B, T, _ = gaze.shape
        gaze_feat = self.gaze_proj(gaze)
        pos_embed = self.pos_embed[:, :T, :]              # Shape: [T, D]           # Shape: [1, T, D]
        pos_embed = pos_embed.expand(B, -1, -1)        # Shape: [B, T, D]
        pos_embed = pos_embed.to(gaze_feat.device)
        gaze_feat += pos_embed

        if B>1:
            class_tokens = self.class_tokens[scenario_id].unsqueeze(0)
            class_tokens = class_tokens.expand(B, -1, -1)   
            gaze_feat = torch.cat([class_tokens, gaze_feat], dim=1)
        else:
            class_tokens = self.class_tokens[scenario_id].unsqueeze(1)
            gaze_feat = torch.cat([class_tokens, gaze_feat], dim=1)

        gaze_feat = self.gaze_encoder(gaze_feat)  # [B, T, 768]

        return gaze_feat


@MODEL_REGISTRY.register()
class SkillSightTeacher(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(SkillSightTeacher, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.use_gaze_attention = cfg.MODEL.USE_GAZE_ATTENTION
        self.crop_gaze = cfg.MODEL.CROP_GAZE  # Whether to crop gaze coordinates
        self.gaze_att_layer = cfg.MODEL.GAZE_ATT_LAYER  # Layer to apply gaze attention
        self.vision_query = cfg.MODEL.VISION_QUERY  # 'with_time' or 'without_time'
        self.att_std = cfg.MODEL.ATT_STD  # Standard deviation for attention bias
        self.share_encoder = cfg.MODEL.SHARE_ENCODER
        self.use_gaze_seq = cfg.MODEL.USE_GAZE_SEQ  # Whether to use gaze sequence features
        self.gaze_head_length = cfg.MODEL.GAZE_HEAD_LENGTH  # Length of gaze feature vector
        self.num_cls = cfg.MODEL.NUM_CLASSES

        self.load_timesformer(cfg, **kwargs)
        # self.load_timesformer2(cfg, **kwargs)

        self.gaze_encoder = GazeEncoderClass(length=self.gaze_head_length) 
        self.sep_pred_head = False
        self.load_pred_head(sep_pred_head=self.sep_pred_head)
        if self.crop_gaze:
            self.load_crop_gaze_model(cfg, **kwargs)
            feat_dim = 768
            self.image_seq_encoder = ImageSeqEncoder(feat_dim=feat_dim)

    def load_pred_head(self, sep_pred_head=False):
        input_dim = 768 * (1 + self.crop_gaze + self.use_gaze_seq)  # 768 for vision, 768 for cropped vision, 768 for gaze sequence
        if not sep_pred_head:
            self.fc1 = nn.Linear(input_dim, 768)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(768, self.num_cls)
        else:
            self.fc1s = nn.ModuleList([nn.Linear(input_dim, 768) for _ in range(6)])
            self.relu = nn.ReLU()
            self.fc2s = nn.ModuleList([nn.Linear(768, self.num_cls) for _ in range(6)])

    def load_timesformer(self, cfg, **kwargs):
        patch_size = 16
        self.gaze_weight = nn.Parameter(torch.zeros(6), requires_grad=True)
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, gaze_att_weight=cfg.MODEL.GAZE_ATT_WEIGHT, **kwargs)
        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

        for param in self.model.head.parameters():
            param.requires_grad = False
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def load_crop_gaze_model(self, cfg, **kwargs):
        if cfg.MODEL.IMG_MODEL != "dinov2":
            raise ValueError("This cleaned SkillSight teacher copy only supports MODEL.IMG_MODEL=dinov2")
        self.img_model = "dinov2"
        self.clip_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def _select_gaze_sequence_features(self, gaze):
        # Keep the original selected gaze channels, then adapt to the
        # checkpoint/configured gaze encoder width. TimeSDinoGaze checkpoints
        # use 15 channels, while some generated CSVs expose fewer selected
        # columns after the original slicing.
        gaze = torch.cat([gaze[:, :, 0:13], gaze[:, :, 15:]], dim=-1)
        target_dim = self.gaze_encoder.gaze_proj.in_features
        cur_dim = gaze.shape[-1]
        if cur_dim < target_dim:
            gaze = F.pad(gaze, (0, target_dim - cur_dim))
        elif cur_dim > target_dim:
            gaze = gaze[:, :, :target_dim]
        return gaze

    def get_forward_features(self, x, x_cropped, gaze, scenario_id):
        B, T, _ = gaze.shape

        if self.use_gaze_attention:
            gaze_weight = F.relu(self.gaze_weight[scenario_id])
            # print( "Vision transformer gaze, gaze weight:", type(gaze_weight), gaze_weight)
            vision_feat = self.model.forward_gaze_att(x, gaze[:,:,-3:-1], layer = self.gaze_att_layer, std = self.att_std, gaze_weight = gaze_weight)
            vision_feat = vision_feat.unsqueeze(1)
        else:
            vision_feat = self.model.forward_features(x)
            vision_feat = vision_feat.unsqueeze(1)

        if self.crop_gaze:
            # clip
            b, c, t, h, w = x_cropped.shape
            x_reshaped = x_cropped.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            vision_feat_crop = self.clip_model(x_reshaped)
            d = vision_feat_crop.shape[-1]
            vision_feat_crop = vision_feat_crop.reshape(b, t, d)
            if len(scenario_id) > 1:
                scenario_id = scenario_id[0]
            vision_feat_crop = self.image_seq_encoder(vision_feat_crop)  # [B, 1, 768]

        #gaze feat
        # gaze_feat = self.gaze_encoder.forward_features(gaze, scenario_id)  # [B, T, 768]
        gaze = self._select_gaze_sequence_features(gaze)
        gaze_feat = self.gaze_encoder(gaze, scenario_id)


        x = vision_feat[:,0]
        if self.crop_gaze:
            x = torch.cat((x, vision_feat_crop[:,0]), dim=-1)
        if self.use_gaze_seq:
            x = torch.cat((x, gaze_feat), dim=-1)

        if self.sep_pred_head:
            x = self.fc1s[scenario_id](x)  # [B, 512]
            x = self.relu(x)
            x = self.fc2s[scenario_id](x)
        else:
            x = self.fc1(x)  # [B, 512]
            x = self.relu(x)
            x = self.fc2(x)
        return vision_feat[:,0], gaze_feat, vision_feat_crop[:,0], x

    def forward_features(self, x, x_cropped, gaze, scenario_id):
        B, T, _ = gaze.shape

        if self.use_gaze_attention:
            gaze_weight = F.relu(self.gaze_weight[scenario_id])
            vision_feat = self.model.forward_gaze_att(x, gaze[:,:,-3:-1], layer = self.gaze_att_layer, std = self.att_std, gaze_weight = gaze_weight)
            vision_feat = vision_feat.unsqueeze(1)
        else:
            vision_feat = self.model.forward_features(x)
            vision_feat = vision_feat.unsqueeze(1)

        if self.crop_gaze:
            # clip
            b, c, t, h, w = x_cropped.shape
            x_reshaped = x_cropped.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            vision_feat_crop = self.clip_model(x_reshaped)
            d = vision_feat_crop.shape[-1]
            vision_feat_crop = vision_feat_crop.reshape(b, t, d)
            if len(scenario_id) > 1:
                scenario_id = scenario_id[0]
            vision_feat_crop = self.image_seq_encoder(vision_feat_crop)  # [B, 1, 768]

        #gaze feat
        # gaze_feat = self.gaze_encoder.forward_features(gaze, scenario_id)  # [B, T, 768]
        if self.use_gaze_seq:
            gaze = self._select_gaze_sequence_features(gaze)
            gaze_feat = self.gaze_encoder(gaze, scenario_id)


        x = vision_feat[:,0]
        if self.crop_gaze:
            x = torch.cat((x, vision_feat_crop[:,0]), dim=-1)
        if self.use_gaze_seq:
            x = torch.cat((x, gaze_feat), dim=-1)

        if self.sep_pred_head:
            x = self.fc1s[scenario_id](x)  # [B, 512]
        else:
            x = self.fc1(x)  # [B, 512]
        return x

    def forward_and_layer_features(self, x, x_cropped, gaze, scenario_id, layer_idx):
        B, T, _ = gaze.shape

        gaze_weight = F.relu(self.gaze_weight[scenario_id])
        vision_feat, layer_feat = self.model.forward_and_layer_features(x, gaze[:,:,-3:-1], layer_idx, layer = self.gaze_att_layer, std = self.att_std, gaze_weight = gaze_weight)
        vision_feat = vision_feat.unsqueeze(1)
        
        if self.crop_gaze:
            # clip
            b, c, t, h, w = x_cropped.shape
            x_reshaped = x_cropped.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            vision_feat_crop = self.clip_model(x_reshaped)
            d = vision_feat_crop.shape[-1]
            vision_feat_crop = vision_feat_crop.reshape(b, t, d)
            if len(scenario_id) > 1:
                scenario_id = scenario_id[0]
            vision_feat_crop = self.image_seq_encoder(vision_feat_crop)  # [B, 1, 768]

        #gaze feat
        # gaze_feat = self.gaze_encoder.forward_features(gaze, scenario_id)  # [B, T, 768]
        gaze_feat = self.gaze_encoder(gaze, scenario_id)


        x = vision_feat[:,0]
        if self.crop_gaze:
            x = torch.cat((x, vision_feat_crop[:,0]), dim=-1)
        if self.use_gaze_seq:
            x = torch.cat((x, gaze_feat), dim=-1)

        x = self.fc1(x)  # [B, 512]
        x = self.relu(x)
        x = self.fc2(x)
        return x, layer_feat

    def forward(self, x, x_cropped, gaze, scenario_id):
        x = self.forward_features(x, x_cropped, gaze, scenario_id)
        x = self.relu(x)
        if self.sep_pred_head:
            x = self.fc2s[scenario_id](x)
        else:
            x = self.fc2(x)
        return x

    def forward_and_features(self, x, x_cropped, gaze, scenario_id):
        """
        For debugging purposes, returns the features before the final classification layer.
        """
        feat = self.forward_features(x, x_cropped, gaze, scenario_id)
        x = self.relu(feat)
        x = self.fc2(x)
        return x, feat

