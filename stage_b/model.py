"""ROI-based Swin-Temporal Transformer (Section III-B, Fig. 6).

Each of the T = 16 ROI frames goes through a shared Swin-T; global average
pooling over the 7x7 final tokens gives z_t in R^768. A learned positional
table is added, two post-norm Transformer encoder blocks (8 heads, FFN 3072,
GELU, dropout 0.1) mix the frames, the 16 outputs are mean-pooled and a
768 -> 384 -> 1 MLP gives the clip logit.
"""
import math

import timm
import torch
from torch import nn


def sinusoidal_table(length, dim):
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        w = self.score(x).softmax(dim=1)
        return (w * x).sum(dim=1)


class SwinTemporal(nn.Module):
    def __init__(self, backbone="swin_tiny_patch4_window7_224", pretrained=True,
                 num_frames=16, dim=768, depth=2, heads=8, ffn_dim=3072, dropout=0.1,
                 head_hidden=384, temporal_pool="mean", pos_encoding="learned"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        # the "hidden 384" ablation row needs a projection; 768 matches Swin-T and needs none
        self.proj = nn.Identity() if dim == feat_dim else nn.Linear(feat_dim, dim)

        self.temporal_pool = temporal_pool
        n_tokens = num_frames + (temporal_pool == "cls")
        if pos_encoding == "learned":
            self.pos = nn.Parameter(torch.empty(1, n_tokens, dim))
            nn.init.normal_(self.pos, std=0.02)
        else:
            self.register_buffer("pos", sinusoidal_table(n_tokens, dim).unsqueeze(0))
        if temporal_pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.cls, std=0.02)
        elif temporal_pool == "attn":
            self.attn_pool = AttentionPool(dim)

        layer = nn.TransformerEncoderLayer(dim, heads, ffn_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=False)
        self.temporal = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(dim, head_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(head_hidden, 1))

    def frame_features(self, clips):
        """(B, T, 3, H, W) -> (B, T, 768)"""
        b, t = clips.shape[:2]
        return self.backbone(clips.flatten(0, 1)).view(b, t, -1)

    def temporal_forward(self, z):
        x = self.proj(z)
        if self.temporal_pool == "cls":
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        x = self.temporal(x + self.pos)
        if self.temporal_pool == "mean":
            g = x.mean(dim=1)
        elif self.temporal_pool == "max":
            g = x.max(dim=1).values
        elif self.temporal_pool == "attn":
            g = self.attn_pool(x)
        else:
            g = x[:, 0]
        return self.head(g).squeeze(-1)

    def forward(self, clips):
        if self.backbone_frozen:
            with torch.no_grad():
                z = self.frame_features(clips)
        else:
            z = self.frame_features(clips)
        return self.temporal_forward(z)

    @property
    def backbone_frozen(self):
        return not any(p.requires_grad for p in self.backbone.parameters())

    def freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self


class FrameHead(nn.Module):
    """Temporary head for the adaptation phase (Section III-B-4): the frame
    descriptors are mean-pooled over the clip and classified by an MLP, so the
    backbone moves towards ROI crops without learning temporal structure. The
    head is discarded afterwards."""

    def __init__(self, backbone, hidden=384):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(backbone.num_features, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, 1))

    def forward(self, clips):
        b, t = clips.shape[:2]
        z = self.backbone(clips.flatten(0, 1)).view(b, t, -1)
        return self.head(z.mean(dim=1)).squeeze(-1)


def count_parameters(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
