"""VOSA model.

Stage 1 (VOSAStage1): single-epoch model. Signal-wise attention over 25 signals (+CLS)
per second, then time-wise attention over 30 s.
Stage 2 (VOSAStage2): multi-epoch head over the frozen Stage-1 features of epochs
t-1, t, t+1 (90 s); predicts the 30 per-second labels of epoch t.

Labels: 0 Normal, 1 Hypopnea, 2 Apnea. Layer names match the released checkpoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

ROW_HEIGHT = 43   # px per signal row
SEC_WIDTH = 64    # px per second (1920 / 30)
N_SEC = 30        # seconds per epoch


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        att = F.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,
                 forward_drop_p: float = 0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 64, n_classes: int = 3):
        super().__init__(nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes))


# ---------------------------------------------------------------------------
# Stage 1 : intra-epoch
# ---------------------------------------------------------------------------
class SignalPatchEmbedding(nn.Module):
    """Image -> (signal row x 1 s) patch tokens, plus one CLS token per second."""

    def __init__(self, emb_size=768, num_signals=25):
        super().__init__()
        self.em_projection = nn.Linear(ROW_HEIGHT * SEC_WIDTH, emb_size)
        self.positions = nn.Parameter(torch.randn((num_signals + 1, emb_size)))
        self.cls_token = nn.Parameter(torch.randn((1, N_SEC, emb_size)))
        self.ns = num_signals

    def forward(self, x: Tensor):                       # [B, 1080, 1920]
        b = x.shape[0]
        x = x[:, 0:self.ns * ROW_HEIGHT, :]
        patches = rearrange(x, "b (p h) (s w) -> b p s (h w)", p=self.ns, h=ROW_HEIGHT, s=N_SEC, w=SEC_WIDTH)
        emb = self.em_projection(patches)               # [B, P, 30, E]
        cls_tokens = repeat(self.cls_token, "n s e -> b n s e", b=b)
        emb = torch.cat([cls_tokens, emb], dim=1)       # [B, P+1, 30, E]
        emb += repeat(self.positions, "p e -> b p s e", b=b, s=N_SEC)
        return rearrange(emb, "b p s e -> (b s) p e")  # [B*30, P+1, E]


class TimePatchEmbedding(nn.Module):
    """Signal-wise features -> per-signal sequences of 30 s."""

    def __init__(self, emb_size=64, num_signals=25):
        super().__init__()
        self.em_projection = nn.Linear(768, emb_size)
        self.positions = nn.Parameter(torch.randn((N_SEC, emb_size)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # unused; kept for checkpoint keys
        self.ns = num_signals

    def forward(self, x: Tensor):                       # [B*30, P+1, 768]
        emb = self.em_projection(x)
        emb = rearrange(emb, "(b s) p e -> b s p e", s=N_SEC)
        emb += repeat(self.positions, "s e -> b s p e", b=emb.shape[0], p=self.ns + 1)
        return rearrange(emb, "b s p e -> (b p) s e")  # [B*(P+1), 30, 64]


class SelectCLS(nn.Module):
    """Keeps the CLS row -> [B, 30, E]. ``time_slice`` selects the centre epoch in Stage 2."""

    def __init__(self, num_signals=25, time_slice=None):
        super().__init__()
        self.np = num_signals + 1
        self.time_slice = time_slice

    def forward(self, x):
        if self.time_slice is not None:
            x = x[:, self.time_slice[0]:self.time_slice[1], :]
        x = rearrange(x, "(b p) s e -> b p s e", p=self.np)
        return x[:, 0]


class VOSAStage1(nn.Sequential):
    """[B, 1080, 1920] -> [B, 30, n_classes]"""

    def __init__(self, num_signals: int = 25, n_classes: int = 3, emb_size: int = 768,
                 depth: int = 3, drop_p: float = 0.):
        super().__init__(
            SignalPatchEmbedding(emb_size, num_signals),
            TransformerEncoder(drop_p=drop_p, forward_drop_p=drop_p),
            TimePatchEmbedding(64, num_signals),
            TransformerEncoder(depth, emb_size=64, drop_p=drop_p, forward_drop_p=drop_p),
            SelectCLS(num_signals),
            ClassificationHead(64, n_classes),
        )

    def get_intermediate_representation(self, x: Tensor) -> Tensor:
        """Signal-wise encoder output: [B*30, P+1, 768]."""
        return self[1](self[0](x))


# ---------------------------------------------------------------------------
# Stage 2 : inter-epoch
# ---------------------------------------------------------------------------
class ContextPatchEmbedding(nn.Module):
    def __init__(self, emb_size=64, num_signals=25, n_context=3):
        super().__init__()
        self.em_projection = nn.Linear(768, emb_size)
        self.positions = nn.Parameter(torch.randn((n_context * N_SEC, emb_size)))
        self.ns = num_signals

    def forward(self, x: Tensor):                       # [B*30, 3, P+1, 768]
        emb = self.em_projection(x)
        emb = rearrange(emb, "(b s) im p e -> b (im s) p e", s=N_SEC)  # [B, 90, P+1, 64]
        emb += repeat(self.positions, "s e -> b s p e", b=emb.shape[0], p=self.ns + 1)
        return rearrange(emb, "b s p e -> (b p) s e")


class VOSAStage2(nn.Sequential):
    """Stage-1 features of 3 epochs [B*30, 3, P+1, 768] -> [B, 30, n_classes] of the centre epoch"""

    def __init__(self, num_signals: int = 25, n_classes: int = 3, depth: int = 2, drop_p: float = 0.1):
        super().__init__(
            ContextPatchEmbedding(64, num_signals),
            TransformerEncoder(depth=depth, emb_size=64, drop_p=drop_p, forward_drop_p=drop_p),
            SelectCLS(num_signals, time_slice=(N_SEC, 2 * N_SEC)),
            ClassificationHead(64, n_classes),
        )


class Stage1Encoder(nn.Module):
    """Stage-1 feature extractor as a module (so it can be wrapped in nn.DataParallel)."""

    def __init__(self, stage1: VOSAStage1):
        super().__init__()
        self.stage1 = stage1

    def forward(self, x):
        return self.stage1.get_intermediate_representation(x)


class VOSA(nn.Module):
    """Images of epochs t-1, t, t+1 -> [B, 30, 3] logits of epoch t."""

    def __init__(self, num_signals: int = 25):
        super().__init__()
        self.encoder = Stage1Encoder(VOSAStage1(num_signals=num_signals))
        self.head = VOSAStage2(num_signals=num_signals)

    def forward(self, x_prev, x, x_next):
        feats = torch.stack([self.encoder(v) for v in (x_prev, x, x_next)], dim=1)
        return self.head(feats)


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def load_weights(model: nn.Module, path: str, map_location="cpu"):
    """Loads a checkpoint, with or without the DataParallel 'module.' prefix."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    ckpt = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(ckpt)
    return model
