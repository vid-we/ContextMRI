"""
schema_conditioner.py

'Arm B' conditioning pathway for the CLIP-string vs. XML-string vs.
schema-aware ablation described in the accompanying thesis chapter.

Where Arm A (utils.row_to_text_string_xml / row_to_text_string_skm_tea_xml)
keeps the frozen CLIP tokenizer + CLIPTextModel and only changes the string
that is tokenized, Arm B removes the free-text tokenizer entirely. Metadata
is kept as typed fields (categorical / continuous / a variable-length
pathology set) for as long as possible and each field is embedded with a
purpose-built encoder:

  - categorical fields (anatomy, contrast, sequence, sex)
        -> learned embedding table, the same mechanism DiT uses for class
           labels (Peebles & Xie, "Scalable Diffusion Models with
           Transformers", ICCV 2023) and FT-Transformer uses for categorical
           columns (Gorishniy et al., "Revisiting Deep Learning Models for
           Tabular Data", NeurIPS 2021).
  - continuous fields (slice, TR, TE, TI, flip angle, age)
        -> sinusoidal / Fourier features + MLP, the same functional form as
           the diffusion timestep embedding (Ho, Jain & Abbeel, "Denoising
           Diffusion Probabilistic Models", NeurIPS 2020), applied here to
           physical acquisition parameters instead of the timestep.
  - the pathology set (variable-length, open vocabulary)
        -> hashed multi-label embedding (feature hashing: Weinberger et al.,
           "Feature Hashing for Large Scale Multitask Learning", ICML 2009),
           mean-pooled into a single token, so no fixed pathology vocabulary
           has to be enumerated up front.

Every field produces exactly one 768-d token (matching
UNet2DConditionModel's cross_attention_dim in config_mri.json). The output
is a (batch, num_fields, 768) tensor that is passed to the UNet as
`encoder_hidden_states`, in the exact same role CLIP's (batch, 77, 768)
output played before. UNet2DConditionModel's cross-attention blocks
(SimpleCrossAttnDownBlock2D / SimpleCrossAttnUpBlock2D /
UNetMidBlock2DSimpleCrossAttn) attend over keys/values of arbitrary sequence
length, so this is a drop-in replacement for the conditioning tensor and
does NOT require changing the UNet architecture or config_mri.json.

Unlike the frozen CLIP encoder, SchemaConditioner is trainable (it is
initialized from scratch, since there is no pretrained checkpoint for it) and
must be optimized jointly with the UNet - see train_mri.py's
`--conditioning_mode schema` branch.
"""

import hashlib
import math

import torch
import torch.nn as nn


def _stable_hash(text: str, buckets: int) -> int:
    """Deterministic string hash (unlike Python's built-in hash(), which is
    randomized per-process via PYTHONHASHSEED unless disabled). Needed so the
    same pathology name always maps to the same bucket across training and
    inference / across separate processes."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


class FourierFeatures(nn.Module):
    """Sinusoidal embedding for a single normalized scalar, following the
    standard transformer/diffusion positional-encoding form used for
    timestep embeddings (Ho et al., 2020)."""

    def __init__(self, num_frequencies: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) already normalized to roughly [-1, 1]
        half = self.num_frequencies
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=x.device).float() / half
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)


class CategoricalField(nn.Module):
    """One learned embedding table per categorical field. Index 0 is reserved
    for 'missing' (dropped out / not provided, the schema analogue of CLIP's
    empty-string CFG conditioning), index 1 for out-of-vocabulary values."""

    MISSING, UNK = 0, 1

    def __init__(self, vocab, embed_dim: int, out_dim: int):
        super().__init__()
        self.vocab = {v: i + 2 for i, v in enumerate(vocab)}
        self.embed = nn.Embedding(len(vocab) + 2, embed_dim, padding_idx=self.MISSING)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def _index(self, value):
        if value is None:
            return self.MISSING
        return self.vocab.get(value, self.UNK)

    def forward(self, values, device) -> torch.Tensor:
        idx = torch.tensor([self._index(v) for v in values], device=device, dtype=torch.long)
        return self.proj(self.embed(idx))


class ContinuousField(nn.Module):
    """Fourier-feature encoder for one scalar field, with a learned 'missing'
    vector used when the field was dropped (mirrors the 50% MR-parameter
    dropout already used for the CLIP-string prompts in utils.py)."""

    def __init__(self, out_dim: int, num_frequencies: int = 32, value_range=None):
        super().__init__()
        self.fourier = FourierFeatures(num_frequencies)
        self.missing_embed = nn.Parameter(torch.zeros(out_dim))
        self.proj = nn.Sequential(
            nn.Linear(2 * num_frequencies, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        self.value_range = value_range  # (lo, hi) for normalization to [-1, 1]

    def forward(self, values, device) -> torch.Tensor:
        rows = []
        present_mask = []
        for v in values:
            present_mask.append(v is not None)
            rows.append(0.0 if v is None else float(v))

        vals = torch.tensor(rows, device=device, dtype=torch.float32)
        if self.value_range is not None:
            lo, hi = self.value_range
            vals = 2.0 * (vals - lo) / max(hi - lo, 1e-6) - 1.0

        feats = self.proj(self.fourier(vals))  # (B, out_dim)
        missing = torch.tensor(present_mask, device=device).logical_not().unsqueeze(-1)
        return torch.where(missing, self.missing_embed.unsqueeze(0).expand_as(feats), feats)


class PathologySetField(nn.Module):
    """Variable-length, open-vocabulary pathology set -> fixed-size hashed
    embedding table, mean-pooled into a single token (feature hashing;
    Weinberger et al., ICML 2009). Bucket 0 means 'no pathology reported'."""

    def __init__(self, out_dim: int, hash_buckets: int = 512):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.embed = nn.Embedding(hash_buckets + 1, out_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, pathology_lists, device) -> torch.Tensor:
        pooled = []
        for names in pathology_lists:
            if not names:
                idx = torch.tensor([0], device=device, dtype=torch.long)
            else:
                idx = torch.tensor(
                    [1 + _stable_hash(n, self.hash_buckets) for n in names], device=device, dtype=torch.long
                )
            pooled.append(self.embed(idx).mean(dim=0))
        return self.proj(torch.stack(pooled, dim=0))


class SchemaConditioner(nn.Module):
    """Full metadata -> token-sequence conditioner. See module docstring."""

    # Extend these vocabularies/ranges if training on more datasets/anatomies
    # than fastMRI knee+brain and SKM-TEA (Table 2 in Chung et al., 2025).
    CATEGORICAL_VOCABS = {
        "anatomy": ["Knee", "Brain"],
        "contrast": ["PD", "PDFS", "T1", "T1PRE", "T1POST", "T2", "FLAIR"],
        "sequence": ["Turbospinecho", "Flash", "Qdess"],
        "sex": ["M", "F"],
    }
    CONTINUOUS_RANGES = {
        "slice": (0, 160),
        "TR": (0, 16000),
        "TE": (0, 130),
        "TI": (0, 2500),
        "flip_angle": (0, 180),
        "age": (0, 100),
    }

    def __init__(self, out_dim: int = 768, num_frequencies: int = 32, hash_buckets: int = 512):
        super().__init__()
        self.out_dim = out_dim

        self.categorical = nn.ModuleDict(
            {name: CategoricalField(vocab, embed_dim=out_dim, out_dim=out_dim)
             for name, vocab in self.CATEGORICAL_VOCABS.items()}
        )
        self.continuous = nn.ModuleDict(
            {name: ContinuousField(out_dim, num_frequencies, value_range=rng)
             for name, rng in self.CONTINUOUS_RANGES.items()}
        )
        self.pathology = PathologySetField(out_dim, hash_buckets=hash_buckets)

        self.field_order = list(self.CATEGORICAL_VOCABS) + list(self.CONTINUOUS_RANGES) + ["pathology"]
        # Learned field-type embedding so cross-attention can distinguish
        # "this token is TR" from "this token is TE", etc. (fields have no
        # inherent order the way word position does in a sentence).
        self.field_type_embed = nn.Embedding(len(self.field_order), out_dim)

    def forward(self, metadata_batch, device=None) -> torch.Tensor:
        """
        metadata_batch: list[dict], one dict per sample, e.g. as produced by
        utils.row_to_metadata_dict / row_to_metadata_dict_skm_tea. A missing
        key or a None value both mean "field not provided".

        Returns (B, num_fields, out_dim) - pass directly as
        `encoder_hidden_states` to UNet2DConditionModel.
        """
        if device is None:
            device = self.field_type_embed.weight.device

        tokens = []
        for i, name in enumerate(self.field_order):
            if name == "pathology":
                tok = self.pathology([m.get("pathologies", []) for m in metadata_batch], device)
            elif name in self.categorical:
                tok = self.categorical[name]([m.get(name) for m in metadata_batch], device)
            else:
                tok = self.continuous[name]([m.get(name) for m in metadata_batch], device)
            tokens.append(tok + self.field_type_embed.weight[i].unsqueeze(0))

        return torch.stack(tokens, dim=1)

    @torch.no_grad()
    def null_tokens(self, batch_size: int, device=None) -> torch.Tensor:
        """All-fields-missing batch - the schema-conditioning analogue of
        encoding the empty string "" with CLIP for classifier-free guidance's
        unconditional branch."""
        return self.forward([{} for _ in range(batch_size)], device=device)
