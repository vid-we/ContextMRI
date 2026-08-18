"""
unimed_clip_conditioner.py

'UniMed-CLIP encoder-swap' arm: same plain-text prompt string as the
baseline (utils.row_to_text_string), but tokenized and encoded by
UniMed-CLIP's text tower (BiomedBERT, contrastively pretrained on 5.3M
medical image-text pairs across 6 modalities including MRI - Khattak et al.,
arXiv:2412.10372) instead of OpenAI's frozen CLIP.

This is a pure encoder swap: only the module producing `encoder_hidden_states`
changes; the string fed into it and the UNet are untouched. Compare this to:
  - clip_xml    (utils.row_to_text_string_xml): keeps CLIP, changes the string
  - schema_conditioner.SchemaConditioner: removes the text encoder pathway
    entirely in favor of typed per-field embeddings
so each arm isolates a different variable relative to the clip_plain baseline.

WHY the base (not large) text encoder, and why this needs verifying on your
machine before trusting it silently:
  UniMed-CLIP ships 3 checkpoints (see the repo's README table). The
  ViT-B-16-quickgelu checkpoint pairs with BiomedBERT-BASE
  (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract), hidden_size=768 -
  the SAME width as UNet2DConditionModel's cross_attention_dim in
  config_mri.json, so token embeddings can be fed to the UNet with no added
  projection layer (verified against the repo's actual hf_model.py: when
  d_model == output_dim, HFTextEncoder.proj is nn.Identity()). The two
  ViT-L-14 checkpoints pair with BiomedBERT-LARGE (hidden_size=1024), which
  would need a trainable 1024->768 projection - use `--unimed_projection` if
  you go that route (a small extra trainable component, more comparable in
  spirit to the schema arm's approach than to a clean encoder swap).

SETUP (do this before importing this module):
    git clone https://github.com/mbzuai-oryx/UniMed-CLIP.git
    cd UniMed-CLIP
    wget https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt   # only if you re-train UniMed-CLIP yourself
    # for inference-only use (what we need here), instead download one of the
    # released UniMed-CLIP checkpoints linked in the repo README's
    # "Pre-trained Models" table (SharePoint link - no wget one-liner, this
    # is a manual download) and note its local path.
Then point --unimed_clip_repo_path at the cloned repo and
--unimed_clip_checkpoint at the downloaded .pt file (see train_mri.py args).

I have NOT been able to run this end-to-end myself (no internet access to
huggingface.co or the SharePoint checkpoint host in the environment I
implemented this in) - I validated the shape-level mechanics
(output_tokens=True -> per-token 768-d hidden states, CLS token excluded)
directly against the repo's real hf_model.py source with a locally
constructed BertConfig, but the actual checkpoint loading path below is
untested. Run tests/test_unimed_clip_smoke.py first and read any error
messages carefully - version drift in the repo (it's research code, not a
versioned pip package) is the most likely failure point.
"""

import os
import sys

import torch
import torch.nn as nn


class UniMedCLIPTextEncoder(nn.Module):
    def __init__(
        self,
        unimed_clip_repo_path: str,
        checkpoint_path: str,
        model_name: str = "ViT-B-16-quickgelu",
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        context_length: int = 77,
        device: str = "cpu",
        expected_hidden_dim: int = 768,
    ):
        super().__init__()

        src_path = os.path.join(unimed_clip_repo_path, "src")
        if not os.path.isdir(src_path):
            raise FileNotFoundError(
                f"{src_path} not found - unimed_clip_repo_path should point at your local "
                f"clone of https://github.com/mbzuai-oryx/UniMed-CLIP (the folder containing 'src/')."
            )
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # DW, 17.8.2026: PyTorch >=2.6 defaults torch.load(weights_only=True), which checkpoint doesn't fullfil
        #import numpy as np
        #torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        #torch.serialization.add_safe_globals([np.dtype])
        # Imported lazily (only once the repo's src/ is on sys.path) since
        # this is UniMed-CLIP's own vendored/modified open_clip, not the
        # pip-installable open_clip_torch package.
        from open_clip import create_model_and_transforms, get_mean_std

        mean, std = get_mean_std()
        model, _, _ = create_model_and_transforms(
            model_name,
            checkpoint_path,
            precision="fp32",
            device=device,
            force_quick_gelu=True,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=text_encoder_name,
        )

        # Only the text tower is needed for conditioning the MRI UNet - the
        # vision tower (image encoder) is unused here and can be dropped.
        del model.visual
        #DW self.text_tower = model.text
        self.text_tower = model.text_encoder

        # HFTextEncoder.forward() normally returns just the pooled/projected
        # contrastive embedding (one vector per prompt) - the same role
        # CLIP's `text_projection` output plays for image-text similarity.
        # We instead want the PER-TOKEN hidden states (like CLIPTextModel's
        # last_hidden_state) for cross-attention conditioning, which is what
        # output_tokens=True switches the forward() return value to.
        self.text_tower.output_tokens = True

        self.hidden_dim = self.text_tower.transformer.config.hidden_size
        if self.hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"UniMed-CLIP text encoder hidden_dim={self.hidden_dim} != "
                f"expected {expected_hidden_dim} (UNet cross_attention_dim). "
                f"You likely loaded a BiomedBERT-LARGE checkpoint (hidden_dim=1024). "
                f"Either switch to the base checkpoint+text encoder pair "
                f"(ViT-B-16-quickgelu / BiomedNLP-BiomedBERT-base-uncased-abstract), "
                f"or add a trainable projection layer before using this as encoder_hidden_states."
            )

        from open_clip import HFTokenizer
        self.tokenizer = HFTokenizer(text_encoder_name, context_length=context_length)
        self.context_length = context_length

        self.requires_grad_(False)
        self.text_tower.eval()

    @torch.no_grad()
    def forward(self, prompts, device=None, max_length=None):
        """Same role as train_mri.py's encode_text(): list[str] in, a
        (B, seq_len, hidden_dim) tensor out, ready for encoder_hidden_states.
        Frozen (no_grad), exactly like the OpenAI-CLIP path it replaces."""
        if device is None:
            device = next(self.parameters()).device
        input_ids = self.tokenizer(prompts, context_length=max_length or self.context_length)
        input_ids = input_ids.to(device)
        _pooled, tokens = self.text_tower(input_ids)
        return tokens
