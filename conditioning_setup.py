"""
conditioning_setup.py

Shared helper for recon_mri.py, inference.py, and any custom experiment
sweep script (e.g. run_knee_experiments_*.py): given --conditioning_mode and
its associated args, builds the kwargs needed to construct a
MRIDiffusionPipeline for that arm.

Not used by train_mri.py, which already builds these inline.

Usage in your own script:
    from conditioning_setup import add_conditioning_args, build_pipeline_kwargs
    parser = add_conditioning_args(argparse.ArgumentParser())
    ...
    args = parser.parse_args()
    ...
    pipeline_kwargs = build_pipeline_kwargs(args, args.pretrained_model_name_or_path,
                                             cross_attention_dim=768, device=device)
    pipeline = MRIDiffusionPipeline(unet=unet, scheduler=scheduler, config_path=..., **pipeline_kwargs)
"""
import torch
from transformers import CLIPTokenizer, CLIPTextModel

from pipeline_mri import TextEncoderConditioner, CallableConditioner
from unimed_clip_conditioner import UniMedCLIPTextEncoder
from schema_conditioner import SchemaConditioner


def add_conditioning_args(parser):
    """Adds --conditioning_mode and the unimed_clip_*/metadata_encoder_*
    args shared across recon_mri.py / inference.py / custom sweep scripts.
    Call this on your own argparse.ArgumentParser before parse_args()."""
    parser.add_argument(
        "--conditioning_mode", type=str, default="clip_plain",
        choices=["clip_plain", "clip_xml", "unimed_clip", "schema"],
        help="Which conditioning arm's checkpoint to load. clip_plain and clip_xml use "
             "the exact same OpenAI CLIP tokenizer/text_encoder - they only differ in the "
             "prompt STRING, which your script builds, not this helper.",
    )
    parser.add_argument("--unimed_clip_repo_path", type=str, default=None,
                         help="Local clone of https://github.com/mbzuai-oryx/UniMed-CLIP. Required for unimed_clip.")
    parser.add_argument("--unimed_clip_checkpoint", type=str, default=None,
                         help="Path to a downloaded UniMed-CLIP .pt checkpoint. Required for unimed_clip.")
    parser.add_argument("--unimed_clip_model_name", type=str, default="ViT-B-16-quickgelu")
    parser.add_argument("--unimed_clip_text_encoder_name", type=str,
                         default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
    parser.add_argument("--unimed_clip_context_length", type=int, default=77)
    parser.add_argument(
        "--metadata_encoder_checkpoint", type=str, default=None,
        help="Path to metadata_encoder_weights.pth (schema mode only). SchemaConditioner is "
             "trained from scratch alongside the UNet (train_mri.py --conditioning_mode schema), "
             "so - unlike CLIP/UniMed-CLIP - there's no separate pretrained checkpoint; this must "
             "be the file saved next to that run's unet_weights.pth.",
    )
    return parser


def build_pipeline_kwargs(args, pretrained_model_name_or_path, cross_attention_dim, device):
    """Returns a dict to unpack into MRIDiffusionPipeline(...).
    clip_plain / clip_xml -> {"text_encoder":..., "tokenizer":...}   (unchanged from before)
    unimed_clip / schema  -> {"conditioner": ...}
    """
    mode = args.conditioning_mode

    if mode in ("clip_plain", "clip_xml"):
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        text_encoder.eval().to(device)
        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    elif mode == "unimed_clip":
        if not args.unimed_clip_repo_path or not args.unimed_clip_checkpoint:
            raise ValueError(
                "--unimed_clip_repo_path and --unimed_clip_checkpoint are both required "
                "for --conditioning_mode unimed_clip."
            )
        unimed_encoder = UniMedCLIPTextEncoder(
            unimed_clip_repo_path=args.unimed_clip_repo_path,
            checkpoint_path=args.unimed_clip_checkpoint,
            model_name=args.unimed_clip_model_name,
            text_encoder_name=args.unimed_clip_text_encoder_name,
            context_length=args.unimed_clip_context_length,
            device="cpu",  # moved to `device` explicitly right after
            expected_hidden_dim=cross_attention_dim,
        )
        unimed_encoder.to(device)
        return {"conditioner": CallableConditioner(unimed_encoder)}

    elif mode == "schema":
        if not args.metadata_encoder_checkpoint:
            raise ValueError(
                "--metadata_encoder_checkpoint is required for --conditioning_mode schema "
                "(the metadata_encoder_weights.pth saved by the matching train_mri.py run)."
            )
        schema = SchemaConditioner(out_dim=cross_attention_dim)
        state_dict = torch.load(args.metadata_encoder_checkpoint, map_location="cpu")
        schema.load_state_dict(state_dict)
        schema.eval()
        schema.to(device)
        return {"conditioner": CallableConditioner(schema, empty_value={})}

    else:
        raise ValueError(f"Unknown conditioning_mode {mode!r}")
