import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from pipeline_mri import MRIDiffusionPipeline
import pandas as pd
from mri.utils import real_to_nchw_comp, clear
# DW
from utils import (
    row_to_text_string_skm_tea, row_to_text_string,
    row_to_text_string_xml, row_to_text_string_skm_tea_xml,
    row_to_metadata_dict, row_to_metadata_dict_skm_tea,
    user_input_to_metadata_dict, save_image,
)
from conditioning_setup import add_conditioning_args, build_pipeline_kwargs
# end

def main(args):
    
    #DW, 12.4.2026: added ()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if args.mri_type == "fastmri":
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="fastmri")
    elif args.mri_type == "skm-tea":
        unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="skm-tea")
    else:
        raise ValueError(f"Not supported mri data type {args.mri_type}")

    unet.eval()
    # DW
    unet.to(device)

    pipeline_kwargs = build_pipeline_kwargs(
        args, args.pretrained_model_name_or_path,
        cross_attention_dim=unet.config.cross_attention_dim, device=device,
    )
    # end
    pipeline = MRIDiffusionPipeline(
                unet=unet,
                scheduler=noise_scheduler,
                config_path=args.model_config,
                **pipeline_kwargs, # DW
            )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
    pipeline.scheduler.eta = args.eta

    # DW
    is_schema = args.conditioning_mode == "schema"
    is_xml = args.conditioning_mode == "clip_xml"
    # clip_plain and unimed_clip share the same plain-string formatter -
    # only the encoder differs, which build_pipeline_kwargs() already handled.
    string_formatter = (row_to_text_string_xml, row_to_text_string_skm_tea_xml) if is_xml \
        else (row_to_text_string, row_to_text_string_skm_tea)
    dict_formatter = (row_to_metadata_dict, row_to_metadata_dict_skm_tea)
    # end
    
    # Conditional MRI generation
    if args.use_auto:
        row_index = 150
        if args.mri_type == "fastmri":
            df = pd.read_csv("./assets/fastmri/knee/metadata_val.csv")
            row = df.iloc[row_index]
            prompts = dict_formatter[0](row, p=1.0) if is_schema else string_formatter[0](row, p=1.0)

        elif args.mri_type == "skm-tea":
            df = pd.read_csv("./assets/skm-tea/metadata_val.csv")
            row = df.iloc[row_index]
            prompts = dict_formatter[1](row, p=1.0) if is_schema else string_formatter[1](row, p=1.0)
    # DW
    elif is_schema:
        prompts = user_input_to_metadata_dict(
            anatomy=args.anatomy, slice_number=args.slice_number, contrast=args.contrast,
            pathology=args.pathology.split(",") if args.pathology else None,
            sequence=args.sequence, TR=args.TR, TE=args.TE, TI=args.TI, flip_angle=args.flip_angle,
        )
    #end
    else:
        prompts = args.meta_prompt
    print(f'Generated Image from metadata: {prompts} with CFG scale {args.cfg_scale}')
        
    generated_mri = pipeline(
        prompt=[prompts],
        guidance_scale=args.cfg_scale,
    )["images"]
    
    # Convert 2-channel image into grayscale
    output = np.abs(real_to_nchw_comp(generated_mri))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(output, f"{args.output_dir}/sample.png")
    print(f'Successfully generated image from metadata, saved in {args.output_dir}/sample.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="MRI Inference")
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--model_config', type=str, default="./configs/model_index.json")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="./MRI_checkpoint")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--mri_type', type=str, choices=["fastmri", "skm-tea"], default="fastmri")
    parser.add_argument("--use_auto", type=bool, default=True, help="Recommend to use auto generation of metadata to sync the training text distribution")
    parser.add_argument("--meta_prompt", type=str, default="", help="Use customized metadata prompt for generation (clip_plain/clip_xml/unimed_clip only - ignored for schema, use the individual --anatomy/--slice_number/... flags instead). Please match the format of the auto generation")
    # DW 
    # schema-mode manual entry (ignored unless --conditioning_mode schema and --use_auto False)
    parser.add_argument("--anatomy", type=str, default="Knee")
    parser.add_argument("--slice_number", type=float, default=0)
    parser.add_argument("--contrast", type=str, default="PD")
    parser.add_argument("--pathology", type=str, default=None, help="Comma-separated pathology names, e.g. 'Meniscus Tear,Bone-Subchondral edema'")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--TR", type=float, default=None)
    parser.add_argument("--TE", type=float, default=None)
    parser.add_argument("--TI", type=float, default=None)
    parser.add_argument("--flip_angle", type=float, default=None)
    parser = add_conditioning_args(parser)
    # end
    args = parser.parse_args()
    main(args)