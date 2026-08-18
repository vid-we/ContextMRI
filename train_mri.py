import torch
from torch.nn import functional as F

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import json
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from dataset_mri import MRIDataset
# DW added
from unimed_clip_conditioner import UniMedCLIPTextEncoder

import diffusers
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
)
from diffusers.utils.torch_utils import is_compiled_module
import wandb

logging.basicConfig(level=logging.INFO)  # Set the level to INFO
logger = get_logger(__name__)

def collate_fn(examples):
    pixel_values = [example["image"] for example in examples]
    prompts = [example["prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch

def encode_text(prompt, tokenizer, model, device, max_length=None):
    if max_length is not None:
        tokenizer_max_length = max_length
    else:
        tokenizer_max_length = tokenizer.model_max_length

    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_max_length)
    input_ids = inputs.input_ids.to(device)
    attention_mask = None

    with torch.no_grad():
        text_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    return text_embeddings


def main(args):
    # Accelerate config
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # warning
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # DW: conditioning_mode for training:
    #   "clip_plain"   - baseline: frozen OpenAI CLIP text encoder, comma-separated string
    #   "clip_xml"     - keep CLIP, reformat the string as XML
    #   "unimed_clip"  - keep the SAME plain string as clip_plain, swap the frozen encoder for UniMed-CLIP's BiomedBERT text tower
    mode = args.conditioning_mode
    uses_clip = mode in ("clip_plain", "clip_xml")
    uses_unimed = mode == "unimed_clip"
    # end
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    unet_config_path = "./configs/unet/config_mri.json"
    with open(unet_config_path, "r") as f:
        unet_config = json.load(f) 

    unet = UNet2DConditionModel.from_config(unet_config)
    
    unet.requires_grad_(True)
    #DW added
    tokenizer, text_encoder, unimed_encoder = None, None, None

    if uses_clip:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        text_encoder.requires_grad_(False)  # frozen, as in the original ContextMRI recipe
    elif uses_unimed:
        unimed_encoder = UniMedCLIPTextEncoder(
            unimed_clip_repo_path=args.unimed_clip_repo_path,
            checkpoint_path=args.unimed_clip_checkpoint,
            model_name=args.unimed_clip_model_name,
            text_encoder_name=args.unimed_clip_text_encoder_name,
            context_length=args.unimed_clip_context_length,
            device="cpu",  # moved to accelerator.device explicitly below
            expected_hidden_dim=unet_config["cross_attention_dim"],
        )  # already frozen internally (requires_grad_(False) in its __init__)
    else:
        raise ValueError(f"Unknown conditioning_mode {mode!r}")
    # end

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    unet.to(accelerator.device)
    # DW added
    if uses_clip:
        text_encoder.to(accelerator.device, dtype=weight_dtype)  # frozen -> safe to cast permanently
    elif uses_unimed:
        unimed_encoder.to(accelerator.device, dtype=weight_dtype)  # frozen -> safe to cast permanently
    # end
    ema_unet = AveragedModel(unet, multi_avg_fn=get_ema_multi_avg_fn(0.999)) # For 0.999 decay



    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_to_save = model.state_dict()  # Save full transformer weights
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                # Pop weights to avoid duplicate saving
                weights.pop()

            # Save the full weights of each model
            torch.save(unet_to_save, os.path.join(output_dir, "unet_weights.pth"))

            # Save EMA weights
            ema_weights = ema_unet.module.state_dict() if hasattr(ema_unet, "module") else ema_unet.state_dict()
            torch.save(ema_weights, os.path.join(output_dir, "unet_ema_0.999_weights.pth"))

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected load model: {model.__class__}")

        # Load full weights from the saved state dicts
        unet_state_dict = torch.load(os.path.join(input_dir, "unet_weights.pth"))
        unet_.load_state_dict(unet_state_dict)
        # DW added
        to_cast = [unet_]
        # end

        # If mixed precision is being used, ensure the model weights are in float32
        if args.mixed_precision == "fp16":
            # Upcast trainable parameters to fp32
            cast_training_params(to_cast)

        del unet_state_dict
        torch.cuda.empty_cache()


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    trainable_models = [unet] # DW
    if args.mixed_precision == "fp16":
        cast_training_params(trainable_models, dtype=torch.float32)

    unet_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    unet_parameters_with_lr = {"params": unet_parameters, "lr": args.learning_rate}
    params_to_optimize = [unet_parameters_with_lr]
    
    if not args.optimizer.lower() == "adamw":
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        raise ValueError(f"Not supported {args.optimizer.lower()} optimizer")

    train_dataset = MRIDataset(
        metadata_file_knee=args.mri_metadata_dir_knee,
        metadata_file_brain=args.mri_metadata_dir_brain,
        train=True,
        conditioning_mode=args.conditioning_mode, # DW
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # DW, 14.8.2026 replaced
        # collate_fn=lambda examples: collate_fn(examples),
        collate_fn=collate_fn,
        # DW, 14.8.2026 from 8 to 0
        num_workers=8,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # DW added
    # clip_plain / clip_xml / unimed_clip: the encoder is frozen (already
    # placed on accelerator.device above), so only the UNet needs DDP wrapping.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # end

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "DeepFloyD-IF-M-MRI"
        accelerator.init_trackers(tracker_name, config=vars(args))

     # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if accelerator.is_main_process:
        wandb.login()
        wandb.init(project="ContextMRI")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[2])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        # unimed_encoder / text_encoder are frozen and stay in .eval() throughout
        # (unimed_encoder was already set to .eval() in its own __init__).

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [unet]
            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch["pixel_values"]
                prompts = batch["prompts"]

                # DW added
                if uses_clip:
                    # clip_plain / clip_xml: prompts is a list[str] (the only
                    # difference between the two arms happened upstream, in
                    # which utils.row_to_text_string* the dataset called).
                    condition_embeddings = encode_text(
                        prompts, tokenizer, text_encoder, device=accelerator.device, max_length=77
                    )
                else:
                    # unimed_clip: prompts is the SAME list[str] as clip_plain;
                    # only the encoder producing condition_embeddings differs.
                    condition_embeddings = unimed_encoder(
                        prompts, device=accelerator.device, max_length=args.unimed_clip_context_length
                    )

                condition_embeddings = condition_embeddings.contiguous()
                # end 
                model_input = pixel_values.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                model_pred = unet(noisy_model_input, timestep=timesteps, encoder_hidden_states=condition_embeddings).sample # DW

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean") # epsilon matching

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA
                if accelerator.sync_gradients:
                    ema_unet.update_parameters(unet)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit` 
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                wandb.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]})

            if global_step >= args.max_train_steps:
                break


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MRI foundation training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--mri_metadata_dir_knee",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--mri_metadata_dir_brain",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mri-result",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp32"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        # DW added
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="clip_plain",
        choices=["clip_plain", "clip_xml", "unimed_clip"],
        help=(
            "Which metadata-conditioning arm to train: "
            "'clip_plain' = original comma-separated string through frozen OpenAI CLIP (baseline); "
            "'clip_xml' = XML-structured string through the SAME frozen CLIP; "
            "'unimed_clip' = SAME plain string as clip_plain, through a frozen UniMed-CLIP "
            "(BiomedBERT) text encoder instead of OpenAI CLIP."
        ),
    )
    parser.add_argument(
        "--unimed_clip_repo_path", type=str, default=None,
        help="Path to your local clone of https://github.com/mbzuai-oryx/UniMed-CLIP "
             "(the folder containing 'src/'). Required if --conditioning_mode unimed_clip.",
    )
    parser.add_argument(
        "--unimed_clip_checkpoint", type=str, default=None,
        help="Path to a downloaded UniMed-CLIP .pt checkpoint (see the repo's README "
             "'Pre-trained Models' table). Required if --conditioning_mode unimed_clip.",
    )
    parser.add_argument(
        "--unimed_clip_model_name", type=str, default="ViT-B-16-quickgelu",
        help="open_clip model name matching the checkpoint. Use the base (not -L-14) variant "
             "to keep the text encoder at hidden_dim=768, matching cross_attention_dim with no projection.",
    )
    parser.add_argument(
        "--unimed_clip_text_encoder_name", type=str,
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        help="HF text encoder name matching --unimed_clip_checkpoint's text tower.",
    )
    parser.add_argument(
        "--unimed_clip_context_length", type=int, default=77,
        help="Token sequence length for UniMed-CLIP's tokenizer. 77 (matching CLIP's) keeps "
             "the comparison to clip_plain apples-to-apples; UniMed-CLIP itself defaults to 256.",
    )
    # end
  

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # DW edited
    if args.mri_metadata_dir_knee is None:
        raise ValueError(
            "Specify --mri_metadata_dir_knee. --mri_metadata_dir_brain is optional (omit it "
            "entirely for a knee-only run - MRIDataset already treats a missing brain CSV as "
            "'no brain data' rather than an error; the original stricter check that required "
            "both was relaxed here for exactly that use case)."
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
