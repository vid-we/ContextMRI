from diffusers import UNet2DConditionModel
import json
import torch

# Load config
with open("configs/unet/config_mri.json", "r") as f:  
    config = json.load(f)

# Instantiate model
model = UNet2DConditionModel.from_config(config)

# Calc parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
