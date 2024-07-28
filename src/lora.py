from ultralytics import YOLO
import torch

# Load your model
model = YOLO('model.pt')

# Assuming you've added LoRA modules to 'model' and named them with 'lora' in their names
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]

print(lora_params)
print(model.state_dict().keys())