import ultralytics
import torch

model = ultralytics.YOLO("./model.pt")

example = torch.rand(1, 3, 640, 640)

traceed_script = torch.jit.trace(model.model, example)

traceed_script.save("torchscript_model.pt")