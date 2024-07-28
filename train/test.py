import torch
from ultralytics import YOLO


def main():
    model = YOLO("model.onnx", task="classify")

    outs = model.predict(["src/test.jpg"], device='cpu')
    for out in outs:
        print(model.names[out.probs.top1])
        print(out.probs.top5conf)
    
if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
