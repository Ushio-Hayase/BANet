from ultralytics import YOLO


def main():
    model = YOLO("model.pt")

    outs = model(["src/test.jpg"])
    for out in outs:
        print(model.names[out.probs.top1])
        print(out.probs.top1conf.item())
    
if __name__ == "__main__":
    main()
