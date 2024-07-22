from ultralytics import YOLO


def main():
    model = YOLO("src/best.pt")

    outs = model(["src/test.jpg", "src/test1.jpg", "src/test2.jpg", "src/test3.jpg"])
    for out in outs:
        print(model.names[out.probs.top1])
        print(out.probs.top1conf.item())
    
if __name__ == "__main__":
    main()
