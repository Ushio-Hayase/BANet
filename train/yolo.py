from ultralytics import YOLO

def train():
    model = YOLO("src/yolov8x-cls.pt")
    model.task = "classify"
    model.train(data="src/data/dataset", imgsz = 640,epochs=19,batch=0.8,device=0)

def export():
    model = YOLO("model.pt")
    model.export(format='torchscript')

if __name__ == "__main__":
    export()
