import os
import tkinter
import shutil
from tkinter.ttk import Progressbar
import torch
from ultralytics import YOLO

def classify(path: list[str]) -> list[tuple[str]]:
    """
    반환된 배열의 요소의 첫번째는 경로
    두번째는 분류된 클래스 이름
    세번째는 확률
    """
    results = model.predict(path) # 예측
    
    out = [] # 반환할 배열

    for i, r in enumerate(results):
        out.append((path[i],model.names[r.probs.top1],
                    str(r.probs.top1conf.item()))) # 분류한 라벨과 확률 추가
        
    return out

def load_file_path(root: str) -> list[str]:
    paths = []

    for (root, directories, files) in os.walk(root):
        for file in files:
            file_path = os.path.join(root, file)
            paths.append(file_path)
        
    return paths
    
def click():
    input_path = input_entry.get() 
    output_path = output_entry.get()

    progress.start(100)
    progress_var = "진행중"
    files_path = load_file_path(input_path)
    results = classify(files_path)

    for result in results:
        if float(result[3]) > 0.3: # 확률이 0.3이 넘는다면
            shutil.copy2(result[0], os.path.join(output_path, result[2])) # 파일 복사
        else:
            shutil.copy2(result[0], os.path.join(output_path, "기타"))
    
    progress.stop()
    progress_var = "완료"



if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    tk = tkinter.Tk()
    tk.title("BANet")
    
    model = YOLO(os.path.join(os.path.dirname(__file__),"model.pt")).to(device=device)

    progress_var = tkinter.Variable()
    progress_var = "대기중"

    progress_label = tkinter.Label(tk, textvariable=progress_var).grid(row=3, column=0)
    input_label = tkinter.Label(tk, text="분류할 이미지 폴더 경로").grid(row=0, column=0) 
    output_label = tkinter.Label(tk, text="저장할 폴더 경로").grid(row=1, column=0) 

    input_entry = tkinter.Entry(tk).grid(row=0, column=1) # 분류할 이미지 폴더 경로
    output_entry = tkinter.Entry(tk).grid(row=1, column=1) # 저장할 이미지 폴더 경로

    btn = tkinter.Button(tk, text="분류", command=click).grid(row=3, column=1) # 분류 시작 버튼
    progress = Progressbar(tk, maximum=100, mode="determinate").grid(row=4)

    tk.mainloop()