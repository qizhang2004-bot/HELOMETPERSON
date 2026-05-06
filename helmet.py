from ultralytics import YOLO
import cv2
import os
import sys
import torch
import platform


def get_device():
    """根据系统自动选择设备：Mac优先MPS，Windows优先CUDA"""
    system = platform.system()

    if system == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print(f"系统: macOS → 使用 MPS (GPU加速)")
            return "mps"
        else:
            print(f"系统: macOS → MPS不可用，使用 CPU")
            return "cpu"

    elif system == "Windows":
        if torch.cuda.is_available():
            print(f"系统: Windows → 使用 CUDA GPU")
            return "cuda"
        else:
            print(f"系统: Windows → CUDA不可用，使用 CPU")
            return "cpu"

    else:
        if torch.cuda.is_available():
            print(f"系统: {system} → 使用 CUDA GPU")
            return "cuda"
        else:
            print(f"系统: {system} → 使用 CPU")
            return "cpu"


def get_model_path():
    """获取模型路径，兼容PyInstaller打包"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, 'exp.pt')


# 初始化模型
model_path = get_model_path()
device = get_device()
model = YOLO(model_path, task='detect')

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率（1920x1080）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 验证实际分辨率
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"模型: {model_path}")
print(f"设备: {device}")
print(f"请求分辨率: 1920x1080")
print(f"实际分辨率: {actual_w}x{actual_h}")
print("按 'q' 键退出\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每帧都检测，输入尺寸320（速度快）
    results = model(frame, imgsz=1056, stream=True, device=device)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow('安全帽检测', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
