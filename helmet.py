from ultralytics import YOLO
import cv2
import os
import sys

def get_model_path():
    """获取模型路径，兼容PyInstaller打包"""
    if getattr(sys, 'frozen', False):
        # PyInstaller打包后，从临时目录读取
        base_path = sys._MEIPASS
    else:
        # 正常运行，从脚本所在目录读取
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, 'exp.torchscript')

model_path = get_model_path()
model = YOLO(model_path, task='detect')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated = r.plot()
        cv2.imshow('安全帽检测', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
