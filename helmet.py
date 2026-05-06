from ultralytics import YOLO
import cv2
import os
import sys
import platform


def get_device():
    """根据系统自动选择设备"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        if sys.platform == "darwin" and hasattr(sys, ".version_info"):
            try:
                import torch
                if torch.backends.mps.is_available():
                    print(f"使用 GPU 加速 (MPS)")
                    return "mps"
            except:
                pass
        print("使用 CPU")
        return "cpu"
    elif system == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                print(f"使用 GPU 加速 (CUDA)")
                return "cuda"
        except:
            pass
        print("使用 CPU")
        return "cpu"
    else:
        print("使用 CPU")
        return "cpu"


def get_model_path():
    """获取模型路径，兼容PyInstaller打包"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, 'exp.pt')


def open_camera():
    """尝试不同的摄像头后端"""
    # 尝试默认后端
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    # Windows上尝试DirectShow
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    return cap


# ===== 启动信息 =====
print("=" * 50)
print("安全帽检测系统 启动中...")
print("=" * 50)

# 初始化模型
model_path = get_model_path()
print(f"模型路径: {model_path}")

device = get_device()

if not os.path.exists(model_path):
    print(f"\n❌ 错误：找不到模型文件 {model_path}")
    print("请确保 exp.pt 文件在程序同目录下")
    input("\n按回车退出...")
    sys.exit(1)

print(f"正在加载模型...")

try:
    model = YOLO(model_path, task='detect')
    print(f"✅ 模型加载成功")
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    input("\n按回车退出...")
    sys.exit(1)

# 打开摄像头
print("\n正在打开摄像头...")
cap = open_camera()

if not cap.isOpened():
    print("\n❌ 错误：无法打开摄像头！")
    print("请检查：")
    print("  1. 摄像头是否连接")
    print("  2. 是否有其他程序正在使用摄像头")
    print("  3. 摄像头驱动是否正常")
    input("\n按回车退出...")
    sys.exit(1)

# 获取摄像头信息
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"✅ 摄像头已打开")
print(f"   分辨率: {actual_w} x {actual_h}")
print(f"   FPS: {fps}")
print(f"   设备: {device}")
print("\n" + "=" * 50)
print("检测中... 按 'q' 键退出")
print("=" * 50)

frame_count = 0
fps_display = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("\n⚠️ 摄像头读取失败，重新尝试...")
        continue
    
    frame_count += 1
    
    # 执行检测
    results = model(frame, imgsz=640, stream=True, device=device)
    
    annotated_frame = frame.copy()
    for r in results:
        annotated_frame = r.plot()
        # 打印检测结果
        if r.boxes is not None and len(r.boxes) > 0:
            labels = r.names
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = labels[cls]
                if frame_count % 30 == 0:  # 每30帧打印一次
                    print(f"检测到: {label} ({conf:.0%})")
    
    cv2.imshow('安全帽检测', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n已退出")
