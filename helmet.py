from ultralytics import YOLO
import cv2
import os
# m = 
model_path = os.path.join(os.path.dirname(__file__), "exp.torchscript")


model = YOLO(model_path) # 换成你的模型路径
cap = cv2.VideoCapture(0) # 0 是默认摄像头

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