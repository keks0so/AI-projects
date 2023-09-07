
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
results = model.train(data="cocopalm.yaml", epochs=600, device=0)