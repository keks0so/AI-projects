from ultralytics import YOLO
import os


#model_path = os.path.join( 'runs', 'detect', 'train3', 'weights', 'best.pt')
model_path = 'E:/Work/JUPYTER/Hands/runs/detect/train3/weights/best.pt'
model = YOLO(model_path)

#image = os.path.join('testdata', 'images', 'frame_13.jpg')
image = 'E:/Work/JUPYTER/Hands/testdata/images/frame_13.jpg'

results = model(image)[0]
print(f"Рук: {len(results)}")