from ultralytics import YOLO
import os


project_dir = os.getcwd()
yolo_dataset_dir = os.path.join(project_dir,"data/dataset")
data_path = os.path.join(yolo_dataset_dir,'data.yaml')
model_path= os.path.join(project_dir,"models")



model = YOLO("yolov8n-seg.pt")
result = model.train(data=data_path,epochs=100,imgsz=640,batch=8,device=0, project=model_path)