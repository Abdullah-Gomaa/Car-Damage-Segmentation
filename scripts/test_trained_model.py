from ultralytics import YOLO
import os
from convert_dataset_format import MaskToYoloV8Converter

loader = MaskToYoloV8Converter()

project_dir = os.getcwd()
yolo_dataset_dir = os.path.join(project_dir,"data/dataset")
data_path = os.path.join(yolo_dataset_dir,'data.yaml')
model_path= os.path.join(project_dir,"models")

imgs = loader.load_imgs()

trained_model = YOLO(os.path.join(model_path, "train/weights/best.pt"))
res = trained_model.predict(source=imgs['test'],save=True, imgsz=640)