from roboflow import Roboflow
import shutil


key_file = "roboflow_api_key.txt"
with open(key_file, 'r') as f:
    for line in f:
        api_key = line



rf = Roboflow(api_key=api_key)
project = rf.workspace("project-p5nyc").project("car-damages-v3gyz")
version = project.version(5)
dataset = version.download("png-mask-semantic")

shutil.move("Car-Damages-5", "./data")