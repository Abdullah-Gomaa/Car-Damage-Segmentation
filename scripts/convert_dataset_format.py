import os
import shutil
import cv2 as cv 
import numpy as np
import glob
import torch

 



class MaskToYoloV8Converter:
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = "data/Car-Damages-5"
        self.data_dir = os.path.join(f"{os.getcwd()}","data")
        self.dataset_dir = os.path.join(f"{os.getcwd()}", self.dataset_dir)
        self.yolo_dataset_dir = ""
        print(self.project_dir)
        print(self.dataset_dir)
        print(self.data_dir)
    def convert_mask(self,mask_path, min_area = 50,pixel_to_class=None):
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        h,w = mask.shape

        label_lines = []

        # Detect all unique pixel values in the mask (ignore background=0)
        unique_vals = np.unique(mask)
        unique_vals = [v for v in unique_vals if v != 0]
        if pixel_to_class is None:
            pixel_to_class = {v: i for i, v in enumerate(sorted(unique_vals))}
            
        for pixel_val, class_id in pixel_to_class.items():
            binary = np.where(mask == pixel_val, 255, 0).astype("uint8")
            # Find Contours
            contours, _ = cv.findContours(binary,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            for contour in contours:
                area = cv.contourArea(contour)

                if area < min_area:
                    continue

                # flatten contour to be in yolo format (x1 y1 x2 y2 .... xn yn)
                contour = contour.reshape(-1,2)

                # Normalize with width and height
                poly = []
                for x,y in contour:
                    x = x/w
                    y = y/h
                    poly.append(x)
                    poly.append(y)

                # Check if valid poly
                if len(poly) < 6: # Less than 3 points
                    continue
                
                # Add Polygon Segment 
                line = f"{class_id} " + " ".join(f"{point:.6f}" for point in poly)   
                
                label_lines.append(line)

        return label_lines
    



    def prepare_dirs(self,data_dir=None, nc = 2, class_names = ["No Damage", "Severe Damage"]):
        # root dir
        if data_dir == None:
            dataset_dir = self.data_dir
        
        dataset_dir = "dataset"
        if dataset_dir not in os.listdir(data_dir):
            os.mkdir(os.path.join(data_dir,dataset_dir))
            
        else:
            print("dataset dir already exists")
        dataset_dir = os.path.join(data_dir,dataset_dir)
    
        # sub dir
        sub_dirs = ["train", "valid", "test"]
        for dir in sub_dirs:
            if dir not in os.listdir(dataset_dir):
                os.mkdir(os.path.join(dataset_dir,dir))
                os.mkdir(os.path.join(dataset_dir,os.path.join(dir,"images")))
                os.mkdir(os.path.join(dataset_dir,os.path.join(dir,"labels")))
            else:
                print(f"{dir} dir already exists")

        # Create data.yaml file
        yaml_text = f""" 
    train: {(os.path.join(dataset_dir,os.path.join("train","images")))}
    val:   {os.path.join(dataset_dir,os.path.join("valid","images"))}
    test:  {os.path.join(dataset_dir,os.path.join("test","images"))}
    nc: {nc}
    names: {class_names}
            """
        with open(os.path.join(dataset_dir,'data.yaml'), 'w') as dy:
            dy.write(yaml_text) 

        self.yolo_dataset_dir = dataset_dir
        return self.yolo_dataset_dir

    def load_imgs(self,dataset_dir=None, from_yolo=False):
        imgs = {}

        if dataset_dir== None:
            dataset_dir = self.dataset_dir

        if from_yolo:
            train_imgs = glob.glob(os.path.join(dataset_dir,'train/images/*.jpg'))
            valid_imgs = glob.glob(os.path.join(dataset_dir,'valid/images/*.jpg'))
            test_imgs = glob.glob(os.path.join(dataset_dir,'test/images/*.jpg'))
        else:
            train_imgs = glob.glob(os.path.join(dataset_dir,'train/*.jpg'))
            valid_imgs = glob.glob(os.path.join(dataset_dir,'valid/*.jpg'))
            test_imgs = glob.glob(os.path.join(dataset_dir,'test/*.jpg'))

        imgs['train'] = sorted(train_imgs)
        imgs['valid'] = sorted(valid_imgs)
        imgs['test'] = sorted(test_imgs)


        print(len(imgs['train']))
        print(len(imgs['valid']))
        print(len(imgs['test']))

        return imgs

    def load_masks(self, dataset_dir= None, from_yolo=False):
        masks = {}
        if dataset_dir== None:
            dataset_dir = self.dataset_dir
        if from_yolo:
            train_masks = glob.glob(os.path.join(dataset_dir,'train/labels/*.png'))
            valid_masks = glob.glob(os.path.join(dataset_dir,'valid/labels/*.png'))
            test_masks = glob.glob(os.path.join(dataset_dir,'test/labels/*.png'))
        else: 
            train_masks = glob.glob(os.path.join(dataset_dir,'train/*.png'))
            valid_masks = glob.glob(os.path.join(dataset_dir,'valid/*.png'))
            test_masks = glob.glob(os.path.join(dataset_dir,'test/*.png'))
        masks['train'] = sorted(train_masks)
        masks['valid'] = sorted(valid_masks)
        masks['test'] = sorted(test_masks)

        print(len(masks['train']))
        print(len(masks['valid']))
        print(len(masks['test']))

        return masks


    def move_dataset(self, dataset_dir=None, dataset_yolo=None):
        if dataset_dir== None:
            dataset_dir = self.dataset_dir

        if dataset_yolo== None:
            dataset_yolo = self.yolo_dataset_dir

        imgs = self.load_imgs(dataset_dir)
        masks = self.load_masks(dataset_dir)

        dirs = ['train', 'valid', 'test']
        for dir in dirs:
            dst_i = os.path.join(dataset_yolo,f"{dir}/images") 
            for img in imgs[dir]:
                if img not in os.listdir(dst_i):
                    shutil.copy(img, os.path.join(dst_i))

            dst_m = os.path.join(dataset_yolo,f"{dir}/labels")
            for mask in masks[dir]:
                lines = self.convert_mask(mask, pixel_to_class={3:0, 4:1})

                base_name = os.path.splitext(os.path.basename(mask))[0]
                base_name = base_name.replace("_mask","")
                label_file = os.path.join(dst_m, f"{base_name}.txt")

                # Write lines to .txt
                with open(label_file, "w") as f:
                    for line in lines:
                        f.write(line + "\n")

    def convert(self, dataset_dir=None):
        if dataset_dir== None:
            dataset_dir = self.dataset_dir


        dataset_yolo = self.prepare_dirs()
        self.yolo_dataset_dir = dataset_yolo

        self.move_dataset(dataset_dir)



if __name__ == "__main__":
    convertor = MaskToYoloV8Converter()
    convertor.convert()