from src.detection.entity import DataIngestionConfig
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import os
import cv2
import json
import shutil
from tqdm import tqdm
from src.detection import logging

class DataIngestion:
    def __init__(self, DataIngestionConfig =  DataIngestionConfig):
        logging.info("-" * 20  + f'Data Ingestion stage started '+ "-" * 20)   
        self.dataingestion = DataIngestionConfig
        self.curr_dir = Path(os.getcwd())
        self.input_dir = Path(self.dataingestion.temp_dir)
        self.tranform_dir = Path(self.dataingestion.transform_dir)


    def convert_bbox_coco2yolo(self,img_width, img_height, bbox):
        
        x_tl, y_tl, w, h = bbox

        dw = 1.0 / img_width
        dh = 1.0 / img_height

        x_center = x_tl + w / 2.0
        y_center = y_tl + h / 2.0

        x = x_center * dw
        y = y_center * dh
        w = w * dw
        h = h * dh

        return [x, y, w, h]


    
    def convert_coco_json_to_yolo_txt(self):
     
        for files in os.listdir(self.input_dir):
            file_path = os.path.join(self.curr_dir,self.input_dir,files)
            image_dir = os.path.join(file_path,'images')
            json_dir = os.path.join(file_path,'annotations')
            json_file_path = os.path.join(json_dir,'instances_default.json')        
            output_file_path = os.path.join(self.curr_dir,self.tranform_dir,files)
            output_file_path1 = os.path.join(output_file_path,'output')
            output_file_path2 = os.path.join(output_file_path,'images')
            os.makedirs(output_file_path2, exist_ok=True)
            os.makedirs(output_file_path1, exist_ok=True)
            os.makedirs(output_file_path,exist_ok=True)
                
            with open(json_file_path) as f:
                json_data = json.load(f)
                if len(os.listdir(output_file_path1))==0:

                    # write _darknet.labels, which holds names of all classes (one class per line)
                    label_file = Path(os.path.join(output_file_path, "_darknet.labels"))
                    with open(label_file, "w") as f:
                        print('\n')
                        for category in tqdm(json_data["categories"], desc="Categories"):
                            category_name = category["name"]
                            print(f"******************Conversion Started for {category_name}************************")
                            f.write(f"{category_name}\n")

                    for image in tqdm(json_data["images"], desc=f"Annotation txt for {category_name} image"):
                        img_id = image["id"]
                        img_name = image["file_name"]
                        img_width = image["width"]
                        img_height = image["height"]

                        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
                        anno_txt = os.path.join(output_file_path1, img_name.split(".")[0] + ".txt")
                        with open(anno_txt, "w") as f:
                            for anno in anno_in_image:
                                category = anno["category_id"]
                                bbox_COCO = anno["bbox"]
                                x, y, w, h = self.convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    print(f"******************Conversion to YOLO format finished for {category_name}************************")
                    print('\n')
                else:
                    print('Files already converted to YOLO format')       


    def file_migrations(self):
        for files in os.listdir(self.input_dir):
            file_path = os.path.join(self.curr_dir,self.input_dir,files)
            image_dir = os.path.join(file_path,'images')
            image_final_path = os.path.join(self.curr_dir,self.tranform_dir,files,'images')
            
            if len(os.listdir(image_final_path)) == 0:                
                for image_name in os.listdir(image_dir):
                    old_image_path = os.path.join(image_dir, image_name)
                    os.makedirs(image_final_path,exist_ok=True)
                    shutil.copy(old_image_path, image_final_path)
            else:
                print('Image already exists')        
            # logging.info(f'loading mask images  for {folder_name} at directory {self.input_dir} ') 
            # logging.info(f'creating the output path for {folder_name} at directory {file_dir}')
            # logging.info(f'Reading each files from {folder_name} folder')
            # logging.info(f'Iterating over the files in the directory {image_path} for {folder_name}')
            # logging.info(f'Finding the polygon for each {folder_name} image')
            # logging.info(f'Storing the polygon for each {folder_name} image at directory {file_dir}')
            logging.info("-" * 20  + f'Data Ingestion stage completed successfully' + "-" * 20)
   