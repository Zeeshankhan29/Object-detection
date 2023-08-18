from src.detection.entity import ModelTrainingConfig
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import os
import cv2
import shutil
from src.detection import logging


class ModelTraining:
    def __init__(self,ModelTrainingConfig = ModelTrainingConfig):
        self.modeltraining = ModelTrainingConfig
        self.curr_dir = Path(os.getcwd())
        self.yolo_config_dir = Path(self.modeltraining.yolo_config_dir)
        

    def yolo_train(self,epochs_value =5):
        '''Epochs value depends on the accuracy required, the more the epoch the more
        the accuracy'''

        # Yolo base model
        model = YOLO('yolov8n.pt') 
        self.yolo_config_dir1 = Path(os.path.join(self.curr_dir,self.yolo_config_dir))
        
        '''Loading the yolo config file'''
        self.yolo_config_path = Path(os.path.join(self.yolo_config_dir1,'task-seg.yaml'))
        
        
        logging.info('-'*20 +f"Model Training stage started " +'-'*20)
        logging.info(f'Data training from {self.yolo_config_path} directory')
        
        # Train the model
        model.train(data=self.yolo_config_path, epochs=epochs_value, imgsz=640)  
        
        logging.info('-'*20 +f"Model Training stage Completed " +'-'*20)
        
        