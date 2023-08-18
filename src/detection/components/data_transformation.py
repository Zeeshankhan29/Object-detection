from src.detection.entity import DataTransformationConfig
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import os
import cv2
import shutil
from src.detection import logging
import re

class DataTransformation:
    def __init__(self, DataTransformConfig =  DataTransformationConfig):
        logging.info('-'*20 +f"Data Transformation stage started " +'-'*20)
        self.datatransformation = DataTransformConfig
        self.curr_dir = Path(os.getcwd())
        self.transform_dir = Path(self.datatransformation.transform_dir)
        self.train_label_dir = Path(self.datatransformation.train_label_dir)
        self.test_label_dir = Path(self.datatransformation.test_label_dir)
        self.train_image_dir = Path(self.datatransformation.train_image_dir)
        self.test_image_dir = Path(self.datatransformation.test_image_dir)
        self.train_img_dir_final = Path(os.path.join(self.curr_dir,self.train_image_dir))
        self.text_img_dir_final = Path(os.path.join(self.curr_dir,self.test_image_dir))
        self.train_label_dir_final = Path(os.path.join(self.curr_dir,self.train_label_dir))
        self.text_label_dir_final = Path(os.path.join(self.curr_dir,self.test_label_dir))



    def rename_sort(self):
        image_dir = os.path.join(self.curr_dir, self.transform_dir)
        for file in os.listdir(image_dir):
            image_dir1 = os.path.join(image_dir, file, 'images')
            label_dir1 = os.path.join(image_dir, file, 'output')
            
            for file_name in os.listdir(image_dir1):
                file_parts = file_name.strip().split('.')
                result = [re.sub(r'^\d+', '', string) for string in file_parts]
                image_old_name = os.path.join(image_dir1, file_name)
                image_new_name = os.path.join(image_dir1, result[0] + '.png')
                
                if not os.path.exists(image_new_name):
                    os.rename(image_old_name, image_new_name)
                    logging.info(f"{file} file renamed ")
                else:
                    logging.info(f"{file} file already exists, skipping renaming.")

            for file_name1 in os.listdir(label_dir1):
                file_parts1 = file_name1.strip().split('.')
                result1 = [re.sub(r'^\d+', '', string) for string in file_parts1]
                image_old_name1 = os.path.join(label_dir1, file_name1)
                image_new_name1 = os.path.join(label_dir1, result1[0] + '.txt')
                
                if not os.path.exists(image_new_name1):
                    os.rename(image_old_name1, image_new_name1)
                    logging.info(f"{file} label file renamed ")
                else:
                    logging.info(f"{file} label file already exists, skipping renaming.")


    def split_train_test(self):
        image = []
        labels = []
        for files in os.listdir(self.transform_dir):
            file_path = os.path.join(self.curr_dir,self.transform_dir,files)
            image_dir = os.path.join(file_path,'images')
            output_dir = os.path.join(file_path,'output')

            for file in os.listdir(image_dir):
                image.append(file)
                sorted_images = sorted(image)

            for file in os.listdir(output_dir):
                labels.append(file)
                sorted_labels = sorted(labels)

        #Split the data to train and test for each classes
        from sklearn.model_selection import train_test_split
        img_train_data, img_test_data = train_test_split(sorted_images, test_size=0.2, random_state=42)
        text_train_data, text_test_data = train_test_split(sorted_labels, test_size=0.2, random_state=42)
        
        for file in os.listdir(self.transform_dir):
            image_dir = Path(os.path.join(self.curr_dir,self.transform_dir,file,'images'))
            label_dir = Path(os.path.join(self.curr_dir,self.transform_dir,file,'output'))
            for image_file in img_train_data:
                image_file_dir = os.path.join(image_dir,image_file)
                if os.path.exists(image_file_dir) == True:
                    shutil.copy(image_file_dir,self.train_img_dir_final)
            for image_file1 in img_test_data:
                image_file_dir1 = os.path.join(image_dir,image_file1)
                if os.path.exists(image_file_dir1) == True:
                    shutil.copy(image_file_dir1,self.text_img_dir_final)

            

            for text_file in text_train_data:
                label_train_path = os.path.join(label_dir,text_file)
                if os.path.exists(label_train_path) == True:
                    shutil.copy(label_train_path,self.train_label_dir_final)

            for text_file1 in text_test_data:
                label_test_path = os.path.join(label_dir,text_file1)
                if os.path.exists(label_test_path) == True:
                    shutil.copy(label_test_path,self.text_label_dir_final)           



    def class_no_sort1(self, old_label=1, new_label=0):
        train_label_dir = Path(os.path.join(self.curr_dir, self.train_label_dir))
        test_label_dir = Path(os.path.join(self.curr_dir, self.test_label_dir))
        prefixes = ['Apple Fuji', 'Mosambi']
        prefixes.sort()

        for prefix_index, prefix in enumerate(prefixes):
            for file in os.listdir(train_label_dir):
                if file.startswith(prefix):
                    file_path = os.path.join(train_label_dir, file)
                    basename = os.path.basename(file_path).split('_')[0]
                    with open(file_path, 'r+') as f:
                        lines = f.readlines()
                        f.seek(0)
                        updated_lines = []
                        for line in lines:
                            numbers = line.strip().split()
                            class_label = int(numbers[0])
                            if class_label == old_label:
                                numbers[0] = str(new_label + prefix_index)
                            updated_line = ' '.join(numbers) + '\n'
                            updated_lines.append(updated_line)
                        f.writelines(updated_lines)
                        f.truncate()
                    print(f"Label Replacement completed for {basename}")

            for file in os.listdir(test_label_dir):
                if file.startswith(prefix):
                    file_path = os.path.join(test_label_dir, file)
                    basename = os.path.basename(file_path).split('_')[0]
                    with open(file_path, 'r+') as f:
                        lines = f.readlines()
                        f.seek(0)
                        updated_lines = []
                        for line in lines:
                            numbers = line.strip().split()
                            class_label = int(numbers[0])
                            if class_label == old_label:
                                numbers[0] = str(new_label + prefix_index )
                            updated_line = ' '.join(numbers) + '\n'
                            updated_lines.append(updated_line)
                        f.writelines(updated_lines)
                        f.truncate()
                    print(f"Label Replacement completed for {basename}")
        
        # logging.info('filter the labels with .txt format ')
        # logging.info(f'loading images for {file} at directory {file_path1}')
        # logging.info('filter the Images  with .png format ')
        logging.info('-'*20 +f"Data Transformation stage Completed" +'-'*20)
        