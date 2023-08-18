from src.detection.utils import read_yaml,create_directories
from src.detection.constants import CONFIG_FILE_PATH
from src.detection.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainingConfig
from box import ConfigBox
from src.detection import logging


class Configuration:
    def __init__(self,config_file_path=CONFIG_FILE_PATH):
        logging.info(f'loading config yaml configuration file: {config_file_path}')
        self.config = read_yaml(config_file_path)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directories([config.temp_dir])
        create_directories([config.transform_dir])
       


        data_ingestion  = DataIngestionConfig(
                             temp_dir= config.temp_dir,
                             transform_dir= config.transform_dir
)
        return data_ingestion
    
    def get_data_transformation_config(self):
        config = self.config.data_transformation
        create_directories([config.transform_dir])
        create_directories([config.train_label_dir])
        create_directories([config.test_label_dir])
        create_directories([config.train_image_dir])
        create_directories([config.test_image_dir])
        create_directories([config.yolo_config_dir])
        

        data_transformation  = DataTransformationConfig(transform_dir=config.transform_dir,
                                                        train_label_dir=config.train_label_dir,
                                                        test_label_dir=config.test_label_dir,
                                                        train_image_dir=config.train_image_dir,
                                                        test_image_dir=config.test_image_dir,
                                                        yolo_config_dir=config.yolo_config_dir
                                                        )
        return data_transformation
    
    def get_model_training_config(self):
        config = self.config.model_training
        create_directories([config.yolo_config_dir])

        model_training = ModelTrainingConfig(yolo_config_dir=config.yolo_config_dir)

        return model_training