from src.detection.components import DataIngestion,DataTransformation,ModelTraining
from src.detection.config import Configuration
from src.detection import logger





def main1():
    config = Configuration()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    data_transformation.rename_sort()
    data_transformation.split_train_test()
    data_transformation.class_no_sort1()

if __name__ =='__main__':
    try:
        main1()
    except Exception as e:
        logger.exception(e)