from src.detection.components import DataIngestion,DataTransformation,ModelTraining
from src.detection.config import Configuration
from src.detection import logger



def main():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.convert_coco_json_to_yolo_txt()
    data_ingestion.file_migrations()


def main1():
    config = Configuration()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    data_transformation.rename_sort()
    data_transformation.split_train_test()
    data_transformation.class_no_sort1()
    


def main2():
    config = Configuration()
    model_training_config = config.get_model_training_config()
    model_training = ModelTraining(model_training_config)
    model_training.yolo_train(200)

if __name__ =='__main__':
    try:
        main()
        main1()
        main2()
    except Exception as e:
        logger.exception(e)
