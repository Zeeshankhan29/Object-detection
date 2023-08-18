
from src.detection.components import DataIngestion,DataTransformation,ModelTraining
from src.detection.config import Configuration
from src.detection import logger




def main2():
    config = Configuration()
    model_training_config = config.get_model_training_config()
    model_training = ModelTraining(model_training_config)
    model_training.yolo_train(50)

if __name__ =='__main__':
    try:
        main2()
    except Exception as e:
        logger.exception(e)
