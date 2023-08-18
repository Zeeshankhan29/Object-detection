from src.detection.config import Configuration
from src.detection.components import DataIngestion
from src.detection import logger


def main():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.convert_coco_json_to_yolo_txt()
    data_ingestion.file_migrations()


if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)