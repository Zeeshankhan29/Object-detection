from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    temp_dir : Path
    transform_dir : Path
    # labels_dir : Path
    # temp_mask_dir : Path
    # original_image_dir : Path





@dataclass(frozen=True)
class DataTransformationConfig:
    transform_dir : Path
    train_label_dir :Path
    test_label_dir:Path
    train_image_dir : Path
    test_image_dir : Path
    yolo_config_dir: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    yolo_config_dir: Path
    

# @dataclass(frozen=True)
# class ModelPusherConfig:
#     pickle_dir : Path
#     s3_bucket_pickle : 