import os
from pathlib import Path

import torch
import torchvision.transforms as T

TRANSFORMS = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICEWIN = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = ["resnet50", "efficientnet_v2_s"]

# Dataset paths (override with PLANTNET_DATASET_DIR env var)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.getenv("PLANTNET_DATASET_DIR", PROJECT_ROOT / "plantnet_300K")
)

IMAGES_TRAIN_DIR = DATASET_ROOT / "images_train"
IMAGES_VAL_DIR = DATASET_ROOT / "images_val"
IMAGES_TEST_DIR = DATASET_ROOT / "images_test"

METADATA_DIR = DATASET_ROOT / "metaData"
IMAGES_PATH_DF = METADATA_DIR / "dataWithImages.csv"
FEATURES = METADATA_DIR / "featuresWithPaths.pkl"
FEATURES_INDEXES_L2 = METADATA_DIR / "indexedImagesFeaturesData.idx"
NAMES_JSON = METADATA_DIR / "names.json"

WILL_RETURN_IMAGE_COUNT = 5
