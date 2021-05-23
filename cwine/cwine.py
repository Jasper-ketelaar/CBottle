import os
import sys
import skimage.io
import skimage.color
import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Directory to store COCO images
COCO_DIR = os.path.join(ROOT_DIR, "coco_data")
# Bottle class ID all we care about
BOTTLE_ID = 40


class BottleConfig(Config):
    NAME = "bottles"

    # GPU Available
    GPU_COUNT = 1

    # Amount of images per GPU
    IMAGES_PER_GPU = 2

    # Amount of steps per epoch (simply over-training on already trained class so low number)
    STEPS_PER_EPOCH = 20

    # Validation steps can be lower too
    VALIDATION_STEPS = 10

    # Background class + wine
    NUM_CLASSES = 1 + 1

    # Require 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class BottleDataset(coco.CocoDataset):

    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def prepare_bottles(self):
        self.load_coco(COCO_DIR, self.subset, "2017", class_ids=[44], auto_download=True)
        self.prepare()


def train_bottle_model():
    config = BottleConfig()
    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    dataset_train = BottleDataset("train")
    dataset_train.prepare_bottles()

    dataset_val = BottleDataset("val")
    dataset_val.prepare_bottles()

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=5,
        layers='heads'
    )

    return model


if __name__ == '__main__':
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    bottle_model = train_bottle_model()
