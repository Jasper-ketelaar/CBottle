import os
import sys
import skimage.io
import skimage.color
import tensorflow as tf
from pycocotools.coco import COCO

import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
import numpy as np

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
    NUM_CLASSES = 1 + 1


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class BottleDataset(coco.CocoDataset):

    def __init__(self, subset):
        super().__init__()
        self.subset = subset

    def prepare_bottles(self):
        self.load_coco(COCO_DIR, self.subset, "2017", class_ids=[44], max_id=10000)
        self.prepare()


def train_bottle_model():
    config = BottleConfig()
    annotations = os.path.join(COCO_DIR, 'annotations')
    annotate_2017 = os.path.join(annotations, 'instances_train2014.json')
    ct = COCO(annotate_2017)
    print(ct.getCatIds(['bottle']))

    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(COCO_DIR, "train", class_ids=[44], max_id=50000, auto_download=True)
    dataset_train.prepare()

    dataset_val = coco.CocoDataset()
    dataset_val.load_coco(COCO_DIR, "val", class_ids=[44], max_id=50000, auto_download=True)
    dataset_val.prepare()

    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=100,
        layers='heads'
    )

    return model


def get_inference_model():
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model


def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def test_detect():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    loaded_model = get_inference_model()
    test_image = os.path.join(IMAGE_DIR, "input_sample.jpg")
    image = skimage.io.imread(test_image)
    if image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    results = loaded_model.detect([image], verbose=1)
    if len(results) == 0:
        return
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                coco.CLASS_NAMES, r['scores'])
    splash = color_splash(image, r['masks'])
    skimage.io.imsave(test_image.replace('images', 'splashes'), splash)


if __name__ == '__main__':
    train_bottle_model()
