import os
import sys
import cwine
from cwine.model.training import BottleDataset
import cv2
import mrcnn.model as modellib
import numpy as np
import skimage.color
import skimage.io
from mrcnn import utils
from mrcnn import visualize
from mrcnn.config import Config
from pycocotools.coco import COCO

import coco

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Directory to store COCO images
COCO_DIR = '/media/jasper/projects/Projects/cbottle/'
ANNOTATIONS = os.path.join(COCO_DIR, 'annotations')
# Bottle class ID all we care about
BOTTLE_ID = 40

"""
31/05/2021 Meeting:

- For augmentation we are at risk of creating context clues, make sure we have enough base images
- 
"""


class BottleConfig(Config):
    NAME = "bottles"
    NUM_CLASSES = 1 + 1


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def train_bottle_model():
    config = BottleConfig()
    annotate_2017 = os.path.join(ANNOTATIONS, 'instances_train2017.json')
    ct = COCO(annotate_2017)

    dataset_train = coco.CocoDataset()
    dataset_train.load_coco(COCO_DIR, "train", year="2017", class_ids=[44], max_id=50000, auto_download=True)
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


def augment_picker():
    annotate_2017 = os.path.join(ANNOTATIONS, 'instances_train2017.json')
    ct = COCO(annotate_2017)
    wine_images = ct.getImgIds(catIds=[44])
    for wine_image_id in wine_images:
        img_ann = ct.imgToAnns[wine_image_id]
        for seg in img_ann:
            if int(seg['category_id']) == 44 and float(seg['area']) > 1000:
                wine_image_id_str = str(wine_image_id)
                zeros_padding = 12 - len(wine_image_id_str)
                file_path = os.path.join(COCO_DIR, 'train2017',
                                         f'{"".join(["0" for _ in range(zeros_padding)])}{wine_image_id_str}.jpg')
                img = cv2.imread(file_path)
                cv2.imshow(file_path, img)
                key = cv2.waitKey(0)
                if key == 13:
                    print('Saving image to augment base')
                    cv2.imwrite(os.path.join(IMAGE_DIR, 'augment', f'{wine_image_id_str}.jpg'), img)

                cv2.destroyWindow(file_path)
                break


if __name__ == '__main__':
    augment_model = get_inference_model()
    bs = BottleDataset(IMAGE_DIR)
    bs.augment_from_inference(augment_model)
