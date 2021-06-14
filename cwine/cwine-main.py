import json
import os
from typing import Optional

import cv2
import mrcnn.model as modellib
import numpy as np
import skimage.color
import skimage.io
from mrcnn import utils
from pycocotools.coco import COCO

from cwine.model.keypoints import get_bottle_crop, KeypointModel
from cwine.model.training import BottleConfig, BottleDataset, BottleAugmentation

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
WINE_BOTTLE_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_wine_bottle.h5")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
# Directory to store COCO images
COCO_DIR = '/media/jasper/projects/Projects/cbottle/'
ANNOTATIONS = os.path.join(COCO_DIR, 'annotations')
# Bottle class ID all we care about
BOTTLE_ID = 40


class InferenceConfig(BottleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def train_bottle_model():
    config = BottleConfig()

    # annotate_2017 = os.path.join(ANNOTATIONS, 'instances_train2017.json')
    # ct = COCO(annotate_2017)

    dataset_train = BottleDataset()
    dataset_train.load_bottles(COCO_DIR)
    dataset_train.prepare()

    # dataset_train = coco.CocoDataset()
    # dataset_train.load_coco(COCO_DIR, "train", year="2017", class_ids=[44], max_id=50000, auto_download=True)

    dataset_val = BottleDataset()
    dataset_val.load_bottles(COCO_DIR, train=False)
    dataset_val.prepare()

    # dataset_val = coco.CocoDataset()
    # dataset_val.load_coco(COCO_DIR, "val", class_ids=[44], max_id=50000, auto_download=True)

    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers='heads'
    )

    return model


def get_inference_model(weights_path: Optional[str] = COCO_MODEL_PATH):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    if weights_path is None:
        weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)
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


def test_detect(loaded_model, kp_model, image_name="test-new.jpg"):
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    test_image = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(test_image)
    crop = get_bottle_crop(image, loaded_model)
    kp_model.compute_sift_descriptors(crop)


def augment_picker():
    annotate_2017 = os.path.join(ANNOTATIONS, 'instances_train2017.json')
    ct = COCO(annotate_2017)
    wine_images = ct.getImgIds(catIds=[44])
    last_key = 0
    for wine_image_id in wine_images:
        img_ann = ct.imgToAnns[wine_image_id]
        for seg in img_ann:
            if int(seg['category_id']) == 44 and float(seg['area']) > 1000:
                wine_image_id_str = str(wine_image_id)
                zeros_padding = 12 - len(wine_image_id_str)
                file_path = os.path.join(COCO_DIR, 'train2017',
                                         f'{"".join(["0" for _ in range(zeros_padding)])}{wine_image_id_str}.jpg')
                img = cv2.imread(file_path)
                cv2.imshow(f'{last_key} - {file_path}', img)
                key = cv2.waitKey(0)
                if key == 13:
                    print('Saving image to augment base')
                    cv2.imwrite(os.path.join(IMAGE_DIR, 'augment', f'{wine_image_id_str}.jpg'), img)
                last_key = key
                cv2.destroyAllWindows()
                break


def segment_picker(subset):
    annotate_2017 = os.path.join(ANNOTATIONS, f'instances_{subset}2017.json')
    ct = COCO(annotate_2017)

    wine_images = ct.getImgIds(catIds=[44])
    last_key = 0
    picked_images = dict()

    def _inner_dump():
        dump_path = os.path.join(ANNOTATIONS, f'wine_bottles_{subset}2017.json')
        with open(dump_path, 'w') as wb_dump:
            json.dump(picked_images, wb_dump)

    for wine_image_id in wine_images:
        img_ann = ct.imgToAnns[wine_image_id]
        img_ann_updated = []
        add = True
        for seg in img_ann:
            if int(seg['category_id']) == 44 and float(seg['area']) > 1000:
                segmentation = seg['segmentation']
                if not isinstance(segmentation, list):
                    continue
                seg_poly = seg['segmentation'][0]
                poly_2d = []
                for i in range(0, len(seg_poly), 2):
                    x = seg_poly[i]
                    y = seg_poly[i + 1]
                    poly_2d.append([int(x), int(y)])
                poly_2d = np.array(poly_2d)
                poly_2d.reshape((-1, 1, 2))
                wine_image_id_str = str(wine_image_id)
                zeros_padding = 12 - len(wine_image_id_str)
                file_path = os.path.join(COCO_DIR, f'{subset}2017',
                                         f'{"".join(["0" for _ in range(zeros_padding)])}{wine_image_id_str}.jpg')
                img = cv2.imread(file_path)
                cv2.polylines(img, [poly_2d], True, (0, 255, 0))
                cv2.imshow(f'{last_key} - {file_path}', img)
                key = cv2.waitKey(0)
                last_key = key

                if key == 81:
                    print('Not marking as wine bottle')
                elif key == 83:
                    print('Marking as wine bottle')
                    img_ann_updated.append(seg)
                elif key == 13:
                    cv2.destroyAllWindows()
                    _inner_dump()
                    return
                else:
                    add = False
                    cv2.destroyAllWindows()
                    print("Don't use this image")
                    break
        if add:
            picked_images[wine_image_id] = img_ann_updated
        cv2.destroyAllWindows()
    _inner_dump()


def test_inference_model():
    wbm = get_inference_model(weights_path=None)
    detect_path = os.path.join(IMAGE_DIR, 'detect_test')
    for file in os.listdir(os.path.join(IMAGE_DIR, 'detect_test')):
        test_detect(wbm, image_name=os.path.join('detect_test', file))


def run_bottle_augmentation(coco=True):
    ba = BottleAugmentation(IMAGE_DIR)
    wbm = get_inference_model() if coco else get_inference_model(weights_path=None)
    ba.augment_from_inference(wbm)


if __name__ == '__main__':
    kp_index_model = KeypointModel(IMAGE_DIR)
    kp_index_model.prepare()
    kp_index_model.train_index()
    # test_inference_model()
    # ba = BottleAugmentation(IMAGE_DIR)
    # ba.augment_from_inference(inference)

    # segment_picker('val')

    # train_bottle_model()

# Dataset
# - Images from coco with modified annotations
# - Augmented images from coco with own wine bottles
# - Validation set based on local images and inference model detection results/custom handmade polygons

# Training
# - 0.01 learning rate
# - Frozen head layers

# Concerns
# - Optimal scale for keypoints
# - Indexing the known keypoint feature vectors
# -
