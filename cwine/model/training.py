import json
import os
import random

import cv2
import mrcnn.model as mlib
import numpy as np
import skimage.io
from mrcnn.config import Config
from mrcnn.utils import Dataset
from pycocotools import mask


def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    # img_copy = img
    # x_crop = 0 if x >= 0 else -x
    # alpha_s = img_overlay[:, :, 3] / 255.0
    # img_overlay = img_overlay[:, x_crop:, :]
    # y1, y2 = y, y + img_overlay.shape[0]
    # x1, x2 = max(x, x_crop), max(x, x_crop) + img_overlay.shape[1]
    #
    # alpha_l = 1.0 - alpha_s
    # alpha_s = img_overlay[:, :, 3] / 255.0
    #
    # for c in range(0, 3):
    #     img_copy[y1:y2, x1:x2, c] = (alpha_s * img_overlay[:, :, c] +
    #                                  alpha_l * img_copy[y1:y2, x1:x2, c])
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


class BottleConfig(Config):
    NAME = "wine_bottle"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


def convert_annotation_to_rle(ann, height, width):
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = mask.frPyObjects(segm, height, width)
        rle = mask.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = mask.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    return rle


def convert_annotation_to_mask(ann, height, width):
    rle = convert_annotation_to_rle(ann, height, width)
    m = mask.decode(rle)
    return m


class BottleDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.source = "wine_bottle"

    def load_bottles(self, dataset_dir, train=True):
        self.add_class("wine_bottle", 1, "wine_bottle")
        subset = "train" if train else "val"
        annotations = f"{dataset_dir}/annotations/wine_bottles_{subset}2017.json"
        images = f"{dataset_dir}/{subset}2017/"
        with open(annotations) as annotation_file:
            annotations = json.load(annotation_file)
        for annotation in annotations:
            image_id_str = str(annotation)
            zeros_padding = 12 - len(image_id_str)
            image_path = os.path.join(images, f"{'0' * zeros_padding}{image_id_str}.jpg")
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            annotation_arr = annotations[annotation]
            self.add_image(self.source, annotation, image_path, width=width, height=height,
                           annotations=annotation_arr)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        instance_masks = []
        class_ids = []
        for annotation in annotations:
            m = convert_annotation_to_mask(annotation, image_info["height"],
                                           image_info["width"])

            instance_masks.append(m)
            class_ids.append(1)

        if class_ids:
            stacked_mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return stacked_mask, class_ids
        else:
            return np.empty([0, 0, 0]), np.empty([0])

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "wine_bottle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class BottleAugmentation:

    def __init__(self, images_path):
        self.images_path = images_path
        self.augment_path = os.path.join(self.images_path, 'augment')
        self.augments = dict()
        self.bottles = []
        self._load_augment_images()
        self._load_bottle_images()

    def _load_augment_image(self, file: str):
        print(file)
        file_id = int(file[:-4])
        img = cv2.imread(os.path.join(self.augment_path, file))
        self.augments[file_id] = img

    def _load_augment_images(self):
        for files in next(os.walk(self.augment_path)):
            for file in files:
                if len(file) <= 1:
                    continue
                self._load_augment_image(file)

    def _load_bottle_images(self):
        for files_or_dirs in next(os.walk(self.images_path)):
            for file_or_dir in files_or_dirs:
                relative_path = os.path.join(self.images_path, file_or_dir)
                if os.path.isdir(relative_path):
                    break

                self.bottles.append(relative_path)

    def _load_cropped_bottle_image(self, index, angle):
        bottle_img = cv2.imread(self.bottles[index], cv2.IMREAD_UNCHANGED)
        if angle != 0:
            bottle_width = bottle_img.shape[1]
            bottle_height = bottle_img.shape[0]
            bottle_center = bottle_width / 2, bottle_height / 2
            rot = cv2.getRotationMatrix2D(bottle_center, -angle, 1)
            bottle_img = cv2.warpAffine(bottle_img, rot, (bottle_width, bottle_height))
        if bottle_img.shape[2] > 3:
            trans_mask = bottle_img[:, :, 3] == 0
            bottle_img[trans_mask] = [255, 255, 255, 255]
            bottle_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cntr = contours[0]
        x, y, w, h = cv2.boundingRect(cntr)

        new_img = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2BGRA)
        new_img[:, :, 3] = mask
        crop = new_img[y:y + h, x:x + w]
        return crop

    def _overwrite_mask_with_bottle(self, img, mask, bottle_index, splash_color=(75, 25, 240)):
        img_copy = img.copy()
        contour = []

        for row_index, row in enumerate(mask):
            for mask_index, mask_value in enumerate(row):
                if mask_value:
                    contour.append((mask_index, row_index))

        rect = cv2.minAreaRect(np.array(contour))
        angle = rect[2] - 90
        center = rect[0]
        crop_width = int(rect[1][0])
        crop_height = int(rect[1][1])
        x_offset = round(center[0] - (crop_height / 2))
        y_offset = round(center[1] - (crop_width / 2))

        bottle = self._load_cropped_bottle_image(bottle_index, angle)
        bottle = cv2.resize(bottle, (crop_height, crop_width), interpolation=cv2.INTER_LINEAR)
        alpha_mask = bottle[:, :, 3] / 255.0

        overlay_image_alpha(
            img_copy, bottle[:, :, :3],
            x_offset, y_offset,
            alpha_mask
        )

        cv2.imshow('overlayed', img_copy)
        cv2.waitKey(0)

    def augment_from_inference(self, inf_model: mlib.MaskRCNN):
        for augment in self.augments.keys():
            augment_img = self.augments[augment]

            results = inf_model.detect([augment_img])[0]
            class_ids = results['class_ids']
            if inf_model.config.NUM_CLASSES > 2:
                only_bottles = [x != 40 for x in class_ids]
            else:
                only_bottles = [False for _ in class_ids]
            masked_bottle_scores = np.ma.masked_array(data=results['scores'], mask=only_bottles,
                                                      dtype=np.float32, copy=True, fill_value=0.0)
            masked_bottle_indices = np.argwhere(masked_bottle_scores)

            if len(masked_bottle_indices) == 0:
                continue
            bottle_indices = np.concatenate(masked_bottle_indices, axis=0)
            rand_bottles = [random.randrange(0, len(self.bottles)) for _ in range(len(bottle_indices))]
            masks = results['masks']
            cv2.imshow('aug_pre', augment_img)
            for index, bottle in enumerate(bottle_indices):
                rand_bottle_index = rand_bottles[index]
                bottle_mask = masks[:, :, bottle]
                # self._overwrite_mask_with_bottle(augment_img, bottle_mask, rand_bottle_index)
                height = 0
                width = 0
                y_offset = 0
                x_offset = float('inf')
                for row_index, row in enumerate(bottle_mask):
                    row_sum = 0
                    curr_x_offset = 0
                    for mask_index, mask_value in enumerate(row):
                        if not mask_value:
                            if curr_x_offset > 0:
                                break
                            continue
                        if row_sum == 0:
                            curr_x_offset = mask_index - 1
                        row_sum += 1

                    if row_sum > 0:
                        if height == 0:
                            y_offset = row_index - 1
                        height += 1
                        x_offset = min(curr_x_offset, x_offset)
                        width = max(row_sum, width)

                    elif y_offset > 0:
                        break

                augment_img[bottle_mask, :] = 0
                height = round(height * 1.1)
                width = round(width * 1.1)
                bottle_img = self._load_cropped_bottle_image(rand_bottle_index, 0)
                bottle_img = cv2.resize(bottle_img, dsize=(width, height), interpolation=cv2.INTER_AREA)
                alpha_mask = bottle_img[:, :, 3] / 255.0

                augment_img[bottle_mask, :] = 0
                overlay_image_alpha(
                    augment_img, bottle_img[:, :, :3],
                    x_offset, y_offset,
                    alpha_mask
                )
            cv2.imshow('aug_post', augment_img)

            last_key = cv2.waitKey()
            while last_key != 13 and last_key != 27:
                last_key = cv2.waitKey()
            if last_key == 13:
                cv2.imwrite(os.path.join(self.augment_path, 'report', f'{augment}.jpg'), augment_img)

            cv2.destroyAllWindows()
