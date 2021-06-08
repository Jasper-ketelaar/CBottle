import os
import random

import cv2
import mrcnn.model as mlib
import numpy as np
from mrcnn.config import Config
from mrcnn.utils import Dataset


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
    NAME = "bottles"
    NUM_CLASSES = 1 + 1


class BottleDataset(Dataset):

    def __init__(self, images_path):
        super(BottleDataset, self).__init__()
        self.images_path = images_path
        self.augment_path = os.path.join(self.images_path, 'augment')
        self.augments = dict()
        self.bottles = []
        self._load_augment_images()
        self._load_bottle_images()

    def _load_augment_image(self, file: str):
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

    def _load_cropped_bottle_image(self, index):
        bottle_img = cv2.imread(self.bottles[index], cv2.IMREAD_UNCHANGED)
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

    def load_augmentation_base(self):
        # self.add_class("wine", 1, "wine_bottle")
        pass

    def augment_from_inference(self, inf_model: mlib.MaskRCNN):
        for augment in self.augments.keys():
            augment_img = self.augments[augment]

            results = inf_model.detect([augment_img])[0]
            class_ids = results['class_ids']
            only_bottles = [x != 40 for x in class_ids]
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
                height = round(height * 1.05)
                bottle_img = self._load_cropped_bottle_image(rand_bottle_index)
                bottle_img = cv2.resize(bottle_img, dsize=(width, height), interpolation=cv2.INTER_AREA)
                alpha_mask = bottle_img[:, :, 3] / 255.0
                _, thresh = cv2.threshold(augment_img, 0, 255, cv2.THRESH_BINARY)
                t_gray = cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(t_gray, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                cnt = contours[-1]
                rect = cv2.minAreaRect(cnt)
                angle = rect[3]
                augment_img[bottle_mask, :] = 255
                overlay_image_alpha(
                    augment_img, bottle_img[:, :, :3],
                    x_offset, y_offset,
                    alpha_mask
                )

            while cv2.waitKey(0) != 13:
                pass

            cv2.destroyAllWindows()
