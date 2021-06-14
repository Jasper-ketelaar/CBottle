import os
from glob import glob
from typing import Optional

import cv2
import faiss
import numpy as np
from tqdm import tqdm

SIFT_DIMENSIONS = 128
NUM_FEATURES = 100
DEST_HEIGHT = 512


class KeypointModel:
    index: Optional[faiss.IndexIDMap]

    def __init__(self, bottle_path, dimensions=SIFT_DIMENSIONS, features=NUM_FEATURES):
        self.dimensions = dimensions
        self.features = features
        self.bottle_path = bottle_path
        self.index = None
        self.sift = None

    def prepare(self, new_index=True):
        if new_index:
            self.index = faiss.index_factory(SIFT_DIMENSIONS, "IDMap,PCA128,IVF2048,PQ16")
        else:
            self.index = faiss.read_index('cache/keypoints.index')
        if faiss.get_num_gpus() > 0:
            print("FAISS Indexing on GPU")
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)

    def train_index(self):
        ids_length = 0
        index_dict = dict()
        ids = None
        features = np.matrix([])
        files = glob(os.path.join(self.bottle_path, '*.png')) + glob(os.path.join(self.bottle_path, '*.jpg'))
        print(f'{len(files)} bottle images to index')

        for bottle_file in tqdm(files[:250]):
            bottle_sku = bottle_file[bottle_file.rindex(os.path.sep) + 1:bottle_file.rindex('-')]
            error, feature = self.compute_sift_descriptors(bottle_file)
            if error != 0 or not feature.any():
                continue
            index_dict.update({
                ids_length: (bottle_sku, feature)
            })
            ids_list = np.linspace(ids_length, ids_length, num=feature.shape[0], dtype=np.int64)
            ids_length += 1
            if ids is None:
                features = feature
                ids = ids_list
            else:
                ids = np.hstack((ids, ids_list))
                features = np.vstack((features, feature))

        print("Done computing sift features for all images, training index")
        if not features.any():
            return

        if self.index.is_trained:
            return

        self.index.train(features)
        print("Trained the index on all features")
        self.index.add_with_ids(features, ids)
        print("Added all features to the index")
        if not os.path.exists('cache'):
            os.mkdir('cache')
        faiss.write_index(self.index, 'cache/keypoints.index')

    def compute_sift_descriptors(self, image):
        parsed = image
        if isinstance(image, str):
            parsed = cv2.imread(image, cv2.IMREAD_COLOR)

        height, width = parsed.shape[:2]
        scale = DEST_HEIGHT / height
        if scale != 1:
            # TODO: Confirm necessary considering scale invariance
            parsed = cv2.resize(parsed, (int(height * scale), int(width * scale)))

        gray_cvt = cv2.cvtColor(parsed, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray_cvt, None)
        feature_matrix = np.matrix(des)

        # Error code, keypoint descriptor matrix
        return 0, feature_matrix

    def search(self, ids, features, k=1):
        results = []
        for id_, feature in zip(ids, features):
            scores, neighbours = [], []
            if feature.size > 0:
                scores, neighbours = self.index.search(feature, k=k)
            num_results, num_dimension = neighbours.shape
            for i in range(num_results):
                unique = np.unique(neighbours[i]).tolist()
                for f_id in unique:
                    if f_id == -1:
                        continue
                    # TODO: Rerank results based on desc occurrences


def get_bottle_crop(image, model):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    results = model.detect([image])
    if len(results) == 0:
        return
    r = results[0]
    if 'scores' not in r or len(r['scores']) == 0:
        return image
    best_score_idx = np.argmax(r['scores'])
    mask = r['masks'][:, :, best_score_idx]
    contour = []
    for row_index, row in enumerate(mask):
        for mask_index, mask_value in enumerate(row):
            if mask_value:
                contour.append((mask_index, row_index))
    contour = np.array(contour)
    mask_crop = np.zeros_like(image)
    cv2.drawContours(mask_crop, [contour], 0, 255, -1)
    out = np.zeros_like(image)
    out[mask, :] = image[mask, :]
    x, y, w, h = cv2.boundingRect(contour)
    return out[y:y + h, x:x + w]
