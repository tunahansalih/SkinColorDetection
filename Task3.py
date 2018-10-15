import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

class K_Means:
    def __init__(self, k=3, tolerance=0.1, max_iteration=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iteration = max_iteration

    def apply_color_range_to_image(self, original_image, color_range):
        binary_mask = np.full_like(original_image[:, :, 0], True)
        for i in range(3):
            binary_mask = np.where(original_image[:, :, i] < color_range[i][1], binary_mask, False)
            binary_mask = np.where(original_image[:, :, i] > color_range[i][0], binary_mask, False)

            return binary_mask

    def get_masked_image(self, image, binary_mask):
        return np.where(np.stack([binary_mask] * 3, -1), image, np.zeros_like(image))

    def euclidean_dist(self, p1, p2):
        return np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

    def get_ranges(self, cluster):
        ranges = [[255, 0] for _ in range(3)]
        for point in cluster:
            for i in range(3):
                ranges[i][0] = min(ranges[i][0], point[i])
                ranges[i][1] = max(ranges[i][1], point[i])
        return ranges

    def run_on_image(self, img):
        points = np.reshape(np.array(img, np.int), (img.shape[0] * img.shape[1], img.shape[2]))
        cluster_centers = [points[i] for i in np.random.choice(range(points.shape[0]), self.k)]
        for i in range(self.max_iteration):
            print(i)
            clusters = [[[]] for i in range(self.k)]
            [clusters[np.argmin([self.euclidean_dist(p, c) for c in cluster_centers])].append(p) for p in points]
            new_cluster_centers = [np.mean(cluster, 0) for cluster in clusters]
            if np.sum(np.divide(np.abs(np.subtract(cluster_centers, new_cluster_centers)),
                                cluster_centers))*100 < self.tolerance:
                break
            cluster_centers = new_cluster_centers
            print(cluster_centers)
            print([len(cluster) for cluster in clusters])
        return [self.get_ranges(cluster) for cluster in clusters]


ORIG_IMAGE_FOLDER = os.path.join('Images', 'Original Images')
orig_filenames = [img for img in glob.glob(os.path.join(ORIG_IMAGE_FOLDER, "*.jpg"))]
orig_filenames.sort()
orig_filenames = orig_filenames[10:]

range_dicts = []
for f in orig_filenames:
    img = cv2.imread(f)
    print(f)
    k = 7
    ranges_dict = {}
    kmeans = K_Means(k=k, tolerance=0.1, max_iteration=10)
    ranges = kmeans.run_on_image(img)
    ranges_dict["k"] = k
    ranges_dict["filename"] = f
    ranges_dict["ranges"] = ranges
    range_dicts.append(ranges_dict)
    with open('ranges%s.p' % os.path.basename(f), 'wb') as outfile:
        pickle.dump(range_dicts, outfile)
