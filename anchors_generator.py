"""
A module that calculates anchors basing on IoU Kmeans.

"""
import os
import random
import copy
import pickle
import numpy as np
import tensorflow as tf
from helpers import iou_distances
random.seed(42)
tf.keras.utils.set_random_seed(42)


class AnchorsGenerator():
    def __init__(self) -> None:

        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")


    def load_bboxes(self) -> list:
        """ Returns list of bounding boxes. 

        Returns:
            np.ndarray: list of width and height for all bounding boxes.
        """
        with open(os.path.join(self.training_set_dir, "Targets.pickle"), "rb") as input_file:
            targets: list = pickle.load(input_file)
        
        bbox_list: list = [box[-2:] for sample in targets for box in sample['objects']]

        return bbox_list

    
    def anchor_kmeans(self, n_clusters: int, max_iter: int) -> np.ndarray:
        """ Calculates anchors basing on IoU Kmeans.

        Args:
            n_clusters (int): number of clusters to create.
            max_iter (int): number of max algorithm iterations.

        Returns:
            np.ndarray: final centroids (anchors).
        """
        bbox_list: np.ndarray = np.array(self.load_bboxes())

        unique_bbox_list: np.ndarray = np.unique(bbox_list, axis=0)
        np.random.seed(42)
        idx: np.typing.ArrayLike = np.random.choice(len(unique_bbox_list), n_clusters, replace=False)
        centroids: np.ndarray = copy.deepcopy(unique_bbox_list[idx])
        avg_iou: np.floating = np.float64(0)
                 
        last_step:int = 0
        for step in range(max_iter):
            iou_list: np.ndarray = iou_distances(bbox_list, centroids)

            new_cluster: np.ndarray = np.argmax(iou_list, axis=1)
            avg_iou = np.mean(iou_list[np.arange(iou_list.shape[0]), new_cluster])

            old_centroids: np.typing.ArrayLike = copy.deepcopy(centroids)
            centroids_list: list = []
            for k in range(n_clusters):
                grouped_bbox_list: np.typing.ArrayLike = bbox_list[np.where(new_cluster == k)]
                centroids_list.append([np.mean(grouped_bbox_list[:,0]), np.mean(grouped_bbox_list[:,1])])
            centroids = np.stack(centroids_list, axis=0)

            delta_centroids: np.typing.ArrayLike = np.abs(old_centroids - centroids)
            if np.max(delta_centroids) < 0.01:
                last_step = step
                break

        area_array: np.typing.ArrayLike = np.prod(centroids, axis=1)
        area_idx: np.typing.ArrayLike = np.argsort(area_array)
        centroids = centroids[area_idx]

        centroids = centroids.astype(int)
        print(f"Avg IoU for all bounding boxes is {avg_iou}, after {last_step} steps")
        
        return centroids
    