"""
_summary_

"""
import os
import random
import copy
import pickle
import numpy as np
import tensorflow as tf


class AnchorsGenerator():
    def __init__(self) -> None:

        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")


    def load_bboxes(self) -> np.typing.ArrayLike:
        """Returns list of bounding boxes. 

        Returns:
            np.typing.ArrayLike: list of width and height for all bounding boxes.
        """
        with open(os.path.join(self.training_set_dir, "Targets.pickle"), "rb") as input_file:
            targets: list = pickle.load(input_file)
        
        bbox_list: list = [box[-2:] for sample in targets for box in sample['objects']]

        return np.array(bbox_list)

    
    def iou_distances(self, bbox_list: np.typing.ArrayLike, centroid: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Calculates intersection over union with bounding boxes.

        Args:
            bbox_list (np.typing.ArrayLike): list of width and height for all bounding boxes.
            centroid (np.typing.ArrayLike): centroids for calculating intersection over union with bounding boxes.

        Returns:
            np.typing.ArrayLike: intersection over union with bounding boxes
        """
        w_box: np.typing.ArrayLike = bbox_list[:,0]
        h_box: np.typing.ArrayLike = bbox_list[:,1]
        w_centroid: np.typing.ArrayLike = np.full(len(w_box), centroid[0])
        h_centroid: np.typing.ArrayLike = np.full(len(h_box), centroid[1])

        w_intersection: np.typing.ArrayLike = np.minimum(w_box, w_centroid)
        h_intersection: np.typing.ArrayLike = np.minimum(h_box, h_centroid)

        area_box: np.typing.ArrayLike = w_box * h_box
        area_centroid: np.typing.ArrayLike = w_centroid * h_centroid
        area_intersection: np.typing.ArrayLike = w_intersection * h_intersection

        iou_array: np.typing.ArrayLike = area_intersection / (area_box + area_centroid - area_intersection)

        return iou_array


    def anchor_kmeans(self, n_clusters: int, max_iter: int) -> np.typing.ArrayLike:
        """Calculates anchors basing on IoU Kmeans.

        Args:
            n_clusters (int): number of clusters to create.
            max_iter (int): number of max algorithm iterations.

        Returns:
            np.typing.ArrayLike: final centroids (anchors).
        """
        bbox_list: np.typing.ArrayLike = np.array(self.load_bboxes())

        unique_bbox_list: np.typing.ArrayLike = np.unique(bbox_list, axis=0)
        np.random.seed(42)
        idx: np.typing.ArrayLike = np.random.choice(len(unique_bbox_list), n_clusters, replace=False)
        centroids: np.typing.ArrayLike = copy.deepcopy(unique_bbox_list[idx])
        selected_values: float = 0
                 
        for _ in range(max_iter):
            iou_list: list = []
            for centroid in centroids:
                iou: np.typing.ArrayLike = self.iou_distances(bbox_list, centroid)
                iou_list.append(iou)

            iou_array: np.typing.ArrayLike = np.stack(iou_list, axis=0)
            new_cluster: np.typing.ArrayLike = np.argmax(iou_array, axis=0)
            selected_values = np.mean(iou_array[new_cluster, np.arange(iou_array.shape[1])])

            old_centroids: np.typing.ArrayLike = copy.deepcopy(centroids)
            centroids = []
            for k in range(n_clusters):
                tmp: np.typing.ArrayLike = bbox_list[np.where(new_cluster == k)]
                centroids.append([np.mean(tmp[:,0]), np.mean(tmp[:,1])])
            centroids = np.stack(centroids, axis=0)

            delta_centroids: np.typing.ArrayLike = np.abs(old_centroids - centroids)
            if np.max(delta_centroids) < 0.01:
                break

        area_array: np.typing.ArrayLike = np.prod(centroids, axis=1)
        area_idx: np.typing.ArrayLike = np.argsort(area_array)
        centroids = centroids[area_idx]

        centroids = centroids.astype(int)
        print(f"Avg IoU for all bounding boxes is {selected_values}")
        
        return centroids



class TrainingAssistant():

    def __init__(self) -> None:
        
        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")

    def load_input_chunk(self):
        pass

    def build_nn(self):
        pass

    def train_model(self):
        pass
