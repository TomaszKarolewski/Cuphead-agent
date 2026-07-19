"""
Helper functions.

"""
import numpy as np

def iou_distances(bbox_list: np.ndarray, anchors_list: np.ndarray) -> np.ndarray:
    """ Calculates intersection over union with bounding boxes.

    Args:
        bbox_list (np.ndarray): list of width and height for all bounding boxes.
        anchors_list (np.ndarray): anchors for calculating intersection over union with bounding boxes.

    Returns:
        np.ndarray: intersection over union with bounding boxes.
    """
    w_box: np.ndarray = bbox_list[:,0]
    h_box: np.ndarray = bbox_list[:,1]
    
    w_anchor: np.ndarray = np.tile(anchors_list[:,0], (len(w_box), 1))
    h_anchor: np.ndarray = np.tile(anchors_list[:,1], (len(h_box), 1))

    w_intersection: np.ndarray = np.minimum(w_box[:, np.newaxis], w_anchor)
    h_intersection: np.ndarray = np.minimum(h_box[:, np.newaxis], h_anchor)

    area_box: np.ndarray = w_box * h_box
    area_anchor: np.ndarray = w_anchor * h_anchor
    area_intersection: np.ndarray = w_intersection * h_intersection

    iou_array: np.ndarray = area_intersection / (area_box[:, np.newaxis] + area_anchor - area_intersection)

    return iou_array
