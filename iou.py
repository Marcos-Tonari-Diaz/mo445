import os
import numpy as np
from sklearn.metrics import jaccard_score
from PIL import Image


def calculate_mean_iou(results_arr, truelabels_arr):
    iou_arr = []
    for result, truelabel in zip(results_arr, truelabels_arr):
        #print(result.shape)
        #print(truelabel.shape)
        iou_arr.append(jaccard_score(truelabel, result, zero_division=1.0))
    return np.mean(iou_arr)


def image_to_label_array(path):
    image = Image.open(path)
    image_array = np.asarray(image)
    # convert rgb to bw
    if (len(image_array.shape) == 3):
        image_sum = image_array.sum(axis=2)
        if (image_sum.max() != 0):
            image_array = image_sum/image_sum.max()
        else:
            image_array = image_sum
    else:
        image_array = image_array / 255
    #print(image_array.shape)
    #image_array = image_array.astype(int)

    return image_array.ravel()


def calculate_detection_iou():
    results_arr = []
    truelabels_arr = []
    results_folder = "detection_comp"
    truelabels_folder = "truelabels"
    for img_file in os.listdir(results_folder):
        results_arr.append(image_to_label_array(results_folder+"/"+img_file))
    for img_file in os.listdir(truelabels_folder):
        truelabels_arr.append(image_to_label_array(truelabels_folder+"/"+img_file))
    #print(results_arr[0].shape)
    #print(truelabels_arr[0].shape)
    #print([el.shape for el in truelabels_arr])
    return calculate_mean_iou(results_arr, truelabels_arr)
    #return calculate_mean_iou(truelabels_arr, truelabels_arr)

mean_iou = calculate_detection_iou()
print("detection mean IOU: {}".format(mean_iou))