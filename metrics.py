import os
import numpy as np
from sklearn.metrics import jaccard_score
from scipy.spatial import distance
from PIL import Image
import json

def calculate_mean_iou(results_arr, truelabels_arr):
    iou_arr = []
    for result, truelabel in zip(results_arr, truelabels_arr):
        #print("iou: " + str(jaccard_score(truelabel, result)))
        # print(len(truelabels_arr))
        # print(len(results_arr))
        # print(result.shape)
        # print(truelabel.shape)
        # discard the case where the prediction and truelabel are both empty
        if not (sum(result) == 0 and sum(truelabel) == 0):
            iou_arr.append(jaccard_score(truelabel, result, zero_division=1.0))
    return np.mean(iou_arr)


def calculate_mean_dice(results_arr, truelabels_arr):
    jaccard = calculate_mean_iou(results_arr, truelabels_arr)
    return 2*jaccard / (1 + jaccard)


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
    # print(image_array.shape)
    # image_array = image_array.astype(int)
    return image_array.ravel()


def load_arrays_from_images():
    results_arr = []
    truelabels_arr = []
    results_folder = "detection_comp"
    truelabels_folder = "truelabels"
    for img_file in sorted(os.listdir(results_folder)):
        #print(img_file)
        results_arr.append(image_to_label_array(results_folder+"/"+img_file))
    for img_file in sorted(os.listdir(truelabels_folder)):
        #print(img_file)
        truelabels_arr.append(image_to_label_array(
            truelabels_folder+"/"+img_file))
    return results_arr, truelabels_arr

def load_specified_arrays():
    results_arr = []
    truelabels_arr = []
    results_folder = "detection_comp"
    truelabels_folder = "truelabels"
    results_arr.append(image_to_label_array(results_folder+"/000005_layer3.png"))
    results_arr.append(image_to_label_array(results_folder+"/000030_layer3.png"))
    results_arr.append(image_to_label_array(results_folder+"/000054_layer3.png"))
    results_arr.append(image_to_label_array(results_folder+"/000058_layer3.png"))
    truelabels_arr.append(image_to_label_array(truelabels_folder+"/000005.png"))
    truelabels_arr.append(image_to_label_array(truelabels_folder+"/000030.png"))
    truelabels_arr.append(image_to_label_array(truelabels_folder+"/000054.png"))
    truelabels_arr.append(image_to_label_array(truelabels_folder+"/000058.png"))
    return results_arr, truelabels_arr


def calculate_metrics():
    results_arr, truelabels_arr = load_arrays_from_images()
    #results_arr, truelabels_arr = load_specified_arrays()
    mean_iou = calculate_mean_iou(results_arr, truelabels_arr)
    mean_dice = calculate_mean_dice(results_arr, truelabels_arr)
    return mean_iou, mean_dice


mean_iou, mean_dice = calculate_metrics()

with open('jsonAux.json', 'r') as file_json:
    jsonAux = json.load(file_json)

jsonAux['IoU'] = mean_iou
jsonAux['DICE'] = mean_dice

with open('jsonAux.json', 'w') as file_json:
    json.dump(jsonAux, file_json)

print("detection mean IOU: {}, DICE: {}".format(mean_iou, mean_dice))
