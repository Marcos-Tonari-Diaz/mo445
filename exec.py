import os
import sys
import shutil
from PIL import Image
import numpy as np
import scipy


def calculate_mean_iou(results_arr, truelabels_arr):
    iou_arr = []
    for result, truelabel in zip(results_arr, truelabels_arr):
        iou_arr.append(scipy.metrics.jaccard_score(truelabel, result))
    return np.mean(iou_arr)


def image_to_1darray(path):
    image = Image.open(path)
    image_array = np.asarray(image)
    return image_array.ravel()


def calculate_detection_iou():
    results_arr = []
    truelabels_arr = []
    for img_file in os.listdir("detection_comp"):
        results_arr.append(image_to_1darray(img_file))
    for img_file in os.listdir("truelabels"):
        truelabels_arr.append(image_to_1darray(img_file))
    return calculate_mean_iou(results_arr, truelabels_arr)


def delete_folder_if_found(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


if (len(sys.argv) != 4):
    print("python exec <P1> <P2>")
    print("P1: number of layers (if negative, do not encode layers again)")
    print("P2: layer for the results")
    print("P3: model_type (0, 1, 2)")
    exit()

# Teste de exclusão de folders

delete_folder_if_found("bag")
delete_folder_if_found("boxes")
delete_folder_if_found("filtered")
delete_folder_if_found("flim")
delete_folder_if_found("salie")
delete_folder_if_found("layer0")
delete_folder_if_found("layer1")
delete_folder_if_found("layer2")
delete_folder_if_found("layer3")


nlayers = int(sys.argv[1])
target_layer = int(sys.argv[2])
model_type = int(sys.argv[3])

os.system("preproc images 1.5 filtered")
npts_per_marker = 1
line = "bag_of_feature_points filtered markers {} bag".format(npts_per_marker)
os.system(line)

for layer in range(1, nlayers+1):
    line = "create_layer_model bag arch.json {} flim".format(layer)
    extract_line = "extract layer{}".format(layer)
    os.system(line)
    if (model_type == 0):
        line = "encode_layer arch.json {} flim".format(layer)
        os.system(line)
        os.system(extract_line)
    else:
        line = "merge_layer_models arch.json {} flim".format(layer)
        os.system(line)
        line = "encode_merged_layer arch.json {} flim".format(layer)
        os.system(line)
        os.system(extract_line)

line = "decode_layer {} arch.json flim {} salie".format(
    target_layer, model_type)
extract_line = "extract layer{}".format(target_layer)
os.system(line)
os.system(extract_line)

line = "detection salie {} boxes".format(target_layer)
os.system(line)
mean_iou = calculate_detection_iou()
print("detection mean IOU: {}".format(mean_iou))

# line = "delineation salie {} objs".format(target_layer)
# os.system(line)
