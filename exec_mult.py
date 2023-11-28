import os
import sys
import shutil
from PIL import Image
import numpy as np
import scipy
import json
import random

def delete_folder_if_found(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def move_markers(numberMarkers, originFolder, destinyFolder, listOfMarkers):

    numbersChosen = random.sample(listOfMarkers, numberMarkers)
    for number in numbersChosen:
        auxPath = ''
        if number < 10:
            auxPath = originFolder + f'/00000{number}-seeds.txt'
        else:
            auxPath = originFolder + f'/0000{number}-seeds.txt'
        shutil.move(auxPath, destinyFolder)
    
    return numbersChosen

def duplicate_create_json():

    try:
        with open('jsonResults.json', 'r') as file_json:
            jsonResults = json.load(file_json)

        with open('jsonResultsLastRun.json', 'w') as file_json2:
            json.dump(jsonResults, file_json2)

    except:
        pass


def move_markers_back(originFolder, destinyFolder):
    files = [f for f in os.listdir(destinyFolder) if os.path.isfile(os.path.join(destinyFolder, f))]

    for file in files:
        auxPath = destinyFolder + f'/{file}'
        shutil.move(auxPath, originFolder)

    
if (len(sys.argv) != 7):
    print("python exec <P1> <P2> <P3> <P4> <P5>")
    print("P1: number of layers (if negative, do not encode layers again)")
    print("P2: layer for the results")
    print("P3: model_type (0, 1, 2)")
    print("P4: makersType (1: point, 2: line)")
    print("P5: number of models to run (looping)")
    print("P6: number of markers (int < 10)")
    exit()


listOfMarkers = [1,2,3,4,5,6,7,10,11,14,16,18,20,22,24,26,28,30,32,
                 34,35,38,40,42,44,46,48,50,52,54,56,58,60,62,64,
                 66,67,70,71,74,75,78,79,81,84,86,87,89,92,93,96]

nlayers = int(sys.argv[1])
target_layer = int(sys.argv[2])
model_type = int(sys.argv[3])
makersType = int(sys.argv[4])
numberOfLoops = int(sys.argv[5])
numberMarkers = int(sys.argv[6])

# Definição de qual marcador usar e de que folder deve-se pegar as marcações

if makersType == 1:
    originFolder = '/home/osboxes/Desktop/mo445/all_markers_point'
else:
    originFolder = '/home/osboxes/Desktop/mo445/all_markers_line'

destinyFolder = '/home/osboxes/Desktop/mo445/markers'

duplicate_create_json()

for i in range(numberOfLoops):

    
    numbersChosen = move_markers(numberMarkers, originFolder, destinyFolder, listOfMarkers)

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
    delete_folder_if_found("layer4")


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

    # Salvar resultado do treinamento
    jsonAux = {'nlayers': nlayers,
            'target_layer': target_layer,
            'model_type': model_type,
            'makers_type' : makersType,
            'markers': numbersChosen}

    with open('jsonAux.json', 'w') as file_json:
        json.dump(jsonAux, file_json)

    line = "python3 iou.py"
    os.system(line)

    with open('jsonAux.json', 'r') as file_json:
        jsonAux = json.load(file_json)

    with open('jsonResults.json', 'r') as file_json2:
        jsonResults = json.load(file_json2)

    if not jsonResults:
        print('Entrou')
        jsonResults['0'] = jsonAux

    else:
        print('Entrou2')
        maxKey = max(map(int, jsonResults.keys()))
        jsonResults[f'{maxKey+1}'] = jsonAux

    with open('jsonResults.json', 'w') as file_json:
        json.dump(jsonResults, file_json)

    move_markers_back(originFolder, destinyFolder)
    # line = "delineation salie {} objs".format(target_layer)
    # os.system(line)
