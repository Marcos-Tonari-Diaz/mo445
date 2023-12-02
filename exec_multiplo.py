import os
import sys
import shutil
import numpy as np
import scipy
import json
    
def delete_folder_if_found(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)


def move_markers(originFolder, destinyFolder, listOfMarkers):

    for number in listOfMarkers:
        auxPath = ''
        if number < 10:
            auxPath = originFolder + f'/00000{number}-seeds.txt'
        else:
            auxPath = originFolder + f'/0000{number}-seeds.txt'
        shutil.move(auxPath, destinyFolder)
    
    return listOfMarkers

def duplicate_create_json():

    try:
        with open('jsonResults.json', 'r') as file_json:
            jsonResults = json.load(file_json)

        with open('jsonResultsLastRun.json', 'w') as file_json2:
            json.dump(jsonResults, file_json2)

        jsonResults = {}
        with open('jsonResults.json', 'w') as file_json3:
            json.dump(jsonResults, file_json3)

    except:
        pass

if (len(sys.argv) != 4):
    print("python exec <P1> <P2>")
    print("P1: number of layers (if negative, do not encode layers again)")
    print("P2: layer for the results")
    print("P3: model_type (0, 1, 2)")
    exit()

delete_folder_if_found("filtered")

'''listOfMarkers = [1,2,3,4,5,6,7,10,11,14,16,18,20,22,24,26,28,30,32,
                 34,35,38,40,42,44,46,48,50,52,54,56,58,60,62,64,
                 66,67,70,71,74,75,78,79,81,84,86,87,89,92,93,96]'''

listOfMarkers = [1,2,3,4,5,6,7,10,11,14,16,18,20,22,24,28,32,
                 34,35,38,40,44,46,48,50,52,54,56,58,60,62,64,
                 66,67,70,71,74,75,78,79,81,84,86,87,89,92,93,96]

markersAlreadyChosen = [26, 30, 42]

originFolder = '/home/osboxes/Desktop/Antigo/mo445/all_markers_line'
destinyFolder = '/home/osboxes/Desktop/Antigo/mo445/markers'

nlayers      = int(sys.argv[1])
target_layer = int(sys.argv[2])
model_type   = int(sys.argv[3])

os.system("preproc images 1.5 filtered")

duplicate_create_json()

for number in listOfMarkers:
    # Teste de exclusão de folders
    delete_folder_if_found("bag")
    delete_folder_if_found("boxes")
    delete_folder_if_found("flim")
    delete_folder_if_found("salie")
    delete_folder_if_found("layer0")
    delete_folder_if_found("layer1")
    delete_folder_if_found("layer2")
    delete_folder_if_found("layer3")
    delete_folder_if_found("detection_comp")

    numbersChosen = []
    numbersChosen.append(number)

    numbersChosen = move_markers(originFolder, destinyFolder, numbersChosen)

    numbersChosen += markersAlreadyChosen

    npts_per_marker = 1
    line = "bag_of_feature_points filtered markers {} bag".format(npts_per_marker)
    os.system(line)

    for layer in range(1,nlayers+1):
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

    line = "decode_layer {} arch.json flim {} salie".format(target_layer, model_type)
    extract_line = "extract layer{}".format(target_layer)
    os.system(line)
    os.system(extract_line)

    line = "detection salie {} boxes".format(target_layer)
    os.system(line)

    # Salvar resultado do treinamento
    jsonAux = {'nlayers': nlayers,
            'target_layer': target_layer,
            'model_type': model_type,
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
        jsonResults['0'] = jsonAux

    else:
        maxKey = max(map(int, jsonResults.keys()))
        jsonResults[f'{maxKey+1}'] = jsonAux

    with open('jsonResults.json', 'w') as file_json:
        json.dump(jsonResults, file_json)

    numbersChosen = move_markers(destinyFolder, originFolder, [number])

# line = "delineation salie {} objs".format(target_layer)
# os.system(line)
        
