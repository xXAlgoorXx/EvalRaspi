import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from utils import loadJson,saveJson
from hailoEval_utils import get_pred_hailo,evalModel_hailo, HailoCLIPImage, HailoCLIPText, getCorrespondingGemm, getCorrespondingTextEmb, getModelfiles,HailofastCLIPImage
from utils import get_max_class_with_threshold,get_trueClass,find_majority_element,printAndSaveHeatmap



def getMaxThresholdInOut(df_5patch):
    step = 0.01
    th_list = np.arange(0.0,1.0 + step,step)
    th_dict = {}
    for th in th_list:
        df = df_5patch.copy()
        df['y_predIO'] = df.apply(get_max_class_with_threshold, axis=1, threshold=th)

        # set the outdoor classes to 0 when the image was classified as indoor 
        # set the indoor classes to 0 when the image was classified as outdoor 
        df.loc[df['y_predIO'] == 'In', ['Out_Constr', 'Out_Urban', 'Forest']] = 0 
        df.loc[df['y_predIO'] == 'Out', ['In_Arch', 'In_Constr']] = 0

        # create the new column y_predIO
        columns = ['In_Arch', 'In_Constr', 'Out_Constr', 'Out_Urban', 'Forest']
        df['y_pred'] = df[columns].idxmax(axis=1)

        # evaluate performance of model
        y_test = df['ClassTrue']
        y_pred = df['y_pred']

        # majority counts
        y_test_s = []
        majority_pred = []

        # # iterate through the input array in chunks of 5
        # for i in range(0, len(y_test), 5):

        #     patches = y_test[i:i+5]
        #     majority_element = find_majority_element(patches)
        #     y_test_s.append(majority_element)

        #     patches = y_pred[i:i+5]
        #     majority_element = find_majority_element(patches)
        #     majority_pred.append(majority_element)

        # majority counts
        y_test_s = y_test
        majority_pred = y_pred

        
        # conpute indoor/outdoor classification accuracy score
        replacements = {
            "In_Arch": "In",
            "In_Constr": "In",
            "Forest": "Out",
            "Out_Constr": "Out",
            "Out_Urban": "Out"
        }

        IO_pred = [replacements.get(item, item) for item in majority_pred]
        IO_true = [replacements.get(item, item) for item in y_test_s]

        accuracy = accuracy_score(IO_true, IO_pred)
        th_dict[th] = accuracy
        
    return th_dict

def getMaxThresholdOutdoor(df_5patch):
    step = 0.01
    th_list = np.arange(0.0,1.0 + step,step)
    th_dict = {}
    for th in th_list:
        df = df_5patch.copy()
        df['y_predOutdoor'] = df.apply(get_max_class_with_threshold_outdoor, axis=1, threshold=th)

        # evaluate performance of model
        y_test = df['ClassTrue']
        y_pred = df['y_predOutdoor']

        # majority counts
        y_test_s = y_test
        majority_pred = y_pred

        IO_pred = [item for item in majority_pred]
        IO_true = [item for item in y_test_s]

        accuracy = accuracy_score(IO_true, IO_pred)
        th_dict[th] = accuracy

    return th_dict

def getMaxThresholdIndoor(df_5patch):
    step = 0.01
    th_list = np.arange(0.0,1.0 + step,step)
    th_dict = {}
    for th in th_list:
        df = df_5patch.copy()
        df['y_predIndoor'] = df.apply(get_max_class_with_threshold_indoor, axis=1, threshold=th)

        # evaluate performance of model
        y_test = df['ClassTrue']
        y_pred = df['y_predIndoor']

        # majority counts
        y_test_s = y_test
        majority_pred = y_pred

        IO_pred = [item for item in majority_pred]
        IO_true = [item for item in y_test_s]

        accuracy = accuracy_score(IO_true, IO_pred)
        th_dict[th] = accuracy
        
    return th_dict

def printAndSaveTh(th_dict, model, outputfolder,name, use_5_Scentens = False):
    th_list = []
    acc_list = []
    max_th = 0
    max_acc = 0
    for key,value in th_dict.items():
        th_list.append(key)
        acc_list.append(value)
        if value > max_acc:
            max_acc = value
            max_th = key
        
    if use_5_Scentens:
        figname = f'Threshold {name} 5S({model})'
        saveName = f"Threshold_5S_{name}({model})"
    else:
        figname = f'Threshold {name} ({model})'
        saveName = f"Threshold_{name}({model})"

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     "font.size": 12
    # })

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.plot(th_list,acc_list)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(max_th,max_acc,'ro') 
    plt.text(0.1, 0.1, f'max acc: {max_acc:.3}\nth: {max_th:.3}', bbox={
        'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    plt.title(figname)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(outputfolder / saveName, dpi = 600, bbox_inches='tight')
    print(f"Saved figure {saveName} at {outputfolder}")
    
    plt.clf()

def getxImagesPerClass(dataFolder,x):
    """
    Images in output dict is randomly shuffled
    """
    # define second degree classes 
    in_arch = [7,10,18,27,29,32,36,1,28,6,33,40,30,31,24]#[7,18,31]#
    out_constr = [8,16,22]#[16]#
    in_constr = [9,13,39,12] #[12]#
    out_urb = [2,20,38,26,15,42,44,4,23]#[15,2,23]#
    out_forest = [17]
    
    in_arch_list = []
    out_constr_list = []
    in_constr_list = []
    out_urb_list = []
    out_forest_list = []
    
    files = os.listdir(dataFolder)
    for file in files:
        classNumber = int(file.split("_")[1])
        if classNumber in in_arch:
            in_arch_list.append(file)
            continue
        if classNumber in out_constr:
            out_constr_list.append(file)
            continue
        if classNumber in in_constr:
            in_constr_list.append(file)
            continue
        if classNumber in out_urb:
            out_urb_list.append(file)
            continue
        if classNumber in out_forest:
            out_forest_list.append(file)
            continue
        print(f"{file} no match")    
    
    random.shuffle(in_arch_list)
    random.shuffle(out_constr_list)
    random.shuffle(in_constr_list)
    random.shuffle(out_urb_list)
    random.shuffle(out_forest_list)
    
    in_arch_list = in_arch_list[:x]
    out_constr_list = out_constr_list[:x]
    in_constr_list = in_constr_list[:x]
    out_urb_list = out_urb_list[:x]
    out_forest_list = out_forest_list[:x]
    
    imagedict = {"in_arch":in_arch_list,
                 "out_constr":out_constr_list,
                 "in_constr":in_constr_list,
                 "out_urb":out_urb_list,
                 "out_forest":out_forest_list
                 }
    
    return imagedict

def getCalbirationData(dataFolder,x):
    """
    Get calibration data
    """
    imagedict = getxImagesPerClass(dataFolder,x)
    calibData = []
    for value in imagedict.values():
        calibData.append(value)
    calibData = [item for imageList in calibData for item in imageList]#flatten list
    return calibData

def getCalibrationPrediction(calibdata,hailoModelText,hailoModelImage,use5Scentens = False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''

    in_list = []
    out_list = []
    in_arch_list = []
    in_constr_list = []
    out_constr_list = []
    out_urb_list = []
    out_for_list = []
    scene_list = []
    img_list = []
    
    # Loop through each file
    for file in tqdm(calibdata,desc="Calibration data",position=1):
        
        # Read the image
        image_path = os.path.join(str(Dataset5Patch), file)
        image = hailoModelImage.performPreprocess(Image.open(image_path)).unsqueeze(0).numpy()
        
        ### FIRST DEGREE ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,1)

        score_in = probs[0][0]
        score_out = probs[0][1]

        ### SECOND DEGREE (in) ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,2)

        score_in_arch = (probs[0][0] + probs[0][1] + probs[0][2] + probs[0][3] + probs[0][4] + probs[0][5] + probs[0][6]) 
        score_in_constr = (probs[0][7] + probs[0][8] + probs[0][9]) 

        ### SECOND DEGREE (out) ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,3)
            
        score_out_constr = probs[0][0] 
        score_out_urb = (probs[0][1] + probs[0][2] + probs[0][3] + probs[0][4] + probs[0][5]) 
        score_out_for = probs[0][6] 

        in_list.append(score_in)
        out_list.append(score_out)
        in_arch_list.append(score_in_arch)
        in_constr_list.append(score_in_constr)
        out_constr_list.append(score_out_constr)
        out_urb_list.append(score_out_urb)
        out_for_list.append(score_out_for)
        scene_list.append(int(os.path.basename(file).split('.')[0].split('_')[1]))
        img_list.append(int(os.path.basename(file).split('.')[0].split('_')[2]))

    df_pred = pd.DataFrame({'Scene': scene_list, 'Image': img_list, 'In_Arch': in_arch_list, 'In_Constr': in_constr_list, 'Out_Constr': out_constr_list, 'Out_Urban': out_urb_list, 'Forest': out_for_list, 'In': in_list, 'Out': out_list})
    return df_pred

def get_max_class_with_threshold_indoor(row, threshold):
    in_prob = row['In_Arch']

    # if 'In' probability is greater than the threshold, classify as 'In'
    if in_prob > threshold:
        return 'In_Arch'
    else:
        return 'In_Constr'

def get_max_class_with_threshold_outdoor(row, threshold):
    Out_Constr_prob = row["Out_Constr"]
    Out_Urban_prob = row["Out_Urban"]
    Forest_prob = row["Forest"]
    
    if Out_Constr_prob > Out_Urban_prob:
        if Forest_prob > threshold:
            return "Forest"
        else:
            return "Out_Constr"
    else:
        if Forest_prob > threshold:
            return "Forest"
        else:
            return "Out_Constr"
            
    # # Check
    # if Forest_prob > threshold:
    #     return "Forest"
    # elif Out_Constr_prob > Out_Urban_prob:
    #     return "Out_Constr"
    # else:
    #     return "Out_Urban"
    
    # # Check which probability is above the threshold and return the corresponding class
    # if Out_Constr_prob > threshold and Out_Constr_prob >= max(Out_Urban_prob, Forest_prob):
    #     return "Out_Constr"
    # elif Out_Urban_prob > threshold and Out_Urban_prob >= max(Out_Constr_prob, Forest_prob):
    #     return "Out_Urban"
    # elif Forest_prob > threshold:
    #     return "Forest"
    # else:
    #     return "Below_Threshold"

def main():
    models_list = next(os.walk(models_path), (None, [], None))[1]
    
    json_path = calibrationData_path / "calibData.json"
    
    if os.path.exists(json_path):
        calibData = loadJson(json_path)
    else:
        calibData = getxImagesPerClass(Dataset5Patch,40)
        saveJson(calibData,str(json_path))
        
    allClassData = np.array([item for item in calibData.values()]).flatten()
    indoor_in_arch = calibData["in_arch"]
    indoor_in_constr = calibData["in_constr"]
    indoorData = np.array(indoor_in_arch + indoor_in_constr).flatten()
    outdoor_out_urb = calibData["out_urb"]
    outdoor_out_constr = calibData["out_constr"]
    outdoor_out_forest = calibData["out_forest"]
    outdoorData = np.array(outdoor_out_urb+outdoor_out_constr+outdoor_out_forest).flatten()
    
    
    # models_list = [model for model in models_list if "Tiny" in model]
    for model_folder in models_list:
        folder_path = models_path + "/" + model_folder
        gemm_path, hef_path, textemb_path, textemb5S_path,OnnxPostp_path = getModelfiles(folder_path)
        emb_path = textemb5S_path
        modelname = model_folder
        print(f"Model name:{modelname}")
        print(f"Gemm path:{OnnxPostp_path}")
        print(f"Emb path:{emb_path}")
        hailoImagemodel = HailofastCLIPImage(hef_path,OnnxPostp_path)
        hailoTextmodel = HailoCLIPText(emb_path)
        
        use5Scentens = hailoTextmodel.getuse5Scentens()

        # check if csv already exists
        
        print("Outdoor/Indoor")
        df_calibAll = getCalibrationPrediction(allClassData,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
        df_calibAll = get_trueClass(df_calibAll)
        
        print("Indoor")
        df_calibIn = getCalibrationPrediction(indoorData,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
        df_calibIn = get_trueClass(df_calibIn)
        
        print("Outdoor")
        df_calibOut = getCalibrationPrediction(outdoorData,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
        df_calibOut = get_trueClass(df_calibOut)
        
        del hailoImagemodel
        
        # Indoor / Outdoor
        df = df_calibAll.copy()
        maxThdict_all = getMaxThresholdInOut(df)
        printAndSaveTh(maxThdict_all,modelname,outputPath_InOut,"InOut",use_5_Scentens = use5Scentens)
        
        # Indoor
        df = df_calibIn.copy()
        maxThdict_in = getMaxThresholdIndoor(df)
        printAndSaveTh(maxThdict_in,modelname,outputPath_Indoor,"Indoor",use_5_Scentens = use5Scentens)
        
        # Outdoor       
        df = df_calibOut.copy()
        maxThdict_out = getMaxThresholdOutdoor(df)
        printAndSaveTh(maxThdict_out,modelname,outputPath_Outdoor,"Outdoor",use_5_Scentens = use5Scentens)


if __name__ == "__main__":
    # Pathes
    outputPath_InOut = Path("Evaluation/Data/Hailo/calibration/Indoor_outdoor")
    outputPath_Indoor = Path("Evaluation/Data/Hailo/calibration/Indoor")
    outputPath_Outdoor = Path("Evaluation/Data/Hailo/calibration/Outdoor")
    Dataset5Patch224px = Path("candolle_5patches_224px")
    Dataset5Patch = Path("candolle_5patch")
    tinyClipModels = Path("tinyClipModels")
    models_path = "Evaluation/models"
    hefFolder_path = "Evaluation/resources/hef" # get from for loop
    postprocess_path = "Evaluation/resources/json/gemm_layer_RN50x4.json"
    textEmbeddings_path = "Evaluation/resources/json/textEmbeddings_RN50x4.json"
    calibrationData_path = Path("calibrationData")
    calibrationOutout_Path = Path("Evaluation/Data/Hailo/calibration")
    main()