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
from cpuEval_utils import get_max_class_with_threshold,get_trueClass,find_majority_element,printAndSaveHeatmap
from hailoEval_utils import get_pred_hailo,evalModel_hailo, HailoCLIPImage, HailoCLIPText, getCorrespondingGemm, getCorrespondingTextEmb, getModelfiles
from cpuEval_utils import get_max_class_with_threshold,get_trueClass,find_majority_element,printAndSaveHeatmap

# Pathes
outputPath = Path("Evaluation/Data/Hailo")
Dataset5Patch224px = Path("candolle_5patches_224px")
Dataset5Patch = Path("candolle_5patch")
tinyClipModels = Path("tinyClipModels")
models_path = "Evaluation/models"
hefFolder_path = "Evaluation/resources/hef" # get from for loop
postprocess_path = "Evaluation/resources/json/gemm_layer_RN50x4.json"
textEmbeddings_path = "Evaluation/resources/json/textEmbeddings_RN50x4.json"
calibrationData_path = Path("calibrationData")
calibrationOutout_Path = Path("Evaluation/Data/Hailo/calibration")

def getMaxThreshold(df_5patch):
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

        # iterate through the input array in chunks of 5
        for i in range(0, len(y_test), 5):

            patches = y_test[i:i+5]
            majority_element = find_majority_element(patches)
            y_test_s.append(majority_element)

            patches = y_pred[i:i+5]
            majority_element = find_majority_element(patches)
            majority_pred.append(majority_element)

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
    
def printAndSaveTh(th_dict, model, outputfolder, use_5_Scentens = False):
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
        figname = f'Threshold Outdoor/Indoor 5S({model})'
        saveName = f"Threshold_5S_({model})"
    else:
        figname = f'Threshold Outdoor/Indoor ({model})'
        saveName = f"Threshold_({model})"

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
    plt.savefig(calibrationOutout_Path / saveName, dpi = 600, bbox_inches='tight')
    print(f"Saved figure {saveName} at {outputPath}")
    
    plt.clf()

def getxImagesPerClass(dataFolder,x):
    """
    Images in output dict is randomly shuffled
    """
    # define second degree classes 
    in_arch = [7,10,18,27,29,32,36,1,28,6,33,40,30,31,24]#[7,18,31]#
    out_constr_res = [8,16,22]#[16]#
    in_constr_res = [9,13,39,12] #[12]#
    out_urb = [2,20,38,26,15,42,44,4,23]#[15,2,23]#
    out_forest = [17]
    
    in_arch_list = []
    out_constr_res_list = []
    in_constr_res_list = []
    out_urb_list = []
    out_forest_list = []
    
    files = os.listdir(dataFolder)
    for file in files:
        classNumber = int(file.split("_")[1])
        if classNumber in in_arch:
            in_arch_list.append(file)
            continue
        if classNumber in out_constr_res:
            out_constr_res_list.append(file)
            continue
        if classNumber in in_constr_res:
            in_constr_res_list.append(file)
            continue
        if classNumber in out_urb:
            out_urb_list.append(file)
            continue
        if classNumber in out_forest:
            out_forest_list.append(file)
            continue
        print(f"{file} no match")    
    
    random.shuffle(in_arch_list)
    random.shuffle(out_constr_res_list)
    random.shuffle(in_constr_res_list)
    random.shuffle(out_urb_list)
    random.shuffle(out_forest_list)
    
    in_arch_list = in_arch_list[:x]
    out_constr_res_list = out_constr_res_list[:x]
    in_constr_res_list = in_constr_res_list[:x]
    out_urb_list = out_urb_list[:x]
    out_forest_list = out_forest_list[:x]
    
    imagedict = {"in_arch":in_arch_list,
                 "out_constr_res":out_constr_res_list,
                 "in_constr_res":in_constr_res_list,
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
        image = hailoModelImage.performPreprocess(Image.open(image_path)).unsqueeze(0)
        
        ### FIRST DEGREE ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,1,False)

        score_in = probs[0][0]
        score_out = probs[0][1]

        ### SECOND DEGREE (in) ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,2,False)

        score_in_arch = (probs[0][0] + probs[0][1] + probs[0][2] + probs[0][3] + probs[0][4] + probs[0][5] + probs[0][6]) 
        score_in_constr = (probs[0][7] + probs[0][8] + probs[0][9]) 

        ### SECOND DEGREE (out) ###
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,3,False)
            
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

def main():
    models_list = next(os.walk(models_path), (None, [], None))[1]
    json_path = calibrationData_path / "calibData.json"
    
    if os.path.exists(json_path):
        calibData = loadJson(json_path)["data"]
    else:
        calibData = getCalbirationData(Dataset5Patch,40)
        calibsave= {"data":calibData}
        saveJson(calibsave,str(json_path))
    
    # Only TinyClip
    # models_list = [model for model in models_list if "Tiny" in model]
    for model_folder in models_list:
        folder_path = models_path + "/" + model_folder
        gemm_path, hef_path, textemb_path, textemb5S_path = getModelfiles(folder_path)
        modelname = model_folder
        print(f"Model name:{modelname}")
        print(f"Gemm path:{gemm_path}")
        print(f"Emb path:{textemb_path}")
        hailoImagemodel = HailoCLIPImage(hef_path,gemm_path)
        hailoTextmodel = HailoCLIPText(textemb_path)
        
        use5Scentens = hailoTextmodel.getuse5Scentens()

        # Path to csv
        if use5Scentens:
            csv_path_calib = calibrationData_path / f'calib_{modelname}_5patches_5scentens.csv'
        else:
            csv_path_calib = calibrationData_path / f'calib_{modelname}_5patches.csv'

        # check if csv already exists
        if os.path.exists(csv_path_calib):
            df_5patch = get_trueClass(pd.read_csv(csv_path_calib))
        else:                
            df_calib = getCalibrationPrediction(calibData,hailoModelImage=hailoImagemodel,hailoModelText=hailoTextmodel,use5Scentens = False)
            df_calib.to_csv(csv_path_calib, index=False)
            df_5patch = get_trueClass(pd.read_csv(csv_path_calib))
        df = df_5patch.copy()
        maxThdict = getMaxThreshold(df)
        printAndSaveTh(maxThdict,modelname,outputPath)

    
if __name__ == "__main__":
    main()