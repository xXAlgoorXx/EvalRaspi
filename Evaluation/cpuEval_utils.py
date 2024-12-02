import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from collections import Counter
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as T
from torcheval.metrics import Throughput

def get_pred(input_folder, text1, text2, text3, preprocess,model,use5Scentens = False):

    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    model.eval()
    # List all files in the input folder
    files = os.listdir(input_folder)

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
    for file in tqdm(files,desc="Files",position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        ts = time.monotonic()
        ### FIRST DEGREE ###
        text = text1
        
        probs = evalModel(model,image,text,False)

        score_in = probs[0][0]
        score_out = probs[0][1]

        ### SECOND DEGREE (in) ###
        text = text2
        probs = evalModel(model,image,text,use5Scentens)

        score_in_arch = (probs[0][0] + probs[0][1] + probs[0][2] + probs[0][3] + probs[0][4] + probs[0][5] + probs[0][6]) 
        score_in_constr = (probs[0][7] + probs[0][8] + probs[0][9]) 

        ### SECOND DEGREE (out) ###
        text = text3
        probs = evalModel(model,image,text,use5Scentens)
            
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

def get_throughput(input_folder, text1, text2, text3, preprocess,model):

    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    model.eval()
    metric = Throughput()
    # List all files in the input folder
    files = os.listdir(input_folder)

    # use less files for faster computing
    files = files[0:100]

    # Loop through each file
    for file in tqdm(files,desc="Files",position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        ### FIRST DEGREE ###
        ts = time.monotonic()
        text = text1
        probs = evalModel(model,image,text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (in) ###
        ts = time.monotonic()
        text = text2
        probs = evalModel(model,image,text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (out) ###
        ts = time.monotonic()
        text = text3
        probs = evalModel(model,image,text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

    
    return metric.compute()


def evalModel(model,image,text,use_5_Scentens = False):
    with torch.no_grad():
            # Encode image and text features separately
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            # Normalize features for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (scaled by 100)
            logits_per_image = 100.0 * (image_features @ text_features.T)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            
            if use_5_Scentens:
                new_size = probs.shape[1] // 5
                probs = np.array([probs[0, i:i+5].sum() for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)


    return probs

def get_trueClass(df):
    '''
    Function reads and stores the image features and true class in a dataframe
    In: Dataframe with the class probability
    Out: Dataframe with an additional column 'ClassTrue' 
    '''
    df['Image'] = df['Scene'].astype(str) + '_' + df['Image'].astype(str)


    # define second degree classes 
    in_arch = [7,10,18,27,29,32,36,1,28,6,33,40,30,31,24]#[7,18,31]#
    out_constr_res = [8,16,22]#[16]#
    in_constr_res = [9,13,39,12] #[12]#
    out_urb = [2,20,38,26,15,42,44,4,23]#[15,2,23]#
    out_forest = [17]


    # add second degree classes
    df['ClassTrue'] = np.select(
        [
            df['Scene'].isin(in_arch),
            df['Scene'].isin(out_constr_res),
            df['Scene'].isin(in_constr_res),
            df['Scene'].isin(out_urb),
            df['Scene'].isin(out_forest)
        ],
        ['In_Arch','Out_Constr', 'In_Constr','Out_Urban','Forest'],
        default='Other'
    )

    df.drop(df[df['ClassTrue'] == 'Other'].index, inplace=True)
    df.drop('Scene', axis=1, inplace=True)

    # uncomment for evaluation only on test set
    #df.drop(df[df['Image'].isin(['17_3', '17_4', '17_5', '17_6', '17_7', '17_8', '17_9', '17_10', '17_11', 
    #                         '17_12', '17_13', '17_14', '17_15', '17_16', '17_17', '17_18', '17_19', 
    #                         '17_20', '17_21', '17_22', '17_23', '17_24'])].index, inplace=True)


    return df

    # create the new column y_predIO with the prediction of level 1 classes
def get_max_class_with_threshold(row, threshold):
    in_prob = row['In']
    
    # if 'In' probability is greater than the threshold, classify as 'In'
    if in_prob > threshold:
        return 'In'
    else:
        return 'Out'
    
'''
Majority voting for the 5 patches dataset
'''
# function to find the majority element in a list
def find_majority_element(lst):
    count = Counter(lst)
    return count.most_common(1)[0][0]

def printAndSaveHeatmap(df, model, outputfolder, use_5_Scentens = False):
    # define specific sequence of class labels
    class_sequence = ['In_Arch', 'In_Constr', 'Out_Constr', 'Forest', 'Out_Urban']

    label_encoder = LabelEncoder()
    df['Class3new_encoded'] = label_encoder.fit_transform(df['ClassTrue'])
    df['y_pred_encoded'] = label_encoder.transform(df['y_pred'])
    label_encoder.fit(class_sequence)
    # compute the confusion matrix
    cm = confusion_matrix(df['Class3new_encoded'], df['y_pred_encoded'])

    # get the indices of class_sequence
    class_indices = label_encoder.transform(class_sequence)

    # reorder the confusion matrix
    cm_ordered = cm[np.ix_(class_indices, class_indices)]
    if use_5_Scentens:
        figname = f'Confusion Matrix 5S({model})'
    else:
        figname = f'Confusion Matrix ({model})'

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Times New Roman",
    #     "font.size": 12
    # })

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ordered, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_sequence, yticklabels=class_sequence)

    plt.title(figname)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(outputfolder / figname,dpi = 600, bbox_inches='tight')
    plt.clf()

