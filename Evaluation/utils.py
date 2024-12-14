from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, PILToTensor
import torch
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn import Softmax
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import json

IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')

class ThroughputMetric:
    def __init__(self):
        self.data = []

    def update(self, num_processed, time):
        """
        iterations per second
        """
        self.data.append(num_processed / time)

    def getMean(self):
        return np.mean(self.data)

    def getStd(self):
        return np.std(self.data)

    def compute(self):
        return self.getMean(), self.getStd()


def get_trueClass(df):
    '''
    Function reads and stores the image features and true class in a dataframe
    In: Dataframe with the class probability
    Out: Dataframe with an additional column 'ClassTrue' 
    '''
    df['Image'] = df['Scene'].astype(str) + '_' + df['Image'].astype(str)

    # define second degree classes
    in_arch = [7, 10, 18, 27, 29, 32, 36, 1,
               28, 6, 33, 40, 30, 31, 24]  # [7,18,31]#
    out_constr_res = [8, 16, 22]  # [16]#
    in_constr_res = [9, 13, 39, 12]  # [12]#
    out_urb = [2, 20, 38, 26, 15, 42, 44, 4, 23]  # [15,2,23]#
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
        ['In_Arch', 'Out_Constr', 'In_Constr', 'Out_Urban', 'Forest'],
        default='Other'
    )

    df.drop(df[df['ClassTrue'] == 'Other'].index, inplace=True)
    df.drop('Scene', axis=1, inplace=True)

    # uncomment for evaluation only on test set
    # df.drop(df[df['Image'].isin(['17_3', '17_4', '17_5', '17_6', '17_7', '17_8', '17_9', '17_10', '17_11',
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


def get_max_class_with_threshold2(row, threshold):
    in_prob = row['In']
    out_prob = row['Out']
    softmax = Softmax(dim=-1)
    in_prob, out_prob = softmax(torch.tensor([in_prob, out_prob]))

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


def printAndSaveHeatmap(df, model, outputfolder, use_5_Scentens=False):
    # define specific sequence of class labels
    class_sequence = ['In_Arch', 'In_Constr',
                      'Out_Constr', 'Forest', 'Out_Urban']

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
    plt.savefig(outputfolder / figname, dpi=600, bbox_inches='tight')
    plt.clf()

def loadJson(path):
    # Open and read the JSON file
    with open(path, 'r') as file:
        data = json.load(file)

    return data

def saveJson(data,path):
    with open(path, 'w') as f:
        json.dump(data, f)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px):
    """
    n_px: input resolution of the network
    """
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def printClassificationReport(df,model):
    # define specific sequence of class labels
    class_sequence = ['In_Constr', 'In_Arch', 'Out_Constr', 'Forest', 'Out_Urban']

    label_encoder = LabelEncoder()
    df['Class3new_encoded'] = label_encoder.fit_transform(df['ClassTrue'])
    df['y_pred_encoded'] = label_encoder.transform(df['y_pred'])
    label_encoder.fit(class_sequence)
    # compute the confusion matrix
    y_true = df['Class3new_encoded']
    y_pred = df['y_pred_encoded']
    print(f"== {model} ==")
    print(classification_report(y_true, y_pred,zero_division=0, target_names=class_sequence))
    # precision, recall, fscore, support = precision_recall_fscore_support(y_true,y_pred)
    
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    return classification_report(y_true, y_pred,zero_division=0, target_names=class_sequence,output_dict=True)

def printAndSaveClassReport(classReportDict, modelname, outputFolder):
    # Ensure outputFolder is a Path object
    outputFolder = Path(outputFolder)
    outputFolder.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    
    supportLabels = ["In_Arch","In_Constr","Forest","Out_Constr","Out_Urban"]
    # Data preparation
    x = np.arange(len(classReportDict))  # Positions for class groups
    grouplabels = list(classReportDict.keys())
    width = 1.0/(4 + 1) # Width of each bar
    multiplier = 0
    
    supportAmount = 0
    for label in supportLabels:
        classDict = classReportDict[label]
        supportAmount += classDict.get('support', 0)
    
    # Create bars
    for label in next(iter(classReportDict.values())).keys():
        measurements = [values[label] * 100/ supportAmount if label == 'support' else values[label] * 100 for values in classReportDict.values()]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=label)
        ax.bar_label(rects, fmt='%.1f', padding=3) 
        multiplier += 1

    # Add labels, title, and legend
    ax.set_ylabel("Performance (%)")
    ax.set_title(f'Class Report - {modelname}')
    ax.set_xticks(x + (multiplier - 1) * width / 2, grouplabels)
    ax.legend(loc='lower right', ncols=1)
    ax.set_ylim(0, 110)
    ax.set_yticks([0,20,40,60,80,100])
    
    # Save the figure
    figname = f"ClassReport_{modelname}.png"
    fig.savefig(outputFolder / figname, dpi=600, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free resources
