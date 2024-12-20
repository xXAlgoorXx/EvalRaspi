import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torch
import time
import pandas as pd
from torcheval.metrics import Throughput
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn import Softmax
import matplotlib.pyplot as plt
import seaborn as sns
import clip
import open_clip
from utils import ThroughputMetric

def get_modelnames():
    # Evaluate every Resnet model
    resnetModels = []
    for clipmodel in clip.available_models():
        if "RN" in clipmodel:
            resnetModels.append(clipmodel)

    print("Open Clip Models:", open_clip.list_models())
    for clipmodel in open_clip.list_models():
        if "ResNet" in clipmodel and "Tiny" in clipmodel:
            resnetModels.append(clipmodel)
    return resnetModels


# def get_pred2(input_folder, text1, text2, text3, preprocess, model, use5Scentens=False):
#     '''
#     Function that calculates the probability that each image belongs to each class
#     In: path of the image folder, tokenized text prompts 
#     Out: dataframe with the probability scores for each image
#     '''
#     model.eval()
#     # List all files in the input folder
#     files = os.listdir(input_folder)
#     text = torch.cat((text1, text2, text3), 0)
#     startNames2 = len(text1)
#     startNames3 = len(text1) + len(text2)
#     in_list = []
#     out_list = []
#     in_arch_list = []
#     in_constr_list = []
#     out_constr_list = []
#     out_urb_list = []
#     out_for_list = []
#     scene_list = []
#     img_list = []
#     # Loop through each file
#     for file in tqdm(files, desc="Files", position=1):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         # Read the image
#         image_path = os.path.join(input_folder, file)
#         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#         ts = time.monotonic()

#         probs = evalModel(model, image, text, False)

#         score_in = probs[0][0]

#         score_out = probs[0][1]

#         score_in_arch = (probs[0][startNames2 + 0]
#                          + probs[0][startNames2 + 1]
#                          + probs[0][startNames2 + 2]
#                          + probs[0][startNames2 + 3]
#                          + probs[0][startNames2 + 4]
#                          + probs[0][startNames2 + 5]
#                          + probs[0][startNames2 + 6])

#         score_in_constr = (probs[0][startNames2 + 7]
#                            + probs[0][startNames2 + 8]
#                            + probs[0][startNames2 + 9])

#         score_out_constr = probs[0][startNames3]

#         score_out_urb = (probs[0][startNames3 + 1]
#                          + probs[0][startNames3 + 2]
#                          + probs[0][startNames3 + 3]
#                          + probs[0][startNames3 + 4]
#                          + probs[0][startNames3 + 5])

#         score_out_for = probs[0][startNames3 + 6]

#         in_list.append(score_in)
#         out_list.append(score_out)
#         in_arch_list.append(score_in_arch)
#         in_constr_list.append(score_in_constr)
#         out_constr_list.append(score_out_constr)
#         out_urb_list.append(score_out_urb)
#         out_for_list.append(score_out_for)
#         scene_list.append(int(os.path.basename(
#             file).split('.')[0].split('_')[1]))
#         img_list.append(int(os.path.basename(
#             file).split('.')[0].split('_')[2]))

#     df_pred = pd.DataFrame({'Scene': scene_list,
#                             'Image': img_list,
#                             'In_Arch': in_arch_list,
#                             'In_Constr': in_constr_list,
#                             'Out_Constr': out_constr_list,
#                             'Out_Urban': out_urb_list,
#                             'Forest': out_for_list,
#                             'In': in_list,
#                             'Out': out_list})
#     return df_pred


def get_pred(input_folder, text1, text2, text3, preprocess, model, use5Scentens=False):
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
    for file in tqdm(files, desc="Files", position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        ts = time.monotonic()
        ### FIRST DEGREE ###
        text = text1

        probs = evalModel(model, image, text, False)

        score_in = probs[0][0]
        score_out = probs[0][1]

        ### SECOND DEGREE (in) ###
        text = text2
        probs = evalModel(model, image, text, use5Scentens)

        score_in_arch = (probs[0][0]
                         + probs[0][1]
                         + probs[0][2]
                         + probs[0][3]
                         + probs[0][4]
                         + probs[0][5]
                         + probs[0][6])

        score_in_constr = (probs[0][7]
                           + probs[0][8]
                           + probs[0][9])

        ### SECOND DEGREE (out) ###
        text = text3
        probs = evalModel(model, image, text, use5Scentens)

        score_out_constr = probs[0][0]

        score_out_urb = (probs[0][1]
                         + probs[0][2]
                         + probs[0][3]
                         + probs[0][4]
                         + probs[0][5])

        score_out_for = probs[0][6]

        in_list.append(score_in)
        out_list.append(score_out)
        in_arch_list.append(score_in_arch)
        in_constr_list.append(score_in_constr)
        out_constr_list.append(score_out_constr)
        out_urb_list.append(score_out_urb)
        out_for_list.append(score_out_for)
        scene_list.append(int(os.path.basename(
            file).split('.')[0].split('_')[1]))
        img_list.append(int(os.path.basename(
            file).split('.')[0].split('_')[2]))

    df_pred = pd.DataFrame({'Scene': scene_list,
                            'Image': img_list,
                            'In_Arch': in_arch_list,
                            'In_Constr': in_constr_list,
                            'Out_Constr': out_constr_list,
                            'Out_Urban': out_urb_list,
                            'Forest': out_for_list,
                            'In': in_list,
                            'Out': out_list})
    return df_pred


def get_throughput(input_folder, text1, text2, text3, preprocess, model):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    model.eval()
    metric = ThroughputMetric()
    # List all files in the input folder
    files = os.listdir(input_folder)

    # use less files for faster computing
    files = files[0:100]

    # Loop through each file
    for file in tqdm(files, desc="Files", position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        ### FIRST DEGREE ###
        ts = time.monotonic()
        text = text1
        probs = evalModel(model, image, text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (in) ###
        ts = time.monotonic()
        text = text2
        probs = evalModel(model, image, text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (out) ###
        ts = time.monotonic()
        text = text3
        probs = evalModel(model, image, text)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

    return metric.compute()


def evalModel(model, image, text, use_5_Scentens=False):
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

        # uncomment when processing 5 sentences
        if use_5_Scentens:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum()
                             for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

    return probs


def get_throughput_image(input_folder, text1, text2, text3, preprocess, model):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    model.eval()
    metric = ThroughputMetric()
    # List all files in the input folder
    files = os.listdir(input_folder)

    # use less files for faster computing
    files = files[0:100]

    # Loop through each file
    for file in tqdm(files, desc="Files", position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Read the image
        image_path = os.path.join(input_folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        ### FIRST DEGREE ###
        text = text1
        probs = evalModel_image(model, image, text, metric)
        ### SECOND DEGREE (in) ###
        text = text2
        probs = evalModel_image(model, image, text, metric)

        ### SECOND DEGREE (out) ###
        text = text3
        probs = evalModel_image(model, image, text, metric)

    return metric.compute()


def evalModel_image(model, image, text, metric, use_5_Scentens=False,):
    with torch.no_grad():
        # Encode image and text features separately
        ts = time.monotonic()
        image_features = model.encode_image(image)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)
        text_features = model.encode_text(text)

        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (scaled by 100)
        logits_per_image = 100.0 * (image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # uncomment when processing 5 sentences
        if use_5_Scentens:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum()
                             for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

    return probs

