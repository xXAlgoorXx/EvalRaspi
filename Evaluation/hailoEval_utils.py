import numpy as np
import onnxruntime as ort
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
)
from pathlib import Path
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time
import re
from PIL import Image
import torch
from torcheval.metrics import Throughput
from os import walk
from utils import loadJson,transform

def getCorrespondingGemm(modelName):
    """
    Get gemm layer json path
    """
    gemmFolder_path = "Evaluation/resources/gemmLayer"

    filenames = next(walk(gemmFolder_path), (None, None, []))[2]
    modelName = modelName.split(".")[0]
    for filename in filenames:
        if modelName in filename:
            gemmFile = filename
    gemmFile_path = gemmFolder_path + "/" + gemmFile
    
    return gemmFile_path

def getCorrespondingTextEmb(modelName):
    """
    Get text embeddings json path
    """
    textFolder_path = "Evaluation/resources/TextEmbeddings"
    filenames = next(walk(textFolder_path), (None, None, []))[2]
    modelName = modelName.split(".")[0]
    for filename in filenames:
        if modelName in filename:
            textFile = filename
    textFile_path = textFolder_path + "/" + textFile
    
    return textFile_path
    
def get_pred_hailo(input_folder,hailoModelText,hailoModelImage,use5Scentens = False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
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
        
        # Read the image
        image_path = os.path.join(input_folder, file)
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

def get_throughput_hailo(input_folder,hailoModelText,hailoModelImage,use5Scentens = False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
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
        image = hailoModelImage.performPreprocess(Image.open(image_path)).unsqueeze(0)
        
        ### FIRST DEGREE ###
        ts = time.monotonic()
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,1)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (in) ###
        ts = time.monotonic()
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,2)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

        ### SECOND DEGREE (out) ###
        ts = time.monotonic()
        probs = evalModel_hailo(hailoModelImage,hailoModelText,image,3)
        elapsed_time = time.monotonic() - ts
        metric.update(1, elapsed_time)

    
    return metric.compute()

def evalModel_hailo(model_i,model_t,image,level,use_5_Scentens = False):
    
    # Encode image and text features separately
    image_features = model_i.encode_image(image)
    text_features = model_t.encode_text(level,use_5_Scentens)
    probs = model_t.clalcProbs(image_features,text_features,level)

    return probs

class HailoCLIPText:
    
    def __init__(self,json_path: str):
        """
        Textembeddings
        inputsize
        """
        self.json_path = json_path
        self.loadTextembeddings(json_path)
        
    def getModelName(self):
        match = re.search(r"_([A-Za-z0-9]+)\.json$", self.json_path)
        if match:
            name = match.group(1)
            return name

    def encode_text(self,level,use5Scentens):
        if level == 1:
            return self.getEmbeddingsLvl1(use5Scentens)
        
        if level == 2:
            return self.getEmbeddingsLvl2(use5Scentens)
        
        if level == 3:
            return self.getEmbeddingsLvl3(use5Scentens)

    def loadTextembeddings(self,json_path:str):
        self.allEmbeddings = loadJson(json_path)
        self.embeddingsLvl1 = self.allEmbeddings["1"]
        self.embeddingsLvl2 = self.allEmbeddings["2"]
        self.embeddingsLvl3 = self.allEmbeddings["3"]
        self.use5Scentens = self.allEmbeddings["Use5Scentens"]

    def getEmbeddingsLvl1(self):
        return(np.array(self.embeddingsLvl1["embeddings"]))

    def getEmbeddingsLvl2(self):
        return(np.array(self.embeddingsLvl2["embeddings"]))

    def getEmbeddingsLvl3(self):
        return(np.array(self.embeddingsLvl3["embeddings"]))
    
    def getEmbeddingsLvl1(self,use5Scentens):
        return(np.array(self.embeddingsLvl1["embeddings"]))

    def getEmbeddingsLvl2(self,use5Scentens):
        return(np.array(self.embeddingsLvl2["embeddings"]))

    def getEmbeddingsLvl3(self,use5Scentens):
        return(np.array(self.embeddingsLvl3["embeddings"]))

    def getLabelsLvl1(self):
        return(list(self.embeddingsLvl1["Discription"]))

    def getLabelsLvl2(self):
        return(list(self.embeddingsLvl2["Discription"]))

    def getLabelsLvl3(self):
        return(list(self.embeddingsLvl3["Discription"]))
    
    def getuse5Scentens(self):
        return self.use5Scentens
    
    # def calclvl1Probs(self,image_features):
    #     return self._clalcProbs(image_features,self.getEmbeddingsLvl1())

    # def calclvl2Probs(self,image_features):
    #     return self._clalcProbs(image_features,self.getEmbeddingsLvl2())

    # def calclvl3Probs(self,image_features):
    #     return self._clalcProbs(image_features,self.getEmbeddingsLvl3())

    def clalcProbs(self,image_features,text_features,level):
        text_features  = torch.from_numpy(text_features)
        image_features = torch.from_numpy(image_features)
        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity (scaled by 100)
        logits_per_image = 100.0 * (image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[np.newaxis,:] # new axis so its same out like CLIP

        # uncomment when processing 5 sentences
        if self.use5Scentens and level != 1:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum() for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

        return probs

class HailoCLIPImage:
    def __init__(self,hef_path,postprocessjson_path = None):
        self.hef = HEF(hef_path)
        self.postprocess = None
        if postprocessjson_path != None:
            self.postprocess = RestOfGraph(postprocessjson_path)
        input_info = (self.hef.get_input_vstream_infos()[0]).shape[0]
        self.preprocess = transform(input_info)
            
    def loadPostprocess(self,postprocessjson_path):
        post_process = RestOfGraph(postprocessjson_path)
        self.postprocess = post_process
        
    def encode_image(self,image):
        """
        Same like clip model:
        model.encode_image(image)
        """
        result = self._run_inference(image)
        result = np.array(list(result.values())[0]).squeeze()
        posP_image = self.postprocess(result)
        return posP_image
    
    def performPostprocess(self,input):
        """
        calculation of teh cut off graph
        """
        return self.postprocess(input)
    
    def performPreprocess(self,input):
        """
        Like preprocess from:
        model,preprocess = clip.load(model)
        """
        return self.preprocess(input)
    
    def _run_inference(self,image):
        # Setup device and configure model
        with VDevice() as device:
            
            # Configure network
            config_params = ConfigureParams.create_from_hef(
                hef=self.hef, 
                interface=HailoStreamInterface.PCIe
            )
            configuration = device.configure(self.hef, config_params)
            network_group = configuration[0]
            
            # Setup streams
            input_params = InputVStreamParams.make(
                network_group, 
                format_type=FormatType.FLOAT32
            )
            
            output_params = OutputVStreamParams.make(
                network_group, 
                format_type=FormatType.FLOAT32
            )
            
            # Prepare input data
            input_info = self.hef.get_input_vstream_infos()[0]
            dataset = image.detach().numpy().astype(np.float32,order='F')
            dataset = np.transpose(dataset, (0,2, 3, 1)).astype(np.float32,order='F')
            
            # Run inference
            with network_group.activate():
                with InferVStreams(network_group, input_params, output_params) as pipeline:
                    results = pipeline.infer({input_info.name: np.ascontiguousarray(dataset)})
                    return results
                      

class RestOfGraph:
    """
    GemmLayer which got cut off
    """
    
    def __init__(self,weightJson_path):
        self.json = loadJson(weightJson_path)
        self.bias = np.array(self.json["bias"])
        self.weight = np.array(self.json["weights"])
        
    def __call__(self,input):
        # input = np.array(list(input.values())[0]).squeeze()
        result = np.dot(input, self.weight.T) + self.bias
        return result



if __name__ == "__main__":
    from PIL import Image
    import clip
    import torch
    
    def printEval(image_features, text_features):
        image_features = torch.Tensor(image_features)
        text_features = torch.Tensor(text_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(5)
        # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{names2[index]:>16s}: {100 * value.item():.2f}%")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gemmLayerjson_path = "Evaluation/resources/json/gemm_layer_RN50x4.json"
    hef_path = 'Evaluation/resources/hef/Modified_RN50x4.hef'
    onnx_path = "Evaluation/resources/onnx/modified_RN50x4_simple.onnx"
    
    hailoInference = HailoCLIPImage(hef_path,gemmLayerjson_path)
    model, preprocess = clip.load("RN50x4", device=device)
    names2 = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]
    names2 = ["car","dog","cat","house","wrench","Blue"]
     
    # Example usage
    image_path = "Evaluation/testImg/pexels-mikebirdy-170811.jpg"
    image_pre = Image.open(image_path)
    image = preprocess(image_pre).unsqueeze(0).to(device)
    image = hailoInference.performPreprocess(image_pre).unsqueeze(0)
    text_inputs = clip.tokenize(names2).to(device)
     
    #CLIP
    with torch.no_grad():
        imageEmb_clip = model.encode_image(image).detach().numpy().squeeze()
        text_features = model.encode_text(text_inputs)
    
    #ONNX
    session = ort.InferenceSession(onnx_path)
    imageEmb_onnx = session.run(None, {"onnx::Cast_0": image.numpy()})
    imageEmb_onnx = hailoInference.performPostprocess(np.array(imageEmb_onnx).flatten())
    
    #HAILO
    imageEmb_hailo = hailoInference.encode_image(image)
    
    # Print results
    print("\n=== CLIP ===")
    print(f"Shape: {imageEmb_clip.shape}")
    printEval(imageEmb_clip,text_features)
    
    print("\n=== ONNX ===")
    print(f"Shape: {imageEmb_onnx.shape}")
    diff = imageEmb_onnx - imageEmb_clip
    print(f"Mean diff to CLIP: {np.mean(diff)}")
    print(f"Var diff to CLIP: {np.var(diff)}")
    printEval(imageEmb_onnx,text_features)
    
    print("\n=== Hailo ===")
    print(f"Shape: {imageEmb_hailo.shape}")
    diff = imageEmb_onnx - imageEmb_hailo
    print(f"Mean diff to CLIP: {np.mean(diff)}")
    print(f"Var diff to CLIP: {np.var(diff)}")
    printEval(imageEmb_hailo,text_features)
    

    