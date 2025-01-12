import numpy as np
import onnxruntime as ort
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,HailoSchedulingAlgorithm,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
from loguru import logger
from functools import partial
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
from utils import loadJson,transform,ThroughputMetric
from typing import List, Generator, Optional, Tuple, Dict
import queue
import onnx

def getModelfiles(folder_path):
    files = os.listdir(folder_path)
    gemm_path = None
    hef_path = None
    textemb_path = None
    textemb5S_path = None
    for file in files:
        stem = file.split("/")[-1]
        
        if stem.split(".")[-1] == "json":
            filename = stem.split(".")[0]
            if "gemmLayer" in filename:
                gemm_path = folder_path + "/" + file
                continue
            
            if "_5S" in filename:
                textemb5S_path = folder_path + "/" + file
                continue
            
            else:
                textemb_path = folder_path + "/" + file
                continue
  
        if stem.split(".")[-1] == "hef":
            hef_path = folder_path + "/" + file
            continue
        
        if "RestOf" in stem.split(".")[0]:
            OnnxPostp_path = folder_path + "/" + file
            continue
            
    return gemm_path, hef_path, textemb_path, textemb5S_path,OnnxPostp_path

def getCorrespondingGemm(modelName):
    """
    Get gemm layer json path
    """
    gemmFolder_path = "Evaluation/resources/gemmLayer"
    gemmFile = "str"
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
    textFile = "str"
    textFolder_path = "Evaluation/resources/TextEmbeddings"
    filenames = next(walk(textFolder_path), (None, None, []))[2]
    modelName = modelName.split(".")[0]
    if "TinyCLIP" not in modelName:
        modelName = modelName.split("_")[1]
        
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
    files = files

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

def get_throughput_hailo(input_folder,hailoModelText,hailoModelImage,use5Scentens = False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
    metric = ThroughputMetric()
    # List all files in the input folder
    files = os.listdir(input_folder)

    # use less files for faster computing
    files = files[:100]

    # Loop through each file
    for file in tqdm(files,desc="Files",position=1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Read the image
        image_path = os.path.join(input_folder, file)
        image = hailoModelImage.performPreprocess(Image.open(image_path)).unsqueeze(0).numpy()
        
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

def evalModel_hailo(model_i,model_t,image,level):
    
    # Encode image and text features separately
    image_features = model_i.encode_image(image)
    text_features = model_t.encode_text(level)
    probs = model_t.clalcProbs(image_features,text_features,level)

    return probs

def get_throughput_hailo_image(input_folder,hailoModelText,hailoModelImage,use5Scentens = False):
    '''
    Function that calculates the probability that each image belongs to each class
    In: path of the image folder, tokenized text prompts 
    Out: dataframe with the probability scores for each image
    '''
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
        image = hailoModelImage.performPreprocess(Image.open(image_path)).unsqueeze(0).numpy()

        ### FIRST DEGREE ###
        probs = evalModel_hailo_image(hailoModelImage,hailoModelText,image,1,metric)

        ### SECOND DEGREE (in) ###
        probs = evalModel_hailo_image(hailoModelImage,hailoModelText,image,2,metric)

        ### SECOND DEGREE (out) ###
        probs = evalModel_hailo_image(hailoModelImage,hailoModelText,image,3,metric)
    
    return metric.compute()


def evalModel_hailo_image(model_i,model_t,image,level,metric,use_5_Scentens = False):
    
    # Encode image and text features separately
    ts = time.monotonic()
    image_features = model_i.encode_image(image)
    elapsed_time = time.monotonic() - ts
    metric.update(1, elapsed_time)
    text_features = model_t.encode_text(level)
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
        if self.use5Scentens:
            name = self.json_path.split("/")[-1]
            name = name.split(".")[0]
            name  = name.split("_")[1]
        else:
            name = self.json_path.split("/")[-1]
            name = name.split(".")[0]
            name  = name.split("_")[1:]
        return name

    def encode_text(self,level):
        if level == 1:
            return self.getEmbeddingsLvl1()
        
        if level == 2:
            return self.getEmbeddingsLvl2()
        
        if level == 3:
            return self.getEmbeddingsLvl3()

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

    def getLabelsLvl1(self):
        return(list(self.embeddingsLvl1["Discription"]))

    def getLabelsLvl2(self):
        return(list(self.embeddingsLvl2["Discription"]))

    def getLabelsLvl3(self):
        return(list(self.embeddingsLvl3["Discription"]))
    
    def getuse5Scentens(self):
        return self.use5Scentens

    def clalcProbs(self,image_features,text_features,level):
        text_features  = torch.from_numpy(text_features).type(torch.float32)
        image_features = torch.from_numpy(image_features).type(torch.float32)
        # Normalize features for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity (scaled by 100)
        logits_per_image = 100.0 * (image_features @ text_features.T)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy() # new axis so its same output like CLIP

        # uncomment when processing 5 sentences
        if self.use5Scentens and level != 1:
            new_size = probs.shape[1] // 5
            probs = np.array([probs[0, i:i+5].sum() for i in range(0, probs.shape[1], 5)]).reshape(1, new_size)

        return probs

class HailoCLIPImage:
    def __init__(self,hef_path,postprocessjson_path = None):
        self.hef = HEF(hef_path)
        self.isTinyClip = False
        if "TinyCLIP" in hef_path:
            self.isTinyClip = True
        self.postprocess = None
        if postprocessjson_path != None:
            self.postprocess = RestOfGraph(postprocessjson_path)
        input_info = (self.hef.get_input_vstream_infos()[0]).shape[0]
        self.preprocess = transform(input_info)
            
    def loadPostprocess(self,postprocessjson_path):
        post_process = RestOfGraph(postprocessjson_path)
        self.postprocess = post_process
        
    def onlyHEF(self,image):
        result = self._run_inference(image)
        result = np.array(list(result.values())[0]).squeeze()
        return result
        
    def encode_image(self,image):
        """
        Same like clip model:
        model.encode_image(image)
        """
        result = self._run_inference(image)
        result = np.array(list(result.values())[0]).squeeze()
        if self.isTinyClip:
            result = result[0,:]
        posP_image = self.postprocess(result)
        return posP_image
    
    def performPostprocess(self,input):
        """
        calculation of the cut off graph
        """
        return self.postprocess(input)
    
    def performPreprocess(self,input):
        """
        Like preprocess from:
        model,preprocess = clip.load(model)
        """
        return self.preprocess(input)
    
    def _run_inference(self,image):
        """
        Inferance code for Hailo
        """
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
                format_type = FormatType.FLOAT32
            )
            
            output_params = OutputVStreamParams.make(
                network_group, 
                format_type = FormatType.FLOAT32
            )
            
            # Prepare input data
            input_info = self.hef.get_input_vstream_infos()[0]
            dataset = image.astype(np.float32,order='F')
            dataset = np.transpose(dataset, (0,2, 3, 1)).astype(np.float32,order='F')
            
            # Run inference
            with network_group.activate():
                with InferVStreams(network_group, input_params, output_params) as pipeline:
                    results = pipeline.infer({input_info.name: np.ascontiguousarray(dataset)})
                    return results

class HailofastCLIPImage:
    """
    Initialises target on creation means the resourcess are acquierd on creation of object.
    """
    def __init__(
        self, hef_path: str, postprocess_Path, batch_size: int = 1,
        send_original_frame: bool = False, is_data_batched:bool = False,
        isnewCut:bool = False) -> None:

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        params = VDevice.create_params()    
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        
        self.outputname = self.infer_model.output_names
        output_type = "FLOAT32"
        output_type = {self.outputname[0]:output_type} # for easyier handling of the output types
        input_type = "FLOAT32"
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.isTinyClip = False
        if "TinyCLIP" in hef_path:
            self.isTinyClip = True
        self.output_type = output_type
        self.send_original_frame = send_original_frame
        self.is_data_batched = is_data_batched
        
        self.preprocess = transform(self.get_input_shape()[0])
        if postprocess_Path != None:
            if "json" in postprocess_Path.lower():
                self.postprocess = RestOfGraph(postprocess_Path)
                self.jsonPostprocess = True
            else:
                self.postprocess = RestOfGraphOnnx(postprocess_Path,isnewCut)
                self.jsonPostprocess = False
                
    def __del__(self):
        """
        Destructor to free device
        """
        print("\nRelease Device")
        self.target.release()
        
    def loadPostprocess(self,postprocessjson_path):
        post_process = RestOfGraph(postprocessjson_path)
        self.postprocess = post_process
    
    def encode_image(self,image):
        """
        image: image as np array
        """
        # image = self.performPreprocess(image).unsqueeze(0).numpy()
        image = np.transpose(image, (0,2, 3, 1)).astype(np.float32,order='F')
        image = np.expand_dims(image.flatten(), axis=0)
        self.input_queue.put(image)
        self.run()
        result = self.output_queue.get()[1] # output is from queue is (input,ouput)
        if self.isTinyClip and self.jsonPostprocess:
            result = result.squeeze(0)[0,:]
        imageEncoding = self.performPostprocess(result.squeeze())
        return imageEncoding
    
    def onlyHEF(self,image):
        """
        Give data direct to model without preprocess
        """
        self.input_queue.put(image)
        self.run()
        result = self.output_queue.get()[1]
        return result
    
    def performPreprocess(self,input):
        """
        Like preprocess from:
        model,preprocess = clip.load(model)
        """
        return self.preprocess(input)
    
    def performPostprocess(self,input):
        """
        calculation of the cut off graph
        """
        return self.postprocess(input)
    
    def callback(
        self, completion_info, bindings_list: list, input_batch: list,
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                # If the model has a single output, return the output buffer. 
                # Else, return a dictionary of output buffers, where the keys are the output names.
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                self.output_queue.put((input_batch[i], result))

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input
        
    def run(self,metric = None) -> None:
        with self.infer_model.configure() as configured_infer_model:
            
            batch_data = self.input_queue.get()
            preprocessed_batch = batch_data

            bindings_list = []
            
            # if data is batched create bindings for every batch
            if self.is_data_batched:
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)
            else:
                bindings = self._create_bindings(configured_infer_model)
                bindings.input().set_buffer(np.array(preprocessed_batch))
                bindings_list.append(bindings)

            configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            job = configured_infer_model.run_async(
                bindings_list, partial(
                    self.callback,
                    input_batch=preprocessed_batch,
                    bindings_list=bindings_list
                )
            )
            job.wait(10000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )
        
    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(self.outputname[0]).set_format_type(
                getattr(FormatType, output_type)
            )
      
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
        if result.shape[0] == 50:
            result = result[0,:]
        return result

class RestOfGraphOnnx:
    """
    GemmLayer which got cut off
    """
    def __init__(self,onnx_path,isnewCut=False):
        self.isnewCut = isnewCut
        self.session = ort.InferenceSession(onnx_path)
        model = onnx.load(onnx_path)
        output =[node.name for node in model.graph.output]

        self.input_name = [node.name for node in model.graph.input][0]
        
        print('Inputs: ', self.input_name)
        
    def __call__(self,input):
        input = np.array(list(input)).squeeze()
        if input.ndim == 1:
            input = input[np.newaxis,:]
        elif input.ndim == 2 and self.isnewCut == True:
            input = input[:,np.newaxis,:]
        result = self.session.run(None, {self.input_name: input})
        result = np.array(result).squeeze(0)
        return result

if __name__ == "__main__":
    from PIL import Image
    import clip
    import torch
    import open_clip
    
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
    gemmLayerjson_path = "Evaluation/models/TinyClip19M/gemmLayer_TinyCLIP-ResNet-19M.json"
    hef_path = 'Evaluation/models/TinyClip19M/TinyCLIP-ResNet-19M.hef'
    onnx_path = "Evaluation/models/TinyClip19M/TinyCLIP-ResNet-19M.onnx"
    model_name = "TinyCLIP-ResNet-19M-Text-19M"
    hailoInference = HailoCLIPImage(hef_path,gemmLayerjson_path)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=f"tinyClipModels/{model_name}-LAION400M.pt"
    )
    
    names2 = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]
    
    names2 = ["car","dog","cat","house","wrench","Blue"]
    
    # Example usage
    image_path = "Evaluation/testImg/pexels-mikebirdy-170811.jpg"
    image_pre = Image.open(image_path)
    image = preprocess_val(image_pre).unsqueeze(0).to(device)
    #image = hailoInference.performPreprocess(image_pre).unsqueeze(0)
    tokenizer = open_clip.get_tokenizer(model_name)
    text_inputs = tokenizer(names2).to(device)
     
    # CLIP
    with torch.no_grad():
        imageEmb_clip = model.encode_image(image).detach().numpy().squeeze()
        text_features = model.encode_text(text_inputs)
    
    # ONNX
    session = ort.InferenceSession(onnx_path)
    imageEmb_onnx = np.array(session.run(None, {"input": image.numpy()})).squeeze()
    #imageEmb_onnx = hailoInference.performPostprocess(np.array(imageEmb_onnx).flatten())
    
    # HAILO
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
    

    