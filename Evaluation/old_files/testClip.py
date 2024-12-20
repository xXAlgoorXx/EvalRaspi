from hailoEval_utils import HailoCLIPImage, HailoCLIPText
from PIL import Image
import clip
import torch
import open_clip
from Evaluation.old_files.hailoEval_image import CLIPHailoInference
from hailoEval_utils import HailoCLIPImage,RestOfGraph
from PIL import Image
from utils import loadJson,transform
from time import time
import numpy as np


gemmLayerjson_path = "Evaluation/models/RN101/gemmLayer_CLIP_RN101.json"
hef_path = "Evaluation/models/RN101/CLIP_RN101.hef"
onnx_path = "Evaluation/resources/temp/modified_TinyCLIP-ResNet-19M_Multioutput.onnx"
testImg_path = "Evaluation/testImg/pexels-mikebirdy-170811.jpg"
batch_size = 1
img = Image.open(testImg_path)

hailo_old = HailoCLIPImage(hef_path,gemmLayerjson_path)

start = time()
preprop_img = hailo_old.performPreprocess(img).unsqueeze(0)
img_encoded = hailo_old.encode_image(preprop_img)
print(f"Process time:{time()-start}")

hailo_inference = CLIPHailoInference(hef_path, gemmLayerjson_path, batch_size,
                                     input_type = "FLOAT32", output_type = {"CLIP_RN101/matmul2":"FLOAT32"})

start = time()
img_encoded2 = hailo_inference.encode_image(img)
print(f"Process time:{time()-start}")

print(f"Diff:{img_encoded-img_encoded2}")