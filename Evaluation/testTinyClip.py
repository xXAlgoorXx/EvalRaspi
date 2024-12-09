from PIL import Image
import clip
import torch
import numpy as np
import onnxruntime as ort

from hailoEval_utils import HailoCLIPImage

gemmLayerjson_path = "Evaluation/models/TinyClip19M/gemmLayer_TinyCLIP-ResNet-19M.json"
hef_path = 'Evaluation/models/TinyClip19M/TinyCLIP-ResNet-19M.hef'
onnx_path = "Evaluation/resources/temp/modified_TinyCLIP-ResNet-19M_Multioutput.onnx"

hailoInference = HailoCLIPImage(hef_path,gemmLayerjson_path)

# Example usage
image_path = "Evaluation/testImg/pexels-mikebirdy-170811.jpg"
image_pre = Image.open(image_path)
image = hailoInference.performPreprocess(image_pre).unsqueeze(0)

# ONNX
session = ort.InferenceSession(onnx_path)

# HAILO
imageEmb_hailo = hailoInference.onlyHEF(image)
imageEmb_hailo_reshaped = np.array(imageEmb_hailo).reshape((22,50,64))

imageEmb_onnx = session.run(None, {"input": image.numpy()})
transpose_ouput = imageEmb_onnx[1]
reshape_8_ouput = imageEmb_onnx[2]
reshape_7_ouput = imageEmb_onnx[3]
Gemm_output = imageEmb_onnx[4]
ouput = imageEmb_onnx[0]

transpose_ouput = transpose_ouput.squeeze(1)

imageEmb_onnx = np.array(transpose_ouput).squeeze(0)
imageEmb_onnx = imageEmb_onnx.transpose(2,0,1,3)

print("Transpose output")
print(imageEmb_onnx-transpose_ouput)

print("Reshape 7 output")
imageEmb_onnx = imageEmb_onnx.reshape(50,-1)
print(imageEmb_onnx-reshape_7_ouput)

print("Gemm output")
onnx_ouput = hailoInference.performPostprocess(imageEmb_onnx)
print(onnx_ouput-Gemm_output)

print("Reshape 8 output")
print(reshape_8_ouput-onnx_ouput)

print("==============")
onnx_ouput = onnx_ouput[0,:]
print(ouput-onnx_ouput)