import pandas as pd
import numpy as np
import os
import sys
import clip
from PIL import Image
from hailoEval_utils import get_pred_hailo,get_throughput_hailo, HailoCLIPImage, HailoCLIPText,getModelfiles,get_throughput_hailo_image,HailofastCLIPImage

"""
Check output from HEF file
"""

# Own modules
cwd = os.getcwd()
newPath = cwd + "/python"
print(newPath)
sys.path.append(newPath)
import matplotlib.pyplot as plt
folder_path = "Evaluation/models/RN50"
model_name = "RN50"

inLabels = ["architectural", "office", "residential", "school", "manufacturing",
              "cellar", "laboratory", "construction site", "mining", "tunnel"]

outlabels = ["construction site", "town", "city",
              "country side", "alley", "parking lot", "forest"]


names2 = ["architectural", "office", "residential", "school", "manufacturing",
            "cellar", "laboratory", "construction site", "mining", "tunnel"]
names3 = ["construction site", "town", "city",
            "country side", "alley", "parking lot", "forest"]

gemm_path, hef_path, textemb_path, textemb5S_path, OnnxPostp_path = getModelfiles(folder_path)

# set text embeddings and post process
textEmb = textemb_path
postP = OnnxPostp_path
modelname = folder_path
print(f"Model name:{modelname}")
print(f"Postprocess path:{postP}")
print(f"Emb path:{textEmb}")

hailoImagemodel = HailofastCLIPImage(hef_path, postP,isnewCut=False)
hailoTextmodel = HailoCLIPText(textEmb)
level = 3
outlabels = hailoTextmodel.getLabelsLvl3()

# Read the image
image_path = os.path.join("Evaluation/testImg/panorama_00002_0014_2.jpg")
image = hailoImagemodel.performPreprocess(Image.open(image_path)).unsqueeze(0).numpy()
image_features = hailoImagemodel.encode_image(image)
text_features = hailoTextmodel.encode_text(level)
probs = hailoTextmodel.clalcProbs(image_features,text_features,level)

f,axes = plt.subplots(1,1,figsize = (6,6))
axes.bar(outlabels,probs[0])
axes.set_ylim([0,1])
axes.set_title("HEF")
f.tight_layout()
plt.savefig("Evaluation/testImg/"+ f"compareProbs_{model_name}_hailo", dpi = 600)
plt.show()
np.save("Evaluation/testImg/HEFprobs",probs)